#!/usr/bin/env python3
"""
ETL Job: update_teamrankings_predictive

Fetch TeamRankings' college football predictive ratings and append a changed
snapshot. The source page is public HTML, so pull_date is part of the natural
key, but unchanged rankings are skipped to avoid redundant daily copies.
"""

from __future__ import annotations

import csv
import io
import re
import sys
from datetime import date, datetime
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import psycopg

from etl.common_config import load_config
from etl.common_http import build_retry_session
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult


STEP_NAME = "update_teamrankings_predictive"
BASE_URL = "https://www.teamrankings.com/college-football/ranking/predictive-by-other"
TABLE = "public.teamrankings_predictive_ratings"
MAP_TABLE = "public.teamrankings_team_map"
PULL_DATE_TIMEZONE = "America/New_York"
USER_AGENT = "BG.Analytics.CFB ETL/1.0 (+https://github.com/Bgillgrist/BG.Analytics.CFB)"

COLS = [
    "season",
    "pull_date",
    "rank",
    "source_team",
    "team",
    "record",
    "wins",
    "losses",
    "ties",
    "rating",
    "vs_1_10_record",
    "vs_11_25_record",
    "vs_26_40_record",
    "hi_rank",
    "lo_rank",
    "last_rank",
    "source_url",
]

TEXT_COLS = {
    "source_team",
    "team",
    "record",
    "vs_1_10_record",
    "vs_11_25_record",
    "vs_26_40_record",
    "source_url",
}
NUMERIC_OR_DATE_COLS = [c for c in COLS if c not in TEXT_COLS]
FORCE_NULL_SQL = ", ".join(NUMERIC_OR_DATE_COLS)
SNAPSHOT_COMPARISON_COLS = [
    col for col in COLS if col not in {"season", "pull_date", "source_url"}
]

COPY_SQL = f"""
COPY temp_teamrankings_predictive_ratings (
  {", ".join(COLS)}
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL ({FORCE_NULL_SQL})
)
"""

CREATE_MAP_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {MAP_TABLE} (
  teamrankings_team TEXT PRIMARY KEY,
  cfbd_team TEXT NOT NULL,
  active BOOLEAN NOT NULL DEFAULT TRUE,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE {MAP_TABLE}
  ADD COLUMN IF NOT EXISTS teamrankings_team TEXT,
  ADD COLUMN IF NOT EXISTS cfbd_team TEXT,
  ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS notes TEXT,
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE INDEX IF NOT EXISTS idx_teamrankings_team_map_cfbd
ON {MAP_TABLE} (cfbd_team);
"""

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  season INTEGER NOT NULL,
  pull_date DATE NOT NULL,
  rank INTEGER,
  source_team TEXT NOT NULL,
  team TEXT NOT NULL,
  record TEXT,
  wins INTEGER,
  losses INTEGER,
  ties INTEGER,
  rating DOUBLE PRECISION,
  vs_1_10_record TEXT,
  vs_11_25_record TEXT,
  vs_26_40_record TEXT,
  hi_rank INTEGER,
  lo_rank INTEGER,
  last_rank INTEGER,
  source_url TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE {TABLE}
  ADD COLUMN IF NOT EXISTS season INTEGER,
  ADD COLUMN IF NOT EXISTS pull_date DATE,
  ADD COLUMN IF NOT EXISTS rank INTEGER,
  ADD COLUMN IF NOT EXISTS source_team TEXT,
  ADD COLUMN IF NOT EXISTS team TEXT,
  ADD COLUMN IF NOT EXISTS record TEXT,
  ADD COLUMN IF NOT EXISTS wins INTEGER,
  ADD COLUMN IF NOT EXISTS losses INTEGER,
  ADD COLUMN IF NOT EXISTS ties INTEGER,
  ADD COLUMN IF NOT EXISTS rating DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS vs_1_10_record TEXT,
  ADD COLUMN IF NOT EXISTS vs_11_25_record TEXT,
  ADD COLUMN IF NOT EXISTS vs_26_40_record TEXT,
  ADD COLUMN IF NOT EXISTS hi_rank INTEGER,
  ADD COLUMN IF NOT EXISTS lo_rank INTEGER,
  ADD COLUMN IF NOT EXISTS last_rank INTEGER,
  ADD COLUMN IF NOT EXISTS source_url TEXT,
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_teamrankings_predictive_unique_pull
ON {TABLE} (season, pull_date, source_team);

CREATE INDEX IF NOT EXISTS idx_teamrankings_predictive_latest_lookup
ON {TABLE} (season, pull_date DESC);

CREATE INDEX IF NOT EXISTS idx_teamrankings_predictive_team_lookup
ON {TABLE} (team, season, pull_date DESC);
"""

CREATE_TEMP_TABLE_SQL = """
CREATE TEMP TABLE temp_teamrankings_predictive_ratings (
  season INTEGER NOT NULL,
  pull_date DATE NOT NULL,
  rank INTEGER,
  source_team TEXT NOT NULL,
  team TEXT NOT NULL,
  record TEXT,
  wins INTEGER,
  losses INTEGER,
  ties INTEGER,
  rating DOUBLE PRECISION,
  vs_1_10_record TEXT,
  vs_11_25_record TEXT,
  vs_26_40_record TEXT,
  hi_rank INTEGER,
  lo_rank INTEGER,
  last_rank INTEGER,
  source_url TEXT
) ON COMMIT DROP;
"""

INSERT_FROM_TEMP_SQL = f"""
INSERT INTO {TABLE} (
  {", ".join(COLS)}
)
SELECT
  {", ".join(COLS)}
FROM temp_teamrankings_predictive_ratings
ON CONFLICT (season, pull_date, source_team) DO NOTHING;
"""

HEADER_MAP = {
    "rank": "rank",
    "team": "team",
    "rating": "rating",
    "vs_1_10_record": "v1-10",
    "vs_11_25_record": "v11-25",
    "vs_26_40_record": "v26-40",
    "hi_rank": "hi",
    "lo_rank": "lo",
    "last_rank": "last",
}
TEAM_RECORD_RE = re.compile(r"^(?P<team>.+?)\s*\((?P<record>\d+\s*-\s*\d+(?:\s*-\s*\d+)?)\)\s*$")
NULL_TOKENS = {"--", "-", "N/A", "NA", "NR"}


def _current_pull_date() -> date:
    return datetime.now(ZoneInfo(PULL_DATE_TIMEZONE)).date()


def _clean_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())


def _normalize_header(value: str) -> str:
    text = _clean_text(value).casefold()
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9+-]+", "", text)


def _normalize_map_key(value: str) -> str:
    return _clean_text(value).casefold()


def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = _clean_text(str(value))
    if not text or text in NULL_TOKENS:
        return None
    text = text.replace(",", "").replace("%", "")
    return float(text)


def _parse_nullable_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(str(value))
    return None if not text or text in NULL_TOKENS else text


def _parse_int(value: Any) -> Optional[int]:
    parsed = _parse_number(value)
    if parsed is None:
        return None
    return int(parsed)


def _parse_team_cell(value: str) -> tuple[str, Optional[str], Optional[int], Optional[int], Optional[int]]:
    text = _clean_text(value)
    match = TEAM_RECORD_RE.match(text)
    if not match:
        return text, None, None, None, None

    team = _clean_text(match.group("team"))
    record = match.group("record").replace(" ", "")
    parts = [int(part) for part in record.split("-")]
    wins = parts[0] if len(parts) > 0 else None
    losses = parts[1] if len(parts) > 1 else None
    ties = parts[2] if len(parts) > 2 else 0
    return team, record, wins, losses, ties


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: Optional[list[list[str]]] = None
        self._current_row: Optional[list[str]] = None
        self._current_cell_parts: Optional[list[str]] = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "table":
            if self._table_depth == 0:
                self._current_table = []
            self._table_depth += 1
        elif self._table_depth > 0 and tag == "tr":
            self._current_row = []
        elif self._table_depth > 0 and self._current_row is not None and tag in {"td", "th"}:
            self._current_cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_cell_parts is not None:
            self._current_cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._current_cell_parts is not None and self._current_row is not None:
            self._current_row.append(_clean_text("".join(self._current_cell_parts)))
            self._current_cell_parts = None
        elif tag == "tr" and self._current_row is not None:
            if self._current_table is not None and any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
            self._current_cell_parts = None
        elif tag == "table" and self._table_depth > 0:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_table is not None:
                self.tables.append(self._current_table)
                self._current_table = None


def _header_mapping(row: list[str]) -> Optional[dict[str, int]]:
    normalized = [_normalize_header(cell) for cell in row]
    mapping: dict[str, int] = {}
    for output_col, header_label in HEADER_MAP.items():
        if header_label not in normalized:
            return None
        mapping[output_col] = normalized.index(header_label)
    return mapping


def _extract_rankings_rows(html_text: str) -> tuple[list[list[str]], dict[str, int]]:
    parser = _HTMLTableParser()
    parser.feed(html_text)

    for table in parser.tables:
        for row_idx, row in enumerate(table):
            mapping = _header_mapping(row)
            if mapping is not None:
                return table[row_idx + 1 :], mapping

    raise ValueError("Could not find TeamRankings predictive ratings table in HTML.")


def _parse_predictive_html(
    html_text: str,
    season: int,
    pull_date: date,
    source_url: str,
) -> List[Dict[str, Any]]:
    table_rows, mapping = _extract_rankings_rows(html_text)
    rows: List[Dict[str, Any]] = []

    max_idx = max(mapping.values())
    for cells in table_rows:
        if len(cells) <= max_idx:
            continue

        rank = _parse_int(cells[mapping["rank"]])
        if rank is None:
            continue

        source_team, record, wins, losses, ties = _parse_team_cell(cells[mapping["team"]])
        if not source_team:
            continue

        rows.append(
            {
                "season": season,
                "pull_date": pull_date.isoformat(),
                "rank": rank,
                "source_team": source_team,
                "team": None,
                "record": record,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "rating": _parse_number(cells[mapping["rating"]]),
                "vs_1_10_record": _parse_nullable_text(cells[mapping["vs_1_10_record"]]),
                "vs_11_25_record": _parse_nullable_text(cells[mapping["vs_11_25_record"]]),
                "vs_26_40_record": _parse_nullable_text(cells[mapping["vs_26_40_record"]]),
                "hi_rank": _parse_int(cells[mapping["hi_rank"]]),
                "lo_rank": _parse_int(cells[mapping["lo_rank"]]),
                "last_rank": _parse_int(cells[mapping["last_rank"]]),
                "source_url": source_url,
            }
        )

    return rows


def _build_source_url(pull_date: Optional[date] = None) -> str:
    if pull_date is None:
        return BASE_URL
    return f"{BASE_URL}?{urlencode({'date': pull_date.isoformat()})}"


def _fetch_predictive_rows(
    season: int,
    pull_date: date,
    logger,
    allow_missing_table: bool = False,
) -> List[Dict[str, Any]]:
    source_url = _build_source_url(pull_date)
    session = build_retry_session(api_key=None, timeout_seconds=60, total_retries=6)
    session.headers.update(
        {
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": USER_AGENT,
        }
    )

    logger.info(f"Fetching TeamRankings predictive ratings for season={season} pull_date={pull_date}")
    resp = session.get(source_url, timeout=getattr(session, "_etl_timeout", 60))
    if resp.status_code == 404 and allow_missing_table:
        logger.info(f"TeamRankings page returned 404 for pull_date={pull_date}; skipping")
        return []
    if resp.status_code >= 400:
        raise RuntimeError(
            f"TeamRankings GET failed: status={resp.status_code} url={source_url} body={resp.text[:500]}"
        )

    try:
        rows = _parse_predictive_html(resp.text, season, pull_date, source_url)
    except ValueError:
        if allow_missing_table:
            logger.info(f"No TeamRankings predictive table found for pull_date={pull_date}; skipping")
            return []
        raise

    logger.info(f"Fetched {len(rows)} TeamRankings predictive rows.")
    return rows


def _load_active_team_map(pg_dsn: str) -> dict[str, str]:
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_MAP_TABLE_SQL)
            cur.execute(
                f"""
                SELECT teamrankings_team, cfbd_team
                FROM {MAP_TABLE}
                WHERE active IS TRUE
                """
            )
            rows = cur.fetchall()
            cur.execute("COMMIT;")

    return {_normalize_map_key(source): cfbd for source, cfbd in rows}


def _apply_team_map(rows: List[Dict[str, Any]], team_map: dict[str, str]) -> List[Dict[str, Any]]:
    if not team_map:
        raise ValueError(f"{MAP_TABLE} has no active mappings; import the TeamRankings team map before scraping.")

    missing = sorted(
        {
            str(row.get("source_team"))
            for row in rows
            if row.get("source_team") and _normalize_map_key(str(row["source_team"])) not in team_map
        }
    )
    if missing:
        sample = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(f"Found unmapped TeamRankings teams: {sample}{suffix}")

    mapped: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["team"] = team_map[_normalize_map_key(str(row["source_team"]))]
        mapped.append(out)
    return mapped


def _validate_rows(rows: List[Dict[str, Any]], season: int, pull_date: date) -> None:
    if not rows:
        raise ValueError("TeamRankings returned 0 rows; refusing to insert an empty pull.")

    for row in rows:
        missing = [c for c in COLS if c not in row]
        if missing:
            raise ValueError(f"Mapped TeamRankings row missing keys for: {missing}")
        if row.get("season") in (None, ""):
            raise ValueError("Mapped TeamRankings row is missing a season.")
        if row.get("pull_date") in (None, ""):
            raise ValueError("Mapped TeamRankings row is missing a pull_date.")
        if row.get("source_team") in (None, ""):
            raise ValueError("Mapped TeamRankings row is missing a source_team.")
        if row.get("team") in (None, ""):
            raise ValueError("Mapped TeamRankings row is missing a canonical team.")

    wrong_seasons = sorted(
        {
            int(row["season"])
            for row in rows
            if row.get("season") not in (None, "") and int(row["season"]) != season
        }
    )
    if wrong_seasons:
        raise ValueError(f"TeamRankings rows contained seasons other than {season}: {wrong_seasons}")

    wrong_pull_dates = sorted(
        {str(row["pull_date"]) for row in rows if str(row["pull_date"]) != pull_date.isoformat()}
    )
    if wrong_pull_dates:
        raise ValueError(
            f"TeamRankings rows contained pull dates other than {pull_date.isoformat()}: {wrong_pull_dates}"
        )

    seen = set()
    dupes = 0
    for row in rows:
        key = (row.get("season"), row.get("pull_date"), row.get("source_team"))
        if key in seen:
            dupes += 1
        seen.add(key)
    if dupes:
        raise ValueError(
            "Found "
            f"{dupes} duplicate TeamRankings rows on key (season, pull_date, source_team); aborting."
        )

    if all(row.get("rating") is None for row in rows):
        raise ValueError("All TeamRankings predictive ratings are NULL; source schema likely changed.")


def _comparison_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 8)
    return value


def _snapshot_signature(rows: List[Dict[str, Any]]) -> list[tuple[Any, ...]]:
    return sorted(
        tuple(_comparison_value(row.get(col)) for col in SNAPSHOT_COMPARISON_COLS)
        for row in rows
    )


def _latest_prior_snapshot_matches(
    pg_dsn: str,
    season: int,
    pull_date: date,
    rows: List[Dict[str, Any]],
    logger,
) -> tuple[bool, date | None]:
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT pull_date
                FROM {TABLE}
                WHERE season = %s
                  AND pull_date < %s
                GROUP BY pull_date
                ORDER BY pull_date DESC
                LIMIT 1;
                """,
                (season, pull_date),
            )
            found = cur.fetchone()
            if found is None:
                return False, None

            previous_pull_date = found[0]
            cur.execute(
                f"""
                SELECT {", ".join(SNAPSHOT_COMPARISON_COLS)}
                FROM {TABLE}
                WHERE season = %s
                  AND pull_date = %s;
                """,
                (season, previous_pull_date),
            )
            previous_rows = [
                dict(zip(SNAPSHOT_COMPARISON_COLS, values))
                for values in cur.fetchall()
            ]

    matches = _snapshot_signature(rows) == _snapshot_signature(previous_rows)
    if matches:
        logger.info(
            f"{TABLE} latest prior snapshot for season={season} pull_date={previous_pull_date} "
            f"matches pull_date={pull_date}; skipping unchanged rankings"
        )
    return matches, previous_pull_date


def _to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLS})
    return buf.getvalue().encode("utf-8")


def _ensure_table_and_count_existing_pull(
    pg_dsn: str,
    season: int,
    pull_date: date,
    logger,
) -> int:
    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_MAP_TABLE_SQL)
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season = %s AND pull_date = %s;", (season, pull_date))
            existing = int(cur.fetchone()[0])
            cur.execute("COMMIT;")

    if existing:
        logger.info(f"{TABLE} already has {existing} rows for season={season} pull_date={pull_date}; skipping")
    return existing


def _insert_new_pull(
    pg_dsn: str,
    season: int,
    pull_date: date,
    rows: List[Dict[str, Any]],
    logger,
) -> int:
    csv_bytes = _to_csv_bytes(rows)

    with psycopg.connect(pg_dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN;")
            cur.execute(CREATE_MAP_TABLE_SQL)
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TEMP_TABLE_SQL)

            with cur.copy(COPY_SQL) as cp:
                cp.write(csv_bytes)

            cur.execute(INSERT_FROM_TEMP_SQL)
            inserted = int(cur.rowcount or 0)

            cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season = %s AND pull_date = %s;", (season, pull_date))
            pull_count = int(cur.fetchone()[0])
            logger.info(
                f"Inserted {inserted} new rows into {TABLE} for season={season} pull_date={pull_date}; "
                f"pull now has {pull_count} rows"
            )

            cur.execute("COMMIT;")
            return inserted


def run_pull(
    pg_dsn: str,
    season: int,
    pull_date: date,
    run_id: str,
    logger,
    step_name: str = STEP_NAME,
    allow_missing_table: bool = False,
) -> StepResult:
    prefix = format_step_prefix(run_id, step_name)
    source_url = _build_source_url(pull_date)
    logger.info(f"{prefix} start (season={season}, pull_date={pull_date}, source_url={source_url})")

    try:
        with log_timing(logger, f"{prefix} preflight"):
            existing = _ensure_table_and_count_existing_pull(pg_dsn, season, pull_date, logger)
            if existing:
                msg = f"season={season} pull_date={pull_date} already exists with {existing} rows; skipping"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=step_name,
                    season=season,
                    status="skipped",
                    rows_fetched=0,
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                    meta={"pull_date": pull_date.isoformat(), "existing_rows": existing, "source_url": source_url},
                )

        with log_timing(logger, f"{prefix} fetch"):
            rows = _fetch_predictive_rows(season, pull_date, logger, allow_missing_table=allow_missing_table)

        with log_timing(logger, f"{prefix} transform"):
            if not rows:
                msg = f"season={season} pull_date={pull_date} no TeamRankings rows returned; skipping"
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=step_name,
                    season=season,
                    status="skipped",
                    rows_fetched=0,
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                    meta={"pull_date": pull_date.isoformat(), "source_url": source_url},
                )
            team_map = _load_active_team_map(pg_dsn)
            rows = _apply_team_map(rows, team_map)

        with log_timing(logger, f"{prefix} validate"):
            _validate_rows(rows, season, pull_date)

        with log_timing(logger, f"{prefix} compare"):
            unchanged, previous_pull_date = _latest_prior_snapshot_matches(
                pg_dsn=pg_dsn,
                season=season,
                pull_date=pull_date,
                rows=rows,
                logger=logger,
            )
            if unchanged:
                msg = (
                    f"season={season} pull_date={pull_date} unchanged from "
                    f"previous_pull_date={previous_pull_date}; skipping"
                )
                logger.info(f"{prefix} skipped | {msg}")
                return StepResult(
                    step_name=step_name,
                    season=season,
                    status="skipped",
                    rows_fetched=len(rows),
                    rows_deleted=0,
                    rows_inserted=0,
                    message=msg,
                    meta={
                        "pull_date": pull_date.isoformat(),
                        "previous_pull_date": previous_pull_date.isoformat() if previous_pull_date else None,
                        "source_url": source_url,
                        "skip_reason": "unchanged_snapshot",
                    },
                )

        with log_timing(logger, f"{prefix} load"):
            inserted = _insert_new_pull(pg_dsn, season, pull_date, rows, logger)

        msg = f"season={season} pull_date={pull_date} inserted={inserted}"
        logger.info(f"{prefix} success | {msg}")
        return StepResult(
            step_name=step_name,
            season=season,
            status="success",
            rows_fetched=len(rows),
            rows_deleted=0,
            rows_inserted=inserted,
            message=msg,
            meta={"pull_date": pull_date.isoformat(), "source_url": source_url},
        )

    except Exception as e:
        logger.exception(f"{prefix} FAILED: {e}")
        return StepResult(
            step_name=step_name,
            season=season,
            status="failed",
            message="Job failed; see logs for details.",
            meta={"pull_date": pull_date.isoformat(), "source_url": source_url},
            error=str(e),
        )


def run() -> StepResult:
    cfg = load_config()
    logger = setup_logging()
    return run_pull(
        pg_dsn=cfg.pg_dsn,
        season=cfg.season,
        pull_date=_current_pull_date(),
        run_id=cfg.run_id,
        logger=logger,
    )


def main() -> None:
    res = run()
    if res.status not in ("success", "skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pytest

from etl.jobs.nightly_data_updates import update_teamrankings_predictive as tr


PULL_DATE = date(2026, 9, 1)
FIXTURE = Path(__file__).parent / "fixtures" / "teamrankings_predictive_sample.html"


def _mapped_rows() -> list[dict]:
    rows = tr._parse_predictive_html(FIXTURE.read_text(), 2026, PULL_DATE, "https://example.test/source")
    return tr._apply_team_map(
        rows,
        {
            tr._normalize_map_key("Ohio St"): "Ohio State",
            tr._normalize_map_key("S Florida"): "South Florida",
            tr._normalize_map_key("J Madison"): "James Madison",
        },
    )


def test_parse_predictive_html_maps_visible_columns() -> None:
    rows = tr._parse_predictive_html(FIXTURE.read_text(), 2026, PULL_DATE, "https://example.test/source")

    assert rows[0] == {
        "season": 2026,
        "pull_date": "2026-09-01",
        "rank": 1,
        "source_team": "Ohio St",
        "team": None,
        "record": "2-0",
        "wins": 2,
        "losses": 0,
        "ties": 0,
        "rating": 32.1,
        "vs_1_10_record": "1-0",
        "vs_11_25_record": "0-0",
        "vs_26_40_record": "1-0",
        "hi_rank": 1,
        "lo_rank": 2,
        "last_rank": 1,
        "source_url": "https://example.test/source",
    }
    assert rows[1]["source_team"] == "S Florida"
    assert rows[1]["rating"] == -2.6
    assert rows[2]["rating"] is None
    assert rows[2]["vs_11_25_record"] is None
    assert rows[2]["hi_rank"] is None
    assert rows[2]["last_rank"] is None


def test_apply_team_map_adds_canonical_team_names() -> None:
    rows = _mapped_rows()

    assert [row["team"] for row in rows] == ["Ohio State", "South Florida", "James Madison"]


def test_apply_team_map_rejects_unmapped_teams() -> None:
    rows = tr._parse_predictive_html(FIXTURE.read_text(), 2026, PULL_DATE, "https://example.test/source")

    with pytest.raises(ValueError, match="unmapped TeamRankings teams"):
        tr._apply_team_map(rows, {tr._normalize_map_key("Ohio St"): "Ohio State"})


def test_validate_rows_rejects_wrong_season() -> None:
    rows = _mapped_rows()
    rows[0]["season"] = 2025

    with pytest.raises(ValueError, match="seasons other than 2026"):
        tr._validate_rows(rows, 2026, PULL_DATE)


def test_validate_rows_rejects_wrong_pull_date() -> None:
    rows = _mapped_rows()
    rows[0]["pull_date"] = "2026-08-31"

    with pytest.raises(ValueError, match="pull dates other than 2026-09-01"):
        tr._validate_rows(rows, 2026, PULL_DATE)


def test_validate_rows_rejects_duplicate_source_team_for_pull_date() -> None:
    rows = _mapped_rows()
    rows.append(dict(rows[0]))

    with pytest.raises(ValueError, match="duplicate TeamRankings rows"):
        tr._validate_rows(rows, 2026, PULL_DATE)


def test_to_csv_bytes_includes_expected_header() -> None:
    csv_bytes = tr._to_csv_bytes([])

    assert csv_bytes.decode("utf-8") == (
        "season,pull_date,rank,source_team,team,record,wins,losses,ties,rating,"
        "vs_1_10_record,vs_11_25_record,vs_26_40_record,hi_rank,lo_rank,last_rank,source_url\n"
    )


def test_snapshot_signature_ignores_pull_date_and_source_url() -> None:
    rows = _mapped_rows()
    next_day_rows = [
        {
            **row,
            "pull_date": "2026-09-02",
            "source_url": "https://example.test/next-day",
        }
        for row in rows
    ]

    assert tr._snapshot_signature(rows) == tr._snapshot_signature(next_day_rows)


def test_run_pull_skips_existing_pull(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_fetch(*args, **kwargs):
        raise AssertionError("fetch should not run when pull already exists")

    monkeypatch.setattr(tr, "_ensure_table_and_count_existing_pull", lambda *args, **kwargs: 3)
    monkeypatch.setattr(tr, "_fetch_predictive_rows", fail_fetch)

    result = tr.run_pull("postgres://example", 2026, PULL_DATE, "run123", logging.getLogger(__name__))

    assert result.status == "skipped"
    assert result.meta["existing_rows"] == 3


def test_run_pull_skips_unchanged_prior_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_insert(*args, **kwargs):
        raise AssertionError("insert should not run when snapshot is unchanged")

    monkeypatch.setattr(tr, "_ensure_table_and_count_existing_pull", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        tr,
        "_fetch_predictive_rows",
        lambda *args, **kwargs: tr._parse_predictive_html(
            FIXTURE.read_text(),
            2026,
            PULL_DATE,
            "https://example.test/source",
        ),
    )
    monkeypatch.setattr(
        tr,
        "_load_active_team_map",
        lambda *args, **kwargs: {
            tr._normalize_map_key("Ohio St"): "Ohio State",
            tr._normalize_map_key("S Florida"): "South Florida",
            tr._normalize_map_key("J Madison"): "James Madison",
        },
    )
    monkeypatch.setattr(
        tr,
        "_latest_prior_snapshot_matches",
        lambda *args, **kwargs: (True, date(2026, 8, 31)),
    )
    monkeypatch.setattr(tr, "_insert_new_pull", fail_insert)

    result = tr.run_pull("postgres://example", 2026, PULL_DATE, "run123", logging.getLogger(__name__))

    assert result.status == "skipped"
    assert result.rows_fetched == 3
    assert result.meta["skip_reason"] == "unchanged_snapshot"

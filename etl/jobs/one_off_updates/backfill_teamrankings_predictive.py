#!/usr/bin/env python3
"""
ETL Job: backfill_teamrankings_predictive

Backfill TeamRankings predictive-rating snapshots for a date range. Each date
uses the same scrape/validate/load path as the nightly job, including skipping
unchanged snapshots.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import date, timedelta
from typing import Sequence

from etl.common_config import load_config
from etl.common_logging import format_step_prefix, log_timing, setup_logging
from etl.common_types import StepResult
from etl.jobs.nightly_data_updates import update_teamrankings_predictive as predictive


STEP_NAME = "backfill_teamrankings_predictive"


def _parse_date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD date, got {value!r}") from exc


def _iter_pull_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    out: list[date] = []
    current = start_date
    while current <= end_date:
        out.append(current)
        current += timedelta(days=1)
    return out


def _infer_season_for_pull_date(pull_date: date) -> int:
    if pull_date.month <= 3:
        return pull_date.year - 1
    return pull_date.year


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill TeamRankings predictive ratings by pull date.")
    parser.add_argument("--start-date", required=True, type=_parse_date_arg)
    parser.add_argument("--end-date", required=True, type=_parse_date_arg)
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Optional fixed season for every pull date. Omit for multi-season backfills.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=float(os.getenv("TEAMRANKINGS_BACKFILL_SLEEP_SECONDS", "1.0")),
        help="Delay between date fetches.",
    )
    return parser


def run_backfill(
    pg_dsn: str,
    season: int | None,
    start_date: date,
    end_date: date,
    sleep_seconds: float,
    run_id: str,
    logger,
) -> StepResult:
    prefix = format_step_prefix(run_id, STEP_NAME)
    pull_dates = _iter_pull_dates(start_date, end_date)
    season_label = season if season is not None else "auto"
    logger.info(
        f"{prefix} start (season={season_label}, start_date={start_date}, "
        f"end_date={end_date}, dates={len(pull_dates)})"
    )

    results: list[StepResult] = []
    try:
        for idx, pull_date in enumerate(pull_dates):
            pull_season = season if season is not None else _infer_season_for_pull_date(pull_date)
            with log_timing(logger, f"{prefix} pull_date={pull_date}"):
                result = predictive.run_pull(
                    pg_dsn=pg_dsn,
                    season=pull_season,
                    pull_date=pull_date,
                    run_id=run_id,
                    logger=logger,
                    step_name=predictive.STEP_NAME,
                    allow_missing_table=True,
                )
            results.append(result)

            if result.status == "failed":
                logger.error(f"{prefix} stopping after failed pull_date={pull_date}: {result.error}")
                break

            if sleep_seconds > 0 and idx < len(pull_dates) - 1:
                time.sleep(sleep_seconds)

        failed = [result for result in results if result.status == "failed"]
        succeeded = [result for result in results if result.status == "success"]
        skipped = [result for result in results if result.status == "skipped"]
        status = "failed" if failed else "success" if succeeded else "skipped"
        result_season = season if season is not None else _infer_season_for_pull_date(end_date)
        msg = (
            f"season={season_label} dates={len(results)}/{len(pull_dates)} "
            f"success={len(succeeded)} skipped={len(skipped)} failed={len(failed)}"
        )
        logger.info(f"{prefix} {status} | {msg}")
        return StepResult(
            step_name=STEP_NAME,
            season=result_season,
            status=status,
            rows_fetched=sum(result.rows_fetched for result in results),
            rows_deleted=0,
            rows_inserted=sum(result.rows_inserted for result in results),
            message=msg,
            meta={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "dates_attempted": len(results),
                "dates_requested": len(pull_dates),
                "season_mode": "fixed" if season is not None else "auto",
                "seasons_attempted": sorted({result.season for result in results}),
            },
            error=failed[0].error if failed else None,
        )

    except Exception as e:
        logger.exception(f"{prefix} FAILED: {e}")
        return StepResult(
            step_name=STEP_NAME,
            season=season if season is not None else _infer_season_for_pull_date(end_date),
            status="failed",
            message="Backfill failed; see logs for details.",
            meta={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "season_mode": "fixed" if season is not None else "auto",
            },
            error=str(e),
        )


def run(argv: Sequence[str] | None = None) -> StepResult:
    args = _build_parser().parse_args(argv)
    cfg = load_config()
    logger = setup_logging()
    season = args.season if args.season is not None else cfg.season
    return run_backfill(
        pg_dsn=cfg.pg_dsn,
        season=season,
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_seconds=args.sleep_seconds,
        run_id=cfg.run_id,
        logger=logger,
    )


def main() -> None:
    res = run()
    if res.status not in ("success", "skipped"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from datetime import date

from etl.common_types import StepResult
from etl.jobs.one_off_updates import backfill_teamrankings_predictive as backfill


def test_iter_pull_dates_includes_start_and_end() -> None:
    assert backfill._iter_pull_dates(date(2026, 8, 31), date(2026, 9, 2)) == [
        date(2026, 8, 31),
        date(2026, 9, 1),
        date(2026, 9, 2),
    ]


def test_infer_season_for_pull_date_handles_postseason_boundary() -> None:
    assert backfill._infer_season_for_pull_date(date(2026, 3, 31)) == 2025
    assert backfill._infer_season_for_pull_date(date(2026, 4, 1)) == 2026


def test_backfill_uses_nightly_pull_and_aggregates_skips(monkeypatch) -> None:
    calls = []

    def fake_run_pull(**kwargs):
        calls.append(kwargs["pull_date"])
        if kwargs["pull_date"] == date(2026, 9, 1):
            return StepResult(
                step_name="update_teamrankings_predictive",
                season=2026,
                status="skipped",
                message="already exists",
            )
        return StepResult(
            step_name="update_teamrankings_predictive",
            season=2026,
            status="success",
            rows_fetched=3,
            rows_inserted=3,
        )

    monkeypatch.setattr(backfill.predictive, "run_pull", fake_run_pull)

    result = backfill.run_backfill(
        pg_dsn="postgres://example",
        season=2026,
        start_date=date(2026, 8, 31),
        end_date=date(2026, 9, 2),
        sleep_seconds=0,
        run_id="run123",
        logger=logging.getLogger(__name__),
    )

    assert calls == [date(2026, 8, 31), date(2026, 9, 1), date(2026, 9, 2)]
    assert result.status == "success"
    assert result.rows_fetched == 6
    assert result.rows_inserted == 6
    assert result.message == "season=2026 dates=3/3 success=2 skipped=1 failed=0"


def test_backfill_can_infer_season_per_pull_date(monkeypatch) -> None:
    calls = []

    def fake_run_pull(**kwargs):
        calls.append((kwargs["pull_date"], kwargs["season"]))
        return StepResult(
            step_name="update_teamrankings_predictive",
            season=kwargs["season"],
            status="skipped",
            message="unchanged",
        )

    monkeypatch.setattr(backfill.predictive, "run_pull", fake_run_pull)

    result = backfill.run_backfill(
        pg_dsn="postgres://example",
        season=None,
        start_date=date(2026, 3, 31),
        end_date=date(2026, 4, 1),
        sleep_seconds=0,
        run_id="run123",
        logger=logging.getLogger(__name__),
    )

    assert calls == [(date(2026, 3, 31), 2025), (date(2026, 4, 1), 2026)]
    assert result.status == "skipped"
    assert result.message == "season=auto dates=2/2 success=0 skipped=2 failed=0"

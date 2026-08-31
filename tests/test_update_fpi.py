from __future__ import annotations

from datetime import date

import pytest

from etl.jobs.nightly_data_updates import update_fpi as fpi


PULL_DATE = date(2026, 8, 31)


def test_cfbd_to_row_flattens_camel_case_payload() -> None:
    row = fpi._cfbd_to_row(
        {
            "year": 2026,
            "team": "Michigan",
            "conference": "Big Ten",
            "fpi": 27.4,
            "resumeRanks": {
                "gameControl": 2,
                "remainingStrengthOfSchedule": 18,
                "strengthOfSchedule": 22,
                "averageWinProbability": 3,
                "fpi": 1,
                "strengthOfRecord": 4,
            },
            "efficiencies": {
                "specialTeams": 0.8,
                "defense": 1.9,
                "offense": 2.4,
                "overall": 5.1,
            },
        },
        PULL_DATE,
    )

    assert row == {
        "season": 2026,
        "pull_date": "2026-08-31",
        "team": "Michigan",
        "conference": "Big Ten",
        "fpi": 27.4,
        "resume_game_control_rank": 2,
        "resume_remaining_strength_of_schedule_rank": 18,
        "resume_strength_of_schedule_rank": 22,
        "resume_average_win_probability_rank": 3,
        "resume_fpi_rank": 1,
        "resume_strength_of_record_rank": 4,
        "efficiency_special_teams": 0.8,
        "efficiency_defense": 1.9,
        "efficiency_offense": 2.4,
        "efficiency_overall": 5.1,
    }


def test_cfbd_to_row_flattens_snake_case_payload() -> None:
    row = fpi._cfbd_to_row(
        {
            "season": 2026,
            "team": "Georgia",
            "resume_ranks": {"strength_of_record": 5},
            "efficiencies": {"special_teams": 0.4},
        },
        PULL_DATE,
    )

    assert row["season"] == 2026
    assert row["pull_date"] == "2026-08-31"
    assert row["team"] == "Georgia"
    assert row["resume_strength_of_record_rank"] == 5
    assert row["efficiency_special_teams"] == 0.4


def test_validate_rows_rejects_wrong_season() -> None:
    rows = [fpi._cfbd_to_row({"year": 2025, "team": "Michigan"}, PULL_DATE)]

    with pytest.raises(ValueError, match="seasons other than 2026"):
        fpi._validate_rows(rows, 2026, PULL_DATE)


def test_validate_rows_rejects_duplicate_team_for_pull_date() -> None:
    rows = [
        fpi._cfbd_to_row({"year": 2026, "team": "Michigan"}, PULL_DATE),
        fpi._cfbd_to_row({"year": 2026, "team": "Michigan"}, PULL_DATE),
    ]

    with pytest.raises(ValueError, match="duplicate FPI rows"):
        fpi._validate_rows(rows, 2026, PULL_DATE)


def test_to_csv_bytes_includes_expected_header() -> None:
    csv_bytes = fpi._to_csv_bytes([])

    assert csv_bytes.decode("utf-8") == (
        "season,pull_date,team,conference,fpi,resume_game_control_rank,"
        "resume_remaining_strength_of_schedule_rank,resume_strength_of_schedule_rank,"
        "resume_average_win_probability_rank,resume_fpi_rank,resume_strength_of_record_rank,"
        "efficiency_special_teams,efficiency_defense,efficiency_offense,efficiency_overall\n"
    )

from __future__ import annotations

import pytest

from etl.jobs.nightly_data_updates import update_transfer_portal as portal


def test_cfbd_to_row_maps_camel_case_payload() -> None:
    row = portal._cfbd_to_row(
        {
            "season": 2026,
            "firstName": "Jane",
            "lastName": "Doe",
            "position": "QB",
            "origin": "Old State",
            "destination": "New State",
            "transferDate": "2026-01-15T00:00:00Z",
            "rating": 0.91,
            "stars": 4,
            "eligibility": "immediate",
        },
        season=2026,
    )

    assert row == {
        "Season": 2026,
        "FirstName": "Jane",
        "LastName": "Doe",
        "Position": "QB",
        "Origin": "Old State",
        "Destination": "New State",
        "TransferDate": "2026-01-15T00:00:00Z",
        "Rating": 0.91,
        "Stars": 4,
        "Eligibility": "immediate",
    }


def test_cfbd_to_row_maps_snake_case_payload_and_defaults_season() -> None:
    row = portal._cfbd_to_row(
        {
            "first_name": "John",
            "last_name": "Smith",
            "transfer_date": "2026-02-01T00:00:00Z",
            "eligibility": {"value": "pending"},
        },
        season=2026,
    )

    assert row["Season"] == 2026
    assert row["FirstName"] == "John"
    assert row["LastName"] == "Smith"
    assert row["TransferDate"] == "2026-02-01T00:00:00Z"
    assert row["Eligibility"] == '{"value": "pending"}'


def test_validate_rows_allows_empty_payload() -> None:
    portal._validate_rows([], 2026)


def test_validate_rows_rejects_wrong_season() -> None:
    rows = [portal._cfbd_to_row({"season": 2025, "firstName": "Jane"}, season=2026)]

    with pytest.raises(ValueError, match="seasons other than 2026"):
        portal._validate_rows(rows, 2026)


def test_to_csv_bytes_includes_exporter_style_header() -> None:
    csv_bytes = portal._to_csv_bytes([])

    assert csv_bytes.decode("utf-8") == (
        "Season,FirstName,LastName,Position,Origin,Destination,"
        "TransferDate,Rating,Stars,Eligibility\n"
    )

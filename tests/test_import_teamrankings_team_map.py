from __future__ import annotations

from pathlib import Path

import pytest

from etl.jobs.one_off_updates import import_teamrankings_team_map as importer


def test_read_map_csv_accepts_column_aliases(tmp_path: Path) -> None:
    csv_path = tmp_path / "map.csv"
    csv_path.write_text(
        "TeamRankings,School,Active,Notes\n"
        "Ohio St,Ohio State,true,\n"
        "S Florida,South Florida,1,short name\n"
    )

    df = importer._read_map_csv(csv_path)

    assert df.to_dict("records") == [
        {"teamrankings_team": "Ohio St", "cfbd_team": "Ohio State", "active": True, "notes": None},
        {
            "teamrankings_team": "S Florida",
            "cfbd_team": "South Florida",
            "active": True,
            "notes": "short name",
        },
    ]


def test_read_map_csv_accepts_old_dashboard_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "map.csv"
    csv_path.write_text(
        "cfb_name,teamrankings_name\n"
        "Abilene Christian,Abl Christian\n"
        "South Florida,S Florida\n"
    )

    df = importer._read_map_csv(csv_path)

    assert df.to_dict("records") == [
        {
            "teamrankings_team": "Abl Christian",
            "cfbd_team": "Abilene Christian",
            "active": True,
            "notes": None,
        },
        {
            "teamrankings_team": "S Florida",
            "cfbd_team": "South Florida",
            "active": True,
            "notes": None,
        },
    ]


def test_read_map_csv_rejects_duplicate_source_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "map.csv"
    csv_path.write_text("teamrankings_team,cfbd_team\nOhio St,Ohio State\n ohio st ,Ohio State\n")

    with pytest.raises(ValueError, match="duplicate TeamRankings team map rows"):
        importer._read_map_csv(csv_path)

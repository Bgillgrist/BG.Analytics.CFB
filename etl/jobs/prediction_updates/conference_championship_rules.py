#!/usr/bin/env python3
"""
Conference championship selection and title-game simulation rules.

REVIEW STARTS HERE.

Each FBS conference has its own selector function below. The local tester calls
select_conference_championship_teams(), which dispatches by conference name.

Fallback no-division rule stack for conferences without confirmed rules:
  1. Conference winning percentage
  2. Head-to-head winning percentage among tied teams
  3. Winning percentage against common conference opponents
  4. Overall wins
  5. Team strength estimate
  6. Deterministic team name, after a seeded random tiebreaker

This is intentionally organized so each conference can diverge as we confirm
its official tiebreaker language.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable


TeamRecord = dict[str, Any]
HeadToHeadMap = dict[str, dict[str, int]]
SEC_RELATIVE_OFFENSE_CAP = 200.0
SUN_BELT_DIVISIONS: dict[str, str] = {
    "App State": "East",
    "Coastal Carolina": "East",
    "Georgia Southern": "East",
    "Georgia State": "East",
    "James Madison": "East",
    "Marshall": "East",
    "Old Dominion": "East",
    "Arkansas State": "West",
    "Louisiana": "West",
    "ULM": "West",
    "UL Monroe": "West",
    "South Alabama": "West",
    "Southern Miss": "West",
    "Texas State": "West",
    "Troy": "West",
}


def sun_belt_division_for_team(team: str) -> str | None:
    return SUN_BELT_DIVISIONS.get(team)


def select_conference_championship_teams(
    conference: str,
    team_records: list[TeamRecord],
    rng: random.Random,
) -> tuple[str, str]:
    selector = CONFERENCE_RULES.get(conference, select_default_championship_teams)
    return selector(team_records, rng)


def simulate_conference_championship_game(
    first_team: str,
    second_team: str,
    team_metrics: dict[str, TeamRecord],
    rng: random.Random,
) -> str:
    """
    Simulate a neutral-site conference championship game from expected stats.

    Current rating inputs:
      - team_strength: expected win percentage from simulated schedule
      - overall_win_pct: projected overall win percentage before title game
      - conference_win_pct: projected conference win percentage
      - strength_of_schedule: average opponent strength

    We can replace this with a true synthetic game-model prediction once the
    game model exposes a clean neutral-site matchup scorer.
    """
    first_rating = _championship_game_rating(team_metrics[first_team])
    second_rating = _championship_game_rating(team_metrics[second_team])
    first_win_prob = _logistic(4.0 * (first_rating - second_rating))
    return first_team if rng.random() < first_win_prob else second_team


# ---------------------------------------------------------------------------
# Conference-specific selectors
# ---------------------------------------------------------------------------
# REVIEW: ACC
def select_acc_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    ACC no-division championship selector.

    Participants are selected by conference winning percentage. Ties are broken
    with the ACC stack: head-to-head, common opponents, ordered common-opponent
    standings, opponent conference win pct, SportSource Team Rating Score, draw.
    """
    if len(team_records) < 2:
        raise ValueError("At least two ACC teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    participants: list[str] = []
    for group in _standings_groups(list(records_by_team), records_by_team):
        slots = 2 - len(participants)
        if slots <= 0:
            break
        if len(group) <= slots:
            if len(group) == 2:
                participants.extend(_acc_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(group)
            continue
        if slots == 2:
            if len(group) == 2:
                participants.extend(_acc_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(_acc_select_two_from_multi_team_tie(group, records_by_team, rng))
        else:
            participants.append(_acc_select_one_team(group, records_by_team, rng))

    if len(participants) < 2:
        raise RuntimeError("Unable to select two ACC championship game teams.")
    return participants[0], participants[1]


# REVIEW: American Athletic
def select_american_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    American Athletic no-division tiebreaker stack.

    Current implementation follows the supplied American structure:
      1. Highest conference winning percentages determine the two participants.
      2. Ties for first or second use American two-team/multiple-team procedures.
      3. In unbalanced schedules, teams with fewer games can win a tied loss-column
         comparison by defeating all other tied teams when they have at least seven
         conference games.
      4. CFP-ranking and computer-composite steps are wired as data fields. The
         local tester currently has no CFP rank, so computer_composite_score
         falls back to team_strength.

    Ineligible-team and host-location rules are not active until the input data
    includes eligibility flags and the simulator needs home/away title-game sites.
    """
    if len(team_records) < 2:
        raise ValueError("At least two American teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    eligible_teams = [
        team
        for team, record in records_by_team.items()
        if _conference_games_played(record) >= 7
    ]
    if len(eligible_teams) < 2:
        eligible_teams = list(records_by_team)

    seeded: list[str] = []
    remaining = sorted(eligible_teams)
    while len(seeded) < 2 and remaining:
        candidate_group = _american_participant_candidate_group(remaining, records_by_team)
        participant = _american_select_one_team(candidate_group, records_by_team, rng)
        seeded.append(participant)
        remaining = [team for team in remaining if team != participant]

    if len(seeded) < 2:
        raise RuntimeError("Unable to select two American championship game teams.")

    return seeded[0], seeded[1]


# REVIEW: Big 12
def select_big_12_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Big 12 no-division tiebreaker stack.

    Current implementation follows the supplied Big 12 structure:
      A. Two-team ties use head-to-head when played, then common opponents,
         common opponents by standings order, strength of conference schedule,
         adjusted total wins, rating score, coin toss.
      B. Multiple-team ties seed one team, then repeat the process for the
         remaining teams. If reduced to two teams, the two-team procedure is
         applied.

    The local tester currently maps adjusted_total_wins to simulated total
    regular-season wins until FCS/exempt-game adjustments are modeled.
    """
    if len(team_records) < 2:
        raise ValueError("At least two Big 12 teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    standings_groups = _standings_groups(list(records_by_team), records_by_team)
    first_place_group = standings_groups[0]

    if len(first_place_group) == 1:
        first = first_place_group[0]
        if len(standings_groups) == 1:
            remaining = [team for team in records_by_team if team != first]
        else:
            remaining = standings_groups[1]
        second = _big_12_select_one_team(remaining, records_by_team, rng)
        return first, second

    if len(first_place_group) == 2:
        ordered = _big_12_order_two_team_tie(first_place_group, records_by_team, rng)
        return ordered[0], ordered[1]

    return _big_12_select_from_multi_team_first_place(first_place_group, records_by_team, rng)


# REVIEW: Big Ten
def select_big_ten_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Big Ten no-division tiebreaker stack.

    Current implementation follows the supplied Big Ten structure:
      A. Two-team ties use head-to-head when played, then common opponents,
         common opponents by standings order, cumulative opponent conference
         winning percentage, rating score, random draw.
      B. Three-or-more-team first-place ties can select a clear No. 1, select
         two tied No. 1 teams, narrow the candidate group, or advance through
         the next tiebreaker step.

    SportSource Analytics team Rating Score is represented locally by the
    simulator's team_rating_score field until we have the real metric.
    """
    if len(team_records) < 2:
        raise ValueError("At least two Big Ten teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    standings_groups = _standings_groups(list(records_by_team), records_by_team)
    first_place_group = standings_groups[0]

    if len(first_place_group) == 1:
        first = first_place_group[0]
        if len(standings_groups) == 1:
            remaining = [team for team in records_by_team if team != first]
        else:
            remaining = standings_groups[1]
        second = _big_ten_select_one_team(remaining, records_by_team, rng)
        return first, second

    if len(first_place_group) == 2:
        ordered = _big_ten_order_two_team_tie(first_place_group, records_by_team, rng)
        return ordered[0], ordered[1]

    return _big_ten_select_from_multi_team_first_place(first_place_group, records_by_team, rng)


# REVIEW: Conference USA
def select_conference_usa_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Conference USA no-division championship selector.

    Participants are selected by conference winning percentage. Ties are broken
    with the supplied CUSA stack: head-to-head/common opponents, SportSource
    Team Rating Score, ordered common-opponent standings, opponent conference
    win pct, draw.
    """
    return _select_cusa_mac_championship_teams(team_records, rng, "Conference USA")


# REVIEW: Mid-American
def select_mid_american_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Mid-American no-division championship selector.

    Participants are selected by conference winning percentage. Ties are broken
    with the MAC stack: head-to-head/common opponents, SportSource Team Rating
    Score, ordered common-opponent standings, opponent conference win pct, draw.
    """
    return _select_cusa_mac_championship_teams(team_records, rng, "Mid-American")


# REVIEW: Mountain West
def select_mountain_west_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Mountain West no-division tiebreaker stack.

    Current implementation follows the supplied Mountain West structure:
      1. Head-to-head
      2. CFP Selection Committee ranking if ranked teams win the final weekend,
         otherwise composite computer metric rank
      3. Adjusted overall winning percentage, with FCS wins capped at one
      4. Record against the next highest-placed Conference team/group, skipping
         groups not played by every tied team
      5. Winning percentage against common Conference opponents
      6. Commissioner drawing

    The local tester supplies projected CFP ranking and composite rank proxies
    until the actual CFP/computer metric inputs are modeled.
    """
    if len(team_records) < 2:
        raise ValueError("At least two Mountain West teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    participants: list[str] = []
    for group in _standings_groups(list(records_by_team), records_by_team):
        slots = 2 - len(participants)
        if slots <= 0:
            break
        if len(group) <= slots:
            if len(group) == 2:
                participants.extend(_mountain_west_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(group)
            continue
        if slots == 2:
            if len(group) == 2:
                participants.extend(_mountain_west_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(_mountain_west_select_two_from_multi_team_tie(group, records_by_team, rng))
        else:
            participants.append(_mountain_west_select_one_team(group, records_by_team, rng))

    if len(participants) < 2:
        raise RuntimeError("Unable to select two Mountain West championship game teams.")
    return participants[0], participants[1]


# REVIEW: Pac-12
def select_pac_12_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    return _select_top_two_no_divisions(team_records, rng)


# REVIEW: SEC
def select_sec_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    SEC no-division tiebreaker stack.

    Current implementation follows the stated order:
      1. Head-to-head competition among tied teams
      2. Record versus all common Conference opponents among tied teams
      3. Record against highest placed common Conference opponent in the
         Conference standings, proceeding through the standings
      4. Cumulative Conference winning percentage of all Conference opponents
      5. Capped relative total scoring margin versus all Conference opponents
      6. Random draw
    """
    if len(team_records) < 2:
        raise ValueError("At least two SEC teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    standings_groups = _standings_groups(list(records_by_team), records_by_team)
    first_place_group = standings_groups[0]

    if len(first_place_group) == 1:
        first = first_place_group[0]
        if len(standings_groups) == 1:
            remaining = [team for team in records_by_team if team != first]
        else:
            remaining = standings_groups[1]
        second = _sec_select_second_place_team(remaining, records_by_team, rng)
        return first, second

    if len(first_place_group) == 2:
        ordered = _sec_order_two_team_tie(first_place_group, records_by_team, rng)
        return ordered[0], ordered[1]

    return _sec_select_from_multi_team_first_place(first_place_group, records_by_team, rng)


# REVIEW: Sun Belt
def select_sun_belt_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    """
    Sun Belt division-based championship selector.

    Current implementation uses the manual East/West mapping above. Each
    division champion is selected by conference winning percentage across all
    Sun Belt games, then Sun Belt division tiebreakers.

    Ineligible-team and host-location rules are not active until the input data
    includes eligibility flags and the simulator needs home/away title-game sites.
    """
    records_by_team = {record["team"]: record for record in team_records}
    teams_by_division: dict[str, list[str]] = defaultdict(list)
    for record in team_records:
        division = record.get("division") or sun_belt_division_for_team(record["team"])
        if division:
            teams_by_division[str(division)].append(record["team"])

    east_teams = sorted(teams_by_division.get("East", []))
    west_teams = sorted(teams_by_division.get("West", []))
    if not east_teams or not west_teams:
        return _select_top_two_no_divisions(team_records, rng)

    east_champion = _sun_belt_select_division_champion(east_teams, records_by_team, rng)
    west_champion = _sun_belt_select_division_champion(west_teams, records_by_team, rng)
    return east_champion, west_champion


def select_default_championship_teams(team_records: list[TeamRecord], rng: random.Random) -> tuple[str, str]:
    return _select_top_two_no_divisions(team_records, rng)


CONFERENCE_RULES: dict[str, Callable[[list[TeamRecord], random.Random], tuple[str, str]]] = {
    "ACC": select_acc_championship_teams,
    "American Athletic": select_american_championship_teams,
    "Big 12": select_big_12_championship_teams,
    "Big Ten": select_big_ten_championship_teams,
    "Conference USA": select_conference_usa_championship_teams,
    "Mid-American": select_mid_american_championship_teams,
    "Mountain West": select_mountain_west_championship_teams,
    "Pac-12": select_pac_12_championship_teams,
    "SEC": select_sec_championship_teams,
    "Sun Belt": select_sun_belt_championship_teams,
}


# ---------------------------------------------------------------------------
# Shared tiebreaker helpers
# ---------------------------------------------------------------------------
def _select_top_two_no_divisions(
    team_records: list[TeamRecord],
    rng: random.Random,
    *,
    orderer: Callable[[list[str], dict[str, TeamRecord], random.Random], list[str]] | None = None,
) -> tuple[str, str]:
    if len(team_records) < 2:
        raise ValueError("At least two teams are required to select championship game teams.")

    orderer = orderer or _order_tied_group
    records_by_team = {record["team"]: record for record in team_records}
    grouped: dict[float, list[str]] = defaultdict(list)
    for record in team_records:
        grouped[_conference_win_pct(record)].append(record["team"])

    finalists: list[str] = []
    for conference_win_pct in sorted(grouped, reverse=True):
        ordered_group = orderer(grouped[conference_win_pct], records_by_team, rng)
        for team in ordered_group:
            finalists.append(team)
            if len(finalists) == 2:
                return finalists[0], finalists[1]

    raise RuntimeError("Unable to select two championship game teams.")


def _order_tied_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) <= 1:
        return tied_teams

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head)),
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            -float(records_by_team[team].get("overall_wins", 0)),
            -float(records_by_team[team].get("team_strength", 0.5)),
            rng.random(),
            team,
        ),
    )


def _order_sec_tied_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) <= 1:
        return tied_teams

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head)),
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            _negative_tuple(
                _sec_common_opponent_standing_sequence(
                    team,
                    tied_teams,
                    records_by_team,
                    head_to_head,
                )
            ),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            -float(records_by_team[team].get("sec_relative_scoring_margin", 0.0)),
            rng.random(),
            team,
        ),
    )


def _standings_groups(teams: list[str], records_by_team: dict[str, TeamRecord]) -> list[list[str]]:
    grouped: dict[float, list[str]] = defaultdict(list)
    for team in teams:
        grouped[_conference_win_pct(records_by_team[team])].append(team)
    return [sorted(grouped[value]) for value in sorted(grouped, reverse=True)]


def _sec_select_second_place_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _sec_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _sec_select_from_multi_team_second_place(tied_teams, records_by_team, rng, depth)


def _sec_select_from_multi_team_first_place(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> tuple[str, str]:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) < 2:
        raise RuntimeError("SEC first-place tiebreaker needs at least two teams.")
    if len(tied_teams) == 2:
        ordered = _sec_order_two_team_tie(tied_teams, records_by_team, rng)
        return ordered[0], ordered[1]
    if depth > 10:
        drawn = rng.sample(tied_teams, 2)
        return drawn[0], drawn[1]

    h2h_result = _sec_multi_team_head_to_head_result(
        tied_teams,
        records_by_team,
        mode="first",
        rng=rng,
        depth=depth,
    )
    if h2h_result is not None:
        return h2h_result

    head_to_head = _shared_head_to_head(records_by_team)
    common_record_values = {
        team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, list(records_by_team), head_to_head))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_first_values(tied_teams, records_by_team, common_record_values, rng, depth)
    if result is not None:
        return result

    common_standing_values = {
        team: _sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
        for team in tied_teams
    }
    result = _sec_apply_multi_team_first_values(tied_teams, records_by_team, common_standing_values, rng, depth)
    if result is not None:
        return result

    opponent_wp_values = {
        team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_first_values(tied_teams, records_by_team, opponent_wp_values, rng, depth)
    if result is not None:
        return result

    scoring_margin_values = {
        team: float(records_by_team[team].get("sec_relative_scoring_margin", 0.0))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_first_values(tied_teams, records_by_team, scoring_margin_values, rng, depth)
    if result is not None:
        return result

    drawn = rng.sample(tied_teams, 2)
    return drawn[0], drawn[1]


def _sec_select_from_multi_team_second_place(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _sec_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(tied_teams)

    h2h_result = _sec_multi_team_head_to_head_result(
        tied_teams,
        records_by_team,
        mode="second",
        rng=rng,
        depth=depth,
    )
    if h2h_result is not None:
        return h2h_result

    head_to_head = _shared_head_to_head(records_by_team)
    common_record_values = {
        team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, list(records_by_team), head_to_head))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_second_values(tied_teams, records_by_team, common_record_values, rng, depth)
    if result is not None:
        return result

    common_standing_values = {
        team: _sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
        for team in tied_teams
    }
    result = _sec_apply_multi_team_second_values(tied_teams, records_by_team, common_standing_values, rng, depth)
    if result is not None:
        return result

    opponent_wp_values = {
        team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_second_values(tied_teams, records_by_team, opponent_wp_values, rng, depth)
    if result is not None:
        return result

    scoring_margin_values = {
        team: float(records_by_team[team].get("sec_relative_scoring_margin", 0.0))
        for team in tied_teams
    }
    result = _sec_apply_multi_team_second_values(tied_teams, records_by_team, scoring_margin_values, rng, depth)
    if result is not None:
        return result

    return rng.choice(tied_teams)


def _sec_multi_team_head_to_head_result(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    *,
    mode: str,
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | str | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        if mode == "first":
            if len(top_group) == 1:
                first = top_group[0]
                second = _sec_select_second_place_team(
                    [team for team in tied_teams if team != first],
                    records_by_team,
                    rng,
                    depth + 1,
                )
                return first, second
            if len(top_group) == 2:
                ordered = _sec_order_two_team_tie(top_group, records_by_team, rng)
                return ordered[0], ordered[1]
            return _sec_select_from_multi_team_first_place(top_group, records_by_team, rng, depth + 1)

        if len(top_group) == 1:
            return top_group[0]
        return _sec_select_second_place_team(top_group, records_by_team, rng, depth + 1)

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        if mode == "first":
            first = beat_all[0]
            second = _sec_select_second_place_team(
                [team for team in tied_teams if team != first],
                records_by_team,
                rng,
                depth + 1,
            )
            return first, second
        return beat_all[0]

    lost_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(opponent, {}).get(team, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(lost_all) == 1:
        remaining = [team for team in tied_teams if team != lost_all[0]]
        if mode == "first":
            if len(remaining) == 2:
                ordered = _sec_order_two_team_tie(remaining, records_by_team, rng)
                return ordered[0], ordered[1]
            return _sec_select_from_multi_team_first_place(remaining, records_by_team, rng, depth + 1)
        return _sec_select_second_place_team(remaining, records_by_team, rng, depth + 1)

    return None


def _sec_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("SEC two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            _negative_tuple(_sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            -float(records_by_team[team].get("sec_relative_scoring_margin", 0.0)),
            rng.random(),
            team,
        ),
    )


def _sec_apply_multi_team_first_values(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    values_by_team: dict[str, float | tuple[float, ...]],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | None:
    top_group = _top_value_group(values_by_team)
    if len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _sec_select_second_place_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _sec_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return _sec_select_from_multi_team_first_place(top_group, records_by_team, rng, depth + 1)


def _sec_apply_multi_team_second_values(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    values_by_team: dict[str, float | tuple[float, ...]],
    rng: random.Random,
    depth: int,
) -> str | None:
    top_group = _top_value_group(values_by_team)
    if len(top_group) == len(tied_teams):
        return None
    return _sec_select_second_place_team(top_group, records_by_team, rng, depth + 1)


# ---------------------------------------------------------------------------
# American Athletic helpers
# ---------------------------------------------------------------------------
def _american_standings_groups(teams: list[str], records_by_team: dict[str, TeamRecord]) -> list[list[str]]:
    grouped: dict[float, list[str]] = defaultdict(list)
    for team in teams:
        grouped[_conference_win_pct(records_by_team[team])].append(team)
    return [sorted(grouped[value]) for value in sorted(grouped, reverse=True)]


def _american_participant_candidate_group(
    teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str]:
    standings_groups = _american_standings_groups(teams, records_by_team)
    top_win_pct_group = standings_groups[0]
    best_loss_count = min(float(records_by_team[team].get("conference_losses", 0)) for team in teams)
    best_loss_group = sorted(
        team
        for team in teams
        if float(records_by_team[team].get("conference_losses", 0)) == best_loss_count
    )
    if len(best_loss_group) > 1 and any(team in top_win_pct_group for team in best_loss_group):
        return _american_apply_unbalanced_loss_column_precedence(best_loss_group, records_by_team)
    return top_win_pct_group


def _american_apply_unbalanced_loss_column_precedence(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str]:
    head_to_head = _shared_head_to_head(records_by_team)
    max_conference_games = max(_conference_games_played(records_by_team[team]) for team in tied_teams)
    fewer_game_sweepers = [
        team
        for team in tied_teams
        if _conference_games_played(records_by_team[team]) >= 7
        and _conference_games_played(records_by_team[team]) < max_conference_games
        and all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(fewer_game_sweepers) == 1:
        return fewer_game_sweepers
    return tied_teams


def _american_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _american_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _american_select_from_multi_team_tie(tied_teams, records_by_team, rng, depth)


def _american_select_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _american_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    head_to_head_top_group = _american_head_to_head_top_group(candidates, records_by_team)
    result = _american_resolve_multi_team_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("cfp_or_composite", "common_record", "overall_win_pct"):
        values_by_team = _american_value_step(step, candidates, records_by_team)
        result = _american_resolve_multi_team_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            candidates = result

    return rng.choice(candidates)


def _american_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("American two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_american_value_step("cfp_or_composite", tied_teams, records_by_team)[team],
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            -_overall_win_pct(records_by_team[team]),
            rng.random(),
            team,
        ),
    )


def _american_head_to_head_top_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    return None


def _american_resolve_multi_team_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _american_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _american_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "cfp_or_composite":
        ranked_not_lost = {
            team: -float(records_by_team[team]["cfp_rank"])
            for team in tied_teams
            if records_by_team[team].get("cfp_rank") is not None
            and not records_by_team[team].get("lost_final_conference_week")
        }
        if ranked_not_lost:
            return {team: ranked_not_lost.get(team, -999.0) for team in tied_teams}
        return {team: _american_computer_composite_score(records_by_team[team]) for team in tied_teams}
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    if step == "overall_win_pct":
        return {team: _overall_win_pct(records_by_team[team]) for team in tied_teams}
    raise ValueError(f"Unknown American tiebreaker step: {step}")


def _american_computer_composite_score(record: TeamRecord) -> float:
    return float(record.get("computer_composite_score", record.get("team_strength", 0.5)) or 0.0)


# ---------------------------------------------------------------------------
# ACC helpers
# ---------------------------------------------------------------------------
def _acc_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _acc_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _acc_select_one_from_multi_team_tie(tied_teams, records_by_team, rng, depth)


def _acc_select_two_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> list[str]:
    candidates = sorted(set(tied_teams))
    if len(candidates) < 2:
        raise RuntimeError("ACC multi-team tiebreaker needs at least two teams.")
    if len(candidates) == 2:
        return _acc_order_two_team_tie(candidates, records_by_team, rng)
    if depth > 10:
        return rng.sample(candidates, 2)

    h2h_result = _acc_multi_team_head_to_head_group(candidates, records_by_team)
    result = _acc_resolve_multi_team_two_spots_group(
        h2h_result,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, tuple):
        return [result[0], result[1]]
    if isinstance(result, list):
        return _acc_select_two_from_multi_team_tie(result, records_by_team, rng, depth + 1)

    for step in ("common_record", "common_standings", "opponent_wp", "rating"):
        values_by_team = _acc_value_step(step, candidates, records_by_team)
        result = _acc_resolve_multi_team_two_spots_values(
            values_by_team,
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, tuple):
            return [result[0], result[1]]
        if isinstance(result, list):
            return _acc_select_two_from_multi_team_tie(result, records_by_team, rng, depth + 1)

    return rng.sample(candidates, 2)


def _acc_select_one_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _acc_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    h2h_result = _acc_multi_team_head_to_head_group(candidates, records_by_team)
    result = _acc_resolve_multi_team_one_spot_group(
        h2h_result,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return _acc_select_one_from_multi_team_tie(result, records_by_team, rng, depth + 1)

    for step in ("common_record", "common_standings", "opponent_wp", "rating"):
        values_by_team = _acc_value_step(step, candidates, records_by_team)
        result = _acc_resolve_multi_team_one_spot_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            return _acc_select_one_from_multi_team_tie(result, records_by_team, rng, depth + 1)

    return rng.choice(candidates)


def _acc_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("ACC two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            _negative_tuple(_sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            -_team_rating_score(records_by_team[team]),
            rng.random(),
            team,
        ),
    )


def _acc_multi_team_head_to_head_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    lost_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(opponent, {}).get(team, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if lost_all and len(lost_all) < len(tied_teams):
        return [team for team in tied_teams if team not in set(lost_all)]

    return None


def _acc_resolve_multi_team_two_spots_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _acc_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _acc_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _acc_resolve_multi_team_two_spots_values(
    values_by_team: dict[str, float | tuple[float, ...]],
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    top_group = _top_value_group(values_by_team)
    if len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _acc_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _acc_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _acc_resolve_multi_team_one_spot_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _acc_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _acc_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    if step == "common_standings":
        return {
            team: _sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "opponent_wp":
        return {
            team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
            for team in tied_teams
        }
    if step == "rating":
        return {team: _team_rating_score(records_by_team[team]) for team in tied_teams}
    raise ValueError(f"Unknown ACC tiebreaker step: {step}")


# ---------------------------------------------------------------------------
# Conference USA / Mid-American helpers
# ---------------------------------------------------------------------------
def _select_cusa_mac_championship_teams(
    team_records: list[TeamRecord],
    rng: random.Random,
    conference_label: str,
) -> tuple[str, str]:
    if len(team_records) < 2:
        raise ValueError(f"At least two {conference_label} teams are required to select championship game teams.")

    records_by_team = {record["team"]: record for record in team_records}
    participants: list[str] = []
    for group in _standings_groups(list(records_by_team), records_by_team):
        slots = 2 - len(participants)
        if slots <= 0:
            break
        if len(group) <= slots:
            if len(group) == 2:
                participants.extend(_mid_american_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(group)
            continue
        if slots == 2:
            if len(group) == 2:
                participants.extend(_mid_american_order_two_team_tie(group, records_by_team, rng))
            else:
                participants.extend(_mid_american_select_two_from_multi_team_tie(group, records_by_team, rng))
        else:
            participants.append(_mid_american_select_one_team(group, records_by_team, rng))

    if len(participants) < 2:
        raise RuntimeError(f"Unable to select two {conference_label} championship game teams.")
    return participants[0], participants[1]


def _mid_american_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _mid_american_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _mid_american_select_one_from_multi_team_tie(tied_teams, records_by_team, rng, depth)


def _mid_american_select_two_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> list[str]:
    candidates = sorted(set(tied_teams))
    if len(candidates) < 2:
        raise RuntimeError("Mid-American multi-team tiebreaker needs at least two teams.")
    if len(candidates) == 2:
        return _mid_american_order_two_team_tie(candidates, records_by_team, rng)
    if depth > 10:
        return rng.sample(candidates, 2)

    head_to_head_top_group = _mid_american_head_to_head_top_group(candidates, records_by_team)
    result = _mid_american_resolve_multi_team_two_spots_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, tuple):
        return [result[0], result[1]]
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "rating", "common_standings", "opponent_wp"):
        values_by_team = _mid_american_value_step(step, candidates, records_by_team)
        result = _mid_american_resolve_multi_team_two_spots_values(
            values_by_team,
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, tuple):
            return [result[0], result[1]]
        if isinstance(result, list):
            candidates = result

    return rng.sample(candidates, 2)


def _mid_american_select_one_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _mid_american_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    head_to_head_top_group = _mid_american_head_to_head_top_group(candidates, records_by_team)
    result = _mid_american_resolve_multi_team_one_spot_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "rating", "common_standings", "opponent_wp"):
        values_by_team = _mid_american_value_step(step, candidates, records_by_team)
        result = _mid_american_resolve_multi_team_one_spot_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            candidates = result

    return rng.choice(candidates)


def _mid_american_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("Mid-American two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            -_team_rating_score(records_by_team[team]),
            _negative_tuple(_sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            rng.random(),
            team,
        ),
    )


def _mid_american_head_to_head_top_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    return None


def _mid_american_resolve_multi_team_two_spots_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _mid_american_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _mid_american_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _mid_american_resolve_multi_team_two_spots_values(
    values_by_team: dict[str, float | tuple[float, ...]],
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    grouped: dict[float | tuple[float, ...], list[str]] = defaultdict(list)
    for team, value in values_by_team.items():
        grouped[value].append(team)
    ordered_groups = [sorted(grouped[value]) for value in sorted(grouped, reverse=True)]
    top_group = ordered_groups[0]

    if len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        if len(ordered_groups) < 2:
            return None
        second_group = ordered_groups[1]
        if len(second_group) == 1:
            return first, second_group[0]
        second = _mid_american_select_one_team(second_group, records_by_team, rng, depth + 1)
        return first, second
    if len(top_group) == 2:
        ordered = _mid_american_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _mid_american_resolve_multi_team_one_spot_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _mid_american_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _mid_american_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    if step == "rating":
        return {team: _team_rating_score(records_by_team[team]) for team in tied_teams}
    if step == "common_standings":
        return {
            team: _sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "opponent_wp":
        return {
            team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
            for team in tied_teams
        }
    raise ValueError(f"Unknown Mid-American tiebreaker step: {step}")


# ---------------------------------------------------------------------------
# Big 12 helpers
# ---------------------------------------------------------------------------
def _big_12_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _big_12_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _big_12_select_from_multi_team_one_spot(tied_teams, records_by_team, rng, depth)


def _big_12_select_from_multi_team_first_place(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> tuple[str, str]:
    candidates = sorted(set(tied_teams))
    if len(candidates) < 2:
        raise RuntimeError("Big 12 first-place tiebreaker needs at least two teams.")
    if len(candidates) == 2:
        ordered = _big_12_order_two_team_tie(candidates, records_by_team, rng)
        return ordered[0], ordered[1]
    if depth > 10:
        drawn = rng.sample(candidates, 2)
        return drawn[0], drawn[1]

    head_to_head_top_group = _big_12_head_to_head_top_group(candidates, records_by_team)
    result = _big_12_resolve_multi_team_first_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "common_standings", "opponent_wp", "total_wins", "rating"):
        values_by_team = _big_12_value_step(step, candidates, records_by_team)
        result = _big_12_resolve_multi_team_first_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, tuple):
            return result
        if isinstance(result, list):
            candidates = result

    drawn = rng.sample(candidates, 2)
    return drawn[0], drawn[1]


def _big_12_select_from_multi_team_one_spot(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _big_12_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    head_to_head_top_group = _big_12_head_to_head_top_group(candidates, records_by_team)
    result = _big_12_resolve_multi_team_one_spot_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "common_standings", "opponent_wp", "total_wins", "rating"):
        values_by_team = _big_12_value_step(step, candidates, records_by_team)
        result = _big_12_resolve_multi_team_one_spot_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            candidates = result

    return rng.choice(candidates)


def _big_12_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("Big 12 two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            _negative_tuple(_big_12_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            -_big_12_adjusted_total_wins(records_by_team[team]),
            -_big_12_team_rating_score(records_by_team[team]),
            rng.random(),
            team,
        ),
    )


def _big_12_head_to_head_top_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    return None


def _big_12_resolve_multi_team_first_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _big_12_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _big_12_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _big_12_resolve_multi_team_one_spot_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _big_12_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _big_12_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    if step == "common_standings":
        return {
            team: _big_12_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "opponent_wp":
        return {
            team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
            for team in tied_teams
        }
    if step == "total_wins":
        return {team: _big_12_adjusted_total_wins(records_by_team[team]) for team in tied_teams}
    if step == "rating":
        return {team: _big_12_team_rating_score(records_by_team[team]) for team in tied_teams}
    raise ValueError(f"Unknown Big 12 tiebreaker step: {step}")


def _big_12_common_opponent_standing_sequence(
    team: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> tuple[float, ...]:
    tied_set = set(tied_teams)
    possible_opponents = [opponent for opponent in records_by_team if opponent not in tied_set]
    common_opponents = [
        opponent
        for opponent in possible_opponents
        if all(_games_between(tied_team, opponent, head_to_head) > 0 for tied_team in tied_teams)
    ]
    standings_groups: dict[float, list[str]] = defaultdict(list)
    for opponent in common_opponents:
        standings_groups[_conference_win_pct(records_by_team[opponent])].append(opponent)

    sequence: list[float] = []
    for conference_win_pct in sorted(standings_groups, reverse=True):
        opponent_group = standings_groups[conference_win_pct]
        win_pct = _record_against_opponents(team, opponent_group, head_to_head)
        sequence.append(_none_safe(win_pct))
    return tuple(sequence)


def _big_12_adjusted_total_wins(record: TeamRecord) -> float:
    return float(record.get("adjusted_total_wins", record.get("overall_wins", 0)) or 0.0)


def _big_12_team_rating_score(record: TeamRecord) -> float:
    return float(record.get("team_rating_score", record.get("team_strength", 0.5)) or 0.0)


# ---------------------------------------------------------------------------
# Big Ten helpers
# ---------------------------------------------------------------------------
def _big_ten_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _big_ten_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _big_ten_select_from_multi_team_one_spot(tied_teams, records_by_team, rng, depth)


def _big_ten_select_from_multi_team_first_place(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> tuple[str, str]:
    candidates = sorted(set(tied_teams))
    if len(candidates) < 2:
        raise RuntimeError("Big Ten first-place tiebreaker needs at least two teams.")
    if len(candidates) == 2:
        ordered = _big_ten_order_two_team_tie(candidates, records_by_team, rng)
        return ordered[0], ordered[1]
    if depth > 10:
        drawn = rng.sample(candidates, 2)
        return drawn[0], drawn[1]

    head_to_head_top_group = _big_ten_head_to_head_top_group(candidates, records_by_team)
    result = _big_ten_resolve_multi_team_first_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "common_standings", "opponent_wp", "rating"):
        values_by_team = _big_ten_value_step(step, candidates, records_by_team)
        result = _big_ten_resolve_multi_team_first_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, tuple):
            return result
        if isinstance(result, list):
            candidates = result

    drawn = rng.sample(candidates, 2)
    return drawn[0], drawn[1]


def _big_ten_select_from_multi_team_one_spot(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _big_ten_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    head_to_head_top_group = _big_ten_head_to_head_top_group(candidates, records_by_team)
    result = _big_ten_resolve_multi_team_one_spot_top_group(
        head_to_head_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("common_record", "common_standings", "opponent_wp", "rating"):
        values_by_team = _big_ten_value_step(step, candidates, records_by_team)
        result = _big_ten_resolve_multi_team_one_spot_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            candidates = result

    return rng.choice(candidates)


def _big_ten_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("Big Ten two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            _negative_tuple(_sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)),
            -_none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head)),
            -_big_ten_team_rating_score(records_by_team[team]),
            rng.random(),
            team,
        ),
    )


def _big_ten_head_to_head_top_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    return None


def _big_ten_resolve_multi_team_first_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _big_ten_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _big_ten_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _big_ten_resolve_multi_team_one_spot_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _big_ten_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _big_ten_team_rating_score(record: TeamRecord) -> float:
    return float(record.get("team_rating_score", record.get("team_strength", 0.5)) or 0.0)


def _big_ten_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    if step == "common_standings":
        return {
            team: _sec_common_opponent_standing_sequence(team, tied_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "opponent_wp":
        return {
            team: _none_safe(_opponents_cumulative_conference_win_pct(team, records_by_team, head_to_head))
            for team in tied_teams
        }
    if step == "rating":
        return {team: _big_ten_team_rating_score(records_by_team[team]) for team in tied_teams}
    raise ValueError(f"Unknown Big Ten tiebreaker step: {step}")


# ---------------------------------------------------------------------------
# Mountain West helpers
# ---------------------------------------------------------------------------
MOUNTAIN_WEST_CFP_RANK_CUTOFF = 25


def _mountain_west_select_one_team(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    tied_teams = sorted(set(tied_teams))
    if len(tied_teams) == 1:
        return tied_teams[0]
    if len(tied_teams) == 2:
        return _mountain_west_order_two_team_tie(tied_teams, records_by_team, rng)[0]
    return _mountain_west_select_one_from_multi_team_tie(tied_teams, records_by_team, rng, depth)


def _mountain_west_select_two_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> list[str]:
    candidates = sorted(set(tied_teams))
    if len(candidates) < 2:
        raise RuntimeError("Mountain West multi-team tiebreaker needs at least two teams.")
    if len(candidates) == 2:
        return _mountain_west_order_two_team_tie(candidates, records_by_team, rng)
    if depth > 10:
        return rng.sample(candidates, 2)

    h2h_top_group = _mountain_west_head_to_head_top_group(candidates, records_by_team)
    result = _mountain_west_resolve_multi_team_two_spots_top_group(
        h2h_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, tuple):
        return [result[0], result[1]]
    if isinstance(result, list):
        candidates = result

    for step in ("cfp_or_composite", "adjusted_overall_win_pct", "standings_sequence", "common_record"):
        values_by_team = _mountain_west_value_step(step, candidates, records_by_team)
        result = _mountain_west_resolve_multi_team_two_spots_values(
            values_by_team,
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, tuple):
            return [result[0], result[1]]
        if isinstance(result, list):
            candidates = result

    return rng.sample(candidates, 2)


def _mountain_west_select_one_from_multi_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _mountain_west_order_two_team_tie(candidates, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    h2h_top_group = _mountain_west_head_to_head_top_group(candidates, records_by_team)
    result = _mountain_west_resolve_multi_team_one_spot_top_group(
        h2h_top_group,
        candidates,
        records_by_team,
        rng,
        depth,
    )
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        candidates = result

    for step in ("cfp_or_composite", "adjusted_overall_win_pct", "standings_sequence", "common_record"):
        values_by_team = _mountain_west_value_step(step, candidates, records_by_team)
        result = _mountain_west_resolve_multi_team_one_spot_top_group(
            _top_value_group(values_by_team),
            candidates,
            records_by_team,
            rng,
            depth,
        )
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            candidates = result

    return rng.choice(candidates)


def _mountain_west_order_two_team_tie(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("Mountain West two-team tiebreaker requires exactly two teams.")

    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    first, second = tied_teams
    if head_to_head.get(first, {}).get(second, 0) > head_to_head.get(second, {}).get(first, 0):
        return [first, second]
    if head_to_head.get(second, {}).get(first, 0) > head_to_head.get(first, {}).get(second, 0):
        return [second, first]

    return sorted(
        tied_teams,
        key=lambda team: (
            -_mountain_west_value_step("cfp_or_composite", tied_teams, records_by_team)[team],
            -_adjusted_overall_win_pct(records_by_team[team]),
            _negative_tuple(
                _mountain_west_next_standings_opponent_sequence(
                    team,
                    tied_teams,
                    records_by_team,
                    head_to_head,
                )
            ),
            -_none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head)),
            rng.random(),
            team,
        ),
    )


def _mountain_west_head_to_head_top_group(
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[str] | None:
    head_to_head = _shared_head_to_head(records_by_team)
    complete_round_robin = all(
        _games_between(first, second, head_to_head) > 0
        for index, first in enumerate(tied_teams)
        for second in tied_teams[index + 1 :]
    )

    if complete_round_robin:
        h2h_values = {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
        top_group = _top_value_group(h2h_values)
        if len(top_group) == len(tied_teams):
            return None
        return top_group

    beat_all = [
        team
        for team in tied_teams
        if all(head_to_head.get(team, {}).get(opponent, 0) > 0 for opponent in tied_teams if opponent != team)
    ]
    if len(beat_all) == 1:
        return beat_all

    return None


def _mountain_west_resolve_multi_team_two_spots_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        second = _mountain_west_select_one_team(
            [team for team in tied_teams if team != first],
            records_by_team,
            rng,
            depth + 1,
        )
        return first, second
    if len(top_group) == 2:
        ordered = _mountain_west_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _mountain_west_resolve_multi_team_two_spots_values(
    values_by_team: dict[str, float | tuple[float, ...]],
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> tuple[str, str] | list[str] | None:
    grouped: dict[float | tuple[float, ...], list[str]] = defaultdict(list)
    for team, value in values_by_team.items():
        grouped[value].append(team)
    ordered_groups = [sorted(grouped[value]) for value in sorted(grouped, reverse=True)]
    top_group = ordered_groups[0]

    if len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        first = top_group[0]
        if len(ordered_groups) < 2:
            return None
        second_group = ordered_groups[1]
        if len(second_group) == 1:
            return first, second_group[0]
        second = _mountain_west_select_one_team(second_group, records_by_team, rng, depth + 1)
        return first, second
    if len(top_group) == 2:
        ordered = _mountain_west_order_two_team_tie(top_group, records_by_team, rng)
        return ordered[0], ordered[1]
    return top_group


def _mountain_west_resolve_multi_team_one_spot_top_group(
    top_group: list[str] | None,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int,
) -> str | list[str] | None:
    if top_group is None or len(top_group) == len(tied_teams):
        return None
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _mountain_west_order_two_team_tie(top_group, records_by_team, rng)[0]
    return top_group


def _mountain_west_value_step(
    step: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    conference_teams = list(records_by_team)
    if step == "cfp_or_composite":
        cfp_candidates = {
            team: -float(records_by_team[team]["cfp_rank"])
            for team in tied_teams
            if _is_cfp_ranked(records_by_team[team])
            and not records_by_team[team].get("lost_final_conference_week")
        }
        if cfp_candidates:
            return {team: cfp_candidates.get(team, -999.0) for team in tied_teams}
        return {team: -_computer_composite_rank(records_by_team[team]) for team in tied_teams}
    if step == "adjusted_overall_win_pct":
        return {team: _adjusted_overall_win_pct(records_by_team[team]) for team in tied_teams}
    if step == "standings_sequence":
        return {
            team: _mountain_west_next_standings_opponent_sequence(team, tied_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "common_record":
        return {
            team: _none_safe(_common_conference_opponent_win_pct(team, tied_teams, conference_teams, head_to_head))
            for team in tied_teams
        }
    raise ValueError(f"Unknown Mountain West tiebreaker step: {step}")


def _mountain_west_next_standings_opponent_sequence(
    team: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> tuple[float, ...]:
    tied_set = set(tied_teams)
    possible_opponents = [opponent for opponent in records_by_team if opponent not in tied_set]
    standings_groups: dict[float, list[str]] = defaultdict(list)
    for opponent in possible_opponents:
        standings_groups[_conference_win_pct(records_by_team[opponent])].append(opponent)

    sequence: list[float] = []
    for conference_win_pct in sorted(standings_groups, reverse=True):
        opponent_group = standings_groups[conference_win_pct]
        if not all(
            all(_games_between(tied_team, opponent, head_to_head) > 0 for opponent in opponent_group)
            for tied_team in tied_teams
        ):
            continue
        sequence.append(_none_safe(_record_against_opponents(team, opponent_group, head_to_head)))
    return tuple(sequence)


def _is_cfp_ranked(record: TeamRecord) -> bool:
    rank = record.get("cfp_rank")
    if rank is None:
        return False
    return float(rank) <= MOUNTAIN_WEST_CFP_RANK_CUTOFF


def _computer_composite_rank(record: TeamRecord) -> float:
    rank = record.get("computer_composite_rank")
    if rank is not None:
        return float(rank)
    score = float(record.get("computer_composite_score", record.get("team_strength", 0.5)) or 0.0)
    return 1_000.0 - score


def _team_rating_score(record: TeamRecord) -> float:
    return float(record.get("team_rating_score", record.get("team_strength", 0.5)) or 0.0)


# ---------------------------------------------------------------------------
# Sun Belt helpers
# ---------------------------------------------------------------------------
def _sun_belt_select_division_champion(
    division_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> str:
    standings_groups = _standings_groups(division_teams, records_by_team)
    top_group = standings_groups[0]
    if len(top_group) == 1:
        return top_group[0]
    if len(top_group) == 2:
        return _sun_belt_order_two_team_tie(top_group, division_teams, records_by_team, rng)[0]
    return _sun_belt_select_one_from_multi_team_tie(top_group, division_teams, records_by_team, rng)


def _sun_belt_select_one_from_multi_team_tie(
    tied_teams: list[str],
    division_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
    depth: int = 0,
) -> str:
    candidates = sorted(set(tied_teams))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        return _sun_belt_order_two_team_tie(candidates, division_teams, records_by_team, rng)[0]
    if depth > 10:
        return rng.choice(candidates)

    for step in (
        "head_to_head",
        "division_record",
        "division_standings_sequence",
        "common_non_divisional_record",
        "cfp_or_composite",
        "fbs_overall_win_pct",
    ):
        values_by_team = _sun_belt_value_step(step, candidates, division_teams, records_by_team)
        top_group = _top_value_group(values_by_team)
        if len(top_group) == len(candidates):
            continue
        if len(top_group) == 1:
            return top_group[0]
        candidates = top_group

    return rng.choice(candidates)


def _sun_belt_order_two_team_tie(
    tied_teams: list[str],
    division_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    rng: random.Random,
) -> list[str]:
    if len(tied_teams) != 2:
        raise ValueError("Sun Belt two-team tiebreaker requires exactly two teams.")

    values_by_step = [
        _sun_belt_value_step("head_to_head", tied_teams, division_teams, records_by_team),
        _sun_belt_value_step("division_record", tied_teams, division_teams, records_by_team),
        _sun_belt_value_step("division_standings_sequence", tied_teams, division_teams, records_by_team),
        _sun_belt_value_step("common_non_divisional_record", tied_teams, division_teams, records_by_team),
        _sun_belt_value_step("cfp_or_composite", tied_teams, division_teams, records_by_team),
        _sun_belt_value_step("fbs_overall_win_pct", tied_teams, division_teams, records_by_team),
    ]

    return sorted(
        tied_teams,
        key=lambda team: (
            *(_sun_belt_sort_value(values_by_team[team]) for values_by_team in values_by_step),
            rng.random(),
            team,
        ),
    )


def _sun_belt_value_step(
    step: str,
    tied_teams: list[str],
    division_teams: list[str],
    records_by_team: dict[str, TeamRecord],
) -> dict[str, float | tuple[float, ...]]:
    head_to_head = _shared_head_to_head(records_by_team)
    if step == "head_to_head":
        return {
            team: _none_safe(_head_to_head_win_pct(team, tied_teams, head_to_head))
            for team in tied_teams
        }
    if step == "division_record":
        return {team: _division_win_pct(records_by_team[team]) for team in tied_teams}
    if step == "division_standings_sequence":
        return {
            team: _sun_belt_division_standings_sequence(team, tied_teams, division_teams, records_by_team, head_to_head)
            for team in tied_teams
        }
    if step == "common_non_divisional_record":
        return {
            team: _none_safe(_sun_belt_common_non_divisional_win_pct(team, tied_teams, records_by_team, head_to_head))
            for team in tied_teams
        }
    if step == "cfp_or_composite":
        cfp_candidates = {
            team: -float(records_by_team[team]["cfp_rank"])
            for team in tied_teams
            if records_by_team[team].get("cfp_rank") is not None
            and not records_by_team[team].get("lost_final_conference_week")
        }
        if cfp_candidates:
            return {team: cfp_candidates.get(team, -999.0) for team in tied_teams}
        return {team: -_computer_composite_rank(records_by_team[team]) for team in tied_teams}
    if step == "fbs_overall_win_pct":
        return {team: _fbs_overall_win_pct(records_by_team[team]) for team in tied_teams}
    raise ValueError(f"Unknown Sun Belt tiebreaker step: {step}")


def _sun_belt_division_standings_sequence(
    team: str,
    tied_teams: list[str],
    division_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> tuple[float, ...]:
    tied_set = set(tied_teams)
    lower_division_teams = [division_team for division_team in division_teams if division_team not in tied_set]
    standings_groups: dict[float, list[str]] = defaultdict(list)
    for division_team in lower_division_teams:
        standings_groups[_conference_win_pct(records_by_team[division_team])].append(division_team)

    sequence: list[float] = []
    for conference_win_pct in sorted(standings_groups, reverse=True):
        opponent_group = standings_groups[conference_win_pct]
        sequence.append(_none_safe(_record_against_opponents(team, opponent_group, head_to_head)))
    return tuple(sequence)


def _sun_belt_common_non_divisional_win_pct(
    team: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> float | None:
    division = records_by_team[team].get("division") or sun_belt_division_for_team(team)
    possible_opponents = [
        opponent
        for opponent, record in records_by_team.items()
        if opponent not in tied_teams
        and (record.get("division") or sun_belt_division_for_team(opponent)) != division
    ]
    common_opponents = [
        opponent
        for opponent in possible_opponents
        if all(_games_between(tied_team, opponent, head_to_head) > 0 for tied_team in tied_teams)
    ]
    return _record_against_opponents(team, common_opponents, head_to_head)


def _sun_belt_sort_value(value: float | tuple[float, ...]) -> float | tuple[float, ...]:
    if isinstance(value, tuple):
        return _negative_tuple(value)
    return -value


def _top_value_group(values_by_team: dict[str, float | tuple[float, ...]]) -> list[str]:
    best_value = max(values_by_team.values())
    return sorted(team for team, value in values_by_team.items() if value == best_value)


def _conference_win_pct(record: TeamRecord) -> float:
    wins = float(record.get("conference_wins", 0))
    losses = float(record.get("conference_losses", 0))
    games = wins + losses
    if games <= 0:
        return 0.0
    return wins / games


def _conference_games_played(record: TeamRecord) -> int:
    return int(float(record.get("conference_wins", 0)) + float(record.get("conference_losses", 0)))


def _division_win_pct(record: TeamRecord) -> float:
    wins = float(record.get("divisional_wins", 0))
    losses = float(record.get("divisional_losses", 0))
    games = wins + losses
    if games <= 0:
        return 0.0
    return wins / games


def _overall_win_pct(record: TeamRecord) -> float:
    wins = float(record.get("overall_wins", 0))
    losses = float(record.get("overall_losses", 0))
    games = wins + losses
    if games <= 0:
        return 0.0
    return wins / games


def _adjusted_overall_win_pct(record: TeamRecord) -> float:
    wins = float(record.get("adjusted_total_wins", record.get("overall_wins", 0)) or 0.0)
    losses = float(record.get("overall_losses", 0) or 0.0)
    games = float(record.get("overall_wins", 0) or 0.0) + losses
    if games <= 0:
        return 0.0
    return wins / games


def _fbs_overall_win_pct(record: TeamRecord) -> float:
    wins = float(record.get("fbs_wins", 0))
    losses = float(record.get("fbs_losses", 0))
    games = wins + losses
    if games <= 0:
        return 0.0
    return wins / games


def _head_to_head_win_pct(team: str, tied_teams: list[str], head_to_head: HeadToHeadMap) -> float | None:
    wins = 0
    games = 0
    for opponent in tied_teams:
        if opponent == team:
            continue
        wins_against_opponent = head_to_head.get(team, {}).get(opponent, 0)
        losses_against_opponent = head_to_head.get(opponent, {}).get(team, 0)
        wins += wins_against_opponent
        games += wins_against_opponent + losses_against_opponent
    if games == 0:
        return None
    return wins / games


def _common_conference_opponent_win_pct(
    team: str,
    tied_teams: list[str],
    conference_teams: list[str],
    head_to_head: HeadToHeadMap,
) -> float | None:
    tied_set = set(tied_teams)
    possible_opponents = [opponent for opponent in conference_teams if opponent not in tied_set]
    common_opponents = [
        opponent
        for opponent in possible_opponents
        if all(_games_between(tied_team, opponent, head_to_head) > 0 for tied_team in tied_teams)
    ]
    if not common_opponents:
        return None

    wins = 0
    games = 0
    for opponent in common_opponents:
        wins_against_opponent = head_to_head.get(team, {}).get(opponent, 0)
        losses_against_opponent = head_to_head.get(opponent, {}).get(team, 0)
        wins += wins_against_opponent
        games += wins_against_opponent + losses_against_opponent
    if games == 0:
        return None
    return wins / games


def _sec_common_opponent_standing_sequence(
    team: str,
    tied_teams: list[str],
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> tuple[float, ...]:
    tied_set = set(tied_teams)
    possible_opponents = [opponent for opponent in records_by_team if opponent not in tied_set]
    common_opponents = [
        opponent
        for opponent in possible_opponents
        if all(_games_between(tied_team, opponent, head_to_head) > 0 for tied_team in tied_teams)
    ]
    standings_groups: dict[float, list[str]] = defaultdict(list)
    for opponent in common_opponents:
        standings_groups[_conference_win_pct(records_by_team[opponent])].append(opponent)

    sequence: list[float] = []
    for conference_win_pct in sorted(standings_groups, reverse=True):
        opponents = standings_groups[conference_win_pct]
        ordered_opponent_groups = _break_lower_standings_tie_by_head_to_head(opponents, records_by_team)
        for opponent_group in ordered_opponent_groups:
            win_pct = _record_against_opponents(team, opponent_group, head_to_head)
            sequence.append(_none_safe(win_pct))
    return tuple(sequence)


def _break_lower_standings_tie_by_head_to_head(
    tied_opponents: list[str],
    records_by_team: dict[str, TeamRecord],
) -> list[list[str]]:
    if len(tied_opponents) <= 1:
        return [tied_opponents]

    head_to_head = _shared_head_to_head(records_by_team)
    values = {
        team: _none_safe(_head_to_head_win_pct(team, tied_opponents, head_to_head))
        for team in tied_opponents
    }
    grouped: dict[float, list[str]] = defaultdict(list)
    for team, value in values.items():
        grouped[value].append(team)

    if len(grouped) == 1:
        return [tied_opponents]

    return [sorted(grouped[value]) for value in sorted(grouped, reverse=True)]


def _opponents_cumulative_conference_win_pct(
    team: str,
    records_by_team: dict[str, TeamRecord],
    head_to_head: HeadToHeadMap,
) -> float | None:
    opponents = set(head_to_head.get(team, {}))
    for opponent, wins_by_opponent in head_to_head.items():
        if team in wins_by_opponent:
            opponents.add(opponent)

    conference_wins = 0.0
    conference_games = 0.0
    for opponent in opponents:
        record = records_by_team.get(opponent)
        if record is None:
            continue
        wins = float(record.get("conference_wins", 0))
        losses = float(record.get("conference_losses", 0))
        conference_wins += wins
        conference_games += wins + losses
    if conference_games <= 0:
        return None
    return conference_wins / conference_games


def _record_against_opponents(team: str, opponents: list[str], head_to_head: HeadToHeadMap) -> float | None:
    wins = 0
    games = 0
    for opponent in opponents:
        wins_against_opponent = head_to_head.get(team, {}).get(opponent, 0)
        losses_against_opponent = head_to_head.get(opponent, {}).get(team, 0)
        wins += wins_against_opponent
        games += wins_against_opponent + losses_against_opponent
    if games == 0:
        return None
    return wins / games


def _games_between(first: str, second: str, head_to_head: HeadToHeadMap) -> int:
    return head_to_head.get(first, {}).get(second, 0) + head_to_head.get(second, {}).get(first, 0)


def _shared_head_to_head(records_by_team: dict[str, TeamRecord]) -> HeadToHeadMap:
    for record in records_by_team.values():
        return record.get("head_to_head_wins", {})
    return {}


def _championship_game_rating(metrics: TeamRecord) -> float:
    team_strength = float(metrics.get("team_strength") or 0.5)
    overall_win_pct = float(metrics.get("overall_win_pct") or 0.0)
    conference_win_pct = float(metrics.get("conference_win_pct") or 0.0)
    strength_of_schedule = float(metrics.get("strength_of_schedule") or 0.5)
    return (
        team_strength * 0.45
        + overall_win_pct * 0.25
        + conference_win_pct * 0.20
        + strength_of_schedule * 0.10
    )


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _none_safe(value: float | None) -> float:
    if value is None:
        return -1.0
    return value


def _negative_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(-value for value in values)

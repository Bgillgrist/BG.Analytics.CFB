# Prediction Tables Handoff

This handoff covers the snapshot tables created for game predictions, team ratings, ranking projections, and season predictions:

- `public.game_prediction_runs`
- `public.game_predictions_full`
- `public.team_rating_runs`
- `public.team_ratings`
- `public.ranking_projection_runs`
- `public.ranking_projections_full`
- `public.season_prediction_runs`
- `public.season_predictions_full`

The dashboard should treat these as snapshot tables. The detail tables are not latest-only tables; every detail row belongs to a run row. To show the current view, first select the latest successful run, then join to that run's detail rows.

## Pipeline Order

Nightly ETL runs these prediction jobs in this order:

```bash
python -m etl.jobs.prediction_updates.update_game_predictions_full
python -m etl.jobs.prediction_updates.update_team_ratings_full
python -m etl.jobs.prediction_updates.update_ranking_projections
python -m etl.jobs.prediction_updates.update_season_predictions_full
```

The dependency chain is:

```text
game_prediction_runs
  -> game_predictions_full
  -> team_rating_runs.game_prediction_run_id
  -> team_ratings

game_prediction_runs
  -> game_predictions_full
  -> ranking_projection_runs.game_prediction_run_id
  -> ranking_projections_full
  -> season_prediction_runs.ranking_projection_run_id
  -> season_predictions_full
```

This matters for dashboard debugging: a season prediction run is tied to a ranking projection run, and that ranking projection run is tied to a game prediction run. That gives one coherent as-of snapshot for game probabilities, projected rankings, CFP selection, and playoff simulation.

Team rating runs also tie directly to a game prediction run. They use completed FBS-vs-FBS margins as of the run date plus projected margins from that game prediction snapshot, then solve a weighted SRS-style rating and home-field advantage.

## Run Table Pattern

Each `*_runs` table has the same broad lifecycle:

- `running`: metadata row exists while the ETL is working.
- `success`: detail rows were inserted for this run.
- `duplicate`: the ETL ran, produced the same prediction hash as the latest comparable successful snapshot, and skipped inserting duplicate detail rows.
- `failed`: the ETL created a run row but failed before completion.

For dashboard pages, use `status = 'success'`. Do not use `duplicate` runs for detail joins because duplicate runs generally have `inserted_row_count = 0`.

Normal current-state dashboard queries should usually use:

```sql
WHERE season = :season
  AND run_type IN ('nightly', 'manual')
  AND status = 'success'
ORDER BY created_at DESC
LIMIT 1
```

For historical backfill views, use `run_type = 'backfill'` and select by `run_date`.

## Table 1: `public.game_prediction_runs`

One row per attempted game-prediction snapshot.

Primary key:

- `game_prediction_run_id`

Important columns:

- `season`: target season.
- `run_date`: as-of date for the prediction run.
- `run_type`: `nightly`, `manual`, or `backfill`.
- `etl_run_id`: upstream ETL run id.
- `created_at`, `completed_at`: run timing.
- `status`: `running`, `success`, `duplicate`, or `failed`.
- `model_version`: combined game model label.
- `prediction_hash`: hash of the canonical game prediction payload.
- `duplicate_of_run_id`: points to the prior successful run when this run was duplicate.
- `row_count`: number of game rows prepared.
- `inserted_row_count`: number of detail rows inserted.
- `fcs_count`: number of games involving at least one FCS team.
- `notes`: optional run notes.
- `error_message`: failure details.

## Table 2: `public.game_predictions_full`

One row per game in one successful game prediction run.

Primary key:

- `(game_prediction_run_id, gameid)`

Important columns:

- `game_prediction_run_id`: joins to `game_prediction_runs`.
- `gameid`: source `game_data.id`, stored as text.
- `season`, `week`
- `home_team`, `away_team`
- `homepoints`, `awaypoints`: final score when available.
- `homespread`: model spread from the home team's perspective. Negative means home favored.
- `awayspread`: inverse of `homespread`.
- `totalpred`: predicted total points.
- `homewinprob`: probability home team wins.
- `awaywinprob`: probability away team wins.
- `model_version`: row-level game model version.
- `prediction_type`: usually `FBS`; `FCS` means at least one team is FCS.
- `prediction_row_hash`: row-level prediction hash.
- `created_at`: detail insert timestamp.

Dashboard uses:

- Game cards, matchup pages, spreads, totals, win probabilities.
- Historical movement by comparing the same `gameid` across different successful runs.

Latest game predictions:

```sql
WITH latest_run AS (
  SELECT game_prediction_run_id
  FROM public.game_prediction_runs
  WHERE season = :season
    AND run_type IN ('nightly', 'manual')
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT g.*
FROM public.game_predictions_full g
JOIN latest_run r USING (game_prediction_run_id)
ORDER BY g.week, g.home_team, g.away_team;
```

## Table 3: `public.team_rating_runs`

One row per attempted team-rating snapshot.

Primary key:

- `team_rating_run_id`

Important columns:

- `season`, `run_date`, `run_type`
- `etl_run_id`
- `game_prediction_run_id`: the exact game prediction snapshot used for projected margins.
- `created_at`, `completed_at`
- `status`
- `model_version`
- `rating_hash`: hash of the canonical team-rating payload.
- `duplicate_of_run_id`
- `row_count`, `inserted_row_count`
- `completed_game_count`: source games using actual final margins as of `run_date`.
- `projected_game_count`: source games using projected margins.
- `dropped_game_count`: games skipped because no completed margin, spread, or win probability was available.
- `home_field_advantage`: solved HFA from the least-squares system.
- `margin_source`: `completed`, `completed+spread`, `completed+winprob`, `mixed`, etc.
- `notes`, `error_message`

## Table 4: `public.team_ratings`

One row per team in one successful team-rating run.

Primary key:

- `(team_rating_run_id, team)`

Important columns:

- `team_rating_run_id`
- `season`, `run_date`, `run_type`
- `model_version`
- `team`, `conference`, `classification`
- `rank`: rank by `team_rating`, best team is 1.
- `team_rating`: points versus an average FBS team.
- `power_rating`: same value as `team_rating`, kept for dashboard naming compatibility.
- `home_field_advantage`: run-level solved HFA repeated on each detail row.
- `completed_games`, `projected_games`, `total_games`
- `average_margin_signal`, `average_weighted_margin_signal`: team-perspective source margin summaries.
- `completed_game_weight`, `projected_game_weight`, `max_margin_signal`: model constants used by the solve.
- `margin_source`
- `game_prediction_run_id`
- `rating_row_hash`
- `notes`

Latest team ratings:

```sql
WITH latest_run AS (
  SELECT team_rating_run_id
  FROM public.team_rating_runs
  WHERE season = :season
    AND run_type IN ('nightly', 'manual')
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT t.*
FROM public.team_ratings t
JOIN latest_run r USING (team_rating_run_id)
ORDER BY t.rank;
```

## Table 5: `public.ranking_projection_runs`

One row per attempted ranking projection snapshot.

Primary key:

- `ranking_projection_run_id`

Important columns:

- `ranking_projection_run_id`: run id for ranking projection snapshot.
- `season`, `run_date`, `run_type`
- `etl_run_id`
- `game_prediction_run_id`: the exact game prediction snapshot used to project the rest of the season.
- `created_at`, `completed_at`
- `status`
- `model_version`
- `prediction_hash`
- `duplicate_of_run_id`
- `row_count`, `inserted_row_count`
- `notes`, `error_message`

Dashboard/debug use:

- This is the bridge from rankings to game predictions.
- If ranking values look strange, check which `game_prediction_run_id` the ranking run used.

## Table 6: `public.ranking_projections_full`

One row per team in one successful ranking projection run.

Primary key:

- `(ranking_projection_run_id, team)`

Identity columns:

- `ranking_projection_run_id`
- `season`, `run_date`, `run_type`
- `model_version`
- `team`
- `conference`
- `classification`
- `created_at`

Projected rank columns:

- `projected_ap_ranking`: projected current AP ranking.
- `projected_end_ap_ranking`: projected end-of-season AP ranking.
- `projected_cfp_ranking`: projected current CFP ranking.
- `projected_end_cfp_ranking`: projected end-of-season CFP ranking.

Projected score columns:

- `projected_ap_score`
- `projected_end_ap_score`
- `projected_cfp_score`
- `projected_end_cfp_score`

Model component columns:

- `resume_score`: current resume estimate.
- `projected_resume_score`: projected final resume estimate.
- `power_score`: team quality/power component.
- `poll_inertia_score`: ranking inertia from current/previous polls.

Record columns:

- `current_wins`
- `current_losses`
- `current_conference_wins`
- `current_conference_losses`
- `projected_wins`
- `projected_losses`
- `projected_conference_wins`
- `projected_conference_losses`

Poll input columns:

- `current_ap_rank`
- `previous_ap_rank`
- `current_coaches_rank`
- `current_cfp_rank`
- `previous_cfp_rank`

Strength/input columns:

- `strength_of_schedule`
- `remaining_strength_of_schedule`
- `team_strength`
- `talent_score`
- `recruiting_score`
- `returning_production_score`
- `advanced_stats_season`

Other columns:

- `game_prediction_run_id`: copied onto the detail row for convenience.
- `prediction_hash`
- `prediction_type`
- `notes`

Dashboard uses:

- Rankings pages.
- Team profile ranking trend cards.
- CFP selection/debug views.
- Model component breakdowns.

Latest ranking projections:

```sql
WITH latest_run AS (
  SELECT ranking_projection_run_id
  FROM public.ranking_projection_runs
  WHERE season = :season
    AND run_type IN ('nightly', 'manual')
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT r.*
FROM public.ranking_projections_full r
JOIN latest_run lr USING (ranking_projection_run_id)
ORDER BY r.projected_cfp_ranking NULLS LAST, r.projected_ap_ranking NULLS LAST, r.team;
```

## Table 7: `public.season_prediction_runs`

One row per attempted season prediction snapshot.

Primary key:

- `season_prediction_run_id`

Important columns:

- `season_prediction_run_id`
- `season`, `run_date`, `run_type`
- `etl_run_id`
- `ranking_projection_run_id`: the exact ranking projection snapshot used for CFP selection and playoff simulation.
- `created_at`, `completed_at`
- `status`
- `model_version`
- `prediction_hash`
- `duplicate_of_run_id`
- `row_count`, `inserted_row_count`
- `simulations`: number of Monte Carlo simulations.
- `notes`, `error_message`

Dashboard/debug use:

- This is the entry point for season-level team probabilities.
- Join through `ranking_projection_run_id` if you want to explain which rankings snapshot drove CFP odds.

## Table 8: `public.season_predictions_full`

One row per team in one successful season prediction run.

Primary key:

- `(season_prediction_run_id, team)`

Identity columns:

- `season_prediction_run_id`
- `season`, `run_date`, `run_type`
- `model_version`
- `team`
- `conference`
- `division`
- `classification`
- `created_at`

Projected record columns:

- `projected_wins`
- `projected_losses`
- `projected_conference_wins`
- `projected_conference_losses`
- `expected_number_of_wins`

Win distribution columns:

- `probability_0_wins`
- `probability_1_wins`
- `probability_2_wins`
- `probability_3_wins`
- `probability_4_wins`
- `probability_5_wins`
- `probability_6_wins`
- `probability_7_wins`
- `probability_8_wins`
- `probability_9_wins`
- `probability_10_wins`
- `probability_11_wins`
- `probability_12_wins`
- `probability_13_wins`

Conference outcome columns:

- `conference_championship_game_prob`: probability of making conference championship game.
- `conference_champion_prob`: probability of winning conference championship.

CFP columns:

- `playoff_prob`: probability of making the CFP field.
- `cfp_bye_prob`: probability of receiving a first-round CFP bye.
- `cfp_at_large_prob`: probability of making CFP as an at-large.
- `cfp_auto_bid_prob`: probability of making CFP as an automatic bid.
- `national_championship_game_prob`: probability of reaching the national championship game.
- `national_champion_prob`: probability of winning the national championship.

Other probability columns:

- `bowl_eligible_prob`: probability of at least six regular-season wins.

Ranking/resume columns:

- `projected_ap_ranking`
- `projected_cfp_ranking`
- `resume_ranking`
- `strength_of_schedule`
- `remaining_strength_of_schedule`

Other columns:

- `simulations`
- `prediction_hash`
- `prediction_type`
- `notes`

Latest season predictions:

```sql
WITH latest_run AS (
  SELECT season_prediction_run_id
  FROM public.season_prediction_runs
  WHERE season = :season
    AND run_type IN ('nightly', 'manual')
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT s.*
FROM public.season_predictions_full s
JOIN latest_run lr USING (season_prediction_run_id)
ORDER BY s.playoff_prob DESC, s.projected_wins DESC, s.team;
```

## Useful Combined Query

This joins latest season probabilities to the ranking run and game run that produced them.

```sql
WITH latest_season_run AS (
  SELECT *
  FROM public.season_prediction_runs
  WHERE season = :season
    AND run_type IN ('nightly', 'manual')
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT
  sp.team,
  sp.conference,
  sp.projected_wins,
  sp.projected_losses,
  sp.playoff_prob,
  sp.cfp_bye_prob,
  sp.cfp_auto_bid_prob,
  sp.cfp_at_large_prob,
  sp.national_champion_prob,
  sp.projected_cfp_ranking,
  rp.projected_end_cfp_ranking,
  rp.projected_end_cfp_score,
  lsr.season_prediction_run_id,
  lsr.ranking_projection_run_id,
  rpr.game_prediction_run_id
FROM latest_season_run lsr
JOIN public.season_predictions_full sp
  ON sp.season_prediction_run_id = lsr.season_prediction_run_id
LEFT JOIN public.ranking_projection_runs rpr
  ON rpr.ranking_projection_run_id = lsr.ranking_projection_run_id
LEFT JOIN public.ranking_projections_full rp
  ON rp.ranking_projection_run_id = lsr.ranking_projection_run_id
 AND rp.team = sp.team
ORDER BY sp.playoff_prob DESC, sp.national_champion_prob DESC, sp.team;
```

## Manual Run Variables

Game predictions:

- `GAME_PREDICTION_RUN_TYPE`
- `GAME_PREDICTION_RUN_DATE`
- `GAME_PREDICTION_RUN_NOTES`

Ranking projections:

- `RANKING_PROJECTION_RUN_TYPE`
- `RANKING_PROJECTION_RUN_DATE`
- `RANKING_PROJECTION_RUN_NOTES`
- `RANKING_PROJECTION_GAME_PREDICTION_RUN_ID`
- `RANKING_PROJECTION_POLL_THROUGH_WEEK`

Season predictions:

- `SEASON_PREDICTION_RUN_TYPE`
- `SEASON_PREDICTION_RUN_DATE`
- `SEASON_PREDICTION_RUN_NOTES`
- `SEASON_PREDICTION_SIMULATIONS`
- `SEASON_PREDICTION_RANDOM_SEED`
- `SEASON_PREDICTION_RANKING_PROJECTION_RUN_ID`

Use the explicit run id variables when you want to force a ranking or season run to use a specific upstream snapshot.

## Dashboard Guidance

Use run rows as the source of truth for freshness. A detail row's `created_at` tells you when that row was inserted, but the run row tells you whether the run completed successfully and what upstream snapshot it used.

Prefer `run_date` for user-facing "as of" labels. Prefer `created_at` for operational freshness labels.

For current pages, ignore `duplicate` runs and use the latest `success`. For ETL monitoring pages, show all statuses.

For probabilities, values are stored as `0.0` to `1.0`. Format as percentages in the dashboard.

For rankings, lower is better. Use `NULLS LAST` when sorting rank columns.

For season probabilities, remember that conference champion logic is simulated first, then CFP selection and bracket simulation are applied on top of each simulated season.

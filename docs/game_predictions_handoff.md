# Game Predictions Handoff

This document explains the new game prediction snapshot tables, the older latest-only table, and the ETL/modeling logic that fills them. It is written for dashboard work in a separate codebase, so it focuses on the database contract, how to query the data, what each field means, and what assumptions the predictions carry.

## Short Version

The dashboard should treat `public.game_prediction_runs` plus `public.game_predictions_full` as the new prediction system.

`public.game_predictions_full` stores game-level predictions for a specific model run. It is intentionally a snapshot table, not a single current-state table. Every row belongs to one `game_prediction_run_id`, and that run id points back to metadata in `public.game_prediction_runs`.

`public.game_prediction_runs` stores one row per attempted snapshot run. Successful runs have rows in `game_predictions_full`; duplicate runs usually do not. Duplicate runs exist so the ETL can record that it ran, noticed no prediction changes, and avoided inserting another identical copy.

The old `public.game_predictions` table still exists in the repo, but the nightly workflow no longer runs that job. It is a latest-only overwrite table. The active nightly workflow currently runs `etl.jobs.prediction_updates.update_game_predictions_full`, which creates and updates the snapshot tables.

For dashboard work, the safest default is:

1. Read the latest `status = 'success'` run for a season and run type.
2. Join that run to `game_predictions_full`.
3. Use `homewinprob`, `awaywinprob`, `homespread`, `awayspread`, and `totalpred` as the prediction outputs.
4. Treat `homepoints` and `awaypoints` as final scores when present; they may be filled after the original prediction snapshot.

## Active ETL Entry Point

The GitHub Actions workflow `.github/workflows/nightly_etl.yml` runs:

```bash
python -m etl.jobs.prediction_updates.update_game_predictions_full
```

That job runs after the upstream data refresh steps:

- `etl.jobs.nightly_data_updates.update_current_season`
- `etl.jobs.nightly_data_updates.update_team_advanced_stats`
- `etl.jobs.nightly_data_updates.update_team_advanced_game_stats`
- `etl.jobs.nightly_data_updates.update_rankings`
- `etl.jobs.nightly_data_updates.update_betting_odds`
- `etl.jobs.prediction_updates.update_game_predictions_full`

Manual workflow dispatch exposes:

- `season`: optional season override.
- `prediction_run_type`: `manual` or `backfill`.

Scheduled nightly runs set `GAME_PREDICTION_RUN_TYPE=nightly`. Manual dispatch sets it to the selected value.

## Tables

### `public.game_prediction_runs`

This is the run metadata table. It is the entry point for finding usable prediction snapshots.

Schema:

```sql
CREATE TABLE IF NOT EXISTS public.game_prediction_runs (
    game_prediction_run_id UUID PRIMARY KEY,
    season                 INT NOT NULL,
    run_date               DATE NOT NULL,
    run_type               TEXT NOT NULL DEFAULT 'nightly',
    etl_run_id             TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at           TIMESTAMPTZ,
    status                 TEXT NOT NULL DEFAULT 'running',
    model_version          TEXT NOT NULL,
    prediction_hash        TEXT,
    duplicate_of_run_id    UUID REFERENCES public.game_prediction_runs(game_prediction_run_id),
    row_count              INT NOT NULL DEFAULT 0,
    inserted_row_count     INT NOT NULL DEFAULT 0,
    fcs_count              INT NOT NULL DEFAULT 0,
    notes                  TEXT,
    error_message          TEXT,
    CONSTRAINT game_prediction_runs_status_check
      CHECK (status IN ('running', 'success', 'duplicate', 'failed'))
);
```

Important fields:

- `game_prediction_run_id`: UUID for one snapshot attempt. This is the foreign key used by detail rows.
- `season`: target season being scored.
- `run_date`: the as-of date for the run. If `GAME_PREDICTION_RUN_DATE` is supplied, that value is used. Otherwise the job uses the current UTC date.
- `run_type`: usually `nightly`; manual dispatch can use `manual` or `backfill`.
- `etl_run_id`: run id from the common ETL config. Useful for tying back to logs.
- `created_at`: when the run metadata row was created in the database.
- `completed_at`: when the run was marked `success`, `duplicate`, or `failed`.
- `status`: one of `running`, `success`, `duplicate`, or `failed`.
- `model_version`: currently a combined label of the row-level XGBoost model labels.
- `prediction_hash`: SHA-256 hash of the canonical prediction payload for the run. Used to detect duplicate snapshots.
- `duplicate_of_run_id`: populated when this run produced the same prediction hash as the latest comparable successful run.
- `row_count`: number of game predictions prepared for the run, even if the run later becomes a duplicate.
- `inserted_row_count`: number of rows inserted into `game_predictions_full`. This is `0` for duplicate runs.
- `fcs_count`: count of rows involving at least one FCS team.
- `notes`: optional text from `GAME_PREDICTION_RUN_NOTES`.
- `error_message`: populated on failure.

Statuses:

- `success`: this run inserted detail rows into `game_predictions_full`.
- `duplicate`: this run produced the same canonical predictions as a previous comparable successful run and did not insert detail rows.
- `failed`: the job created a run row but failed before completing.
- `running`: transient state while the job is active. If this remains for a long time, the run likely died before being marked failed.

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_game_prediction_runs_lookup
  ON public.game_prediction_runs (season, run_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_prediction_runs_hash
  ON public.game_prediction_runs (season, run_type, prediction_hash)
  WHERE status = 'success';
```

### `public.game_predictions_full`

This is the snapshot detail table. Each row is one game prediction inside one successful run.

Schema:

```sql
CREATE TABLE IF NOT EXISTS public.game_predictions_full (
    game_prediction_run_id UUID NOT NULL
      REFERENCES public.game_prediction_runs(game_prediction_run_id)
      ON DELETE CASCADE,
    gameid                TEXT NOT NULL,
    season                INT NOT NULL,
    week                  INT NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    homepoints            DOUBLE PRECISION,
    awaypoints            DOUBLE PRECISION,
    homespread            DOUBLE PRECISION,
    awayspread            DOUBLE PRECISION,
    totalpred             DOUBLE PRECISION,
    homewinprob           DOUBLE PRECISION,
    awaywinprob           DOUBLE PRECISION,
    model_version         TEXT NOT NULL,
    prediction_type       TEXT NOT NULL DEFAULT 'FBS',
    prediction_row_hash   TEXT NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (game_prediction_run_id, gameid)
);
```

Important fields:

- `game_prediction_run_id`: points to `game_prediction_runs`. Always join through this field when selecting a snapshot.
- `gameid`: source game id from `public.game_data.id`, stored as text.
- `season`: season of the game.
- `week`: week of the game.
- `home_team`: home team name from `game_data.hometeam`.
- `away_team`: away team name from `game_data.awayteam`.
- `homepoints`: actual final home points when known. For future games this is usually `NULL`. The full snapshot job can fill this later for already-inserted prediction rows.
- `awaypoints`: actual final away points when known. Same behavior as `homepoints`.
- `homespread`: model prediction for the home team spread from a betting-display perspective. Positive means the home team is projected as the underdog by that many points; negative means the home team is projected as the favorite by that many points.
- `awayspread`: exact opposite of `homespread`.
- `totalpred`: model prediction for total points in the game.
- `homewinprob`: model probability that the home team wins, from `0.0` to `1.0`.
- `awaywinprob`: exact complement of `homewinprob`, calculated as `1.0 - homewinprob`.
- `model_version`: row-level model version label. Currently one of the FBS/FCS XGBoost labels.
- `prediction_type`: `FBS` or `FCS`. `FCS` means at least one team in the game is classified as FCS.
- `prediction_row_hash`: SHA-256 hash of the canonical row payload, excluding final scores.
- `created_at`: database insert timestamp for the detail row.

Primary key:

- `(game_prediction_run_id, gameid)`

This means the same game can appear in many snapshots, but only once per snapshot.

Indexes:

```sql
CREATE INDEX IF NOT EXISTS idx_game_predictions_full_game_lookup
  ON public.game_predictions_full (season, gameid);

CREATE INDEX IF NOT EXISTS idx_game_predictions_full_team_week
  ON public.game_predictions_full (season, week, home_team, away_team);
```

### Older `public.game_predictions`

The older table is latest-only:

```sql
CREATE TABLE IF NOT EXISTS public.game_predictions (
    gameid        TEXT PRIMARY KEY,
    season        INT NOT NULL,
    week          INT NOT NULL,
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    homepoints    DOUBLE PRECISION,
    awaypoints    DOUBLE PRECISION,
    homespread    DOUBLE PRECISION,
    awayspread    DOUBLE PRECISION,
    totalpred     DOUBLE PRECISION,
    homewinprob   DOUBLE PRECISION,
    awaywinprob   DOUBLE PRECISION,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL DEFAULT 'FBS'
);
```

The old job deletes all rows for the target season and reinserts the current predictions. It is useful for a simple current-state dashboard, but it cannot answer questions like "what did the model think two days ago?" or "how did predictions move over time?" The new full snapshot system can.

## How To Query It From A Dashboard

### Latest Successful Nightly Snapshot

Use this for a normal "current predictions" page backed by the new tables.

```sql
WITH latest_run AS (
  SELECT game_prediction_run_id
  FROM public.game_prediction_runs
  WHERE season = $1
    AND run_type = 'nightly'
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT p.*
FROM public.game_predictions_full p
JOIN latest_run r
  ON r.game_prediction_run_id = p.game_prediction_run_id
ORDER BY p.week, p.gameid;
```

### Latest Successful Snapshot Across Run Types

Use this if the dashboard should show the freshest successful data whether it came from nightly or a manual run.

```sql
WITH latest_run AS (
  SELECT game_prediction_run_id
  FROM public.game_prediction_runs
  WHERE season = $1
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT p.*
FROM public.game_predictions_full p
JOIN latest_run r
  ON r.game_prediction_run_id = p.game_prediction_run_id
ORDER BY p.week, p.gameid;
```

### Include Run Metadata On Each Row

This is often best for dashboard cards because it allows the UI to show "as of" information.

```sql
WITH latest_run AS (
  SELECT *
  FROM public.game_prediction_runs
  WHERE season = $1
    AND run_type = 'nightly'
    AND status = 'success'
  ORDER BY created_at DESC
  LIMIT 1
)
SELECT
  r.game_prediction_run_id,
  r.run_date,
  r.run_type,
  r.created_at AS run_created_at,
  r.completed_at AS run_completed_at,
  r.model_version AS run_model_version,
  r.prediction_hash,
  p.gameid,
  p.season,
  p.week,
  p.home_team,
  p.away_team,
  p.homepoints,
  p.awaypoints,
  p.homespread,
  p.awayspread,
  p.totalpred,
  p.homewinprob,
  p.awaywinprob,
  p.model_version AS row_model_version,
  p.prediction_type,
  p.prediction_row_hash,
  p.created_at AS prediction_created_at
FROM public.game_predictions_full p
JOIN latest_run r
  ON r.game_prediction_run_id = p.game_prediction_run_id
ORDER BY p.week, p.home_team, p.away_team;
```

### One Game Across Snapshots

Use this for movement/history charts.

```sql
SELECT
  r.run_date,
  r.created_at AS run_created_at,
  r.run_type,
  r.status,
  p.gameid,
  p.home_team,
  p.away_team,
  p.homespread,
  p.totalpred,
  p.homewinprob,
  p.awaywinprob,
  p.prediction_row_hash
FROM public.game_prediction_runs r
JOIN public.game_predictions_full p
  ON p.game_prediction_run_id = r.game_prediction_run_id
WHERE r.season = $1
  AND r.status = 'success'
  AND p.gameid = $2
ORDER BY r.created_at;
```

### Show Runs, Including Duplicates

Use this for admin/debug views.

```sql
SELECT
  game_prediction_run_id,
  season,
  run_date,
  run_type,
  created_at,
  completed_at,
  status,
  model_version,
  prediction_hash,
  duplicate_of_run_id,
  row_count,
  inserted_row_count,
  fcs_count,
  notes,
  error_message
FROM public.game_prediction_runs
WHERE season = $1
ORDER BY created_at DESC;
```

Duplicate runs are meaningful metadata but do not have their own detail rows. To display the predictions for a duplicate run, follow `duplicate_of_run_id` to the successful run it references.

## How The Full Snapshot Job Works

The full snapshot job reuses the modeling, training, scoring, and record-shaping functions from `prediction_updates/update_game_predictions.py`. That keeps model output identical between the latest-only table and the full snapshot table.

High-level flow:

1. Load config, including `PG_DSN`, target `SEASON`, and `RUN_ID`.
2. Determine `run_date` from `GAME_PREDICTION_RUN_DATE` or the current UTC date.
3. Determine `run_type` from `GAME_PREDICTION_RUN_TYPE`, defaulting to `nightly`.
4. Build the modeling table from Neon.
5. Ensure `game_prediction_runs` and `game_predictions_full` exist.
6. Update final scores on existing prediction detail rows where scores were previously null.
7. Train models as of `run_date`.
8. Score all current-season games.
9. Filter the scored games based on run type.
10. Convert scored rows to the dashboard-facing record shape.
11. Hash the prediction set.
12. Insert a run metadata row with `status = 'running'`.
13. Compare the new hash with the latest comparable successful run.
14. If unchanged, mark the run `duplicate` and insert no detail rows.
15. If changed, insert the full set of detail rows and mark the run `success`.
16. If anything fails after the run row is created, mark it `failed`.

### Run Date And As-Of Training

`train_models_as_of` trains on:

- All completed games from seasons before the target season.
- Completed games from the target season only when the game date is before `run_date`.

That matters because a snapshot for September 15 should not train on games played September 15 or later. The as-of rule is:

```text
season < current_season
OR (
  season = current_season
  AND game has final score
  AND gamedate < run_date
)
```

### Run Type Filtering

After scoring the current season, the full snapshot job filters rows differently by run type.

For `nightly` and `manual` runs:

```text
NOT (
  gamedate < run_date
  AND homepoints IS NOT NULL
  AND awaypoints IS NOT NULL
)
```

So a normal run stores predictions for every game that was not completed before the
run date. This includes same-date games, even if the score has already landed in
`game_data`, because the model is trained as of the start of that date and
downstream snapshot jobs need one complete probability set for all games not yet
treated as completed.

For `backfill` runs:

```text
gamedate = target_game_date
```

If `target_game_date` is not explicitly passed, it defaults to `run_date + 1 day`. In the automatic season backfill mode, the job loops over every current-season game date and creates a snapshot whose `run_date` is the day before that game date.

### Duplicate Detection

Each record is canonicalized and hashed. The run-level `prediction_hash` is the hash of all canonical records sorted by `gameid`.

Canonicalization:

- Sorts record keys.
- Converts NumPy integers/floats to normal Python values.
- Rounds floats to 4 decimal places.
- Converts NaN to `NULL`.
- Excludes `homepoints` and `awaypoints`.

Scores are excluded so that filling in final scores later does not make the same prediction set appear like a different model snapshot.

For normal non-backfill runs, `manual` and `nightly` snapshots share one duplicate-detection pool. A new manual or nightly hash is compared against the latest successful manual-or-nightly run for the same `season`.

For backfill runs, the new hash is compared against the latest successful run with the same `season`, `run_type`, and `run_date`.

If the hash matches, the current run becomes `duplicate`, points at the previous run through `duplicate_of_run_id`, and inserts no `game_predictions_full` rows.

If the stored run-level hash does not match, the job also re-hashes the latest run's detail rows using the current canonical rules before deciding the snapshot changed. This prevents older snapshots created with a previous hash shape from forcing one extra duplicate insert when the actual predictions are unchanged.

### Final Score Backfill

At the beginning of each full snapshot run, `finalize_completed_scores` finds completed current-season games in the modeling table and updates existing `game_predictions_full` rows:

```sql
UPDATE public.game_predictions_full
SET
    homepoints = %(homepoints)s,
    awaypoints = %(awaypoints)s
WHERE season = %(season)s
  AND gameid = %(gameid)s
  AND (homepoints IS NULL OR awaypoints IS NULL);
```

This means historical prediction snapshots can later receive final scores. The predictions themselves remain unchanged.

For dashboards, this is useful: a past prediction row may have `homepoints` and `awaypoints` even though those values were not known when the prediction was made. The run metadata still tells you when the prediction was created.

## Model Output Semantics

The prediction rows expose only a compact set of outputs:

- Win probabilities.
- Predicted spread.
- Predicted total.
- Actual final score when available.
- Model version and FBS/FCS flag.

The model predicts from the home-team point of view first, then mirrors away-team fields.

### Win Probability

`homewinprob` is the predicted probability that the home team wins.

`awaywinprob` is calculated as:

```text
1.0 - homewinprob
```

Display suggestion:

- Convert to percent in the UI.
- Keep enough precision internally for sorting.
- For display, one decimal place is usually enough.

### Spread

The training target is:

```text
homepoints - awaypoints
```

The scoring function then negates the model output when storing `homespread`:

```text
homespread = -predicted_home_margin
awayspread = -homespread
```

This makes the stored spread align with common betting notation:

- Negative spread means that team is favored.
- Positive spread means that team is an underdog.

Example:

- `homespread = -6.5` means the home team is projected to be favored by 6.5.
- `awayspread = 6.5` means the away team is projected to be an underdog by 6.5.

### Total

`totalpred` is the predicted combined score:

```text
homepoints + awaypoints
```

It is independent of actual final scores, though final score fields can later be populated for completed games.

### Model Version

There are four row-level model labels:

- `xgb_fbs_aware_2026_v2`: FBS-vs-FBS game with both spread and total lines available.
- `xgb_fbs_incomplete_2026_v2`: FBS-vs-FBS game missing one or both betting lines.
- `xgb_fcs_aware_2026_v2`: FBS-vs-FCS game with both spread and total lines available.
- `xgb_fcs_incomplete_2026_v2`: FBS-vs-FCS game missing one or both betting lines.

For 2026, the `_v2` labels mean returning production and recruiting metrics are no
longer used as model covariates. The fields are still available in the modeling table
for future analysis and model experiments.

At the run level, `game_prediction_runs.model_version` is a combined label:

```text
xgb_fbs_aware_2026_v2+xgb_fbs_incomplete_2026_v2+xgb_fcs_aware_2026_v2+xgb_fcs_incomplete_2026_v2
```

Dashboard interpretation:

- Use row-level `model_version` if you want to badge individual predictions.
- Use run-level `model_version` if you want a single metadata label for the selected snapshot.

### Prediction Type

`prediction_type` is:

- `FBS`: both teams are FBS.
- `FCS`: at least one team is FCS.

FCS games are included when one side is FBS and the other side is FCS. Pure FCS-vs-FCS games are not included by the modeling query. FCS games are modeled from the FBS team's perspective, then converted back into the same home/away dashboard fields.

## Modeling Inputs

The modeling table pulls data from these Neon tables:

- `public.game_data`
- `public.betting_odds`
- `public.team_recruiting_rankings`
- `public.team_talent_composite`
- `public.team_returning_production`
- `public.teamrankings_predictive_ratings`

The base game population is:

- Seasons from 2015 through the target season.
- Games with non-null `startdate`.
- Games where at least one team is FBS and the other team is FBS or FCS.

TeamRankings ratings are joined with a strict no-leakage rule:

```text
teamrankings_predictive_ratings.pull_date < game_data.gamedate
```

The selected TeamRankings row is the most recent available snapshot before the game date.

### FBS-vs-FBS Features

The FBS model family uses raw home/away team features:

- `home_teamrankings_rating`
- `away_teamrankings_rating`
- `home_talent`
- `away_talent`
- `neutral_site`

Spread-aware win/spread models also use `avg_spread`. Total-aware total models also use `avg_over_under`.

### FBS-vs-FCS Features

The FCS model family uses only the FBS team's features:

- `fbs_teamrankings_rating`
- `fbs_talent`
- `fbs_is_home`
- `neutral_site`
- `is_fcs_game`

Spread-aware FCS win/margin models also use `fbs_spread`, which converts the home spread into the FBS team's perspective. Total-aware FCS total models also use `avg_over_under`.

## Modeling Strategy

The job trains 12 XGBoost models:

- FBS win with spread.
- FBS win without spread.
- FBS spread with spread.
- FBS spread without spread.
- FBS total with total.
- FBS total without total.
- FCS FBS-team win with spread.
- FCS FBS-team win without spread.
- FCS FBS-team margin with spread.
- FCS FBS-team margin without spread.
- FCS total with total.
- FCS total without total.

Win models use `XGBClassifier`. Spread, margin, and total models use `XGBRegressor`.

Local macOS runs require an arm64 OpenMP runtime for the XGBoost wheel. If model construction fails with a missing `libomp.dylib` message, install an arm64 `libomp` runtime before running the prediction job locally. The GitHub Actions job runs on Ubuntu and installs the Python dependency from `etl/requirements.txt`.

Scoring selection:

- FBS-vs-FBS games use the FBS family.
- FBS-vs-FCS games use the FCS family.
- Win/spread use line-aware models when `avg_spread` exists.
- Win/spread use no-spread models when `avg_spread` is missing.
- Total uses the total-aware model when `avg_over_under` exists.
- Total uses the no-total model when `avg_over_under` is missing.

For FBS-vs-FCS games, the model predicts FBS-team win probability and FBS-team margin. The scoring function maps those predictions back to `homewinprob`, `awaywinprob`, `homespread`, and `awayspread` while preserving betting notation.

## Missing Data And Special Cases

XGBoost handles missing feature values natively, so the prediction job no longer runs advanced-stat prior imputation or Week 1 auxiliary models. FBS-vs-FCS rows only require feature inputs for the FBS side.

## Dashboard Implementation Notes

Recommended dashboard concepts:

- A "latest predictions" view should anchor on one successful run id.
- A movement/history view should plot successful runs over time for a selected `gameid`.
- An admin/debug view can show all runs, including duplicate and failed runs.
- If users ask "as of what date?", show `game_prediction_runs.run_date` and/or `completed_at`.
- If users ask "was this prediction made before the game?", use `run_date`, `created_at`, and the source game date from `game_data`. `game_predictions_full` itself does not store `gamedate`.

Recommended UI formatting:

- `homewinprob` and `awaywinprob`: percent.
- `homespread` and `awayspread`: one decimal place with plus sign for positive values.
- `totalpred`: one decimal place.
- `homepoints` and `awaypoints`: whole-number display when present.
- `prediction_type`: small badge, especially for FCS rows.
- `model_version`: optional technical badge or tooltip.

Important dashboard caveats:

- Duplicate runs have no detail rows. Follow `duplicate_of_run_id` if you intentionally need to display the duplicate run's prediction set.
- Normal nightly/manual snapshots include future unplayed games only. Once a game is played, future snapshots will not include that game, but older rows for that game may have final scores filled in.
- The full table is append-style by successful snapshot. Do not assume one row per game.
- `created_at` on `game_predictions_full` is the insertion time of the detail row, not necessarily the game date.
- `homepoints` and `awaypoints` are excluded from hashes, so final-score filling does not create a new prediction identity.
- The full snapshot job leaves `public.game_predictions` untouched.

## Good Mental Model

Think of the new prediction system like this:

```text
game_prediction_runs
  one row per ETL attempt/snapshot
  says when the snapshot was made, what season/run type it belongs to,
  whether it succeeded, duplicated a previous run, or failed

game_predictions_full
  one row per game inside a successful snapshot
  stores the actual prediction outputs that the dashboard displays
```

The `game_prediction_run_id` is the snapshot boundary. Pick the run first, then read the games.

# ETL Jobs

Jobs are grouped by how they are run:

- `nightly_data_updates/`: recurring source refresh jobs used by the nightly workflow.
- `one_off_updates/`: manual imports for season-scoped, backfill, or full-table replacement.
- `prediction_updates/`: prediction, ranking projection, and season projection snapshot jobs.

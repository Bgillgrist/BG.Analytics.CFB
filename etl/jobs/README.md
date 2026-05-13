# ETL Jobs

Jobs are grouped by how they are run:

- `nightly_data_updates/`: recurring CFBD data refresh jobs used by the nightly workflow.
- `one_off_updates/`: manual CFBD imports for season-scoped or full-table replacement.
- `prediction_updates/`: prediction, ranking projection, and season projection snapshot jobs.

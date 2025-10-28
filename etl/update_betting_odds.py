#!/usr/bin/env python3
import os
import csv
import psycopg
from typing import Iterable, List

# --- Config ---
PG_DSN = os.getenv(
    "PG_DSN",
    "postgresql://neondb_owner:npg_Up1blnYS0oxs@ep-delicate-brook-a4yhfswb-pooler.us-east-1.aws.neon.tech/neondb?sslmode=require",
)
BETTING_TABLE = "public.betting_odds"  # already created
CURRENT_SEASON_CSV = os.getenv(
    "BETTING_CSV",
    "/Users/brennangillgrist/Documents/CFB Git Hub Support/current_season.csv",  # <— change if needed
)

# COPY that matches your CSV columns exactly (quoted identifiers preserve case)
COPY_SQL = f"""
COPY {BETTING_TABLE} (
  "Id",
  "HomeTeam",
  "HomeScore",
  "AwayTeam",
  "AwayScore",
  "FormattedSpread",
  "LineProvider",
  "OverUnder",
  "Spread",
  "OpeningSpread",
  "OpeningOverUnder",
  "HomeMoneyline",
  "AwayMoneyline"
)
FROM STDIN WITH (
  FORMAT CSV,
  HEADER TRUE,
  DELIMITER ',',
  NULL '',
  FORCE_NULL (
    "HomeScore",
    "AwayScore",
    "OverUnder",
    "Spread",
    "OpeningSpread",
    "OpeningOverUnder",
    "HomeMoneyline",
    "AwayMoneyline"
  )
)
"""

def iter_id_batches(csv_path: str, batch_size: int = 5000) -> Iterable[List[int]]:
    """Yield lists of Ids from the CSV in batches (skips blanks/non-numeric)."""
    batch: List[int] = []
    with open(csv_path, newline="") as f:
        rdr = csv.DictReader(f)
        if "Id" not in rdr.fieldnames:
            raise ValueError(f'"Id" column not found in {csv_path}. Headers: {rdr.fieldnames}')
        for row in rdr:
            val = row.get("Id", "").strip()
            if not val:
                continue
            try:
                batch.append(int(val))
            except ValueError:
                # If some Ids are not integers, you can switch to BIGINT::text and cast—but your table uses BIGINT.
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def main(csv_path: str = CURRENT_SEASON_CSV):
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        raise FileNotFoundError(f"CSV not found or empty: {csv_path}")

    print(f"🔄 Nightly betting ETL starting for CSV: {csv_path}")

    with psycopg.connect(PG_DSN) as conn:
        with conn.cursor() as cur:
            # 1) Create a temp table for Ids from the fresh CSV
            print("• Creating temp table temp_betting_ids …")
            cur.execute("CREATE TEMP TABLE temp_betting_ids (\"Id\" BIGINT) ON COMMIT DROP;")

            # 2) Stream Ids into temp table in batches
            print("• Inserting Ids into temp table …")
            total_ids = 0
            for batch in iter_id_batches(csv_path, batch_size=10_000):
                cur.executemany(
                    'INSERT INTO temp_betting_ids ("Id") VALUES (%s)',
                    [(i,) for i in batch],
                )
                total_ids += len(batch)
            print(f"  → Inserted {total_ids:,} Ids into temp table")

            # 3) Delete any rows from betting_odds that match those Ids
            print("• Deleting existing rows in betting_odds matching those Ids …")
            cur.execute(f'''
                DELETE FROM {BETTING_TABLE} bo
                USING temp_betting_ids t
                WHERE bo."Id" = t."Id"
            ''')
            deleted = cur.rowcount
            print(f"  → Deleted {deleted:,} rows from betting_odds")

            # 4) COPY the fresh CSV into betting_odds
            print("• Loading fresh CSV into betting_odds via COPY …")
            with open(csv_path, "rb") as f, cur.copy(COPY_SQL) as cp:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    cp.write(chunk)

            # 5) Commit the transaction
            conn.commit()
            cur.execute(f'SELECT COUNT(*) FROM {BETTING_TABLE}')
            final_count = cur.fetchone()[0]
            print(f"✅ Load complete. betting_odds row count is now: {final_count:,}")

    print("🏁 Nightly betting ETL done.")

if __name__ == "__main__":
    main()

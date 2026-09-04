from __future__ import annotations

from etl import common_config


def test_normalize_pg_dsn_accepts_sqlalchemy_psycopg_url() -> None:
    dsn = "postgresql+psycopg://user:pass@example.com/db?sslmode=require"

    assert common_config._normalize_pg_dsn(dsn) == "postgresql://user:pass@example.com/db?sslmode=require"


def test_normalize_pg_dsn_leaves_plain_postgres_url_unchanged() -> None:
    dsn = "postgresql://user:pass@example.com/db?sslmode=require"

    assert common_config._normalize_pg_dsn(dsn) == dsn

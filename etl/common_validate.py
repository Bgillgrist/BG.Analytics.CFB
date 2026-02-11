from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd


def require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def require_nonempty(df: pd.DataFrame, allow_empty: bool, context: str = "") -> None:
    if df.empty and not allow_empty:
        msg = "DataFrame is empty"
        if context:
            msg += f" ({context})"
        raise ValueError(msg)


def require_no_dupes(df: pd.DataFrame, key_cols: List[str]) -> None:
    if not key_cols:
        return
    dupes = df.duplicated(subset=key_cols).sum()
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on key columns: {key_cols}")


def normalize_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts pandas NaN to Python None where appropriate for SQL insertion.
    """
    return df.where(pd.notnull(df), None)
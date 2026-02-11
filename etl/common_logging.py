from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Iterator


def setup_logging() -> logging.Logger:
    """
    Standard logger used by every ETL job.
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return logging.getLogger("etl")


@contextmanager
def log_timing(logger: logging.Logger, label: str) -> Iterator[float]:
    """
    Usage:
        with log_timing(logger, "fetch CFBD"):
            ...
    """
    start = time.perf_counter()
    logger.info(f"{label}... start")
    try:
        yield start
    finally:
        end = time.perf_counter()
        logger.info(f"{label}... done in {end - start:0.2f}s")


def format_step_prefix(run_id: str, step_name: str) -> str:
    return f"[run_id={run_id}] [step={step_name}]"
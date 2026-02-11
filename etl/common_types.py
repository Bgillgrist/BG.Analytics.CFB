from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class StepResult:
    step_name: str
    season: int
    status: str  # "success" | "skipped" | "failed"
    rows_fetched: int = 0
    rows_deleted: int = 0
    rows_inserted: int = 0
    message: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
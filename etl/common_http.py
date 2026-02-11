from __future__ import annotations

import os
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CFBD_BASE_URL = "https://api.collegefootballdata.com"


def build_retry_session(
    api_key: Optional[str],
    timeout_seconds: int = 30,
    total_retries: int = 6,
    backoff_factor: float = 0.7,
) -> requests.Session:
    """
    One standard HTTP session for every CFBD call.
    Retries transient failures: 429, 5xx, timeouts.
    """
    session = requests.Session()

    retries = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        status=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Standard headers
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    session.headers.update(headers)

    # Store timeout default (we’ll use it in helper below)
    session._etl_timeout = timeout_seconds  # type: ignore[attr-defined]
    return session


def cfbd_get(session: requests.Session, path: str, params: dict | None = None) -> requests.Response:
    """
    Standard GET wrapper that enforces timeout and returns the response.
    """
    timeout = getattr(session, "_etl_timeout", 30)
    url = path if path.startswith("http") else f"{CFBD_BASE_URL}{path}"
    return session.get(url, params=params, timeout=timeout)
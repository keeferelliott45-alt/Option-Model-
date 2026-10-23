from __future__ import annotations

import threading
import time
from urllib.parse import urlparse

import requests

from omodel.config import settings

DEFAULT_UA = "Mozilla/5.0 (compatible; omodel/0.1)"
RETRY_STATUS = {429, 500, 502, 503, 504}

_session = requests.Session()
_lock = threading.Lock()
_last_call: dict[str, float] = {}


def _throttle(host: str, min_interval: float) -> None:
    if min_interval <= 0:
        return
    with _lock:
        wait = _last_call.get(host, 0.0) + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[host] = time.monotonic()


def get(url: str, *, params: dict | None = None, headers: dict | None = None,
        min_interval: float = 0.0, retries: int = 4, timeout: float | None = None) -> requests.Response:
    """GET with per-host throttling and exponential backoff on 429/5xx/network errors.

    Returns the final response (callers handle 404 etc.); raises only if every attempt hit a network error.
    """
    host = urlparse(url).netloc
    hdrs = {"User-Agent": DEFAULT_UA, **(headers or {})}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        _throttle(host, min_interval)
        try:
            resp = _session.get(url, params=params, headers=hdrs, timeout=timeout or settings().http_timeout)
        except requests.RequestException as exc:
            last_exc = exc
        else:
            if resp.status_code not in RETRY_STATUS or attempt == retries:
                return resp
        time.sleep(min(2 ** attempt, 30))
    raise ConnectionError(f"GET {url} failed after {retries + 1} attempts") from last_exc

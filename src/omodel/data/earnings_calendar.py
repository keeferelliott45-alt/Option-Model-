from __future__ import annotations

import sys

import pandas as pd

from omodel.data import http, store

URL = "https://api.nasdaq.com/api/calendar/earnings"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
           "Accept": "application/json"}


def _day(d: pd.Timestamp, refresh: bool) -> list[str]:
    key = str(d.date())
    cached = None if refresh else store.read_json("earnings_cal", key)
    if cached is not None:
        return cached
    resp = http.get(URL, params={"date": key}, headers=HEADERS, min_interval=0.3)
    syms: list[str] = []
    if resp.ok:
        rows = (((resp.json() or {}).get("data") or {}).get("rows")) or []
        syms = [str(r.get("symbol", "")).upper() for r in rows if r.get("symbol")]
    store.write_json(syms, "earnings_cal", key)
    return syms


def upcoming(start: pd.Timestamp, days: int = 120, refresh_after: pd.Timestamp | None = None) -> dict[str, pd.Timestamp]:
    """Ticker -> next scheduled earnings date within `days` business days (Nasdaq calendar, free)."""
    out: dict[str, pd.Timestamp] = {}
    for d in pd.bdate_range(start, periods=days):
        # Calendars fill in over time: refetch dates within the next ~6 weeks each run.
        refresh = refresh_after is not None and d <= refresh_after
        try:
            for s in _day(d, refresh):
                out.setdefault(s, d)
        except Exception as exc:
            print(f"[earnings] {d.date()}: {exc}", file=sys.stderr)
    return out

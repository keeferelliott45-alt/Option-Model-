from __future__ import annotations

import datetime as _dt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from omodel.data import http, store

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
HISTORY_START = pd.Timestamp("1998-01-01")
MIN_INTERVAL = 0.35
DEFAULT_TZ = "America/New_York"
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")
COLUMNS = (*PRICE_COLUMNS, "split_factor")
# Yahoo answers "no such symbol / no data in window" with one of these statuses AND a JSON chart.error. Anything else
# (bare 404 from a CDN, 401 bad crumb, 429, 5xx) must raise rather than be cached as a permanently missing ticker.
_NO_DATA_STATUS = {200, 400, 404}


class LoadError(RuntimeError):
    def __init__(self, failed: list[str]):
        self.failed = failed
        super().__init__(f"{len(failed)} ticker(s) failed to load (successes are cached; rerun to retry): "
                         + ", ".join(failed[:20]))


def yahoo_symbol(ticker: str) -> str:
    t = ticker.strip()
    if "^" in t or "=" in t or t.endswith(".NYB"):
        return t
    return t.replace(".", "-").replace("/", "-")


def _unix(ts: pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    return int((ts.tz_localize("UTC") if ts.tzinfo is None else ts).timestamp())


def _local_dates(seconds, tz: str) -> pd.DatetimeIndex:
    utc = pd.to_datetime(np.asarray(seconds, dtype="int64"), unit="s", utc=True)
    return utc.tz_convert(tz).tz_localize(None).normalize()


def _floats(values, n: int) -> np.ndarray:
    if values is None:
        return np.full(n, np.nan)
    arr = np.array(values, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"Yahoo chart array has {arr.shape} values for {n} timestamps")
    return arr


def _split_factor(dates: pd.DatetimeIndex, splits: dict | None, tz: str) -> np.ndarray:
    """close * split_factor = as-traded close (Yahoo back-adjusts close for every split up to the fetch)."""
    factor = np.ones(len(dates))
    for ev in (splits or {}).values():
        num, den = float(ev.get("numerator") or 0), float(ev.get("denominator") or 0)
        if num > 0 and den > 0:
            ex_date = _local_dates([ev["date"]], tz)[0]
            factor[np.asarray(dates < ex_date)] *= num / den
    return factor


def _parse(result: dict) -> pd.DataFrame | None:
    ts = result.get("timestamp") or []
    if not ts:
        return None
    n = len(ts)
    tz = (result.get("meta") or {}).get("exchangeTimezoneName") or DEFAULT_TZ
    ind = result.get("indicators") or {}
    quote = (ind.get("quote") or [{}])[0] or {}
    adj = ((ind.get("adjclose") or [{}])[0] or {}).get("adjclose")
    data = {k: _floats(quote.get(k), n) for k in ("open", "high", "low", "close", "volume")}
    data["adj_close"] = data["close"].copy() if adj is None else _floats(adj, n)
    dates = _local_dates(ts, tz)
    data["split_factor"] = _split_factor(dates, (result.get("events") or {}).get("splits"), tz)
    df = pd.DataFrame(data, index=pd.DatetimeIndex(dates, name="date"))[list(COLUMNS)]
    df = df.iloc[np.argsort(np.asarray(ts, dtype="int64"), kind="stable")]
    df = df[df["close"] > 0]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df if len(df) else None


def fetch_daily(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    """Daily bars for [start, end]; split_factor is complete only when end reaches the fetch date."""
    params = {
        "period1": _unix(pd.Timestamp(start).normalize()),
        "period2": _unix(pd.Timestamp(end).normalize() + pd.Timedelta(days=1)),
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "true",
    }
    resp = http.get(CHART_URL + yahoo_symbol(ticker), params=params, min_interval=MIN_INTERVAL)
    chart = _chart(resp)
    if resp.status_code in _NO_DATA_STATUS and chart is not None and chart.get("error"):
        return None
    if resp.status_code != 200:
        resp.raise_for_status()
        raise requests.HTTPError(f"unexpected HTTP {resp.status_code} from Yahoo for {ticker}", response=resp)
    if chart is None:
        raise ValueError(f"Yahoo returned no chart JSON for {ticker}")
    results = chart.get("result") or []
    return _parse(results[0]) if results else None


def _chart(resp) -> dict | None:
    try:
        body = resp.json()
    except ValueError:
        return None
    chart = body.get("chart") if isinstance(body, dict) else None
    return chart if isinstance(chart, dict) else None


def load_daily(ticker: str, refresh: bool = False) -> pd.DataFrame | None:
    if not refresh:
        cached = store.read("prices", ticker)
        if cached is not None:
            return cached
        if store.read_json("prices_missing", ticker) is not None:
            return None
    end = pd.Timestamp.today().normalize()
    df = fetch_daily(ticker, HISTORY_START, end)
    if df is None:
        checked = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        store.write_json({"missing": True, "checked": checked}, "prices_missing", ticker)
        return None
    meta = {"source": CHART_URL, "symbol": yahoo_symbol(ticker), "start": str(HISTORY_START.date()),
            "end": str(end.date())}
    store.write(df, "prices", ticker, meta=meta)
    store.path_for("prices_missing", ticker, "json").unlink(missing_ok=True)
    return df


def load_many(tickers: list[str], refresh: bool = False, max_workers: int = 4, *,
              strict: bool = False) -> dict[str, pd.DataFrame]:
    """strict: raise LoadError after every ticker was tried if any raised, so fetch failures never pass as no-data."""
    symbols = list(dict.fromkeys(tickers))
    results: dict[str, pd.DataFrame | None] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(load_daily, t, refresh): t for t in symbols}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception as exc:  # one bad ticker must not abort a 700-ticker load; it is reported below
                failed.append(t)
                print(f"[yahoo] {t} failed: {exc!r}", file=sys.stderr, flush=True)
            if i % 50 == 0:
                print(f"[yahoo] {i}/{len(symbols)} tickers", file=sys.stderr, flush=True)
    out = {t: df for t in symbols if (df := results.get(t)) is not None}
    missing = len(symbols) - len(out) - len(failed)
    msg = f"[yahoo] {len(out)}/{len(symbols)} with data, {missing} missing"
    if failed:
        msg += f", {len(failed)} failed (not cached; rerun to retry): {', '.join(sorted(failed)[:20])}"
    print(msg, file=sys.stderr, flush=True)
    if failed and strict:
        raise LoadError(sorted(failed))
    return out

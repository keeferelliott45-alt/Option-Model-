from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from omodel import holdout
from omodel.data import http, store

log = logging.getLogger(__name__)

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
SNAPSHOT_KIND = "options_snapshots"
EXCHANGE_TZ = "America/New_York"
COLUMNS = ["contract", "root", "expiry", "cp", "strike", "bid", "ask", "mid", "spread_pct", "iv",
           "open_interest", "volume", "delta", "gamma", "vega", "theta", "theo", "last_trade_price",
           "last_trade_time"]
# standard: root matches the underlying's own option root; False marks adjusted/non-standard deliverables
# (e.g. 'AAPL1'), which downstream pricing must filter out.
EXTRA_COLUMNS = ["bid_size", "ask_size", "rho", "standard"]
_NUMERIC = ["bid", "ask", "iv", "open_interest", "volume", "delta", "gamma", "vega", "theta", "theo",
            "last_trade_price", "bid_size", "ask_size", "rho"]
_NOT_FOUND = {403, 404}  # the CDN answers 403, not 404, for symbols it has no chain for
_MIN_INTERVAL = 0.5  # per-host politeness throttle; operational only, never affects results
_OCC = re.compile(r"^(?P<root>.+)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
# Exchange-schedule facts, not tunables: equity options and their underlyings stop printing at 16:00 ET,
# broad ETF/index options at 16:15, and the feed lags 15 minutes.
_CLOSE_PRINT = pd.Timedelta("15:59:00")
_FINAL_QUOTES = pd.Timedelta("16:30:00")
_MAX_STALENESS = pd.Timedelta(days=5)  # spans long weekends and multi-day closures (Sandy 2012)
_CLOCK_SKEW = pd.Timedelta(minutes=5)
_STALE_LOCK_SECONDS = 600.0  # a write holds its lock for well under a second
_NOT_FOUND_ALARM = 0.2  # S&P-500 names nearly all list options: many 403s in one run means throttling


def parse_occ(symbol: str) -> tuple[str, pd.Timestamp, str, float]:
    """Split an OCC symbol (padded or compact) into (root, expiry, 'C'|'P', strike)."""
    m = _OCC.match(symbol.replace(" ", "").upper())
    if m is None:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    expiry = pd.to_datetime(m["ymd"], format="%y%m%d")
    return m["root"], expiry, m["cp"], int(m["strike"]) / 1000.0


def normalize_ticker(ticker: str) -> str:
    """Canonical storage key: upper case, class-share separator as '.' (BRK-B, BRK/B -> BRK.B)."""
    return re.sub(r"[-/]", ".", ticker.strip().upper())


def cboe_symbols(ticker: str) -> list[str]:
    """CBOE symbols to try, in order: 'BRK.B' then 'BRKB' for class shares."""
    t = normalize_ticker(ticker)
    return [t, t.replace(".", "")] if "." in t else [t]


def fetch_chain(ticker: str) -> pd.DataFrame | None:
    """Current 15-min-delayed full chain, or None if CBOE has no chain for the ticker."""
    fetched_at = _now()
    got = _get_payload(ticker)
    if got is None:
        return None
    symbol, payload = got
    data = (payload.get("data") if isinstance(payload, dict) else None) or {}
    options = data.get("options") or []
    if not options:
        return None
    df = _parse_options(pd.DataFrame.from_records(options))
    if df.empty:
        return None
    df["standard"] = df["root"] == symbol.replace(".", "")
    asof = _parse_asof(payload.get("timestamp"), fetched_at)
    # The payload timestamp is when the CDN regenerated the file (it does so pre-open too); the
    # underlying's last trade is what dates the quotes to a trading session.
    quote_time = _parse_quote_time(data.get("last_trade_time"), asof)
    session, complete = _session(quote_time, asof)
    df.attrs = {
        "underlying": normalize_ticker(ticker),
        "cboe_symbol": symbol,
        "underlying_price": _float(data.get("current_price")),
        "asof": asof,
        "quote_time": quote_time,
        "session": session,
        "session_complete": complete,
        # CBOE publishes iv30 in percent but per-contract iv as a decimal; store both as decimals.
        "iv30": _float(data.get("iv30")) / 100.0,
    }
    return df


def snapshot(tickers: list[str], when: pd.Timestamp | None = None, *, require_close: bool = True,
             max_workers: int = 4) -> list[Path]:
    """Store chains under options_snapshots/<session date>; idempotent, returns newly written paths."""
    label = None if when is None else _label(when)
    unique = list(dict.fromkeys(normalize_ticker(t) for t in tickers))

    def run(ticker: str) -> tuple[str, Path | None]:
        try:
            return _snapshot_one(ticker, label, require_close)
        except Exception:
            log.exception("CBOE snapshot failed for %s", ticker)
            return "failed", None

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(pool.map(run, unique))
    counts = Counter(status for status, _ in results)
    level = logging.WARNING if counts["failed"] or counts["not_found"] else logging.INFO
    log.log(level, "CBOE snapshot of %d tickers: %s", len(unique), dict(counts))
    if unique and counts["not_found"] > _NOT_FOUND_ALARM * len(unique):
        log.error("CBOE returned 403/404 for %d of %d tickers; probably throttled, snapshot has gaps",
                  counts["not_found"], len(unique))
    return [p for _, p in results if p is not None]


def load_snapshot(ticker: str, date: pd.Timestamp) -> pd.DataFrame | None:
    """A stored chain with its attrs restored, or None; holdout-dated chains need an unlocked holdout."""
    label, key = _label(date), normalize_ticker(ticker)
    # guard() drops holdout dates while locked and logs the access while unlocked.
    probe = pd.Series(True, index=pd.DatetimeIndex([pd.Timestamp(label)]))
    if holdout.guard(probe, f"{_kind(label)}/{key}").empty:
        raise holdout.HoldoutViolation(
            f"options snapshot {key} {label} is on/after holdout start {holdout.holdout_start().date()}")
    if not _exists(label, key):
        return None
    df = store.read(_kind(label), key)
    meta = store.read_meta(_kind(label), key) or {}
    complete = meta.get("session_complete")
    df.attrs = {
        "underlying": meta.get("underlying", key),
        "cboe_symbol": meta.get("cboe_symbol", key),
        "underlying_price": _float(meta.get("underlying_price")),
        "asof": _meta_ts(meta.get("asof"), "UTC"),
        "quote_time": _meta_ts(meta.get("quote_time"), EXCHANGE_TZ),
        "session": pd.Timestamp(label),
        "session_complete": None if complete is None else bool(complete),
        "iv30": _float(meta.get("iv30")),
    }
    return df


def _get_payload(ticker: str) -> tuple[str, dict] | None:
    for symbol in cboe_symbols(ticker):
        resp = http.get(URL.format(symbol=symbol), min_interval=_MIN_INTERVAL)
        if resp.status_code in _NOT_FOUND:
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"CBOE {symbol}: HTTP {resp.status_code}")
        return symbol, resp.json()
    return None


def _parse_options(raw: pd.DataFrame) -> pd.DataFrame:
    if "option" not in raw:
        return pd.DataFrame(columns=COLUMNS + EXTRA_COLUMNS)
    out = pd.DataFrame({"contract": raw["option"].astype(str).str.replace(" ", "", regex=False).str.upper()})
    parts = out["contract"].str.extract(_OCC)
    out["root"] = parts["root"]
    out["expiry"] = pd.to_datetime(parts["ymd"], format="%y%m%d", errors="coerce")
    out["cp"] = parts["cp"]
    out["strike"] = pd.to_numeric(parts["strike"], errors="coerce") / 1000.0
    for col in _NUMERIC:
        out[col] = pd.to_numeric(raw[col], errors="coerce").astype(float) if col in raw else np.nan
    out["iv"] = out["iv"].where(out["iv"] > 0)  # CBOE reports iv=0 when it has none; 0 would bias averages
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    # Crossed or ask-less quotes are unfillable; a negative spread would rank them as the tightest markets.
    fillable = (out["mid"] > 0) & (out["ask"] >= out["bid"])
    out["spread_pct"] = ((out["ask"] - out["bid"]) / out["mid"]).where(fillable)
    out["standard"] = True
    out["last_trade_time"] = _parse_trade_times(raw["last_trade_time"] if "last_trade_time" in raw else None,
                                                out.index)
    bad = out[["root", "expiry", "cp", "strike"]].isna().any(axis=1)
    if bad.any():
        log.warning("dropped %d unparseable CBOE contracts", int(bad.sum()))
    out = out.loc[~bad].drop_duplicates("contract", keep="last")
    out = out.sort_values(["expiry", "strike", "cp"], kind="stable")
    return out[COLUMNS + EXTRA_COLUMNS].reset_index(drop=True)


def _parse_trade_times(raw: pd.Series | None, index: pd.Index) -> pd.Series:
    if raw is None:
        return pd.Series(pd.NaT, index=index, dtype=f"datetime64[ns, {EXCHANGE_TZ}]")
    ts = pd.to_datetime(raw, errors="coerce", format="ISO8601")
    # Trade times arrive as exchange wall-clock (regular-session trades stamp 09:30-16:00).
    if ts.dt.tz is None:
        return ts.dt.tz_localize(EXCHANGE_TZ, ambiguous="NaT", nonexistent="NaT")
    return ts.dt.tz_convert(EXCHANGE_TZ)


def _parse_ts(raw: Any) -> pd.Timestamp | None:
    # Strings only: a bare number has no unit, and pandas would read epoch seconds as nanoseconds (1970).
    if not isinstance(raw, str):
        return None
    try:
        ts = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts


def _parse_asof(raw: Any, fetched_at: pd.Timestamp) -> pd.Timestamp:
    ts = _parse_ts(raw)
    if ts is None:
        return fetched_at
    # Payload timestamp is UTC: it matches the CDN's Last-Modified (GMT) header to the second.
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if not fetched_at - _MAX_STALENESS <= ts <= fetched_at + _CLOCK_SKEW:
        log.warning("implausible CBOE timestamp %r (fetched %s); using fetch time", raw, fetched_at)
        return fetched_at
    return ts


def _parse_quote_time(raw: Any, asof: pd.Timestamp) -> pd.Timestamp:
    ts = _parse_ts(raw)
    if ts is None:
        return pd.NaT
    ts = ts.tz_localize(EXCHANGE_TZ, ambiguous="NaT", nonexistent="NaT") if ts.tzinfo is None \
        else ts.tz_convert(EXCHANGE_TZ)
    if pd.notna(ts) and ts > asof + _CLOCK_SKEW:
        log.warning("CBOE underlying last trade %r is after the payload time %s; ignored", raw, asof)
        return pd.NaT
    return ts


def _session(quote_time: pd.Timestamp, asof: pd.Timestamp) -> tuple[pd.Timestamp, bool]:
    """Trading session the quotes belong to, and whether they already show that session's close."""
    if pd.isna(quote_time):
        # Without the underlying's last trade the session is a guess (pre-open, holiday): never final.
        return asof.tz_convert(EXCHANGE_TZ).normalize().tz_localize(None), False
    start = quote_time.normalize()
    # Either the closing print is in, or the file was regenerated after every option class closed (half days).
    complete = quote_time >= start + _CLOSE_PRINT or asof >= start + _FINAL_QUOTES
    return start.tz_localize(None), bool(complete)


def _snapshot_one(ticker: str, label: str | None, require_close: bool) -> tuple[str, Path | None]:
    key = normalize_ticker(ticker)
    # Once today's session is stored, any fetch before tomorrow can only return that same session.
    if _exists(label or _label(_now()), key):
        return "exists", None
    df = fetch_chain(ticker)
    if df is None:
        log.warning("no CBOE chain for %s (HTTP 403/404 or empty)", ticker)
        return "not_found", None
    attrs = df.attrs
    session = _label(attrs["session"])
    problem = None
    if label is not None and session != label:
        # Filing quotes under another date leaks the future into the past or passes stale quotes off as fresh.
        problem = f"chain is from session {session}, not {label}"
    elif pd.notna(attrs["quote_time"]) and attrs["asof"] - attrs["quote_time"] > _MAX_STALENESS:
        problem = f"underlying last traded {attrs['quote_time']}, stale at {attrs['asof']}"
    elif require_close and not attrs["session_complete"]:
        problem = f"session {session} not closed yet (last trade {attrs['quote_time']}, file {attrs['asof']})"
    if problem:
        log.warning("%s: %s; not written", ticker, problem)
        return "skipped", None
    with _key_lock(session, key) as locked:
        if not locked:
            log.warning("%s: another run is writing %s; not written", ticker, session)
            return "busy", None
        if _exists(session, key):
            return "exists", None
        return "written", _write_atomic(df, session, key)


def _write_atomic(df: pd.DataFrame, label: str, key: str) -> Path:
    out = df.copy()
    out.attrs = {}  # pyarrow json-encodes attrs and cannot encode Timestamps; the meta sidecar holds them
    meta = {k: _jsonable(v) for k, v in df.attrs.items()}
    staged = store.write(out, f"{SNAPSHOT_KIND}_staging/{uuid.uuid4().hex}", key, meta=meta)
    final = store.path_for(_kind(label), key)
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, final)
    # The sidecar lands last: _exists() treats it as the commit marker, so a crash leaves a re-writable gap.
    os.replace(staged.with_suffix(".meta.json"), final.with_suffix(".meta.json"))
    try:
        staged.parent.rmdir()
    except OSError:
        pass
    return final


@contextmanager
def _key_lock(label: str, key: str) -> Iterator[bool]:
    path = store.path_for(_kind(label), key, "lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if time.time() - path.stat().st_mtime > _STALE_LOCK_SECONDS:
            path.unlink(missing_ok=True)  # left behind by a crashed run
    except FileNotFoundError:
        pass
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        yield False
        return
    try:
        yield True
    finally:
        path.unlink(missing_ok=True)


def _exists(label: str, key: str) -> bool:
    path = store.path_for(_kind(label), key)
    return path.exists() and path.with_suffix(".meta.json").exists()


def _kind(label: str) -> str:
    return f"{SNAPSHOT_KIND}/{label}"


def _label(when: Any) -> str:
    ts = pd.Timestamp(when)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(EXCHANGE_TZ)
    return ts.strftime("%Y-%m-%d")


def _jsonable(v: Any) -> Any:
    if v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return bool(v) if isinstance(v, np.bool_) else v


def _meta_ts(raw: Any, tz: str) -> pd.Timestamp:
    return pd.Timestamp(raw).tz_convert(tz) if raw else pd.NaT


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")

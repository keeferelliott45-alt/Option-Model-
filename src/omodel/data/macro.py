from __future__ import annotations

import datetime as _dt
import io
import re
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from omodel import holdout, params
from omodel.data import http, store, yahoo

MACRO_START_YEAR = params.register(
    "MACRO_START_YEAR", 1998, "convention", "L1",
    "Two years of warm-up before dev_start (2000) so trailing macro features are defined from day one.")
MACRO_FFILL_LIMIT = params.register(
    "MACRO_FFILL_LIMIT", 3, "convention", "L1",
    "Bridge holiday-calendar mismatches between sources only; longer gaps stay NaN so stale macro is never used.")

TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type={curve}&field_tdr_date_value={year}&page&_format=csv")
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
NOMINAL_CURVE = "daily_treasury_yield_curve"
REAL_CURVE = "daily_treasury_real_yield_curve"
REAL_FIRST_YEAR = 2003  # Treasury returns an empty body for real-curve years before this
_CURVE_COLS = {
    NOMINAL_CURVE: {"3 mo": "y3m", "2 yr": "y2", "10 yr": "y10", "30 yr": "y30"},
    REAL_CURVE: {"10 yr": "real10"},
}
_EXPECTED_GAPS = {"y30": range(2002, 2007)}  # no 30-year bond was issued from Feb 2002 to Feb 2006
TREASURY_COLUMNS = ["y3m", "y2", "y10", "y30", "real10"]
CBOE_SOURCES = {"vix": "VIX", "vix3m": "VIX3M", "vix9d": "VIX9D"}
YAHOO_SOURCES = {"hyg": "HYG", "lqd": "LQD", "dxy": "DX-Y.NYB", "gold": "GC=F", "oil": "CL=F",
                 "eurusd": "EURUSD=X", "usdjpy": "JPY=X"}
# Yahoo back-adjusts these for every later distribution, so raw adj_close levels at t depend on payouts after t;
# a total-return index anchored at the first bar depends only on payouts up to t.
TOTAL_RETURN_COLUMNS = ("hyg", "lqd")
# Commodity futures (GC=F, CL=F) print real settlements below zero (crude went negative in Apr 2020); such prints
# are data poison for the panel, so price-like sources get the same positive-price guard FX uses.
PRICE_COLUMNS = ("gold", "oil")
# Yahoo FX bars are stamped 00:00 London (~19:00 ET of the previous US day). 'open' is that snapshot in every era,
# while 'close' was the end-of-bar price until 2010 and a copy of 'open' since, so we use open, one day earlier.
FX_COLUMNS = ("eurusd", "usdjpy")
MACRO_COLUMNS = [*CBOE_SOURCES, *TREASURY_COLUMNS, *YAHOO_SOURCES]
_MIN_INTERVAL = 0.25
_BREAKER_FAILS = 2  # consecutive host-level failures before a Treasury load stops calling the network


class MacroSourceWarning(UserWarning):
    pass


class _HostDown(RuntimeError):
    pass


@dataclass
class _Breaker:
    fails: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def tripped(self) -> bool:
        return self.fails >= _BREAKER_FAILS


def _warn(msg: str) -> None:
    warnings.warn(msg, MacroSourceWarning, stacklevel=3)


def _current_year() -> int:
    return _dt.date.today().year


def _empty(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [], index=pd.DatetimeIndex([], name="date"), dtype=float)


def _fetch_text(url: str) -> str:
    try:
        resp = http.get(url, min_interval=_MIN_INTERVAL)
    except OSError as exc:  # http.get raises only after every retry hit a network error
        raise _HostDown(f"network error for {url}: {exc}") from exc
    if resp.status_code in http.RETRY_STATUS:
        raise _HostDown(f"HTTP {resp.status_code} for {url}")
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} for {url}")
    return resp.text


def _norm(col: str) -> str:
    return re.sub(r"\s+", " ", str(col).strip().lower())


def _parse_dates(raw: pd.Series, what: str) -> pd.DatetimeIndex:
    text = raw.astype(str).str.strip()
    dates = pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    todo = dates.isna() & raw.notna() & (text != "")
    if todo.any():
        dates[todo] = pd.to_datetime(text[todo], format="mixed", errors="coerce")
        bad = todo & dates.isna()
        if bad.any():
            _warn(f"{what}: dropped {int(bad.sum())} rows with unparseable dates, e.g. {text[bad].iloc[0]!r}")
    return pd.DatetimeIndex(dates, name="date")


def _read_csv(text: str, what: str) -> pd.DataFrame:
    """Parse a date-first CSV into a sorted float frame; tolerates preamble lines and an empty body."""
    lines = text.lstrip("﻿").splitlines()
    head = next((i for i, ln in enumerate(lines) if _norm(ln.split(",")[0].strip('"')) == "date"), None)
    if head is None:
        if not text.strip():
            return _empty()
        raise ValueError(f"{what}: no DATE header in response")
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    date_col = df.columns[0]
    out = df.drop(columns=date_col).apply(pd.to_numeric, errors="coerce").astype(float)
    out.index = _parse_dates(df[date_col], what)
    out = out[out.index.notna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _treasury_year(curve: str, year: int, refresh: bool, current_year: int, breaker: _Breaker) -> pd.DataFrame:
    key = f"{curve}_{year}"
    cached = store.read("treasury", key)
    meta = store.read_meta("treasury", key) or {}
    # A year cached while it was still in progress is partial, so only a "complete" copy is final.
    if cached is not None and meta.get("complete") and not refresh:
        return cached
    if breaker.tripped:
        breaker.skipped.append(f"{curve} {year}")
        return cached if cached is not None else _empty()
    url = TREASURY_URL.format(year=year, curve=curve)
    try:
        df = _read_csv(_fetch_text(url), f"treasury {curve} {year}")
    except Exception as exc:  # noqa: BLE001 - one bad year must not sink the whole curve
        breaker.fails = breaker.fails + 1 if isinstance(exc, _HostDown) else 0
        if cached is not None:
            _warn(f"treasury {curve} {year}: refetch failed ({exc}); using copy fetched {meta.get('fetched_at')}")
            return cached
        _warn(f"treasury {curve} {year}: unavailable ({exc})")
        return _empty()
    breaker.fails = 0
    if df.empty:
        if year < current_year:
            _warn(f"treasury {curve} {year}: response had no rows")
        return cached if cached is not None else df
    store.write(df, "treasury", key, meta={"url": url, "year": year, "complete": year < current_year})
    return df


def _select(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    by_norm = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index, columns=list(mapping.values()), dtype=float)
    for src, dst in mapping.items():
        if src in by_norm:
            out[dst] = df[by_norm[src]].astype(float)
    return out


def _curve(curve: str, years: range, refresh: bool, current_year: int, breaker: _Breaker) -> pd.DataFrame:
    mapping = _CURVE_COLS[curve]
    parts = []
    for year in years:
        raw = _treasury_year(curve, year, refresh, current_year, breaker)
        if raw.empty:
            continue
        part = _select(raw, mapping)
        lost = [c for c in part.columns if part[c].isna().all() and year not in _EXPECTED_GAPS.get(c, ())]
        if lost:
            _warn(f"treasury {curve} {year}: no data for {lost}; source columns {list(raw.columns)}")
        parts.append(part)
    if not parts:
        return _empty(list(mapping.values()))
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")]


def _treasury_frame(start_year: int, end_year: int | None, refresh: bool) -> pd.DataFrame:
    current = _current_year()
    end_year = current if end_year is None else min(end_year, current)
    if not holdout.is_unlocked():
        # Years wholly inside the locked holdout would be downloaded and cached only for the guard to drop them.
        end_year = min(end_year, (holdout.holdout_start() - pd.Timedelta(days=1)).year)
    breaker = _Breaker()
    nominal = _curve(NOMINAL_CURVE, range(start_year, end_year + 1), refresh, current, breaker)
    real = _curve(REAL_CURVE, range(max(start_year, REAL_FIRST_YEAR), end_year + 1), refresh, current, breaker)
    if breaker.skipped:
        _warn(f"treasury: host failed {_BREAKER_FAILS} years in a row; no network for {len(breaker.skipped)} "
              f"later curve-years ({breaker.skipped[0]} .. {breaker.skipped[-1]}), cached copies used where present")
    out = pd.concat([nominal, real], axis=1).reindex(columns=TREASURY_COLUMNS).astype(float).sort_index()
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def load_treasury(start_year: int = 1998, end_year: int | None = None, refresh: bool = False) -> pd.DataFrame:
    return holdout.guard(_treasury_frame(start_year, end_year, refresh), what="treasury yields")


def _cboe_close(df: pd.DataFrame) -> pd.Series:
    by_norm = {_norm(c): c for c in df.columns}
    if "close" in by_norm:
        return df[by_norm["close"]]
    numeric = [c for c in df.columns if df[c].notna().any()]
    if not numeric:
        raise ValueError("no numeric column")
    return df[numeric[-1]]


def _cboe_series(name: str, refresh: bool) -> pd.Series:
    sym, col = name.upper(), name.lower()
    cached = store.read("cboe", sym)
    if cached is not None and not refresh:
        return cached[col].rename(col)
    url = CBOE_URL.format(name=sym)
    try:
        s = _cboe_close(_read_csv(_fetch_text(url), f"cboe {sym}")).dropna()
        if s.empty:
            raise ValueError("response had no rows")
    except Exception as exc:
        if cached is None:
            raise
        _warn(f"cboe {sym}: refetch failed ({exc}); using cached copy")
        return cached[col].rename(col)
    out = s.rename(col).astype(float)
    store.write(out.to_frame(), "cboe", sym, meta={"url": url})
    return out


def load_cboe_index(name: str, refresh: bool = False) -> pd.Series:
    return holdout.guard(_cboe_series(name, refresh), what=f"cboe {name.upper()}")


def _column(df: pd.DataFrame | None, ticker: str, wants: tuple[str, ...]) -> pd.Series:
    if df is None or df.empty:
        raise ValueError(f"yahoo returned no data for {ticker}")
    by_norm = {re.sub(r"[^a-z]", "", str(c).lower()): c for c in df.columns}
    for want in wants:
        if want in by_norm:
            return df[by_norm[want]]
    raise ValueError(f"yahoo frame for {ticker} has none of {wants}: {list(df.columns)}")


def _clean(s: pd.Series) -> pd.Series:
    idx = pd.DatetimeIndex(pd.to_datetime(s.index))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(pd.to_numeric(s.to_numpy(), errors="coerce"), index=idx.normalize(), dtype=float)
    s = s[s.index.notna()]
    return s[~s.index.duplicated(keep="last")].sort_index()


def _yahoo_series(col: str, ticker: str, refresh: bool) -> pd.Series:
    df = yahoo.load_daily(ticker, refresh=refresh)
    if col in FX_COLUMNS:
        s = _clean(_column(df, ticker, ("open",)))
        s = s.where(s > 0)
        # Monday bars open on Sunday evening: they drop out rather than leak the weekend into Friday, so Friday
        # rows carry Thursday's evening snapshot through the panel ffill (no Friday-evening bar exists).
        return pd.Series(s.to_numpy(), index=s.index - pd.Timedelta(days=1))
    s = _clean(_column(df, ticker, ("adjclose", "close")))
    if col in PRICE_COLUMNS:
        s = s.where(s > 0)  # same non-positive guard as FX: negative/zero settlements become NaN, ffill bridges them
    if col in TOTAL_RETURN_COLUMNS:
        first = s.first_valid_index()
        if first is None:
            raise ValueError(f"yahoo frame for {ticker} has no valid prices")
        s = 100.0 * s / s.loc[first]
    return s


def _attempt(label: str, fn: Callable[[], pd.Series]) -> pd.Series | None:
    try:
        return _clean(fn())
    except Exception as exc:  # noqa: BLE001 - spec: a dead source degrades to NaN, never aborts the panel
        _warn(f"macro source {label} failed: {exc}; column left all-NaN")
        return None


def load_macro_panel(refresh: bool = False) -> pd.DataFrame:
    """Row t: latest value of each source stamped by ~20:00 ET on t (FX ~19-20:00, dxy/futures ~17:00, VIX 16:15)."""
    start = pd.Timestamp(year=MACRO_START_YEAR, month=1, day=1)
    cols: dict[str, pd.Series | None] = {}
    for col, sym in CBOE_SOURCES.items():
        cols[col] = _attempt(col, lambda sym=sym: _cboe_series(sym, refresh))
    try:
        tsy = _treasury_frame(MACRO_START_YEAR, None, refresh)
    except Exception as exc:  # noqa: BLE001
        _warn(f"macro source treasury failed: {exc}; columns {TREASURY_COLUMNS} left all-NaN")
        tsy = None
    for col in TREASURY_COLUMNS:
        cols[col] = _attempt(col, lambda col=col: tsy[col]) if tsy is not None else None
    for col, tic in YAHOO_SOURCES.items():
        cols[col] = _attempt(col, lambda col=col, tic=tic: _yahoo_series(col, tic, refresh))

    idx = pd.DatetimeIndex([], name="date")
    for s in cols.values():
        if s is not None:
            idx = idx.union(s.dropna().index)
    idx = pd.DatetimeIndex(idx[(idx >= start) & (idx.dayofweek < 5)].sort_values(), name="date")

    panel = pd.DataFrame({c: (s.reindex(idx) if s is not None else np.nan) for c, s in cols.items()},
                         index=idx, columns=MACRO_COLUMNS, dtype=float)
    for c, s in cols.items():
        if s is not None and len(idx) and panel[c].isna().all():
            _warn(f"macro source {c} has no data on/after {start.date()}; column left all-NaN")
    panel = panel.ffill(limit=MACRO_FFILL_LIMIT)
    return holdout.guard(panel, what="macro panel")

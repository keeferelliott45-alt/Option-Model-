from __future__ import annotations

import sys
from typing import Callable

import numpy as np
import pandas as pd

from omodel import holdout, params
from omodel.config import MARKET_ETF, SECTOR_ETFS, settings
from omodel.data import universe, yahoo
from omodel.types import Panel

PANEL_WARMUP_DAYS = params.register(
    "PANEL_WARMUP_DAYS", 550, "convention", "data",
    "Calendar days loaded before dev_start: SECTOR_WINDOW + BETA_WINDOW = 342 trading days (~495 calendar "
    "days) of warm-up plus a holiday buffer.")
REUSE_TOLERANCE_DAYS = params.register(
    "REUSE_TOLERANCE_DAYS", 7, "convention", "data",
    "A member cannot predate its first trade: a price history starting more than ~5 trading days after a "
    "membership spell begins marks the symbol as reused by another company, and that spell's prices are dropped.")

ETF_SYMBOLS = tuple(dict.fromkeys((MARKET_ETF, *SECTOR_ETFS)))
FIELDS = ("close", "open", "raw_close", "volume")

# Hand-verified identity corrections; listing a ticker replaces the late-start heuristic for it. Each [from, to)
# window is where Yahoo's series under that symbol is a different company from the S&P member of that name.
IDENTITY_OVERRIDES: dict[str, tuple[tuple[str, str], ...]] = {
    "ATI": (),                                # Yahoo starts at the 1999-11-29 Allegheny Technologies rename: same firm
    "IR": (("2010-11-17", "2020-03-03"),),    # Yahoo IR before the 2020 rename is Gardner Denver; member was IR plc
    "TT": (("2002-05-13", "2008-06-06"),),    # Yahoo TT pre-2020 is Ingersoll-Rand plc; member was American Standard
    "FOX": (("2015-09-21", "2019-03-20"),),   # Yahoo FOX/FOXA are Fox Corp (2019 spin-off); member was 21st Century Fox
    "FOXA": (("2004-12-20", "2019-03-20"),),
}


def _load(symbols: list[str], refresh: bool,
          loader: Callable[[str], pd.DataFrame | None] | None) -> dict[str, pd.DataFrame]:
    if loader is None:
        return yahoo.load_many(symbols, refresh=refresh, strict=True)
    return {s: df for s in symbols if (df := loader(s)) is not None and len(df)}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = df.set_axis(idx.normalize().rename("date"), axis=0)
    return df[~df.index.duplicated(keep="last")]


def _first_bar(df: pd.DataFrame) -> pd.Timestamp:
    df = _clean(df)
    good = df.index[(df["close"] > 0).to_numpy()]
    return good.min() if len(good) else pd.NaT


def _fields(df: pd.DataFrame, cal: pd.DatetimeIndex) -> dict[str, pd.Series]:
    df = _clean(df).reindex(cal)
    close = df["close"].where(df["close"] > 0)
    adj = df["adj_close"].where(df["adj_close"] > 0) if "adj_close" in df else close
    split = df["split_factor"] if "split_factor" in df else 1.0
    return {
        "close": adj,
        "open": df["open"].where(df["open"] > 0) * adj / close,
        "raw_close": close * split,
        "volume": df["volume"].where(df["volume"] >= 0).astype(float),
    }


def _frames(data: dict[str, pd.DataFrame], symbols: list[str], cal: pd.DatetimeIndex,
            breaks: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    arrays = {f: np.full((len(cal), len(symbols)), np.nan) for f in FIELDS}
    for j, s in enumerate(symbols):
        if s in data:
            for f, values in _fields(data[s], cal).items():
                arrays[f][:, j] = values.to_numpy(dtype=float)
    col = {s: j for j, s in enumerate(symbols)}
    for t, lo, hi in ([] if breaks is None else breaks[["ticker", "start", "end"]].itertuples(index=False)):
        if t in col:
            rows = slice(cal.searchsorted(lo), len(cal) if pd.isna(hi) else cal.searchsorted(hi))
            for a in arrays.values():
                a[rows, col[t]] = np.nan
    cols = pd.Index(symbols, dtype=object)
    return {f: pd.DataFrame(a, index=cal, columns=cols) for f, a in arrays.items()}


def identity_breaks(data: dict[str, pd.DataFrame], membership: pd.DataFrame,
                    floor: pd.Timestamp | None = None) -> pd.DataFrame:
    """Windows [start, end) (NaT end = open) where a symbol's prices are presumed to be another company's."""
    first = pd.Series({t: _first_bar(df) for t, df in data.items()}, dtype="datetime64[ns]")
    sp = membership[membership["ticker"].isin(first.index) & ~membership["ticker"].isin(list(IDENTITY_OVERRIDES))]
    bar = sp["ticker"].map(first)
    # floor = first date the price source covers (the market ETF's first bar): older spells are judged from it.
    judged_from = sp["start_date"] if floor is None else sp["start_date"].clip(lower=pd.Timestamp(floor))
    late = bar > judged_from + pd.Timedelta(days=REUSE_TOLERANCE_DAYS)
    in_spell = sp["end_date"].isna() | (bar < sp["end_date"])
    auto = pd.DataFrame({"ticker": sp["ticker"], "start": sp["start_date"], "end": sp["end_date"],
                         "first_bar": bar, "reason": "late first bar"})[(late & in_spell).to_numpy()]
    manual = pd.DataFrame(
        [(t, pd.Timestamp(lo), pd.Timestamp(hi), first[t], "override")
         for t, wins in IDENTITY_OVERRIDES.items() if t in first.index for lo, hi in wins],
        columns=list(auto.columns))
    parts = [df for df in (auto, manual) if len(df)]
    out = pd.concat(parts, ignore_index=True) if parts else auto
    out = out.astype({"start": "datetime64[ns]", "end": "datetime64[ns]", "first_bar": "datetime64[ns]"})
    return out.sort_values(["ticker", "start"]).reset_index(drop=True)


def _check_etfs(close: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    for s in ETF_SYMBOLS:
        born = _first_bar(data[s])
        if pd.isna(born) or ((close.index >= born).any() and close[s].isna().all()):
            raise ValueError(f"{s} has no usable close in the window after its first bar ({born}): a dead factor "
                             "ETF would silently drop its sector from L2; rerun with refresh=True")


def build_panel(start: pd.Timestamp | None = None, end: pd.Timestamp | None = None,
                tickers: list[str] | None = None, refresh: bool = False,
                loader: Callable[[str], pd.DataFrame | None] | None = None,
                membership: pd.DataFrame | None = None) -> Panel:
    start = pd.Timestamp(start) if start is not None else settings().dev_start - pd.Timedelta(days=PANEL_WARMUP_DAYS)
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    if end < start:
        raise ValueError(f"end {end.date()} is before start {start.date()}")
    # Clip before choosing tickers: index additions made inside a locked holdout must not shape the dev universe.
    sel_end = end if holdout.is_unlocked() else min(end, holdout.holdout_start() - pd.Timedelta(days=1))
    if sel_end < start:
        raise ValueError(f"window {start.date()}..{end.date()} lies inside the locked holdout "
                         f"(starts {holdout.holdout_start().date()})")
    if membership is None:
        membership = universe.load_membership()
    if tickers is None:
        tickers = universe.tickers_between(membership, start, sel_end)

    # ETFs first and strictly: a missing factor ETF must stop the build before ~1,000 stock fetches, not after.
    data = _load(list(ETF_SYMBOLS), refresh, loader)
    missing = [s for s in ETF_SYMBOLS if s not in data]
    if missing:
        raise ValueError(f"no data for factor ETF(s) {missing}; if cached as missing, rerun with refresh=True")
    stocks = list(dict.fromkeys(tickers))
    data |= _load([s for s in stocks if s not in data], refresh, loader)

    mkt = _clean(data[MARKET_ETF])
    days = mkt.index[(mkt["close"] > 0).to_numpy()]
    cal = pd.DatetimeIndex(days[(days >= start) & (days <= sel_end)].sort_values(), name="date")
    if not len(cal):
        raise ValueError(f"no {MARKET_ETF} bars in {start.date()}..{sel_end.date()}: empty trading calendar")

    loaded = [s for s in stocks if s in data]
    breaks = identity_breaks({s: data[s] for s in loaded}, membership, floor=_first_bar(data[MARKET_ETF]))
    hit = breaks[(breaks["start"] <= cal[-1]) & (breaks["end"].isna() | (breaks["end"] > cal[0]))]
    if len(hit):
        print(f"[panel] ignoring prices in {len(hit)} spell(s) where the symbol belongs to another company: "
              + ", ".join(sorted(set(hit["ticker"]))), file=sys.stderr, flush=True)
    px = {f: holdout.guard(df, f"panel.{f}") for f, df in _frames(data, loaded, cal, breaks).items()}
    etf = _frames(data, list(ETF_SYMBOLS), cal)
    _check_etfs(etf["close"], data)
    etf_close = holdout.guard(etf["close"], "panel.etf_close")
    etf_open = holdout.guard(etf["open"], "panel.etf_open")

    keep = px["close"].columns[px["close"].notna().any().to_numpy()]
    px = {f: df[keep] for f, df in px.items()}
    member = holdout.guard(universe.membership_mask(membership, px["close"].index, list(keep)), "panel.member")
    holdout.assert_clean(px["close"].index, "panel")
    return Panel(close=px["close"], open=px["open"], volume=px["volume"], member=member,
                 etf_close=etf_close, etf_open=etf_open, raw_close=px["raw_close"], macro=None)


def coverage(panel: Panel, *, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    if membership is None:
        membership = universe.load_membership()
    full = universe.membership_mask(membership, panel.dates)
    return universe.coverage_by_year(full, panel.close.notna())

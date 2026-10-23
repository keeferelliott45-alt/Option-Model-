from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omodel import params
from omodel.config import MARKET_ETF, settings
from omodel.types import Decomposition, Panel

EVENT_COLUMNS = ["date", "ticker", "pos", "z", "resid", "ret", "mkt_ret", "sec_ret", "sector", "vol_z",
                 "alpha", "beta_mkt", "beta_sec", "sigma", "tot_sigma", "price"]
ENTRIES = ("next_open", "close")


def _grid(df: pd.DataFrame, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    if df.index.equals(index) and df.columns.equals(columns):
        return df
    return df.reindex(index=index, columns=columns)


def _floats(df: pd.DataFrame, index: pd.Index, columns: pd.Index) -> np.ndarray:
    return _grid(df, index, columns).to_numpy(dtype=float)


def _cooldown_keep(r: np.ndarray, c: np.ndarray, cooldown: int) -> np.ndarray:
    keep = np.ones(len(r), dtype=bool)
    if cooldown == 0 or len(r) == 0:
        return keep
    order = np.lexsort((r, c))
    rs, cs = r[order], c[order]
    bounds = np.r_[np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]]), len(rs)]
    kept = np.ones(len(rs), dtype=bool)
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2 or np.all(np.diff(rs[a:b]) > cooldown):
            continue
        last = rs[a]
        for k in range(a + 1, b):
            # Measured from the last KEPT event, so a suppressed event never extends the window.
            if rs[k] - last > cooldown:
                last = rs[k]
            else:
                kept[k] = False
    keep[order] = kept
    return keep


def find_events(dec: Decomposition, panel: Panel, z_threshold: float | None = None, side: str = "down",
                cooldown: int | None = None, min_price: float | None = None) -> pd.DataFrame:
    z_threshold = params.EVENT_Z_THRESHOLD if z_threshold is None else z_threshold
    cooldown = params.EVENT_COOLDOWN if cooldown is None else int(cooldown)
    min_price = params.MIN_PRICE if min_price is None else min_price
    if side not in ("down", "up"):
        raise ValueError(f"side must be 'down' or 'up', got {side!r}")
    if not z_threshold < 0:
        raise ValueError("z_threshold must be negative; side='up' mirrors it as z >= -z_threshold")
    if cooldown < 0:
        raise ValueError("cooldown must be >= 0")

    dates, tickers = panel.dates, panel.close.columns
    z = _floats(dec.z, dates, tickers)
    sigma = _floats(dec.sigma, dates, tickers)
    price = _floats(panel.raw_close if panel.raw_close is not None else panel.close, dates, tickers)
    member = _grid(panel.member, dates, tickers).eq(True).to_numpy()
    hit = z <= z_threshold if side == "down" else z >= -z_threshold
    mask = member & np.isfinite(z) & np.isfinite(sigma) & hit & (price >= min_price)

    r, c = np.nonzero(mask)
    keep = _cooldown_keep(r, c, cooldown)
    r, c = r[keep], c[keep]

    def at(df: pd.DataFrame) -> np.ndarray:
        return _floats(df, dates, tickers)[r, c]

    out = pd.DataFrame({
        "date": dates[r],
        "ticker": np.asarray(tickers, dtype=object)[c],
        "pos": r.astype(np.int64),
        "z": z[r, c],
        "resid": at(dec.resid),
        "ret": at(dec.stock_ret),
        "mkt_ret": dec.mkt_ret.reindex(dates).to_numpy(dtype=float)[r],
        "sec_ret": at(dec.sec_ret),
        "sector": _grid(dec.sector, dates, tickers).to_numpy(dtype=object)[r, c],
        "vol_z": at(dec.vol_z),
        "alpha": at(dec.alpha),
        "beta_mkt": at(dec.beta_mkt),
        "beta_sec": at(dec.beta_sec),
        "sigma": sigma[r, c],
        "tot_sigma": at(dec.tot_sigma),
        "price": price[r, c],
    }, columns=EVENT_COLUMNS)
    return out.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)


def _take(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray, ok: np.ndarray) -> np.ndarray:
    ok = ok & (rows >= 0) & (rows < arr.shape[0]) & (cols >= 0) & (cols < arr.shape[1])
    out = np.full(len(rows), np.nan)
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def _usable(pre_holdout: np.ndarray, pos: np.ndarray) -> np.ndarray:
    n = len(pre_holdout)
    return (pos >= 0) & (pos < n) & pre_holdout[np.clip(pos, 0, n - 1)]


def _dates_at(dates: np.ndarray, pos: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.full(len(pos), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = dates[pos[ok]]
    return out


@dataclass
class _Legs:
    dates: np.ndarray
    pre_holdout: np.ndarray
    p: np.ndarray
    ci: np.ndarray
    mi: np.ndarray
    ki: np.ndarray
    close: np.ndarray
    etf_close: np.ndarray
    entry_date: np.ndarray
    s_in: np.ndarray
    m_in: np.ndarray
    k_in: np.ndarray
    beta_mkt: np.ndarray
    beta_sec: np.ndarray


def _legs(events: pd.DataFrame, panel: Panel, entry: str) -> _Legs:
    if entry not in ENTRIES:
        raise ValueError(f"entry must be one of {ENTRIES}, got {entry!r}")
    dates, tickers = panel.dates, panel.close.columns
    etf_cols = panel.etf_close.columns
    if MARKET_ETF not in etf_cols:
        raise ValueError(f"panel.etf_close has no {MARKET_ETF} column")
    p = dates.get_indexer(pd.DatetimeIndex(events["date"]))
    if (p < 0).any():
        raise ValueError("some event dates are not trading days of panel.dates")
    ci = tickers.get_indexer(events["ticker"])
    if (ci < 0).any():
        raise ValueError("some event tickers are not columns of panel.close")
    n_ev = len(events)
    mi = np.full(n_ev, etf_cols.get_loc(MARKET_ETF))
    ki = etf_cols.get_indexer(events["sector"].astype(object))

    close = panel.close.to_numpy(dtype=float)
    etf_close = _floats(panel.etf_close, dates, etf_cols)
    if entry == "next_open":
        e, s_px, etf_px = p + 1, _floats(panel.open, dates, tickers), _floats(panel.etf_open, dates, etf_cols)
    else:
        e, s_px, etf_px = p, close, etf_close

    date_arr = dates.to_numpy(dtype="datetime64[ns]")
    pre_holdout = np.asarray(dates < settings().holdout_start)
    # An entry print on/after the holdout start is holdout data even though it is only a price.
    entry_ok = _usable(pre_holdout, e)
    return _Legs(dates=date_arr, pre_holdout=pre_holdout, p=p, ci=ci, mi=mi, ki=ki, close=close,
                 etf_close=etf_close, entry_date=_dates_at(date_arr, e, entry_ok),
                 s_in=_take(s_px, e, ci, entry_ok), m_in=_take(etf_px, e, mi, entry_ok),
                 k_in=_take(etf_px, e, ki, entry_ok),
                 beta_mkt=events["beta_mkt"].to_numpy(dtype=float),
                 beta_sec=events["beta_sec"].to_numpy(dtype=float))


def _horizon(legs: _Legs, h: int, cost_bps: float) -> dict[str, np.ndarray]:
    if int(h) != h or h < 1:
        raise ValueError(f"horizon must be a positive integer, got {h!r}")
    x = legs.p + int(h)
    window = _usable(legs.pre_holdout, x)
    s_out = _take(legs.close, x, legs.ci, window)
    m_out = _take(legs.etf_close, x, legs.mi, window)
    k_out = _take(legs.etf_close, x, legs.ki, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw, mkt, sec = np.log(s_out / legs.s_in), np.log(m_out / legs.m_in), np.log(k_out / legs.k_in)
    good = window & np.isfinite(raw) & np.isfinite(mkt) & np.isfinite(sec)
    raw, mkt, sec = (np.where(good, v, np.nan) for v in (raw, mkt, sec))
    abn = raw - legs.beta_mkt * mkt - legs.beta_sec * sec
    round_trip = 2.0 * cost_bps / 1e4
    return {
        "exit_date": _dates_at(legs.dates, x, window),
        "raw": raw, "mkt": mkt, "sec": sec, "abn": abn,
        "net_raw": raw - round_trip, "net_abn": abn - round_trip,
        "missing": window & (np.isnan(legs.s_in) | np.isnan(s_out)),
    }


def add_forward_returns(events: pd.DataFrame, panel: Panel, horizons: tuple[int, ...] | None = None,
                        entry: str | None = None, cost_bps: float | None = None) -> pd.DataFrame:
    horizons = params.HORIZONS if horizons is None else tuple(horizons)
    entry = params.ENTRY if entry is None else entry
    cost_bps = params.COST_BPS_PER_SIDE if cost_bps is None else cost_bps
    legs = _legs(events, panel, entry)
    cols: dict[str, np.ndarray] = {"entry_date": legs.entry_date, "entry_px": legs.s_in}
    for h in horizons:
        for name, values in _horizon(legs, h, cost_bps).items():
            cols[f"{name}_{h}"] = values
    out = events.copy()
    for name, values in cols.items():
        out[name] = values
    return out


def _ffill_rows(a: np.ndarray) -> np.ndarray:
    idx = np.where(np.isfinite(a), np.arange(a.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return a[np.arange(a.shape[0])[:, None], idx]


def position_daily_returns(events: pd.DataFrame, panel: Panel, horizon: int, entry: str | None = None,
                           cost_bps: float | None = None) -> pd.DataFrame:
    entry = params.ENTRY if entry is None else entry
    cost_bps = params.COST_BPS_PER_SIDE if cost_bps is None else cost_bps
    if not events.index.is_unique:
        raise ValueError("events index must be unique: it is the position's event_id")
    legs = _legs(events, panel, entry)
    h = int(horizon)
    v = np.flatnonzero(np.isfinite(_horizon(legs, horizon, cost_bps)["net_abn"]))
    empty = pd.DataFrame({"date": pd.DatetimeIndex([]), "event_id": events.index[:0], "abn": np.empty(0)})
    if len(v) == 0:
        return empty

    rows = legs.p[v, None] + np.arange(1, h + 1)[None, :]

    def daily_log_returns(entry_px: np.ndarray, cols: np.ndarray, arr: np.ndarray) -> np.ndarray:
        px = np.empty((len(v), h + 1))
        px[:, 0] = entry_px[v]
        px[:, 1:] = arr[rows, cols[v, None]]
        with np.errstate(divide="ignore", invalid="ignore"):
            # A mid-holding gap contributes 0 and its move lands on the next print, so the path sums to the label.
            return np.diff(_ffill_rows(np.log(px)), axis=1)

    abn = (daily_log_returns(legs.s_in, legs.ci, legs.close)
           - legs.beta_mkt[v, None] * daily_log_returns(legs.m_in, legs.mi, legs.etf_close)
           - legs.beta_sec[v, None] * daily_log_returns(legs.k_in, legs.ki, legs.etf_close))
    abn[:, 0] -= cost_bps / 1e4
    abn[:, -1] -= cost_bps / 1e4
    out = pd.DataFrame({
        "date": legs.dates[rows].ravel(),
        "event_id": np.repeat(events.index.to_numpy()[v], h),
        "abn": abn.ravel(),
    })
    return out.sort_values(["date", "event_id"], kind="stable").reset_index(drop=True)

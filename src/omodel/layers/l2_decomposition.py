from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import pandas as pd

from omodel import params
from omodel.config import MARKET_ETF, SECTOR_ETFS
from omodel.types import Decomposition, Panel

# Numerical guards, not model choices: they only turn degenerate windows into NaN.
_VAR_FLOOR = 1e-16      # per-observation variance below which a return series counts as constant
_COLLINEAR_TOL = 1e-6   # 1 - corr(mkt, sector)^2 below this makes the 2x2 slope system ill-conditioned
_SSR_RTOL = 1e-12       # residual SS below this fraction of total SS is a degenerate exact fit
_VOL_SD_FLOOR = 1e-9    # log-volume std below this is a constant (stale) volume series

_FIELDS = ("alpha", "beta_mkt", "beta_sec", "resid", "sigma", "z", "sec_ret")


@contextmanager
def _timed(label: str) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    print(f"[l2] {label}: {time.perf_counter() - t0:.1f}s", file=sys.stderr)


def log_returns(px: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    px = px.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(px / px.shift(1))
    return out.replace([np.inf, -np.inf], np.nan)


def _period_ends(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    pos = pd.Series(np.arange(len(index)), index=index).resample(freq).last()
    return pos.dropna().to_numpy(dtype=np.int64)


def _centered(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Correlation is shift-invariant; centring on the window mean avoids cancellation in the raw-sum formula.
    mean = np.where(mask, a, 0.0).sum(0) / np.maximum(mask.sum(0), 1)
    return np.where(mask, a - mean, 0.0)


def _best_sector(x: np.ndarray, y: np.ndarray, min_obs: int) -> np.ndarray:
    mx, my = np.isfinite(x), np.isfinite(y)
    x0, y0 = _centered(x, mx), _centered(y, my)
    fx, fy = mx.astype(float), my.astype(float)
    n = fx.T @ fy
    sx, sy = x0.T @ fy, fx.T @ y0
    with np.errstate(divide="ignore", invalid="ignore"):
        vx = (x0 * x0).T @ fy - sx * sx / n
        vy = fx.T @ (y0 * y0) - sy * sy / n
        cov = x0.T @ y0 - sx * sy / n
        ok = (n >= max(min_obs, 2)) & (vx > n * _VAR_FLOOR) & (vy > n * _VAR_FLOOR)
        corr = np.where(ok, cov / np.sqrt(np.where(ok, vx * vy, 1.0)), -np.inf)
    best = corr.argmax(axis=1).astype(np.int16)
    best[~ok.any(axis=1)] = -1
    return best


def assign_sectors(stock_ret: pd.DataFrame, mkt_ret: pd.Series, etf_ret: pd.DataFrame,
                   window: int | None = None, freq: str | None = None, min_obs: int | None = None) -> pd.DataFrame:
    window = params.SECTOR_WINDOW if window is None else window
    freq = params.SECTOR_REASSIGN_FREQ if freq is None else freq
    min_obs = params.BETA_MIN_OBS if min_obs is None else min_obs
    idx = stock_ret.index
    T, N = stock_ret.shape
    rm = mkt_ret.reindex(idx).to_numpy(dtype=float)[:, None]
    x = stock_ret.to_numpy(dtype=float) - rm
    y = etf_ret.reindex(idx).to_numpy(dtype=float) - rm
    ends = _period_ends(idx, freq)
    ends = ends[ends + 1 < T]
    full = np.full((T, N), -1, dtype=np.int16)
    if len(ends) and y.shape[1]:
        codes = np.empty((len(ends), N), dtype=np.int16)
        for j, p in enumerate(ends):
            lo = max(0, p - window + 1)
            codes[j] = _best_sector(x[lo:p + 1], y[lo:p + 1], min_obs)
        # Assignment computed with data through close d takes effect on the next trading day.
        active = np.searchsorted(ends + 1, np.arange(T), side="right") - 1
        live = active >= 0
        full[live] = codes[active[live]]
    symbols = np.array([*etf_ret.columns, np.nan], dtype=object)
    return pd.DataFrame(symbols[full], index=idx, columns=stock_ret.columns, dtype=object)


def _sector_array(sector: pd.DataFrame, index: pd.Index, columns: pd.Index) -> np.ndarray:
    sec = sector.reindex(index=index, columns=columns).to_numpy(dtype=object)
    # StringDtype frames carry pd.NA, whose == comparison raises; None is the single missing marker.
    return np.where(pd.isna(sec), None, sec)


def _window_sum(a: np.ndarray, hi: np.ndarray, lo: np.ndarray, cols: np.ndarray) -> np.ndarray:
    c = np.empty((a.shape[0] + 1, a.shape[1]))
    c[0] = 0.0
    np.cumsum(a, axis=0, out=c[1:])
    return c[hi, cols] - c[lo, cols]


def _fit_sector(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                window: int, min_obs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rowok = np.isfinite(x1) & np.isfinite(x2)
    m = np.isfinite(y) & rowok[:, None]
    mf = m.astype(float)
    y0 = np.where(m, y, 0.0)
    a1 = np.where(rowok, x1, 0.0)[:, None]
    a2 = np.where(rowok, x2, 0.0)[:, None]
    lo = np.maximum(rows - window, 0)

    def ws(a: np.ndarray) -> np.ndarray:
        return _window_sum(a, rows, lo, cols)

    with np.errstate(divide="ignore", invalid="ignore"):
        n = ws(mf)
        s1, s2, sy = ws(mf * a1), ws(mf * a2), ws(y0)
        x1b, x2b, yb = s1 / n, s2 / n, sy / n
        # Centred cross-products: the intercept is partialled out, leaving a well-scaled 2x2 slope system
        # whose solution is identical to the 3x3 normal equations.
        c11 = ws(mf * (a1 * a1)) - s1 * x1b
        c22 = ws(mf * (a2 * a2)) - s2 * x2b
        c12 = ws(mf * (a1 * a2)) - s1 * x2b
        c1y = ws(y0 * a1) - s1 * yb
        c2y = ws(y0 * a2) - s2 * yb
        cyy = ws(y0 * y0) - sy * yb
        det = c11 * c22 - c12 * c12
        ok = ((n >= max(min_obs, 4)) & (c11 > n * _VAR_FLOOR) & (c22 > n * _VAR_FLOOR)
              & (det > _COLLINEAR_TOL * c11 * c22))
        b1 = (c22 * c1y - c12 * c2y) / det
        b2 = (c11 * c2y - c12 * c1y) / det
        alpha = yb - b1 * x1b - b2 * x2b
        ssr = cyy - b1 * c1y - b2 * c2y
        sigma = np.where(ssr > _SSR_RTOL * cyy, np.sqrt(np.maximum(ssr, 0.0) / (n - 3)), np.nan)
    alpha, b1, b2, sigma = (np.where(ok, v, np.nan) for v in (alpha, b1, b2, sigma))
    return alpha, b1, b2, sigma


def rolling_factor_model(stock_ret: pd.DataFrame, mkt_ret: pd.Series, etf_ret: pd.DataFrame,
                         sector: pd.DataFrame, window: int | None = None,
                         min_obs: int | None = None) -> dict[str, pd.DataFrame]:
    window = params.BETA_WINDOW if window is None else window
    min_obs = params.BETA_MIN_OBS if min_obs is None else min_obs
    idx, names = stock_ret.index, stock_ret.columns
    y = stock_ret.to_numpy(dtype=float)
    x1 = mkt_ret.reindex(idx).to_numpy(dtype=float)
    etf = etf_ret.reindex(idx)
    sec = _sector_array(sector, idx, names)
    out = {k: np.full(y.shape, np.nan) for k in _FIELDS}
    for s in etf.columns:
        hit = sec == s
        sub = np.flatnonzero(hit.any(axis=0))
        if not len(sub):
            continue
        rows, j = np.nonzero(hit[:, sub])
        del hit
        x2 = etf[s].to_numpy(dtype=float)
        alpha, b1, b2, sigma = _fit_sector(y[:, sub], x1, x2, rows, j, window, min_obs)
        cols = sub[j]
        rs = x2[rows]
        out["alpha"][rows, cols] = alpha
        out["beta_mkt"][rows, cols] = b1
        out["beta_sec"][rows, cols] = b2
        out["sigma"][rows, cols] = sigma
        out["sec_ret"][rows, cols] = rs
        out["resid"][rows, cols] = y[rows, cols] - (alpha + b1 * x1[rows] + b2 * rs)
    out["z"] = out["resid"] / out["sigma"]
    frames = {k: pd.DataFrame(v, index=idx, columns=names) for k, v in out.items()}
    frames["tot_sigma"] = stock_ret.shift(1).rolling(window, min_periods=max(min_obs, 2)).std()
    return frames


def volume_z(volume: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    window = params.VOLUME_Z_WINDOW if window is None else window
    v = volume.astype(float)
    lv = np.log(v.where(v > 0))
    prior = lv.shift(1).rolling(window, min_periods=max(window // 2, 2))
    sd = prior.std()
    return (lv - prior.mean()) / sd.where(sd > _VOL_SD_FLOOR)


def peer_z(dec: Decomposition, member: pd.DataFrame, min_peers: int = 3) -> pd.DataFrame:
    z = dec.z
    sec = _sector_array(dec.sector, z.index, z.columns)
    mem =member.reindex(index=z.index, columns=z.columns, fill_value=False).to_numpy()
    mem = np.where(pd.isna(mem), False, mem).astype(bool)
    zv = z.to_numpy(dtype=float)
    base = mem & np.isfinite(zv)
    out = np.full(zv.shape, np.nan)
    for s in pd.unique(sec[pd.notna(sec)]):
        in_s = sec == s
        valid = base & in_s
        own = np.where(valid, zv, 0.0)
        peers_n = valid.sum(axis=1, keepdims=True) - valid
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = (own.sum(axis=1, keepdims=True) - own) / peers_n
        out = np.where(in_s & (peers_n >= min_peers), mean, out)
    return pd.DataFrame(out, index=z.index, columns=z.columns)


def decompose(panel: Panel) -> Decomposition:
    idx = panel.close.index
    # Align prices before differencing: a date missing from etf_close must give NaN, not a multi-day return.
    etf_px = panel.etf_close.reindex(idx)
    with _timed("log returns"):
        stock_ret = log_returns(panel.close)
        mkt_ret = log_returns(etf_px[MARKET_ETF])
        etf_ret = log_returns(etf_px[[s for s in SECTOR_ETFS if s in etf_px.columns]])
    with _timed("assign_sectors"):
        sector = assign_sectors(stock_ret, mkt_ret, etf_ret)
    with _timed("rolling_factor_model"):
        fm = rolling_factor_model(stock_ret, mkt_ret, etf_ret, sector)
    with _timed("volume_z"):
        vol_z = volume_z(panel.volume.reindex(index=idx, columns=stock_ret.columns))
    return Decomposition(stock_ret=stock_ret, mkt_ret=mkt_ret, sector=sector, sec_ret=fm["sec_ret"],
                         alpha=fm["alpha"], beta_mkt=fm["beta_mkt"], beta_sec=fm["beta_sec"], resid=fm["resid"],
                         sigma=fm["sigma"], z=fm["z"], tot_sigma=fm["tot_sigma"], vol_z=vol_z)

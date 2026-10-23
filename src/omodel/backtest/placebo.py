from __future__ import annotations

import numpy as np
import pandas as pd

from omodel import params
from omodel.types import Decomposition, Panel

EVENT_COLUMNS = ["date", "ticker", "pos", "z", "resid", "ret", "mkt_ret", "sec_ret", "sector", "vol_z", "alpha",
                 "beta_mkt", "beta_sec", "sigma", "tot_sigma", "price"]


def _pick(frame: pd.DataFrame, dates: pd.DatetimeIndex, tickers: np.ndarray) -> np.ndarray:
    r = frame.index.get_indexer(dates)
    c = frame.columns.get_indexer(tickers)
    vals = frame.to_numpy()
    numeric = vals.dtype.kind in "biuf"
    out = np.full(len(r), np.nan if numeric else None, dtype=float if numeric else object)
    ok = (r >= 0) & (c >= 0)
    out[ok] = vals[r[ok], c[ok]]
    return out


def _eligible(panel: Panel, dec: Decomposition, tickers: list[str], ev_pos: np.ndarray, ev_col: np.ndarray,
              window: int, min_price: float) -> np.ndarray:
    dates = panel.dates
    member = panel.member.reindex(index=dates, columns=tickers)
    ok = member.eq(True).to_numpy()
    for f in (dec.z, dec.sigma, dec.beta_mkt, dec.beta_sec):
        ok &= np.isfinite(f.reindex(index=dates, columns=tickers).to_numpy(dtype=float))
    price = panel.raw_close if panel.raw_close is not None else panel.close
    ok &= price.reindex(index=dates, columns=tickers).to_numpy(dtype=float) >= min_price  # same filter as events
    ok[: ev_pos.min()] = False
    ok[ev_pos.max() + 1:] = False
    # Exclude each real event day and its own forward window [e, e + window]: past events only. Excluding days
    # BEFORE an event would keep only placebo windows with no future drop in them, which is lookahead.
    marks = np.zeros((len(dates) + 1, len(tickers)), dtype=np.int64)
    np.add.at(marks, (ev_pos, ev_col), 1)
    np.add.at(marks, (np.minimum(ev_pos + window + 1, len(dates)), ev_col), -1)
    return ok & ~(np.cumsum(marks, axis=0)[:-1] > 0)


def _inputs(events: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[str], np.ndarray]:
    ev_pos = dates.get_indexer(pd.DatetimeIndex(pd.to_datetime(events["date"])))
    if (ev_pos < 0).any():
        raise ValueError("event dates missing from panel.dates")
    tickers = sorted(events["ticker"].unique())
    return ev_pos, tickers, pd.Index(tickers).get_indexer(events["ticker"])


def eligible_days(events: pd.DataFrame, panel: Panel, dec: Decomposition, exclude_window: int | None = None,
                  min_price: float | None = None) -> pd.DataFrame:
    """Bool frame (panel.dates x event tickers) of the days placebo_events may draw from."""
    window = params.EVENT_COOLDOWN if exclude_window is None else exclude_window
    min_price = params.MIN_PRICE if min_price is None else min_price
    if window < 0:
        raise ValueError("exclude_window must be non-negative")
    ev_pos, tickers, ev_col = _inputs(events, panel.dates)
    if not len(events):
        return pd.DataFrame(False, index=panel.dates, columns=pd.Index(tickers, dtype=object))
    ok = _eligible(panel, dec, tickers, ev_pos, ev_col, window, min_price)
    return pd.DataFrame(ok, index=panel.dates, columns=pd.Index(tickers, dtype=object))


def _draw(rng: np.random.Generator, cand: np.ndarray, need: int) -> np.ndarray:
    if need <= len(cand):
        return rng.choice(cand, size=need, replace=False)
    # Too few candidates: cycle through fresh permutations so reuse is as even as possible.
    reps = -(-need // len(cand))
    return np.concatenate([rng.permutation(cand) for _ in range(reps)])[:need]


def placebo_events(events: pd.DataFrame, panel: Panel, dec: Decomposition, n_per_event: int = 1,
                   seed: int | None = None, exclude_window: int | None = None, *,
                   min_price: float | None = None) -> pd.DataFrame:
    seed = params.RANDOM_SEED if seed is None else seed
    window = params.EVENT_COOLDOWN if exclude_window is None else exclude_window
    min_price = params.MIN_PRICE if min_price is None else min_price
    if n_per_event < 0 or window < 0:
        raise ValueError("n_per_event and exclude_window must be non-negative")
    dates = panel.dates
    rng = np.random.default_rng(seed)
    ev_pos, tickers, ev_col = _inputs(events, dates)

    rows, cols, srcs = [], [], []
    if len(events) and n_per_event:
        ok = _eligible(panel, dec, tickers, ev_pos, ev_col, window, min_price)
        # Canonical order (ticker, date, input row) so the draw does not depend on how events were sorted.
        order = np.lexsort((np.arange(len(events)), ev_pos, ev_col))
        bounds = np.searchsorted(ev_col[order], np.arange(len(tickers) + 1))
        for j in range(len(tickers)):
            src = order[bounds[j]:bounds[j + 1]]
            cand = np.flatnonzero(ok[:, j])
            if len(cand) == 0:
                continue
            rows.append(_draw(rng, cand, len(src) * n_per_event))
            cols.append(np.full(len(src) * n_per_event, j))
            srcs.append(np.repeat(src, n_per_event))

    pos = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
    src = np.concatenate(srcs) if srcs else np.empty(0, dtype=np.int64)
    d = dates[pos]
    t = np.asarray(tickers, dtype=object)[col]
    price = panel.raw_close if panel.raw_close is not None else panel.close

    out = pd.DataFrame({"date": d, "ticker": t, "pos": pos.astype(np.int64)})
    out["mkt_ret"] = dec.mkt_ret.reindex(d).to_numpy(dtype=float)
    for name, frame in (("z", dec.z), ("resid", dec.resid), ("ret", dec.stock_ret), ("sec_ret", dec.sec_ret),
                        ("sector", dec.sector), ("vol_z", dec.vol_z), ("alpha", dec.alpha),
                        ("beta_mkt", dec.beta_mkt), ("beta_sec", dec.beta_sec), ("sigma", dec.sigma),
                        ("tot_sigma", dec.tot_sigma), ("price", price)):
        out[name] = _pick(frame, d, t)
    out["placebo_of"] = events.index.to_numpy()[src]

    # Match the real events' dtypes so the placebo flows through the same downstream code unchanged.
    for c in EVENT_COLUMNS:
        if c not in events.columns or isinstance(events[c].dtype, pd.CategoricalDtype):
            continue
        if out[c].dtype != events[c].dtype:
            try:
                out[c] = out[c].astype(events[c].dtype)
            except (TypeError, ValueError):
                pass
    out = out[EVENT_COLUMNS + ["placebo_of"]]
    return out.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)

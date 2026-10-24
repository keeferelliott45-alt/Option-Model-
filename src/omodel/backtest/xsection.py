from __future__ import annotations

import numpy as np
import pandas as pd

from omodel.backtest import stats

MIN_STOCKS = 50


def monthly_ic(factor: pd.DataFrame, target: pd.DataFrame, mask: pd.DataFrame,
               dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per formation date: Spearman IC, top-minus-bottom quintile target spread, n stocks."""
    rows = []
    for d in dates:
        f, y, m = factor.loc[d], target.loc[d], mask.loc[d]
        ok = m & f.notna() & y.notna() & np.isfinite(f)
        if ok.sum() < MIN_STOCKS:
            continue
        fr, yv = f[ok].rank(), y[ok]
        ic = np.corrcoef(fr, yv.rank())[0, 1]
        q = pd.qcut(fr, 5, labels=False, duplicates="drop")
        spread = yv[q == q.max()].mean() - yv[q == q.min()].mean()
        rows.append({"date": d, "ic": ic, "spread": spread, "n": int(ok.sum())})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["ic", "spread", "n"])


def evaluate(ic: pd.DataFrame, sign: int, lags: int, dev_end: pd.Timestamp, holdout: tuple,
             subperiods: list[tuple]) -> dict:
    dev = ic[ic.index <= dev_end]
    nw = stats.newey_west(dev["ic"], lags=lags)
    subs = []
    for a, b in subperiods:
        s = dev[(dev.index >= a) & (dev.index <= b)]["ic"]
        subs.append(float(s.mean()) if len(s) else np.nan)
    ho = ic[(ic.index >= holdout[0]) & (ic.index <= holdout[1])]["ic"]
    ho_mean = float(ho.mean()) if len(ho) else np.nan
    return {
        "months": int(len(dev)), "avg_stocks": float(dev["n"].mean()) if len(dev) else np.nan,
        "mean_ic": nw["mean"], "t": nw["t"], "p": nw["p_value"],
        "spread": float(dev["spread"].mean()) if len(dev) else np.nan,
        "sub_means": subs, "subs_right_sign": int(sum(np.sign(s) == sign for s in subs if np.isfinite(s))),
        "holdout_ic": ho_mean, "holdout_months": int(len(ho)),
        "sign_ok": bool(np.sign(nw["mean"]) == sign and abs(nw["t"]) >= 2),
        "holdout_ok": bool(np.isfinite(ho_mean) and np.sign(ho_mean) == sign),
    }

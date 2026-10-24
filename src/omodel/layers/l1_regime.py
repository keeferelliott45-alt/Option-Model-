from __future__ import annotations

import numpy as np
import pandas as pd

from omodel import params
from omodel.types import Panel

VIX_HIGH = params.register("VIX_HIGH", 20.0, "theory", "L1", "Conventional calm/stressed VIX boundary (H2c split).")
CORR_WINDOW = params.register("CORR_WINDOW", 60, "convention", "L1", "Days for average pairwise correlation.")
BREADTH_MA = params.register("BREADTH_MA", 50, "theory", "L1", "Thesis: % of members above 50-day MA.")
TENSION_HALFLIFE = params.register("TENSION_HALFLIFE", 10, "theory", "L1",
                                   "Days for old tension to fade unless refreshed (thesis decay function).")
HIGH_CORR = params.register("HIGH_CORR", 0.35, "convention", "L1", "Label threshold: 'market trading as one'.")
HIGH_TENSION = params.register("HIGH_TENSION", 0.5, "convention", "L1", "Label threshold on the tension z-index.")
Z_MIN_OBS = params.register("REGIME_Z_MIN_OBS", 252, "convention", "L1", "History before an expanding z-score exists.")


def _expanding_z(s: pd.Series) -> pd.Series:
    past = s.shift(1).expanding(min_periods=Z_MIN_OBS)
    return (s - past.mean()) / past.std()


def avg_pairwise_corr(ret: pd.DataFrame, member: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Equal-weight variance identity: rho = (var_EW - mean(var_i)/N) / (mean(sd_i)^2 - mean(var_i)/N)."""
    window = CORR_WINDOW if window is None else window
    r = ret.where(member.reindex_like(ret).eq(True))
    ew = r.mean(axis=1)
    var_p = ew.rolling(window, min_periods=window // 2).var()
    sd = r.rolling(window, min_periods=window // 2).std()
    n = sd.notna().sum(axis=1)
    mean_var = (sd ** 2).mean(axis=1)
    mean_sd = sd.mean(axis=1)
    rho = (var_p - mean_var / n) / (mean_sd ** 2 - mean_var / n)
    return rho.clip(-1, 1)


def regime_features(macro: pd.DataFrame, panel: Panel, stock_ret: pd.DataFrame) -> pd.DataFrame:
    """Daily features known at the close of each date (macro rows are already point-in-time)."""
    m = macro.reindex(panel.dates).ffill(limit=3)
    f = pd.DataFrame(index=panel.dates)
    f["vix"] = m["vix"]
    f["vix_term"] = m["vix"] / m["vix3m"]
    f["backwardation"] = f["vix_term"] > 1
    f["d_y2_5d"] = m["y2"].diff(5)
    f["curve_2s10s"] = m["y10"] - m["y2"]
    f["credit_20d"] = np.log(m["hyg"] / m["lqd"]).diff(20)
    f["dxy_20d"] = np.log(m["dxy"]).diff(20)
    close = panel.close.where(panel.member.eq(True))
    f["breadth"] = (close > close.rolling(BREADTH_MA, min_periods=BREADTH_MA).mean()).sum(axis=1) / close.notna().sum(axis=1)
    f["avg_corr"] = avg_pairwise_corr(stock_ret, panel.member)
    shocks = pd.concat([np.log(m["gold"]).diff(5).abs(), np.log(m["oil"]).diff(5).abs(),
                        m["vix"].diff(5), np.log(m["usdjpy"]).diff().abs().rolling(20).mean()], axis=1)
    z = shocks.apply(_expanding_z).mean(axis=1)
    f["tension"] = z.ewm(halflife=TENSION_HALFLIFE, ignore_na=True).mean()
    return f


def regime_label(row: pd.Series) -> str:
    parts = ["risk-off" if row.get("vix", np.nan) >= VIX_HIGH else "calm"]
    if row.get("backwardation", False):
        parts.append("VIX backwardation (acute stress)")
    corr = row.get("avg_corr", np.nan)
    if np.isfinite(corr):
        parts.append(f"{'high' if corr >= HIGH_CORR else 'normal'} correlation ({corr:.2f})")
    t = row.get("tension", np.nan)
    if np.isfinite(t):
        parts.append(f"tension {'elevated' if t > HIGH_TENSION else 'normal'} ({t:+.2f})")
    return ", ".join(parts)

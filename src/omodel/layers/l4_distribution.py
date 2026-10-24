from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omodel import params

PRIOR_STRENGTH = params.register("L4_PRIOR_STRENGTH", 30, "theory", "L4",
                                 "Pseudo-events: stock- and regime-specific history gets weight n/(n+30).")
UNCERTAINTY_WIDEN = params.register("L4_UNCERTAIN_WIDEN", 1.25, "theory", "L4",
                                    "SOFT/UNKNOWN fundamentals widen the distribution; never shift its center.")
MIN_HISTORY = params.register("L4_MIN_HISTORY", 100, "theory", "L4",
                              "Fewer comparable past events than this and no distribution is stated.")
N_BINS = params.register("L4_BINS", 201, "convention", "L4", "Discrete S_T buckets in the output.")


@dataclass
class Distribution:
    grid: np.ndarray      # S_T bucket centres
    probs: np.ndarray     # probability per bucket, sums to 1
    n_events: int
    n_stock: int
    n_regime: int
    horizon: int
    stats: dict

    def prob_below(self, x: float) -> float:
        return float(self.probs[self.grid < x].sum())


def _samples(hist: pd.DataFrame, horizon: int) -> tuple[np.ndarray, int]:
    """Forward log returns in units of sigma*sqrt(horizon). Noise is unit-free there; drift grows with sqrt(n/h)."""
    h = 15 if horizon <= 22 else 30
    x = hist[f"raw_{h}"].to_numpy(float) / (hist["tot_sigma"].to_numpy(float) * np.sqrt(h))
    ok = np.isfinite(x)
    x = x[ok]
    m = x.mean()
    return (x - m) + m * np.sqrt(horizon / h), ok


def conditional(history: pd.DataFrame, asof: pd.Timestamp, ticker: str, spot: float, sigma_daily: float,
                horizon: int, high_vix: bool | None = None, fund_status: str = "PASS") -> Distribution | None:
    """Empirical S_T distribution after similar past drops, using only events whose exit was before `asof`."""
    h = 15 if horizon <= 22 else 30
    past = history[pd.to_datetime(history[f"exit_date_{h}"]) < asof]
    x, ok = _samples(past, horizon)
    if len(x) < MIN_HISTORY:
        return None
    past = past[ok]
    w = np.full(len(x), 1.0 / len(x))
    n_reg = 0
    if high_vix is not None and "high_vix" in past:
        same = past["high_vix"].to_numpy(bool) == high_vix
        n_reg = int(same.sum())
        if n_reg:
            a = n_reg / (n_reg + PRIOR_STRENGTH)
            w = (1 - a) * w + a * same / n_reg
    own = (past["ticker"] == ticker).to_numpy()
    n_own = int(own.sum())
    if n_own:
        a = n_own / (n_own + PRIOR_STRENGTH)
        w = (1 - a) * w + a * own / n_own
    if fund_status in ("SOFT", "UNKNOWN"):
        c = float(w @ x)
        x = c + (x - c) * UNCERTAINTY_WIDEN
    log_r = x * sigma_daily * np.sqrt(horizon)
    st = spot * np.exp(log_r)
    edges = np.quantile(st, np.linspace(0, 1, N_BINS + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, len(edges) - 2)
    probs = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    centres = np.bincount(idx, weights=w * st, minlength=len(edges) - 1) / np.where(probs > 0, probs, 1)
    keep = probs > 0
    stats = {"mean_ret": float(w @ log_r), "median_ret": float(np.median(log_r)),
             "p_down": float(w[log_r < 0].sum()), "p5": float(np.quantile(log_r, 0.05)),
             "p95": float(np.quantile(log_r, 0.95))}
    return Distribution(centres[keep], probs[keep] / probs[keep].sum(), len(x), n_own, n_reg, horizon, stats)


def thesis_sentence(ev: pd.Series, dist: Distribution, regime: str, veto_reasons: list[str], news: bool | None,
                    trade: str) -> str:
    """Every number traces to a layer: L2 (move split), L1 (regime), L3 (veto), L4 (history), L5 (trade)."""
    explained = ev["ret"] - ev["resid"]
    news_txt = {True: "an 8-K was filed", False: "no 8-K was filed", None: "8-K status unknown"}[news]
    return (f"{ev['ticker']} moved {100 * ev['ret']:+.1f}% on {pd.Timestamp(ev['date']).date()}; market and "
            f"{ev['sector']} explain {100 * explained:+.1f}%, leaving {100 * ev['resid']:+.1f}% unexplained "
            f"(z={ev['z']:.1f}); {news_txt}. Regime: {regime}. Fundamentals: {'; '.join(veto_reasons)}. "
            f"After {dist.n_events} comparable drops ({dist.n_stock} in this stock), the {dist.horizon}-day move "
            f"averaged {100 * dist.stats['mean_ret']:+.1f}% with {100 * dist.stats['p_down']:.0f}% lower "
            f"(5-95%: {100 * dist.stats['p5']:+.1f}% to {100 * dist.stats['p95']:+.1f}%). {trade}")

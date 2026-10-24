from __future__ import annotations

import numpy as np
import pandas as pd

from omodel import params

EWMA_LAMBDA = params.register("EWMA_LAMBDA", 0.94, "theory", "forecast", "RiskMetrics decay for standardising returns.")
FHS_BLOCK = params.register("FHS_BLOCK", 5, "convention", "forecast", "Bootstrap block length (keeps vol clustering).")
FHS_YEARS = params.register("FHS_YEARS", 3, "convention", "forecast", "Years of own history for the return shape.")
SIM_PATHS = params.register("SIM_PATHS", 20000, "convention", "forecast", "Simulated paths per live forecast.")
HAR_FEATURES = ("lrv5", "lrv22", "lrv66", "lvix", "vix_term")
_FLOOR = 1e-4


def rv(ret: pd.DataFrame, k: int) -> pd.DataFrame:
    """Annualised realized vol over the k days ending at t (inclusive)."""
    return np.sqrt(252 * (ret ** 2).rolling(k, min_periods=max(3, int(0.8 * k))).mean())


def har_features(ret: pd.DataFrame, vix: pd.Series, vix3m: pd.Series) -> dict[str, pd.DataFrame]:
    feats = {f"lrv{k}": np.log(rv(ret, k).clip(lower=_FLOOR)) for k in (5, 22, 66)}
    lv = np.log(vix.reindex(ret.index).ffill(limit=3))
    vt = (vix / vix3m).reindex(ret.index).ffill(limit=3)
    feats["lvix"] = pd.DataFrame(np.repeat(lv.to_numpy()[:, None], ret.shape[1], 1), index=ret.index, columns=ret.columns)
    feats["vix_term"] = pd.DataFrame(np.repeat(vt.to_numpy()[:, None], ret.shape[1], 1), index=ret.index, columns=ret.columns)
    return feats


def har_target(ret: pd.DataFrame, h: int = 21) -> pd.DataFrame:
    """log realized vol over t+1 .. t+h."""
    fwd = (ret ** 2).shift(-h).rolling(h, min_periods=int(0.8 * h)).mean()
    return np.log(np.sqrt(252 * fwd).clip(lower=_FLOOR))


class HAR:
    def __init__(self, features: tuple[str, ...] = HAR_FEATURES):
        self.features = features
        self.coef: np.ndarray | None = None
        self.resid_var = np.nan

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HAR":
        ok = X.notna().all(axis=1) & y.notna()
        A = np.column_stack([np.ones(ok.sum()), X.loc[ok, list(self.features)].to_numpy()])
        self.coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        self.resid_var = float(np.var(y[ok].to_numpy() - A @ self.coef, ddof=A.shape[1]))
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        A = np.column_stack([np.ones(len(X)), X[list(self.features)].to_numpy()])
        return A @ self.coef

    def predict_var(self, X: pd.DataFrame) -> np.ndarray:
        """E[vol^2] under a lognormal error: exp(2*mu + 2*s^2)."""
        return np.exp(2 * self.predict_log(X) + 2 * self.resid_var)


def ewma_z(ret: pd.Series | np.ndarray, lam: float | None = None) -> np.ndarray:
    """Returns standardised by an EWMA vol built from strictly earlier returns."""
    lam = EWMA_LAMBDA if lam is None else lam
    r = np.asarray(ret, dtype=float)
    out = np.full(len(r), np.nan)
    finite = r[np.isfinite(r)]
    if len(finite) < 30:
        return out
    var = float(np.mean(finite[:30] ** 2))
    for i, x in enumerate(r):
        if np.isfinite(x):
            if i >= 30 and var > 0:
                out[i] = x / np.sqrt(var)
            var = lam * var + (1 - lam) * x * x
    return out


def bank(z: np.ndarray, h: int, n: int, rng: np.random.Generator, earnings_z: np.ndarray | None = None,
         block: int | None = None) -> np.ndarray:
    """n block-bootstrapped sums of h standardised daily returns; optionally one day is an earnings-day draw."""
    block = FHS_BLOCK if block is None else block
    z = z[np.isfinite(z)]
    if len(z) < 2 * block:
        return np.full(n, np.nan)
    n_blocks = -(-h // block)
    starts = rng.integers(0, len(z) - block + 1, size=(n, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, :h]
    draws = z[idx]
    if earnings_z is not None:
        ez = earnings_z[np.isfinite(earnings_z)]
        if len(ez):
            draws[:, 0] = rng.choice(ez, size=n)
    return draws.sum(axis=1)


def shape(s: np.ndarray, h: int) -> np.ndarray:
    """Centre the bootstrapped sums (drift = 0: a trending history must not leak momentum) and scale to variance h."""
    s = s[np.isfinite(s)]
    s = s - s.mean()
    return s / np.sqrt(np.mean(s ** 2) / h)


CRASH_LOG_RET = params.register("CRASH_LOG_RET", -0.25, "theory", "forecast",
                                "Stage 1/2 crash definition over 63 days; scaled by sqrt(h/63) for other horizons.")


def price_distribution(spot: float, vol_ann: float, h: int, z_hist: np.ndarray, earnings_z: np.ndarray | None = None,
                       n: int | None = None, seed: int = 0, bins: int = 401,
                       crash_ratio: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Discrete S_T distribution (grid, probs) for Layer 5: drift 0, HAR vol, FHS shape, short-interest tail tilt."""
    n = SIM_PATHS if n is None else n
    rng = np.random.default_rng(seed)
    s = bank(z_hist, h, n, rng, earnings_z)
    log_r = vol_ann / np.sqrt(252) * shape(s, h)
    w = np.ones(len(log_r))
    tail = log_r < CRASH_LOG_RET * np.sqrt(h / 63)
    p_tail = tail.mean()
    if crash_ratio != 1.0 and 0 < p_tail < 1:
        new_tail = min(p_tail * crash_ratio, 0.5)
        w[tail] = new_tail / p_tail
        w[~tail] = (1 - new_tail) / (1 - p_tail)
    st = spot * np.exp(log_r)
    edges = np.quantile(st, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.searchsorted(edges, st, side="right") - 1, 0, len(edges) - 2)
    probs = np.bincount(idx, weights=w, minlength=len(edges) - 1)
    centres = np.bincount(idx, weights=w * st, minlength=len(edges) - 1) / np.where(probs > 0, probs, 1)
    keep = probs > 0
    return centres[keep], probs[keep] / probs[keep].sum()

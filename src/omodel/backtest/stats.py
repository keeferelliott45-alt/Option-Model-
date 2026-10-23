from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

from omodel import holdout, params

CLUSTER_FREQ = params.register(
    "CLUSTER_FREQ", "M", "convention", "backtest",
    "Calendar-month date clusters: wider than the 15d primary horizon, so overlapping forward windows "
    "and crash-clustered events mostly share a cluster.")
MIN_EFFECTIVE_CLUSTERS = params.register(
    "MIN_EFFECTIVE_CLUSTERS", 30, "convention", "backtest",
    "Kish effective number of date clusters, (sum c_g)^2 / sum c_g^2, below which cluster-robust CIs and "
    "p-values are anti-conservative (few-cluster guidance of ~20-50, Cameron & Miller 2015); such results "
    "are flagged low-confidence, matching the 30-instance rule.")

_MAX_DRAW_CELLS = 4_000_000  # caps memory of one (reps x clusters) block of bootstrap draw counts
_ZERO_TOL = 1e-12  # relative to the data's scale: a dispersion below this is float noise on a constant series


def _to_datetime(x):
    # pd.to_datetime's cache heuristic iterates Timestamps even on datetime64 input, so skip it when typed.
    return x if pd.api.types.is_datetime64_any_dtype(getattr(x, "dtype", None)) else pd.to_datetime(x)


def _as_days(dates) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(_to_datetime(dates))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def cluster_labels(dates: pd.Series, freq: str | None = None) -> pd.Series:
    s = _to_datetime(dates if isinstance(dates, pd.Series) else pd.Series(dates))
    if s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.to_period(freq or CLUSTER_FREQ).rename("cluster")


def _resolve(reps: int | None, ci: float | None, seed: int | None) -> tuple[int, float, int]:
    return (params.BOOTSTRAP_REPS if reps is None else int(reps),
            params.CI_LEVEL if ci is None else float(ci),
            params.RANDOM_SEED if seed is None else seed)


def _codes(clusters) -> np.ndarray:
    if not isinstance(clusters, (pd.Series, pd.Index, np.ndarray, pd.api.extensions.ExtensionArray)):
        clusters = np.asarray(clusters)
    return pd.factorize(clusters)[0]


def _compact(codes: np.ndarray) -> tuple[np.ndarray, int]:
    uniq, inv = np.unique(codes, return_inverse=True)
    return inv, len(uniq)


def _kish(counts: np.ndarray) -> float:
    s2 = float(np.square(counts).sum())
    return float(counts.sum() ** 2 / s2) if s2 > 0 else 0.0


def _draw_counts(rng: np.random.Generator, g: int, reps: int) -> Iterator[np.ndarray]:
    # Multinomial counts = how often each cluster appears in a with-replacement resample of g clusters.
    chunk = max(1, _MAX_DRAW_CELLS // g)
    probs = np.full(g, 1.0 / g)
    for start in range(0, reps, chunk):
        yield rng.multinomial(g, probs, size=min(chunk, reps - start)).astype(float)


def _p_normal(est: float, se: float) -> float:
    # se == 0 means a degenerate (constant) sample: significance is undefined, never a discovery.
    if not (np.isfinite(est) and np.isfinite(se)) or se <= 0:
        return np.nan
    return float(2 * sps.norm.sf(abs(est) / se))


def _is_zero(sd: float, scale: float) -> bool:
    return not np.isfinite(sd) or sd <= _ZERO_TOL * scale


def _interval(boot: np.ndarray, ci: float, est: float) -> dict:
    boot = boot[np.isfinite(boot)]
    if len(boot) < 2:
        return dict(lo=np.nan, hi=np.nan, se=np.nan, p_value=np.nan)
    alpha = 1 - ci
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    se = float(boot.std(ddof=1))
    if _is_zero(se, float(np.abs(boot).max())):
        se = 0.0
    return dict(lo=float(lo), hi=float(hi), se=se, p_value=_p_normal(est, se))


def cluster_bootstrap_mean(values: pd.Series, clusters: pd.Series, reps: int | None = None,
                           ci: float | None = None, seed: int | None = None) -> dict:
    reps, ci, seed = _resolve(reps, ci, seed)
    v = np.asarray(values, dtype=float)
    codes = _codes(clusters)
    if len(v) != len(codes):
        raise ValueError("values and clusters must have the same length")
    keep = ~np.isnan(v) & (codes >= 0)
    v = v[keep]
    codes, g = _compact(codes[keep])
    n = len(v)
    cnts = np.bincount(codes, minlength=g).astype(float)
    out = dict(mean=float(v.mean()) if n else np.nan, lo=np.nan, hi=np.nan, se=np.nan, p_value=np.nan,
               n=n, n_clusters=g, n_eff_clusters=_kish(cnts))
    if n < 2 or g < 2:
        return out
    sums = np.bincount(codes, weights=v, minlength=g)
    rng = np.random.default_rng(seed)
    boot = np.concatenate([(d @ sums) / (d @ cnts) for d in _draw_counts(rng, g, reps)])
    out.update(_interval(boot, ci, out["mean"]))
    return out


def cluster_bootstrap_diff(values: pd.Series, groups: pd.Series, clusters: pd.Series, high, low,
                           reps: int | None = None, ci: float | None = None, seed: int | None = None) -> dict:
    reps, ci, seed = _resolve(reps, ci, seed)
    v = np.asarray(values, dtype=float)
    grp = pd.Series(groups).reset_index(drop=True)
    is_h = (grp == high).fillna(False).to_numpy(dtype=bool)
    is_l = (grp == low).fillna(False).to_numpy(dtype=bool)
    codes = _codes(clusters)
    if not (len(v) == len(grp) == len(codes)):
        raise ValueError("values, groups and clusters must have the same length")
    keep = ~np.isnan(v) & (codes >= 0) & (is_h | is_l)
    v, is_h, is_l = v[keep], is_h[keep], is_l[keep]
    codes, g = _compact(codes[keep])
    n_h, n_l = int(is_h.sum()), int(is_l.sum())
    m_h = float(v[is_h].mean()) if n_h else np.nan
    m_l = float(v[is_l].mean()) if n_l else np.nan
    c_h = np.bincount(codes, weights=is_h.astype(float), minlength=g)
    c_l = np.bincount(codes, weights=is_l.astype(float), minlength=g)
    out = dict(diff=m_h - m_l, lo=np.nan, hi=np.nan, se=np.nan, p_value=np.nan, n_high=n_h, n_low=n_l,
               mean_high=m_h, mean_low=m_l, n_clusters=g, n_eff_high=_kish(c_h), n_eff_low=_kish(c_l))
    # Each group must span >= 2 clusters, else its resampled mean has no between-cluster variation.
    if n_h < 2 or n_l < 2 or (c_h > 0).sum() < 2 or (c_l > 0).sum() < 2:
        return out
    s_h = np.bincount(codes, weights=np.where(is_h, v, 0.0), minlength=g)
    s_l = np.bincount(codes, weights=np.where(is_l, v, 0.0), minlength=g)
    rng = np.random.default_rng(seed)
    parts = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for d in _draw_counts(rng, g, reps):
            parts.append((d @ s_h) / (d @ c_h) - (d @ s_l) / (d @ c_l))
    out.update(_interval(np.concatenate(parts), ci, out["diff"]))
    return out


def wilson_interval(k: int, n: int, ci: float | None = None) -> tuple[float, float, float]:
    ci = params.CI_LEVEL if ci is None else ci
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"wilson_interval needs 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (np.nan, np.nan, np.nan)
    z = sps.norm.ppf(1 - (1 - ci) / 2)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(p), float(center - half), float(center + half))


def newey_west(series: pd.Series, lags: int | None = None) -> dict:
    lags = params.PRIMARY_HORIZON if lags is None else int(lags)
    y = pd.Series(series, dtype=float).dropna().to_numpy()
    n = len(y)
    out = dict(mean=float(y.mean()) if n else np.nan, se=np.nan, t=np.nan, p_value=np.nan, n=n, lags=lags)
    if n < 2:
        return out
    lags = min(lags, n - 1)
    res = sm.OLS(y, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "kernel": "bartlett"})
    se = float(res.bse[0])
    if _is_zero(se, float(np.abs(y).max())):
        se = 0.0
    out.update(se=se, t=out["mean"] / se if se > 0 else np.nan, p_value=_p_normal(out["mean"], se), lags=lags)
    return out


def calendar_time_portfolio(position_returns: pd.DataFrame, all_dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    df = position_returns
    if "date" in df.columns:
        dates = df["date"]
    elif "date" in df.index.names and df.index.nlevels > 1:
        dates = df.index.get_level_values("date")
    else:
        dates = df.index
    flat = pd.DataFrame({"date": _as_days(dates), "abn": df["abn"].to_numpy(dtype=float)})
    holdout.assert_clean(pd.DatetimeIndex(flat["date"]), what="calendar-time positions")
    out = flat.dropna().groupby("date")["abn"].agg(ret="mean", n_positions="size")
    out.index = pd.DatetimeIndex(out.index, name="date")
    if all_dates is not None:
        cal = _as_days(all_dates).rename("date")
        missing = out.index.difference(cal)
        if len(missing):
            # Reindexing would silently drop these positions from the strategy's return series.
            raise ValueError(f"{len(missing)} position dates are not in all_dates, e.g. {missing[0].date()}")
        out = out.reindex(cal)
        out["ret"] = out["ret"].fillna(0.0)
        out["n_positions"] = out["n_positions"].fillna(0)
    out["n_positions"] = out["n_positions"].astype(int)
    return out


def annualized_sharpe(daily: pd.Series, periods: int = 252) -> float:
    x = pd.Series(daily, dtype=float).dropna()
    if len(x) < 2:
        return np.nan
    sd = float(x.std(ddof=1))
    if _is_zero(sd, float(x.abs().max())):
        return np.nan
    return float(x.mean() / sd * np.sqrt(periods))


def bh_adjust(pvals: dict[str, float], q: float | None = None) -> pd.DataFrame:
    q = params.FDR_Q if q is None else q
    pv = dict(pvals)
    names = list(pv)
    p = np.array([np.nan if pv[k] is None else pv[k] for k in names], dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m:
        idx = np.flatnonzero(ok)
        order = idx[np.argsort(p[idx], kind="stable")]
        ranked = p[order] * m / np.arange(1, m + 1)
        adj[order] = np.minimum(np.minimum.accumulate(ranked[::-1])[::-1], 1.0)
    return pd.DataFrame({"name": names, "p": p, "p_adj": adj, "reject": ok & (adj <= q)})


def summarize(values: pd.Series, dates: pd.Series, label: str = "") -> dict:
    v = np.asarray(values, dtype=float)
    d = _as_days(dates)
    if len(v) != len(d):
        raise ValueError("values and dates must have the same length")
    keep = ~np.isnan(v) & ~d.isna()
    v, d = v[keep], d[keep]
    boot = cluster_bootstrap_mean(v, cluster_labels(pd.Series(d)))
    n = len(v)
    win, win_lo, win_hi = wilson_interval(int((v > 0).sum()), n)
    low = n < params.MIN_SAMPLE_FINDING or boot["n_eff_clusters"] < MIN_EFFECTIVE_CLUSTERS
    return dict(label=label, n=n, mean=boot["mean"], ci_lo=boot["lo"], ci_hi=boot["hi"], se=boot["se"],
                p_value=boot["p_value"], n_clusters=boot["n_clusters"], n_eff_clusters=boot["n_eff_clusters"],
                median=float(np.median(v)) if n else np.nan, win_rate=win, win_lo=win_lo, win_hi=win_hi,
                low_confidence=bool(low))

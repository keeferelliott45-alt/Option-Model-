from __future__ import annotations

import numpy as np
import pandas as pd

from omodel import holdout, params
from omodel.backtest import stats
from omodel.backtest.stats import _as_days, bh_adjust, cluster_bootstrap_diff, cluster_labels, summarize


def _sorted_quantiles(a: np.ndarray, probs: np.ndarray) -> np.ndarray:
    # numpy's default 'linear' quantile on an already-sorted array, without re-sorting it every date.
    h = (len(a) - 1) * probs
    lo = np.floor(h).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    return a[lo] + (h - lo) * (a[hi] - a[lo])


def expanding_bucket(values: pd.Series, dates: pd.Series, n_buckets: int = 3,
                     min_obs: int | None = None) -> pd.Series:
    min_obs = params.EXPANDING_MIN_OBS if min_obs is None else int(min_obs)
    if isinstance(values, pd.Series) and isinstance(dates, pd.Series) and not values.index.equals(dates.index):
        if not (dates.index.is_unique and values.index.isin(dates.index).all()):
            raise ValueError("dates must share values' index labels (or be a positional array)")
        dates = dates.reindex(values.index)
    v = np.asarray(values, dtype=float)
    d = _as_days(dates).to_numpy()
    if len(v) != len(d):
        raise ValueError("values and dates must have the same length")
    out = np.full(len(v), np.nan)
    rows = np.flatnonzero(~np.isnat(d))
    rows = rows[np.argsort(d[rows], kind="stable")]
    _, starts = np.unique(d[rows], return_index=True)
    bounds = np.append(starts, len(rows))
    probs = np.arange(1, n_buckets) / n_buckets
    history = np.empty(0)
    for s, e in zip(bounds[:-1], bounds[1:]):
        idx = rows[s:e]
        vals = v[idx]
        ok = ~np.isnan(vals)
        # Cuts come only from strictly earlier dates: this date's rows are inserted after labelling.
        if len(history) >= min_obs and ok.any():
            cuts = _sorted_quantiles(history, probs)
            out[idx[ok]] = np.searchsorted(cuts, vals[ok], side="left")
        if ok.any():
            new = np.sort(vals[ok])
            history = np.insert(history, np.searchsorted(history, new), new)
    name = values.name if isinstance(values, pd.Series) else None
    index = values.index if isinstance(values, pd.Series) else pd.RangeIndex(len(v))
    return pd.Series(out, index=index, name=name)


def _group_test(events: pd.DataFrame, col: str, metric_col: str, levels: list, labels: list[str]) -> dict:
    holdout.assert_clean(_as_days(events["date"]), what="factor-test events")
    groups = events[col]
    metric = events[metric_col]
    buckets = []
    for level, lab in zip(levels, labels):
        mask = (groups == level).fillna(False).to_numpy(dtype=bool)
        summ = summarize(metric[mask], events["date"][mask], label=f"{col}={lab}")
        buckets.append({"bucket": lab, **summ})
    diff = cluster_bootstrap_diff(metric, groups, cluster_labels(events["date"]), high=levels[-1], low=levels[0])
    n = int(sum(b["n"] for b in buckets))
    n_eff = min(diff["n_eff_high"], diff["n_eff_low"])
    low = any(b["low_confidence"] for b in buckets) or n_eff < stats.MIN_EFFECTIVE_CLUSTERS
    return dict(buckets=buckets, diff=diff["diff"], diff_lo=diff["lo"], diff_hi=diff["hi"],
                p_value=diff["p_value"], n=n, n_eff_clusters=n_eff, low_confidence=bool(low))


def bucket_test(events: pd.DataFrame, bucket_col: str, metric_col: str, n_buckets: int = 3) -> dict:
    levels = list(range(n_buckets))
    return _group_test(events, bucket_col, metric_col, levels, [str(b) for b in levels])


def _flags_as_float(flags: pd.Series) -> np.ndarray:
    known = flags.notna().to_numpy()
    vals = flags[known]
    if pd.api.types.is_bool_dtype(vals.dtype):
        f = vals.to_numpy(dtype=float)
    else:
        try:
            f = pd.to_numeric(vals, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"flag column {flags.name!r} must be boolean or 0/1") from exc
        if not np.isin(f, (0.0, 1.0)).all():
            raise ValueError(f"flag column {flags.name!r} must be boolean or 0/1")
    out = np.full(len(flags), np.nan)
    out[known] = f
    return out


def binary_test(events: pd.DataFrame, flag_col: str, metric_col: str) -> dict:
    ev = events.assign(**{flag_col: _flags_as_float(events[flag_col])})
    return _group_test(ev, flag_col, metric_col, [0.0, 1.0], ["False", "True"])


def factor_battery(events: pd.DataFrame, specs: list[dict], metric_col: str) -> pd.DataFrame:
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("factor spec names must be unique")
    holdout.assert_clean(_as_days(events["date"]), what="factor-battery events")
    rows = []
    for spec in specs:
        kind = spec["kind"]
        if kind == "bucket":
            k = int(spec.get("n_buckets", 3))
            bcol = f"__bucket_{spec['col']}"
            labels = expanding_bucket(events[spec["col"]], events["date"], n_buckets=k, min_obs=spec.get("min_obs"))
            ev = events.assign(**{bcol: labels.to_numpy()})
            res = bucket_test(ev, bcol, metric_col, n_buckets=k)
        elif kind == "binary":
            res = binary_test(events, spec["col"], metric_col)
        else:
            raise ValueError(f"unknown factor kind {kind!r}")
        row = dict(name=spec["name"], kind=kind, n=res["n"], diff=res["diff"], diff_lo=res["diff_lo"],
                   diff_hi=res["diff_hi"], p_value=res["p_value"])
        # Binary groups map to bucket 0 (False) and 1 (True), so 'top minus bottom' reads the same everywhere.
        for i, b in enumerate(res["buckets"]):
            row[f"mean_{i}"] = b["mean"]
            row[f"n_{i}"] = b["n"]
        row["n_eff_clusters"] = min(b["n_eff_clusters"] for b in res["buckets"])
        row["low_confidence"] = res["low_confidence"]
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["name", "kind", "n", "diff", "diff_lo", "diff_hi", "p_value", "p_adj", "reject",
                                     "n_eff_clusters", "low_confidence"])
    # BH counts every test run, but a low-confidence row can never be a discovery: its p-value is unreliable.
    bh = bh_adjust(dict(zip(out["name"], out["p_value"])))
    out.insert(out.columns.get_loc("p_value") + 1, "p_adj", bh["p_adj"].to_numpy())
    reject = bh["reject"].to_numpy() & ~out["low_confidence"].to_numpy(dtype=bool)
    out.insert(out.columns.get_loc("p_adj") + 1, "reject", reject)
    return out

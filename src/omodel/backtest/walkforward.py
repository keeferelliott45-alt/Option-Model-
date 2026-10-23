from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omodel import holdout, params


@dataclass(frozen=True)
class Split:
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


CHOICE_COLUMNS = ["test_start", "test_end", "chosen", "fit_n", "fit_score", "test_n"]


def make_splits(start: pd.Timestamp, end: pd.Timestamp, fit_years: int | None = None,
                test_months: int | None = None) -> list[Split]:
    fit_years = params.WF_FIT_YEARS if fit_years is None else fit_years
    test_months = params.WF_TEST_MONTHS if test_months is None else test_months
    if test_months <= 0 or fit_years <= 0:
        raise ValueError("fit_years and test_months must be positive")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    fit_months = int(round(12 * fit_years))

    # Offsets from the original start, so month-end boundaries never drift (Jan 31 -> Feb 28 -> Mar 28).
    def anchor(months: int) -> pd.Timestamp:
        return start + pd.DateOffset(months=months)

    splits: list[Split] = []
    k = 0
    while (test_start := anchor(fit_months + k * test_months)) < end:
        test_end = min(anchor(fit_months + (k + 1) * test_months), end)
        splits.append(Split(anchor(k * test_months), test_start, test_start, test_end))
        k += 1
    return splits


def _t_score(x: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(x) / (np.std(x, ddof=1) / np.sqrt(len(x))))


def walk_forward(events_by_config: dict[str, pd.DataFrame], metric_col: str, exit_col: str,
                 splits: list[Split], min_n: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    min_n = params.MIN_SAMPLE_FINDING if min_n is None else min_n
    if not events_by_config:
        raise ValueError("events_by_config is empty")

    prepared = {}
    for name, ev in events_by_config.items():
        date = pd.to_datetime(ev["date"]).to_numpy()
        exit_ = pd.to_datetime(ev[exit_col]).to_numpy()
        metric = ev[metric_col].to_numpy(dtype=float)
        prepared[name] = (ev, date, exit_, metric, np.isfinite(metric), np.zeros(len(ev), dtype=bool))

    choices, parts = [], []
    for s in splits:
        fs, fe, ts, te = (np.datetime64(t) for t in (s.fit_start, s.fit_end, s.test_start, s.test_end))
        best, best_score, best_n, max_n = None, -np.inf, 0, 0
        for name, (_, date, exit_, metric, finite, used) in prepared.items():
            # Purge: only labels fully realised before the test window opens may inform the choice.
            fit = (date >= fs) & (date < fe) & (exit_ < ts) & finite
            used |= fit
            n = int(fit.sum())
            max_n = max(max_n, n)
            if n < min_n:
                continue
            score = _t_score(metric[fit])
            if not np.isnan(score) and (best is None or score > best_score):
                best, best_score, best_n = name, score, n
        test_n = 0
        if best is not None:
            ev, date, _, _, finite, used = prepared[best]
            test = (date >= ts) & (date < te) & finite
            used |= test
            test_n = int(test.sum())
            if test_n:
                parts.append(ev.loc[test].assign(config=best, split_test_start=s.test_start))
        # With no qualifying config, fit_n reports the largest fit sample seen, i.e. why none qualified.
        choices.append({"test_start": s.test_start, "test_end": s.test_end, "chosen": best,
                        "fit_n": best_n if best is not None else max_n,
                        "fit_score": best_score if best is not None else np.nan, "test_n": test_n})

    for name, (_, date, exit_, _, _, used) in prepared.items():
        holdout.assert_clean(pd.DatetimeIndex(date[used]), f"walk-forward events[{name}]")
        holdout.assert_clean(pd.DatetimeIndex(exit_[used]), f"walk-forward events[{name}].{exit_col}")

    if parts:
        oos = pd.concat(parts, ignore_index=True)
    else:
        empty = next(iter(events_by_config.values())).iloc[:0].reset_index(drop=True)
        oos = empty.assign(config=pd.Series(dtype=object), split_test_start=pd.Series(dtype="datetime64[ns]"))
    return oos, pd.DataFrame(choices, columns=CHOICE_COLUMNS)

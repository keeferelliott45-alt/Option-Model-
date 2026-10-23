import numpy as np
import pandas as pd
import pytest

from omodel import holdout, params
from omodel.backtest.walkforward import Split, make_splits, walk_forward

T = pd.Timestamp


def _events(seed: int, means: dict[str, float], start="2000-01-03", end="2006-12-29", n=900, hold_days=21):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    out = {}
    for name, mu in means.items():
        date = pd.DatetimeIndex(np.sort(rng.choice(days, n)))
        out[name] = pd.DataFrame({"date": date, "ticker": rng.choice(list("ABCDEFGH"), n),
                                  "exit_date": date + pd.Timedelta(days=hold_days),
                                  "m": rng.normal(mu, 0.05, n)})
    return out


def test_make_splits_arithmetic():
    splits = make_splits(T("2000-01-01"), T("2003-03-01"), fit_years=2, test_months=6)
    assert splits == [
        Split(T("2000-01-01"), T("2002-01-01"), T("2002-01-01"), T("2002-07-01")),
        Split(T("2000-07-01"), T("2002-07-01"), T("2002-07-01"), T("2003-01-01")),
        Split(T("2001-01-01"), T("2003-01-01"), T("2003-01-01"), T("2003-03-01")),
    ]
    assert make_splits(T("2000-01-01"), T("2002-01-01"), fit_years=2, test_months=6) == []


def test_make_splits_defaults_contiguous_and_no_month_end_drift():
    splits = make_splits(T("2000-01-03"), T("2010-01-01"))
    assert splits[0].fit_end - splits[0].fit_start == T("2002-01-03") - T("2000-01-03")
    assert splits[0].test_end == splits[0].test_start + pd.DateOffset(months=params.WF_TEST_MONTHS)
    for a, b in zip(splits, splits[1:]):
        assert a.test_end == b.test_start and a.fit_end == a.test_start
    assert splits[-1].test_end == T("2010-01-01")
    month_end = make_splits(T("2000-01-31"), T("2002-05-01"), fit_years=2, test_months=1)
    assert [s.test_start for s in month_end] == [T("2002-01-31"), T("2002-02-28"), T("2002-03-31"),
                                                 T("2002-04-30")]
    assert month_end[-1].test_end == T("2002-05-01")


def test_purge_excludes_labels_realised_inside_test_window():
    split = Split(T("2000-01-01"), T("2002-01-01"), T("2002-01-01"), T("2002-07-01"))
    dates = pd.bdate_range("2000-01-03", "2001-12-31")[::10]
    rng = np.random.default_rng(1)
    good = pd.DataFrame({"date": dates, "exit_date": dates + pd.Timedelta(days=5),
                         "m": rng.normal(0.003, 0.01, len(dates))})
    leaky = good.copy()
    leaky["m"] = rng.normal(0.0, 0.01, len(dates))
    late = leaky["date"] >= T("2001-01-01")
    leaky.loc[late, "m"] = 0.01
    realised = leaky.copy()
    # Same labels, but realised only inside the test window: they must not inform the choice.
    leaky.loc[late, "exit_date"] = T("2002-02-01")

    def choose(cfg):
        return walk_forward(cfg, "m", "exit_date", [split], min_n=10)[1]

    assert choose({"leaky": realised, "good": good}).loc[0, "chosen"] == "leaky"
    assert choose({"leaky": leaky, "good": good}).loc[0, "chosen"] == "good"
    assert choose({"leaky": leaky}).loc[0, "fit_n"] == int((~late).sum())


def test_choices_invariant_to_test_window_and_future_data():
    ev = _events(7, {"a": 0.002, "b": 0.004, "c": 0.0})
    splits = make_splits(T("2000-01-03"), T("2007-01-01"), fit_years=2, test_months=6)
    _, base = walk_forward(ev, "m", "exit_date", splits)
    assert base["chosen"].notna().all()
    rng = np.random.default_rng(99)
    for s in splits:
        perturbed = {}
        for name, df in ev.items():
            df = df.copy()
            future = df["date"] >= s.test_start
            df.loc[future, "m"] = rng.normal(0, 10, int(future.sum())) * (1 if name == "c" else -1)
            extra = df[future].head(50).assign(m=100.0 if name == "c" else -100.0)
            perturbed[name] = pd.concat([df, extra], ignore_index=True)
        _, alt = walk_forward(perturbed, "m", "exit_date", splits)
        keep = base["test_start"] <= s.test_start
        cols = ["test_start", "chosen", "fit_n", "fit_score"]
        pd.testing.assert_frame_equal(alt.loc[keep, cols], base.loc[keep, cols])


def test_oos_contains_only_chosen_test_window_rows():
    ev = _events(3, {"a": 0.003, "b": 0.003, "c": -0.01})
    ev["a"].loc[ev["a"].index[::7], "m"] = np.nan
    splits = make_splits(T("2000-01-03"), T("2007-01-01"), fit_years=2, test_months=6)
    oos, choices = walk_forward(ev, "m", "exit_date", splits)
    assert len(choices) == len(splits)
    assert len(oos) == choices["test_n"].sum() > 0
    assert np.isfinite(oos["m"]).all()
    assert {"config", "split_test_start"} <= set(oos.columns)
    for s, (_, ch) in zip(splits, choices.iterrows()):
        rows = oos[oos["split_test_start"] == s.test_start]
        assert len(rows) == ch["test_n"]
        assert (rows["date"] >= s.test_start).all() and (rows["date"] < s.test_end).all()
        assert (rows["config"] == ch["chosen"]).all()
        src = ev[ch["chosen"]]
        expected = src[(src["date"] >= s.test_start) & (src["date"] < s.test_end) & src["m"].notna()]
        assert len(rows) == len(expected)
    assert "c" not in set(oos["config"])


def test_min_n_and_ties():
    split = Split(T("2000-01-01"), T("2002-01-01"), T("2002-01-01"), T("2002-07-01"))
    dates = pd.bdate_range("2000-01-03", "2002-06-28")[::5]
    df = pd.DataFrame({"date": dates, "exit_date": dates + pd.Timedelta(days=1),
                       "m": np.random.default_rng(0).normal(0.01, 0.02, len(dates))})
    oos, choices = walk_forward({"x": df, "y": df.copy()}, "m", "exit_date", [split])
    assert choices.loc[0, "chosen"] == "x"
    assert (oos["config"] == "x").all()
    oos, choices = walk_forward({"x": df}, "m", "exit_date", [split], min_n=10_000)
    assert choices.loc[0, "chosen"] is None and choices.loc[0, "test_n"] == 0
    assert np.isnan(choices.loc[0, "fit_score"]) and choices.loc[0, "fit_n"] > 0
    assert oos.empty and {"config", "split_test_start", "m"} <= set(oos.columns)


def test_splits_have_equal_fit_window_lengths():
    splits = make_splits(T("2000-01-31"), T("2003-03-01"), fit_years=2, test_months=6)
    assert len(splits) > 2
    # Equal in months, not raw days: a month-end anchor shifts Feb 29/28 inside the window.
    for a, b in zip(splits, splits[1:]):
        assert a.fit_end == a.fit_start + pd.DateOffset(months=24)
        assert b.fit_end == b.fit_start + pd.DateOffset(months=24)


def test_test_window_end_date_belongs_to_next_split():
    splits = make_splits(T("2000-01-01"), T("2003-03-01"), fit_years=2, test_months=6)
    boundary = splits[0].test_end
    assert boundary == splits[1].test_start
    dates = pd.bdate_range("2000-01-03", "2001-06-30")
    df = pd.DataFrame({"date": dates, "exit_date": dates + pd.Timedelta(days=1),
                       "m": np.random.default_rng(0).normal(0.01, 0.02, len(dates))})
    edge = pd.DataFrame({"date": [boundary], "exit_date": [boundary + pd.Timedelta(days=1)], "m": [0.02]})
    oos, choices = walk_forward({"x": pd.concat([df, edge], ignore_index=True)}, "m", "exit_date", splits,
                                min_n=5)
    this = oos[oos["split_test_start"] == splits[0].test_start]
    nxt = oos[oos["split_test_start"] == splits[1].test_start]
    assert choices.loc[0, "test_n"] == 0 and choices.loc[1, "test_n"] == 1
    assert not (this["date"] == boundary).any()
    assert (nxt["date"] == boundary).sum() == 1


def test_event_realised_before_later_test_start_enters_that_fit_window():
    s0 = Split(T("2000-01-01"), T("2002-01-01"), T("2002-01-01"), T("2002-07-01"))
    s1 = Split(T("2000-07-01"), T("2002-07-01"), T("2002-07-01"), T("2003-01-01"))
    dates = pd.bdate_range("2000-07-03", "2000-12-29")
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"date": dates, "exit_date": dates + pd.Timedelta(days=1),
                         "m": rng.normal(0.003, 0.01, len(dates))})
    # Both sit in s0's test window; only the one realised before s1's test start may inform s1's fit.
    early = pd.DataFrame({"date": [T("2002-02-15")], "exit_date": [T("2002-03-01")], "m": [0.5]})
    late = pd.DataFrame({"date": [T("2002-02-20")], "exit_date": [T("2002-08-01")], "m": [0.5]})
    ev = pd.concat([base, early, late], ignore_index=True)
    _, choices = walk_forward({"x": ev}, "m", "exit_date", [s0, s1], min_n=10)
    assert choices.loc[0, "fit_n"] == len(base)
    assert choices.loc[1, "fit_n"] == len(base) + 1


def test_rejects_holdout_rows():
    h = holdout.holdout_start()
    split = Split(h - pd.DateOffset(years=2), h, h, h + pd.DateOffset(months=6))
    dates = pd.bdate_range(h - pd.DateOffset(years=2), h + pd.DateOffset(months=5))[::3]
    df = pd.DataFrame({"date": dates, "exit_date": dates + pd.Timedelta(days=1),
                       "m": np.random.default_rng(0).normal(0.01, 0.02, len(dates))})
    with pytest.raises(holdout.HoldoutViolation):
        walk_forward({"x": df}, "m", "exit_date", [split])

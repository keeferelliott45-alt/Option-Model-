import time

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from omodel import params
from omodel.backtest import stats as st


def _clustered(rng, n_clusters=60, per=20, mu=0.5, noise=0.3):
    cl = np.repeat(np.arange(n_clusters), per)
    return mu + rng.normal(size=n_clusters)[cl] + noise * rng.normal(size=n_clusters * per), cl


def test_cluster_freq_registered_as_convention():
    p = params.REGISTRY["CLUSTER_FREQ"]
    assert p.value == "M" and p.source == "convention"


def test_cluster_labels_monthly_and_index_preserved():
    dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", None]), index=[10, 11, 12, 13])
    lab = st.cluster_labels(dates)
    assert list(lab.index) == [10, 11, 12, 13]
    assert lab.iloc[0] == lab.iloc[1] != lab.iloc[2]
    assert pd.isna(lab.iloc[3])


def test_cluster_bootstrap_covers_truth_and_is_wider_than_iid():
    rng = np.random.default_rng(1)
    sims, truth = 300, 0.5
    cover_c = cover_n = 0
    ratios = []
    for s in range(sims):
        v, cl = _clustered(rng, mu=truth)
        r = st.cluster_bootstrap_mean(v, cl, reps=500, seed=s)
        se_naive = v.std(ddof=1) / np.sqrt(len(v))
        cover_c += r["lo"] <= truth <= r["hi"]
        cover_n += abs(v.mean() - truth) <= 1.96 * se_naive
        ratios.append((r["hi"] - r["lo"]) / (2 * 1.96 * se_naive))
    assert 0.88 <= cover_c / sims <= 0.99
    assert cover_n / sims < 0.6
    assert np.mean(ratios) > 2.5


def test_cluster_bootstrap_singletons_match_iid_se():
    v = np.random.default_rng(2).normal(size=2000)
    r = st.cluster_bootstrap_mean(v, np.arange(len(v)))
    assert r["se"] == pytest.approx(v.std(ddof=1) / np.sqrt(len(v)), rel=0.1)
    assert r["n"] == 2000 and r["n_clusters"] == 2000


def test_cluster_bootstrap_seed_determinism_and_nan_handling():
    v, cl = _clustered(np.random.default_rng(3))
    v[:5] = np.nan
    a = st.cluster_bootstrap_mean(v, cl, seed=7)
    b = st.cluster_bootstrap_mean(v, cl, seed=7)
    c = st.cluster_bootstrap_mean(v, cl, seed=8)
    assert a == b
    assert a["lo"] != c["lo"]
    assert a["n"] == len(v) - 5
    assert a["mean"] == pytest.approx(np.nanmean(v))
    assert a["lo"] < a["mean"] < a["hi"]


def test_cluster_bootstrap_degenerate_inputs():
    one = st.cluster_bootstrap_mean(pd.Series([1.0, np.nan]), pd.Series([0, 0]))
    assert one["n"] == 1 and one["mean"] == 1.0 and np.isnan(one["lo"]) and np.isnan(one["p_value"])
    single = st.cluster_bootstrap_mean(pd.Series([1.0, 2.0, 3.0]), pd.Series(["a", "a", "a"]))
    assert single["n_clusters"] == 1 and np.isnan(single["se"])
    empty = st.cluster_bootstrap_mean(pd.Series([], dtype=float), pd.Series([], dtype=int))
    assert empty["n"] == 0 and np.isnan(empty["mean"])


def test_cluster_bootstrap_vectorized_speed():
    rng = np.random.default_rng(4)
    v = rng.normal(size=100_000)
    cl = rng.integers(0, 320, size=v.size)
    t0 = time.perf_counter()
    st.cluster_bootstrap_mean(v, cl, reps=2000)
    assert time.perf_counter() - t0 < 2.0


def test_cluster_bootstrap_diff_sanity_and_joint_resampling():
    rng = np.random.default_rng(5)
    n_cl, per = 80, 10
    cl = np.repeat(np.arange(n_cl), per)
    shock = 2.0 * rng.normal(size=n_cl)[cl]  # shared by both groups inside a cluster
    groups = np.tile([0, 1], n_cl * per // 2)
    v = shock + 1.0 * groups + 0.2 * rng.normal(size=cl.size)
    r = st.cluster_bootstrap_diff(v, groups, cl, high=1, low=0)
    assert r["diff"] == pytest.approx(1.0, abs=0.1)
    assert r["lo"] < 1.0 < r["hi"] and r["lo"] > 0 and r["p_value"] < 1e-6
    assert r["n_high"] == r["n_low"] == cl.size // 2
    assert r["mean_high"] - r["mean_low"] == pytest.approx(r["diff"])
    se_h = st.cluster_bootstrap_mean(v[groups == 1], cl[groups == 1])["se"]
    se_l = st.cluster_bootstrap_mean(v[groups == 0], cl[groups == 0])["se"]
    # The common cluster shock cancels in the diff only if clusters are resampled jointly.
    assert r["se"] < 0.2 * np.hypot(se_h, se_l)
    flipped = st.cluster_bootstrap_diff(v, groups, cl, high=0, low=1)
    assert flipped["diff"] == pytest.approx(-r["diff"])
    assert flipped["lo"] == pytest.approx(-r["hi"])


def test_cluster_bootstrap_diff_ignores_other_groups_and_nans():
    v = pd.Series([1.0, 2.0, 3.0, np.nan, 100.0, 5.0, 6.0, 7.0])
    g = pd.Series([1, 1, 1, 1, 2, 0, 0, 0])
    cl = pd.Series([0, 1, 2, 0, 1, 0, 1, 2])
    r = st.cluster_bootstrap_diff(v, g, cl, high=1, low=0, reps=200)
    assert r["n_high"] == 3 and r["n_low"] == 3
    assert r["diff"] == pytest.approx(2.0 - 6.0)


def test_wilson_interval_textbook():
    p, lo, hi = st.wilson_interval(15, 23)
    ref_lo, ref_hi = proportion_confint(15, 23, alpha=0.05, method="wilson")
    assert p == pytest.approx(15 / 23)
    assert (lo, hi) == pytest.approx((ref_lo, ref_hi))
    assert (lo, hi) == pytest.approx((0.4489, 0.8119), abs=1e-4)
    assert all(np.isnan(x) for x in st.wilson_interval(0, 0))


def _ar1(phi, n, rng):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_newey_west_se_grows_with_autocorrelation():
    rng = np.random.default_rng(6)
    res = [st.newey_west(pd.Series(_ar1(phi, 3000, rng))) for phi in (0.0, 0.5, 0.9)]
    ses = [r["se"] for r in res]
    assert ses[0] < ses[1] < ses[2]
    assert res[0]["lags"] == params.PRIMARY_HORIZON
    x = _ar1(0.9, 3000, rng)
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert st.newey_west(x)["se"] > 2.5 * naive
    iid = rng.normal(size=3000)
    assert st.newey_west(iid)["se"] == pytest.approx(iid.std(ddof=1) / np.sqrt(3000), rel=0.15)


def test_newey_west_drops_nan_and_reports_t():
    s = pd.Series([0.1, np.nan, 0.3, 0.2, np.nan, 0.4, 0.1, 0.3])
    r = st.newey_west(s, lags=2)
    assert r["n"] == 6 and r["lags"] == 2
    assert r["mean"] == pytest.approx(s.mean())
    assert r["t"] == pytest.approx(r["mean"] / r["se"])
    assert 0 <= r["p_value"] <= 1
    assert np.isnan(st.newey_west(pd.Series([1.0]))["se"])


def test_calendar_time_portfolio_and_idle_fill():
    d = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-07", "2020-01-07"])
    pos = pd.DataFrame({"event_id": [1, 2, 1, 3, 4], "date": d, "abn": [0.01, 0.03, -0.02, np.nan, 0.05]})
    out = st.calendar_time_portfolio(pos)
    assert list(out.columns) == ["ret", "n_positions"]
    assert out.loc["2020-01-02", "ret"] == pytest.approx(0.02)
    assert out.loc["2020-01-02", "n_positions"] == 2
    assert out.loc["2020-01-07", "n_positions"] == 1
    all_dates = pd.bdate_range("2020-01-02", "2020-01-08")
    full = st.calendar_time_portfolio(pos, all_dates=all_dates)
    assert full.index.equals(pd.DatetimeIndex(all_dates, name="date"))
    assert full.loc["2020-01-06", "ret"] == 0.0 and full.loc["2020-01-06", "n_positions"] == 0
    assert full.loc["2020-01-08", "ret"] == 0.0
    assert full["n_positions"].dtype.kind == "i"


def test_calendar_time_portfolio_accepts_date_index():
    idx = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-03"])
    out = st.calendar_time_portfolio(pd.DataFrame({"abn": [0.0, 0.02, 0.04]}, index=idx))
    assert out["ret"].tolist() == pytest.approx([0.01, 0.04])


def test_annualized_sharpe():
    s = pd.Series([0.01, -0.005, 0.02, 0.0, 0.004])
    assert st.annualized_sharpe(s) == pytest.approx(s.mean() / s.std(ddof=1) * np.sqrt(252))
    assert st.annualized_sharpe(s, periods=12) == pytest.approx(s.mean() / s.std(ddof=1) * np.sqrt(12))
    assert np.isnan(st.annualized_sharpe(pd.Series([0.01, 0.01, 0.01])))
    assert np.isnan(st.annualized_sharpe(pd.Series([0.01])))


def test_bh_adjust_textbook_example():
    # Benjamini & Hochberg (1995) worked example: 15 p-values, q = 0.05 rejects the four smallest.
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459,
         0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.0000]
    names = [f"h{i}" for i in range(len(p))]
    rev = dict(zip(names[::-1], p[::-1]))  # input order must not matter
    df = st.bh_adjust(rev, q=0.05).set_index("name").loc[names]
    assert df["reject"].tolist() == [True] * 4 + [False] * 11
    assert np.allclose(df["p_adj"], multipletests(p, alpha=0.05, method="fdr_bh")[1])
    assert (np.diff(df["p_adj"].to_numpy()) >= 0).all() and df["p_adj"].max() <= 1.0


def test_bh_adjust_nan_passthrough_and_default_q():
    df = st.bh_adjust({"a": 0.01, "b": np.nan, "c": 0.04, "d": None})
    assert list(df.columns) == ["name", "p", "p_adj", "reject"]
    assert list(df["name"]) == ["a", "b", "c", "d"]
    row = df.set_index("name")
    assert np.isnan(row.loc["b", "p_adj"]) and not row.loc["b", "reject"] and not row.loc["d", "reject"]
    # m counts only the two valid p-values.
    assert row.loc["a", "p_adj"] == pytest.approx(0.02) and row.loc["c", "p_adj"] == pytest.approx(0.04)
    assert row.loc["a", "reject"] and row.loc["c", "reject"]  # 0.04 <= FDR_Q = 0.10


def test_summarize_fields_and_low_confidence():
    rng = np.random.default_rng(7)
    dates = pd.Series(pd.bdate_range("2010-01-04", periods=1000)[::5])
    vals = pd.Series(rng.normal(0.01, 0.05, size=200))
    vals.iloc[3] = np.nan
    s = st.summarize(vals, dates, label="all")
    keys = {"label", "n", "mean", "ci_lo", "ci_hi", "se", "p_value", "n_clusters", "n_eff_clusters", "median",
            "win_rate", "win_lo", "win_hi", "low_confidence"}
    assert keys <= set(s)
    assert s["label"] == "all" and s["n"] == 199
    assert s["n_clusters"] == st.cluster_labels(dates).nunique()
    assert s["mean"] == pytest.approx(vals.mean()) and s["median"] == pytest.approx(vals.median())
    assert s["win_rate"] == pytest.approx((vals.dropna() > 0).mean())
    assert s["win_lo"] < s["win_rate"] < s["win_hi"]
    assert s["ci_lo"] < s["mean"] < s["ci_hi"]
    assert not s["low_confidence"]
    small = st.summarize(vals.iloc[:10], dates.iloc[:10])
    assert small["low_confidence"] and small["n"] == 9
    assert st.summarize(pd.Series([], dtype=float), pd.Series([], dtype="datetime64[ns]"))["n"] == 0


def test_min_effective_clusters_registered():
    p = params.REGISTRY["MIN_EFFECTIVE_CLUSTERS"]
    assert p.value == st.MIN_EFFECTIVE_CLUSTERS == 30 and p.source == "convention"


def test_kish_effective_cluster_counts():
    r = st.cluster_bootstrap_mean(pd.Series([1.0, 2.0, 3.0, 4.0, np.nan]), pd.Series(["a", "b", "c", "c", "b"]))
    assert r["n_clusters"] == 3 and r["n_eff_clusters"] == pytest.approx(16 / 6)
    g = pd.Series([1, 1, 1, 0, 0, 0, 0])
    d = st.cluster_bootstrap_diff(pd.Series(np.arange(7.0)), g, pd.Series([0, 1, 1, 0, 1, 2, 3]), high=1, low=0)
    assert d["n_eff_high"] == pytest.approx(9 / 5) and d["n_eff_low"] == pytest.approx(4.0)


def test_summarize_few_crash_clusters_is_low_confidence():
    # Large n concentrated in two crash months: the bootstrap CI is meaningless, so the flag must fire.
    rng = np.random.default_rng(11)
    dates = pd.Series(pd.to_datetime(["2008-10-01"] * 250 + ["2020-03-02"] * 250))
    s = st.summarize(pd.Series(rng.normal(0.01, 0.05, 500)), dates)
    assert s["n"] == 500 and s["n_clusters"] == 2 and s["n_eff_clusters"] == pytest.approx(2.0)
    assert s["low_confidence"]


def test_summarize_low_confidence_boundaries(monkeypatch):
    months = pd.Series(pd.date_range("2005-01-03", periods=60, freq="MS"))
    vals = pd.Series(np.random.default_rng(12).normal(size=60))
    assert st.summarize(vals.iloc[:29], months.iloc[:29])["low_confidence"]
    at30 = st.summarize(vals.iloc[:30], months.iloc[:30])
    assert at30["n"] == 30 and at30["n_eff_clusters"] == pytest.approx(30.0) and not at30["low_confidence"]
    # 60 values in 20 months: enough events, too few effective clusters.
    few = st.summarize(vals, pd.Series(np.repeat(months.iloc[:20].to_numpy(), 3)))
    assert few["n"] == 60 and few["n_eff_clusters"] == pytest.approx(20.0) and few["low_confidence"]
    monkeypatch.setattr(st, "MIN_EFFECTIVE_CLUSTERS", 20)
    assert not st.summarize(vals, pd.Series(np.repeat(months.iloc[:20].to_numpy(), 3)))["low_confidence"]


def test_summarize_drops_nat_dated_rows_everywhere():
    s = st.summarize(pd.Series([1.0, 2.0, 3.0, 100.0]),
                     pd.Series(pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02", None])))
    assert s["n"] == 3 and s["mean"] == pytest.approx(2.0) and s["median"] == pytest.approx(2.0)
    assert s["win_rate"] == 1.0 and s["n_clusters"] == 3


def test_constant_series_are_degenerate_not_significant():
    const = pd.Series([0.1] * 50)
    assert np.isnan(st.annualized_sharpe(const))
    assert np.isnan(st.annualized_sharpe(pd.Series([0.0] * 10)))
    nw = st.newey_west(pd.Series([0.01] * 50))
    assert nw["se"] == 0.0 and np.isnan(nw["t"]) and np.isnan(nw["p_value"])
    boot = st.cluster_bootstrap_mean(const, np.arange(50) // 5)
    assert boot["se"] == 0.0 and np.isnan(boot["p_value"])


def _bartlett_se(y, lags):
    e = y - y.mean()
    n = len(y)
    s = e @ e / n
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    return np.sqrt(s / n)


def test_newey_west_matches_hand_bartlett():
    y = _ar1(0.6, 500, np.random.default_rng(13))
    assert st.newey_west(y, lags=5)["se"] == pytest.approx(_bartlett_se(y, 5), rel=1e-10)
    assert st.newey_west(y)["se"] == pytest.approx(_bartlett_se(y, params.PRIMARY_HORIZON), rel=1e-10)


def test_calendar_time_portfolio_rejects_dates_outside_calendar():
    pos = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-04"]), "abn": [0.01, 0.5]})
    with pytest.raises(ValueError, match="not in all_dates"):
        st.calendar_time_portfolio(pos, all_dates=pd.bdate_range("2020-01-02", "2020-01-08"))


def test_calendar_time_portfolio_normalizes_timestamps():
    pos = pd.DataFrame({"date": pd.to_datetime(["2020-01-02 16:00", "2020-01-03 16:00", "2020-01-03 09:30"]),
                        "abn": [0.01, 0.02, 0.04]})
    full = st.calendar_time_portfolio(pos, all_dates=pd.bdate_range("2020-01-02", "2020-01-06"))
    assert full["n_positions"].tolist() == [1, 2, 0]
    assert full["ret"].tolist() == pytest.approx([0.01, 0.03, 0.0])
    tz = pos.assign(date=pos["date"].dt.tz_localize("America/New_York"))
    pd.testing.assert_frame_equal(st.calendar_time_portfolio(tz), st.calendar_time_portfolio(pos))


def test_calendar_time_portfolio_holdout_guard(monkeypatch):
    from omodel import holdout
    pos = pd.DataFrame({"date": pd.to_datetime(["2025-10-22", "2025-10-23"]), "abn": [0.01, 0.02]})
    with pytest.raises(holdout.HoldoutViolation):
        st.calendar_time_portfolio(pos)
    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    assert st.calendar_time_portfolio(pos)["n_positions"].sum() == 2


def test_wilson_interval_rejects_impossible_counts():
    for k, n in [(5, 3), (-1, 3), (0, -1)]:
        with pytest.raises(ValueError):
            st.wilson_interval(k, n)
    assert st.wilson_interval(3, 3)[0] == 1.0

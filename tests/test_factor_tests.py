import time

import numpy as np
import pandas as pd
import pytest

from omodel import holdout, params
from omodel.backtest import factor_tests as ft
from omodel.backtest import stats as st


def _events(n=3000, n_days=1300, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2005-01-03", periods=n_days)
    date = np.sort(days[rng.integers(0, n_days, n)])
    factor = rng.normal(size=n)
    noise = rng.normal(size=n)
    flag = rng.random(n) < 0.4
    metric = 0.02 * factor + 0.015 * flag + 0.03 * rng.normal(size=n)
    return pd.DataFrame({"date": date, "factor": factor, "noise": noise, "flag": flag, "metric": metric},
                        index=pd.RangeIndex(100, 100 + n))


def _brute_force(v, d, n_buckets, min_obs):
    out = np.full(len(v), np.nan)
    probs = np.arange(1, n_buckets) / n_buckets
    for i in range(len(v)):
        prior = v[(d < d[i]) & ~np.isnan(v)]
        if np.isnan(v[i]) or len(prior) < min_obs:
            continue
        out[i] = np.searchsorted(np.quantile(prior, probs), v[i], side="left")
    return out


def test_expanding_bucket_hand_example():
    v = pd.Series([1.0, 2.0, 3.0, 1.5, 2.0, 3.0], index=list("abcdef"))
    d = pd.Series(pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3), index=v.index)
    b = ft.expanding_bucket(v, d, n_buckets=3, min_obs=3)
    assert list(b.index) == list("abcdef")
    # Prior values {1,2,3}: cuts at 1.667 and 2.333; first-day rows have no prior history.
    assert b.iloc[:3].isna().all()
    assert b.iloc[3:].tolist() == [0.0, 1.0, 2.0]
    assert ft.expanding_bucket(v, d, n_buckets=3, min_obs=1).iloc[:3].isna().all()
    pd.testing.assert_series_equal(ft.expanding_bucket(v, d.to_numpy(), n_buckets=3, min_obs=3), b)
    with pytest.raises(ValueError, match="index"):
        ft.expanding_bucket(v, d.reset_index(drop=True), n_buckets=3, min_obs=3)


def test_expanding_bucket_matches_brute_force_unsorted():
    rng = np.random.default_rng(1)
    n = 600
    d = pd.bdate_range("2010-01-04", periods=40)[rng.integers(0, 40, n)]
    v = rng.normal(size=n)
    v[rng.random(n) < 0.05] = np.nan
    got = ft.expanding_bucket(pd.Series(v), pd.Series(d), n_buckets=4, min_obs=20)
    np.testing.assert_array_equal(got.to_numpy(), _brute_force(v, d.to_numpy(), 4, 20))
    assert set(np.unique(got.dropna())) <= {0, 1, 2, 3}


def test_expanding_bucket_never_sees_same_day_or_future():
    ev = _events()
    base = ft.expanding_bucket(ev["factor"], ev["date"])
    cutoff = ev["date"].iloc[len(ev) // 2]
    future = ev["date"] > cutoff
    perturbed = ev["factor"].copy()
    perturbed[future] = np.random.default_rng(9).normal(5.0, 10.0, size=future.sum())
    extra = pd.Series([-50.0, 50.0], index=[1, 2])
    pv = pd.concat([perturbed, extra])
    pd_ = pd.concat([ev["date"], pd.Series([ev["date"].max()] * 2, index=[1, 2])])
    after = ft.expanding_bucket(pv, pd_)
    assert base[~future].notna().sum() > 1000
    pd.testing.assert_series_equal(after.loc[ev.index[~future]], base[~future], check_names=False)
    assert not after.loc[ev.index[future]].equals(base[future])
    counts = ev.loc[~future, "date"].value_counts()
    multi = np.sort(counts.index[counts.to_numpy() >= 2])
    day = multi[len(multi) // 2]
    same = ev.index[ev["date"] == day]
    assert len(same) >= 2 and base.loc[same].notna().all()
    poked = ev["factor"].copy()
    poked.loc[same[0]] = 1e6
    again = ft.expanding_bucket(poked, ev["date"])
    keep = (ev["date"] <= day) & (ev.index != same[0])
    pd.testing.assert_series_equal(again[keep], base[keep])


def test_expanding_bucket_scales_to_100k_rows():
    rng = np.random.default_rng(2)
    n = 100_000
    d = pd.Series(pd.bdate_range("2000-01-03", periods=7000)[rng.integers(0, 7000, n)])
    t0 = time.perf_counter()
    b = ft.expanding_bucket(pd.Series(rng.normal(size=n)), d)
    assert time.perf_counter() - t0 < 5.0
    assert b.notna().mean() > 0.99


def test_bucket_test_detects_monotone_factor():
    ev = _events()
    ev["b"] = ft.expanding_bucket(ev["factor"], ev["date"])
    res = ft.bucket_test(ev, "b", "metric")
    assert set(res) >= {"buckets", "diff", "diff_lo", "diff_hi", "p_value", "n"}
    assert len(res["buckets"]) == 3
    means = [b["mean"] for b in res["buckets"]]
    assert means[0] < means[1] < means[2]
    assert res["diff"] == pytest.approx(means[2] - means[0])
    assert res["diff_lo"] > 0 and res["p_value"] < 0.01
    assert res["n"] == ev["b"].notna().sum() == sum(b["n"] for b in res["buckets"])


def test_binary_test_sign_convention():
    ev = _events()
    res = ft.binary_test(ev, "flag", "metric")
    m_true = ev.loc[ev["flag"], "metric"].mean()
    m_false = ev.loc[~ev["flag"], "metric"].mean()
    assert res["diff"] == pytest.approx(m_true - m_false)
    assert res["diff"] > 0 and res["diff_lo"] > 0
    assert [b["mean"] for b in res["buckets"]] == pytest.approx([m_false, m_true])
    assert res["n"] == len(ev)
    inverted = ft.binary_test(ev.assign(flag=~ev["flag"]), "flag", "metric")
    assert inverted["diff"] == pytest.approx(-res["diff"])


def test_binary_test_handles_missing_flags():
    ev = _events(n=400)
    flags = ev["flag"].astype(object)
    flags.iloc[:50] = None
    res = ft.binary_test(ev.assign(flag=flags), "flag", "metric")
    assert res["n"] == 350


def test_factor_battery_columns_and_bh():
    ev = _events()
    specs = [{"name": "factor", "col": "factor", "kind": "bucket", "n_buckets": 3},
             {"name": "noise", "col": "noise", "kind": "bucket"},
             {"name": "flag", "col": "flag", "kind": "binary"}]
    out = ft.factor_battery(ev, specs, "metric")
    for col in ["name", "kind", "n", "diff", "diff_lo", "diff_hi", "p_value", "p_adj", "reject",
                "mean_0", "mean_1", "mean_2"]:
        assert col in out.columns
    assert list(out["name"]) == ["factor", "noise", "flag"]
    row = out.set_index("name")
    assert row.loc["factor", "reject"] and row.loc["flag", "reject"]
    assert not row.loc["noise", "reject"]
    assert (out["p_adj"] >= out["p_value"]).all()
    assert np.isnan(row.loc["flag", "mean_2"])
    assert row.loc["flag", "mean_1"] > row.loc["flag", "mean_0"]
    direct = ft.bucket_test(ev.assign(b=ft.expanding_bucket(ev["factor"], ev["date"])), "b", "metric")
    assert row.loc["factor", "diff"] == pytest.approx(direct["diff"])
    with pytest.raises(ValueError):
        ft.factor_battery(ev, specs + [specs[0]], "metric")


def _monthly_shock_events(seed=0, n_months=60, per=50):
    # Factor level and metric both carry a month-wide component, as in a sector/macro shock.
    rng = np.random.default_rng(seed)
    months = pd.date_range("2005-01-01", periods=n_months, freq="MS")
    mi = np.repeat(np.arange(n_months), per)
    n = len(mi)
    date = months[mi] + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    factor = rng.normal(size=n_months)[mi] + 0.5 * rng.normal(size=n)
    metric = rng.normal(0, 0.02, n_months)[mi] + 0.002 * factor + 0.01 * rng.normal(size=n)
    flag = factor > 0.5
    ev = pd.DataFrame({"date": date, "factor": factor, "flag": flag, "metric": metric})
    return ev.sort_values("date", kind="stable").reset_index(drop=True)


def test_expanding_bucket_default_min_obs_is_registry_value():
    rng = np.random.default_rng(3)
    d = pd.bdate_range("2010-01-04", periods=300)[rng.integers(0, 300, 900)]
    v = rng.normal(size=900)
    got = ft.expanding_bucket(pd.Series(v), pd.Series(d))
    np.testing.assert_array_equal(got.to_numpy(), _brute_force(v, d.to_numpy(), 3, params.EXPANDING_MIN_OBS))
    assert got.isna().sum() > ft.expanding_bucket(pd.Series(v), pd.Series(d), min_obs=50).isna().sum()


def test_expanding_bucket_aligns_dates_by_label_and_handles_tz():
    rng = np.random.default_rng(4)
    d = pd.Series(pd.bdate_range("2010-01-04", periods=80)[rng.integers(0, 80, 300)])
    v = pd.Series(rng.normal(size=300))
    base = ft.expanding_bucket(v, d, min_obs=20)
    pd.testing.assert_series_equal(ft.expanding_bucket(v, d.iloc[::-1], min_obs=20), base)
    pd.testing.assert_series_equal(ft.expanding_bucket(v, d.dt.tz_localize("America/New_York"), min_obs=20), base)
    intraday = d + pd.to_timedelta(rng.integers(0, 8, 300), unit="h")
    pd.testing.assert_series_equal(ft.expanding_bucket(v, intraday, min_obs=20), base)


def test_factor_battery_uses_registry_min_obs_unless_spec_overrides():
    ev = _events()
    default = ft.factor_battery(ev, [{"name": "f", "col": "factor", "kind": "bucket"}], "metric")
    direct = ft.bucket_test(ev.assign(b=ft.expanding_bucket(ev["factor"], ev["date"])), "b", "metric")
    assert default.loc[0, "n"] == direct["n"]
    loose = ft.factor_battery(ev, [{"name": "f", "col": "factor", "kind": "bucket", "min_obs": 50}], "metric")
    assert loose.loc[0, "n"] > default.loc[0, "n"]


def test_factor_tests_use_date_clusters():
    ev = _monthly_shock_events()
    ev["b"] = ft.expanding_bucket(ev["factor"], ev["date"])
    res = ft.bucket_test(ev, "b", "metric")
    ref = st.cluster_bootstrap_diff(ev["metric"], ev["b"], st.cluster_labels(ev["date"]), high=2, low=0)
    assert (res["diff_lo"], res["diff_hi"], res["p_value"]) == (ref["lo"], ref["hi"], ref["p_value"])
    iid = st.cluster_bootstrap_diff(ev["metric"], ev["b"], np.arange(len(ev)), high=2, low=0)
    assert (res["diff_hi"] - res["diff_lo"]) > 3 * (iid["hi"] - iid["lo"])
    flag = ft.binary_test(ev, "flag", "metric")
    ref_f = st.cluster_bootstrap_diff(ev["metric"], ev["flag"].astype(float), st.cluster_labels(ev["date"]),
                                      high=1.0, low=0.0)
    assert (flag["diff_lo"], flag["diff_hi"], flag["p_value"]) == (ref_f["lo"], ref_f["hi"], ref_f["p_value"])
    out = ft.factor_battery(ev, [{"name": "f", "col": "factor", "kind": "bucket"}], "metric")
    assert (out.loc[0, "diff_lo"], out.loc[0, "diff_hi"]) == (res["diff_lo"], res["diff_hi"])


def _crash_null_events(rng, n_months=120, crash_p=0.08):
    # A factor whose level rises in crash months, with no effect on the metric (the realistic null).
    months = pd.date_range("2005-01-01", periods=n_months, freq="MS")
    crash = rng.random(n_months) < crash_p
    k = rng.poisson(np.where(crash, 300, 25))
    mi = np.repeat(np.arange(n_months), k)
    n = len(mi)
    date = months[mi] + pd.to_timedelta(rng.integers(0, 28, n), unit="D")
    metric = rng.normal(0, np.where(crash, 0.06, 0.02))[mi] + rng.normal(0, 0.05, n)
    factor = rng.normal(np.where(crash, 2.0, 0.0), 0.5)[mi] + rng.normal(size=n)
    flag = rng.random(n) < np.where(crash, 0.7, 0.3)[mi]
    ev = pd.DataFrame({"date": date, "metric": metric, "factor": factor, "flag": flag})
    return ev.sort_values("date", kind="stable").reset_index(drop=True)


def test_crash_clustered_null_is_flagged_and_never_rejected():
    rng = np.random.default_rng(21)
    specs = [{"name": "factor", "col": "factor", "kind": "bucket"}, {"name": "flag", "col": "flag", "kind": "binary"}]
    sims = 12
    rejects = low = 0
    n_eff = []
    for _ in range(sims):
        out = ft.factor_battery(_crash_null_events(rng), specs, "metric")
        rejects += int(out["reject"].sum())
        low += int(out["low_confidence"].sum())
        n_eff.extend(out["n_eff_clusters"])
    assert rejects == 0
    assert low == 2 * sims
    assert np.median(n_eff) < st.MIN_EFFECTIVE_CLUSTERS < 120


def test_low_confidence_row_is_never_a_discovery():
    ev = _monthly_shock_events(n_months=12, per=400)
    ev["metric"] += 0.05 * ev["flag"]
    out = ft.factor_battery(ev, [{"name": "flag", "col": "flag", "kind": "binary"}], "metric").iloc[0]
    assert out["p_adj"] < params.FDR_Q and out["low_confidence"] and not out["reject"]
    assert out["n_eff_clusters"] <= 12


def test_bucket_and_binary_results_carry_confidence():
    ev = _events()
    res = ft.binary_test(ev, "flag", "metric")
    assert {"n_eff_clusters", "low_confidence"} <= set(res)
    assert not res["low_confidence"] and res["n_eff_clusters"] >= st.MIN_EFFECTIVE_CLUSTERS
    assert all("n_eff_clusters" in b for b in res["buckets"])


def test_binary_test_flag_dtypes():
    ev = _events(n=600)
    as_int = ft.binary_test(ev.assign(flag=ev["flag"].astype(int)), "flag", "metric")
    assert as_int["diff"] == pytest.approx(ft.binary_test(ev, "flag", "metric")["diff"])
    with pytest.raises(ValueError):
        ft.binary_test(ev.assign(flag=ev["flag"].map({True: "True", False: "False"})), "flag", "metric")
    with pytest.raises(ValueError):
        ft.binary_test(ev.assign(flag=ev["flag"].astype(int) * 2), "flag", "metric")


def test_factor_tests_refuse_holdout_dates(monkeypatch):
    ev = _events(n=400)
    late = ev.assign(date=ev["date"] + pd.DateOffset(years=20))
    assert late["date"].max() >= holdout.holdout_start()
    specs = [{"name": "flag", "col": "flag", "kind": "binary"}]
    with pytest.raises(holdout.HoldoutViolation):
        ft.factor_battery(late, specs, "metric")
    with pytest.raises(holdout.HoldoutViolation):
        ft.binary_test(late, "flag", "metric")
    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    assert ft.factor_battery(late, specs, "metric").loc[0, "n"] == 400

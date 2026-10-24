import numpy as np
import pandas as pd
import pytest

from omodel.backtest.synthetic import synthetic_panel
from omodel.experiments import h2bc
from omodel.layers import l1_regime as l1
from omodel.layers import l2_decomposition as l2
from omodel.layers import l3_fundamentals as l3
from omodel.layers import l4_distribution as l4
from omodel.layers import l5_options as l5


def test_avg_corr_known_value():
    rng = np.random.default_rng(0)
    common = rng.normal(size=(400, 1))
    ret = pd.DataFrame(np.sqrt(0.5) * common + np.sqrt(0.5) * rng.normal(size=(400, 30)))
    rho = l1.avg_pairwise_corr(ret, pd.DataFrame(True, index=ret.index, columns=ret.columns), window=300)
    assert rho.iloc[-1] == pytest.approx(0.5, abs=0.08)


def test_regime_features_point_in_time():
    panel, _ = synthetic_panel(n_days=600, n_stocks=30, seed=1)
    idx = pd.bdate_range("2004-01-01", "2008-01-01")
    rng = np.random.default_rng(2)
    macro = pd.DataFrame({c: 20 + rng.normal(size=len(idx)).cumsum() * 0.1 + 50
                          for c in ["vix", "vix3m", "y2", "y10", "hyg", "lqd", "dxy", "gold", "oil", "usdjpy"]},
                         index=idx)
    ret = l2.log_returns(panel.close)
    f1 = l1.regime_features(macro, panel, ret)
    cut = panel.dates[400]
    m2 = macro.copy()
    m2.loc[m2.index > cut] *= 3
    f2 = l1.regime_features(m2, panel, ret)
    pd.testing.assert_frame_equal(f1.loc[:cut], f2.loc[:cut])
    assert "calm" in l1.regime_label(pd.Series({"vix": 15.0}))


def _facts(filed_shift_days=30):
    def q(concept, vals, start="2023-01-01"):
        ends = pd.date_range(start, periods=len(vals), freq="QE")
        return {concept: {"units": {"USD": [
            {"start": str((e - pd.Timedelta(days=90)).date()), "end": str(e.date()), "val": v,
             "filed": str((e + pd.Timedelta(days=filed_shift_days)).date())} for e, v in zip(ends, vals)]}}}
    us = {**q("Revenues", [100] * 4 + [90] * 4), **q("GrossProfit", [50] * 4 + [40] * 4),
          **q("OperatingIncomeLoss", [20] * 8), **q("InterestExpense", [1] * 8)}
    us["LongTermDebt"] = {"units": {"USD": [{"end": "2024-12-31", "val": 100, "filed": "2025-01-30"}]}}
    return {"facts": {"us-gaap": us}}


def test_fundamentals_point_in_time_and_soft_veto():
    fl = l3.facts_long(_facts())
    m_early = l3.metrics(fl, pd.Timestamp("2024-01-15"))  # Q4-2023 not filed yet (filed 2024-01-30)
    m_late = l3.metrics(fl, pd.Timestamp("2025-02-15"))
    assert np.isnan(m_early["revenue"])  # only 3 quarters filed by then: no TTM, no lookahead to Q4
    assert m_late["revenue"] == pytest.approx(360)
    assert m_late["gm"] == pytest.approx(40 / 90)
    v = l3.veto(fl, pd.Timestamp("2025-02-15"), pd.Timestamp("2025-03-20"), pd.DatetimeIndex([]), bullish=True)
    assert v.status == "SOFT" and any("margin" in r for r in v.reasons)


def test_earnings_inside_expiry_is_hard_veto():
    earn = pd.DatetimeIndex(["2025-01-28"])
    v = l3.veto(None, pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-01"), earn, bullish=False)
    assert v.status == "HARD"
    v2 = l3.veto(None, pd.Timestamp("2025-02-10"), pd.Timestamp("2025-03-14"), earn, bullish=False)
    assert v2.status == "UNKNOWN"


def _history(n=400, drift=-0.01):
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2015-01-01", periods=n)
    sig = 0.02
    return pd.DataFrame({"date": dates, "ticker": rng.choice(["A", "B"], n), "tot_sigma": sig,
                         "raw_15": drift + sig * np.sqrt(15) * rng.normal(size=n),
                         "exit_date_15": dates + pd.Timedelta(days=21), "high_vix": rng.random(n) < 0.3})


def test_l4_uses_only_past_and_recovers_drift():
    hist = _history()
    asof = pd.Timestamp("2017-01-01")
    d = l4.conditional(hist, asof, "A", 100.0, 0.02, 15)
    assert d is not None and d.probs.sum() == pytest.approx(1.0)
    assert d.stats["mean_ret"] == pytest.approx(-0.01, abs=0.01)
    future = hist.copy()
    future.loc[future["date"] > pd.Timestamp("2016-06-01"), "raw_15"] += 1.0
    d_future = l4.conditional(future, pd.Timestamp("2016-06-01"), "A", 100.0, 0.02, 15)
    d_base = l4.conditional(hist, pd.Timestamp("2016-06-01"), "A", 100.0, 0.02, 15)
    assert np.allclose(d_future.probs, d_base.probs) and np.allclose(d_future.grid, d_base.grid)
    assert l4.conditional(hist.head(50), asof, "A", 100.0, 0.02, 15) is None


def test_soft_status_widens_not_shifts():
    hist = _history()
    asof = pd.Timestamp("2017-01-01")
    p = l4.conditional(hist, asof, "A", 100.0, 0.02, 15, fund_status="PASS")
    s = l4.conditional(hist, asof, "A", 100.0, 0.02, 15, fund_status="SOFT")
    assert s.stats["mean_ret"] == pytest.approx(p.stats["mean_ret"], abs=1e-9)
    assert (s.stats["p95"] - s.stats["p5"]) > (p.stats["p95"] - p.stats["p5"])


def test_h2bc_settlement_and_refusal():
    legs = (l5.Leg("C", 105.0, -1, 2.0, 2.1, 0.3), l5.Leg("C", 110.0, +1, 0.8, 0.9, 0.15))
    st = l5.Structure("bear_call_credit", pd.Timestamp("2027-01-15"), legs)
    credit = (2.0 - 0.9) * 100 - 4 * l5.COMMISSION
    pnl_below, ml = h2bc.settle(st, 100.0)
    assert pnl_below == pytest.approx(credit)
    pnl_above, _ = h2bc.settle(st, 120.0)
    assert pnl_above == pytest.approx(-ml)
    with pytest.raises(RuntimeError, match="refusing to look early"):
        h2bc.run(today=pd.Timestamp("2027-01-01"))


def test_realized_vol_needs_full_path():
    idx = pd.bdate_range("2026-11-02", periods=30)
    close = pd.Series(100 * np.exp(np.cumsum(np.full(30, 0.01) * np.where(np.arange(30) % 2, 1, -1))), index=idx)
    assert np.isfinite(h2bc.realized_vol(close, idx[0], idx[20]))
    assert np.isnan(h2bc.realized_vol(close, idx[0], idx[-1] + pd.Timedelta(days=30)))

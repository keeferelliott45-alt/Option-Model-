import numpy as np
import pandas as pd
import pytest

from omodel.layers import l5_options as l5

S, T, R, VOL = 100.0, 30 / 365, 0.04, 0.30


def test_put_call_parity_and_iv_roundtrip():
    K = np.array([80.0, 100.0, 120.0])
    c, p = l5.bs_price(S, K, T, VOL, "C", R), l5.bs_price(S, K, T, VOL, "P", R)
    assert np.allclose(c - p, S - K * np.exp(-R * T), atol=1e-9)
    assert np.allclose(l5.implied_vol(c, S, K, T, "C", R), VOL, atol=1e-6)
    assert np.isnan(l5.implied_vol(S + 1, S, 100.0, T, "C", R))  # above the no-arbitrage bound


def test_delta_signs():
    assert 0 < l5.bs_delta(S, 100.0, T, VOL, "C", R) < 1
    assert -1 < l5.bs_delta(S, 100.0, T, VOL, "P", R) < 0


def test_breeden_litzenberger_recovers_flat_smile_lognormal():
    K = np.linspace(70, 130, 13)
    grid, mass = l5.risk_neutral_density(K, np.full(len(K), VOL), S, T, R)
    assert mass.sum() == pytest.approx(1.0)
    assert grid @ mass == pytest.approx(S * np.exp(R * T), rel=2e-3)
    sd = np.sqrt(mass @ np.log(grid) ** 2 - (mass @ np.log(grid)) ** 2)
    assert sd == pytest.approx(VOL * np.sqrt(T), rel=0.03)


def test_skew_shifts_mass_left():
    K = np.linspace(70, 130, 13)
    flat = l5.risk_neutral_density(K, np.full(13, VOL), S, T, R)
    skew = l5.risk_neutral_density(K, VOL - 0.4 * np.log(K / S), S, T, R)
    assert skew[1][skew[0] < 85].sum() > flat[1][flat[0] < 85].sum()


def _quote(cp, k, half_spread=0.0):
    mid = float(l5.bs_price(S, k, T, VOL, cp, R))
    return dict(cp=cp, strike=k, bid=mid - half_spread, ask=mid + half_spread,
                delta=float(l5.bs_delta(S, k, T, VOL, cp, R)))


def test_bear_call_credit_payoff_bounds_and_fair_ev(monkeypatch):
    monkeypatch.setattr(l5, "COMMISSION", 0.0)
    lo, hi = _quote("C", 105.0), _quote("C", 110.0)
    st = l5.Structure("bear_call_credit", pd.Timestamp("2026-12-18"),
                      (l5.Leg(qty=-1, **lo), l5.Leg(qty=+1, **hi)))
    assert st.entry_cost() < 0  # credit
    grid, probs = l5.lognormal_grid(S, T, VOL, mu=R)
    res = l5.evaluate(st, grid, probs)
    assert res["max_loss"] == pytest.approx(500 + st.entry_cost(), rel=1e-6)
    # Priced at fair value with no spread, EV under the pricing measure is ~0 (up to discounting).
    assert abs(res["ev"]) < 0.01 * res["max_loss"]


def test_crossing_the_spread_costs_ev(monkeypatch):
    monkeypatch.setattr(l5, "COMMISSION", 0.0)
    grid, probs = l5.lognormal_grid(S, T, VOL, mu=R)

    def ev(h):
        legs = (l5.Leg(qty=-1, **_quote("C", 105.0, h)), l5.Leg(qty=+1, **_quote("C", 110.0, h)))
        return l5.evaluate(l5.Structure("x", pd.Timestamp("2026-12-18"), legs), grid, probs)["ev"]

    assert ev(0.05) == pytest.approx(ev(0.0) - 2 * 0.05 * 100, abs=1e-6)


def test_vol_premium_makes_short_spread_positive(monkeypatch):
    monkeypatch.setattr(l5, "COMMISSION", 0.0)
    legs = (l5.Leg(qty=-1, **_quote("C", 105.0)), l5.Leg(qty=+1, **_quote("C", 110.0)))
    st = l5.Structure("bear_call_credit", pd.Timestamp("2026-12-18"), legs)
    grid, probs = l5.lognormal_grid(S, T, 0.20, mu=R)  # realized vol below the 30% implied
    assert l5.evaluate(st, grid, probs)["ev"] > 0


def test_delta_buckets():
    assert l5.delta_bucket(0.10) == "0.05-0.20"
    assert l5.delta_bucket(-0.30) == "0.20-0.40"
    assert l5.delta_bucket(0.85) == "0.80+"
    assert l5.delta_bucket(0.01) == "<0.05"


def _chain():
    rows = []
    for cp in ("C", "P"):
        for k in np.arange(90.0, 115.0, 5.0):
            q = _quote(cp, k, 0.02)
            rows.append(dict(expiry=pd.Timestamp("2026-12-18"), open_interest=500, volume=50, **q))
    return pd.DataFrame(rows)


def test_vertical_spreads_are_defined_risk_and_liquid():
    chain = _chain()
    chain.loc[0, "open_interest"] = 0  # illiquid leg must be excluded
    spreads = l5.vertical_spreads(chain, pd.Timestamp("2026-12-18"))
    assert spreads and all(len(s.legs) == 2 and sum(l.qty for l in s.legs) == 0 for s in spreads)
    assert all(l.strike != 90.0 or l.cp != "C" for s in spreads for l in s.legs)
    grid, probs = l5.lognormal_grid(S, T, VOL)
    assert all(np.isfinite(l5.evaluate(s, grid, probs)["max_loss"]) for s in spreads)


def test_scan_expiry_matches_reference_evaluator():
    chain = _chain()
    grid, probs = l5.lognormal_grid(S, T, 0.25)
    fast = l5.scan_expiry(chain, grid, probs).set_index(["name", "legs"])
    for st in l5.vertical_spreads(chain, pd.Timestamp("2026-12-18")):
        ref = l5.evaluate(st, grid, probs)
        legs = "/".join(f"{'+' if l.qty > 0 else '-'}{l.cp}{l.strike:g}" for l in st.legs)
        if (st.name, legs) not in fast.index:
            legs = "/".join(reversed(legs.split("/")))
        row = fast.loc[(st.name, legs)]
        assert row["ev"] == pytest.approx(ref["ev"], abs=0.5)
        assert row["max_loss"] == pytest.approx(ref["max_loss"], abs=0.5)
    assert l5.scan_expiry(chain.iloc[:0], grid, probs).empty


def test_rank_applies_far_otm_hurdle_and_excludes_deep_itm():
    cands = [dict(name="a", ev_pct_risk=0.10, delta=0.10),   # far OTM, below 0.15 hurdle
             dict(name="b", ev_pct_risk=0.10, delta=0.30),
             dict(name="c", ev_pct_risk=0.50, delta=0.85),   # 0.80+ excluded
             dict(name="d", ev_pct_risk=0.20, delta=0.12)]
    assert list(l5.rank(cands)["name"]) == ["d", "b"]

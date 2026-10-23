import numpy as np
import pandas as pd
import pytest

from omodel.backtest.placebo import EVENT_COLUMNS, placebo_events
from omodel.types import Decomposition, Panel

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _world(n_days=400, seed=0, raw=True):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n_days)

    def frame(scale=1.0, loc=0.0):
        return pd.DataFrame(rng.normal(loc, scale, (n_days, len(TICKERS))), index=idx, columns=TICKERS)

    z, sigma, beta_mkt = frame(), frame(0.002, 0.02).abs(), frame(0.2, 1.0)
    for f in (z, sigma, beta_mkt):
        f.iloc[:30] = np.nan
    z.iloc[100:110, 0] = np.nan
    member = pd.DataFrame(True, index=idx, columns=TICKERS)
    member.iloc[200:260, 1] = False
    member.iloc[:, 3] = False
    close = frame(1.0, 50.0)
    panel = Panel(close=close, open=close.copy(), volume=frame(10, 1e6), member=member,
                  etf_close=pd.DataFrame(index=idx), etf_open=pd.DataFrame(index=idx),
                  raw_close=close * 2 if raw else None)
    dec = Decomposition(stock_ret=frame(0.02), mkt_ret=pd.Series(rng.normal(0, 0.01, n_days), index=idx),
                        sector=pd.DataFrame("XLK", index=idx, columns=TICKERS), sec_ret=frame(0.01),
                        alpha=frame(0.001), beta_mkt=beta_mkt, beta_sec=frame(0.2, 0.5), resid=frame(0.02),
                        sigma=sigma, z=z, tot_sigma=frame(0.002, 0.03).abs(), vol_z=frame())
    return panel, dec


def _events(panel, dec, picks):
    rows = []
    for d, t in picks:
        d = pd.Timestamp(d)
        price = panel.raw_close if panel.raw_close is not None else panel.close
        rows.append({"date": d, "ticker": t, "pos": panel.dates.get_loc(d), "z": dec.z.at[d, t],
                     "resid": dec.resid.at[d, t], "ret": dec.stock_ret.at[d, t], "mkt_ret": dec.mkt_ret.at[d],
                     "sec_ret": dec.sec_ret.at[d, t], "sector": dec.sector.at[d, t], "vol_z": dec.vol_z.at[d, t],
                     "alpha": dec.alpha.at[d, t], "beta_mkt": dec.beta_mkt.at[d, t],
                     "beta_sec": dec.beta_sec.at[d, t], "sigma": dec.sigma.at[d, t],
                     "tot_sigma": dec.tot_sigma.at[d, t], "price": price.at[d, t]})
    ev = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    ev["pos"] = ev["pos"].astype(np.int64)
    return ev


def _picks(panel):
    d = panel.dates
    return [(d[60], "AAA"), (d[150], "AAA"), (d[300], "AAA"), (d[90], "BBB"), (d[330], "BBB"),
            (d[120], "CCC"), (d[250], "CCC"), (d[380], "DDD")]


def test_schema_matches_events_and_values_pulled_from_inputs():
    panel, dec = _world()
    ev = _events(panel, dec, _picks(panel))
    pl = placebo_events(ev, panel, dec)
    assert list(pl.columns) == list(ev.columns) + ["placebo_of"]
    pd.testing.assert_series_equal(pl.dtypes.drop("placebo_of"), ev.dtypes)
    assert len(pl) == len(ev) - 1  # DDD is never a member, so it has no placebo days
    assert (ev.loc[pl["placebo_of"], "ticker"].to_numpy() == pl["ticker"].to_numpy()).all()
    rebuilt = _events(panel, dec, list(zip(pl["date"], pl["ticker"])))
    pd.testing.assert_frame_equal(pl.drop(columns="placebo_of"), rebuilt)


def test_price_falls_back_to_close():
    panel, dec = _world(raw=False)
    ev = _events(panel, dec, _picks(panel))
    pl = placebo_events(ev, panel, dec)
    expected = [panel.close.at[d, t] for d, t in zip(pl["date"], pl["ticker"])]
    np.testing.assert_allclose(pl["price"].to_numpy(), expected)


@pytest.mark.parametrize("window", [0, 15, 40])
def test_exclusion_window_and_date_range(window):
    panel, dec = _world()
    ev = _events(panel, dec, _picks(panel))
    pl = placebo_events(ev, panel, dec, n_per_event=4, exclude_window=window)
    assert len(pl) == 4 * (len(ev) - 1)
    for t, grp in pl.groupby("ticker"):
        real = ev.loc[ev["ticker"] == t, "pos"].to_numpy()
        dist = np.abs(grp["pos"].to_numpy()[:, None] - real[None, :]).min(axis=1)
        assert (dist > window).all()
    assert pl["date"].min() >= ev["date"].min() and pl["date"].max() <= ev["date"].max()
    assert not pl.duplicated(["ticker", "date"]).any()


def test_drawn_dates_are_member_days_with_finite_inputs():
    panel, dec = _world()
    ev = _events(panel, dec, _picks(panel))
    pl = placebo_events(ev, panel, dec, n_per_event=10)
    d, t = pl["date"], pl["ticker"]
    assert all(panel.member.at[a, b] for a, b in zip(d, t))
    for col in ("z", "sigma", "beta_mkt", "beta_sec"):
        assert np.isfinite(pl[col]).all()
    assert pl["placebo_of"].value_counts().eq(10).all()


def test_deterministic_and_order_invariant():
    panel, dec = _world()
    ev = _events(panel, dec, _picks(panel))
    a = placebo_events(ev, panel, dec, n_per_event=3, seed=5)
    pd.testing.assert_frame_equal(a, placebo_events(ev, panel, dec, n_per_event=3, seed=5))
    shuffled = ev.sample(frac=1.0, random_state=1)
    pd.testing.assert_frame_equal(a, placebo_events(shuffled, panel, dec, n_per_event=3, seed=5))
    assert not a.equals(placebo_events(ev, panel, dec, n_per_event=3, seed=6))


def test_reuses_dates_only_when_candidates_run_out():
    panel, dec = _world()
    d = panel.dates
    panel.member.loc[:, "CCC"] = False
    panel.member.loc[d[[40, 45, 200]], "CCC"] = True
    ev = _events(panel, dec, [(d[30], "AAA"), (d[100], "CCC"), (d[150], "CCC"), (d[390], "AAA")])
    pl = placebo_events(ev, panel, dec, n_per_event=2, exclude_window=5)
    ccc = pl[pl["ticker"] == "CCC"]
    assert len(ccc) == 4 and set(ccc["date"]) == set(d[[40, 45, 200]])


def test_empty_events():
    panel, dec = _world()
    ev = _events(panel, dec, [])
    pl = placebo_events(ev, panel, dec)
    assert pl.empty and list(pl.columns) == EVENT_COLUMNS + ["placebo_of"]


def test_rejects_event_dates_outside_panel():
    panel, dec = _world()
    ev = _events(panel, dec, _picks(panel))
    ev.loc[0, "date"] = pd.Timestamp("1990-01-01")
    with pytest.raises(ValueError):
        placebo_events(ev, panel, dec)

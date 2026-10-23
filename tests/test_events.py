import math

import numpy as np
import pandas as pd
import pytest

from omodel import params
from omodel.backtest import events as ev
from omodel.config import settings
from omodel.types import Decomposition, Panel

# Column order deliberately not alphabetical: outputs must sort by ticker name, not column position.
TICKERS = ["BBB", "AAA", "CCC"]
ETFS = ["SPY", "XLK", "XLF"]
BETA_MKT = {"BBB": 1.2, "AAA": 0.8, "CCC": 1.5}
BETA_SEC = {"BBB": 0.5, "AAA": 0.3, "CCC": -0.2}
SECTOR = {"BBB": "XLK", "AAA": "XLK", "CCC": "XLF"}
COST = 5.0


def make_panel(start="2020-01-02", n=40, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)

    def walk(cols):
        lr = rng.normal(0.0, 0.02, size=(n, len(cols)))
        return pd.DataFrame(100.0 * np.exp(np.cumsum(lr, axis=0)), index=dates, columns=cols)

    def opens(close):
        gap = np.exp(rng.normal(0.0, 0.01, size=close.shape))
        out = close.shift(1) * gap
        out.iloc[0] = close.iloc[0]
        return out

    close, etf_close = walk(TICKERS), walk(ETFS)
    return Panel(close=close, open=opens(close), volume=pd.DataFrame(1e6, index=dates, columns=TICKERS),
                 member=pd.DataFrame(True, index=dates, columns=TICKERS), etf_close=etf_close,
                 etf_open=opens(etf_close), raw_close=close * 1.5)


def make_dec(panel: Panel, z_points: dict | None = None) -> Decomposition:
    dates, cols = panel.dates, panel.close.columns

    def per_ticker(values):
        return pd.DataFrame([[values[t] for t in cols]] * len(dates), index=dates, columns=cols, dtype=float)

    z = pd.DataFrame(0.0, index=dates, columns=cols)
    for (pos, ticker), value in (z_points or {}).items():
        z.iloc[pos, cols.get_loc(ticker)] = value
    sigma = pd.DataFrame(0.02, index=dates, columns=cols)
    sector = pd.DataFrame([[SECTOR[t] for t in cols]] * len(dates), index=dates, columns=cols, dtype=object)
    etf_ret = np.log(panel.etf_close).diff()
    sec_ret = pd.DataFrame({t: etf_ret[SECTOR[t]] for t in cols})
    return Decomposition(stock_ret=np.log(panel.close).diff(), mkt_ret=etf_ret["SPY"], sector=sector,
                         sec_ret=sec_ret, alpha=pd.DataFrame(1e-4, index=dates, columns=cols),
                         beta_mkt=per_ticker(BETA_MKT), beta_sec=per_ticker(BETA_SEC), resid=z * sigma,
                         sigma=sigma, z=z, tot_sigma=pd.DataFrame(0.025, index=dates, columns=cols),
                         vol_z=pd.DataFrame(0.3, index=dates, columns=cols))


def keys(events):
    return list(zip(events["pos"], events["ticker"]))


def event_rows(panel: Panel, items: list[tuple[int, str]]) -> pd.DataFrame:
    dec = make_dec(panel, {k: -5.0 for k in items})
    out = ev.find_events(dec, panel, cooldown=0)
    assert keys(out) == sorted(items, key=lambda k: (k[0], k[1]))
    return out


# ---------------------------------------------------------------- find_events

def test_find_events_filters_columns_and_values():
    panel = make_panel()
    panel.member.iloc[8, TICKERS.index("BBB")] = False
    panel.raw_close.iloc[9, TICKERS.index("AAA")] = 4.0  # adjusted close ~100 but raw print is sub-$5
    dec = make_dec(panel, {(5, "AAA"): -3.5, (5, "BBB"): -3.0, (6, "CCC"): -2.9, (8, "BBB"): -4.0,
                           (9, "AAA"): -5.0, (12, "CCC"): -4.0, (14, "CCC"): np.nan, (20, "CCC"): 4.0})
    dec.sigma.iloc[12, TICKERS.index("CCC")] = np.nan

    out = ev.find_events(dec, panel, cooldown=0)

    assert list(out.columns) == ev.EVENT_COLUMNS
    assert isinstance(out.index, pd.RangeIndex)
    assert keys(out) == [(5, "AAA"), (5, "BBB")]  # z exactly at threshold is included
    assert out["pos"].dtype == np.int64
    row = out.iloc[0]
    d = panel.dates[5]
    assert row["date"] == d and row["z"] == -3.5
    assert row["ret"] == pytest.approx(math.log(panel.close.at[d, "AAA"] / panel.close.iloc[4]["AAA"]))
    assert row["mkt_ret"] == dec.mkt_ret.iloc[5]
    assert row["sec_ret"] == dec.sec_ret.at[d, "AAA"]
    assert row["resid"] == pytest.approx(-3.5 * 0.02)
    assert (row["sector"], row["beta_mkt"], row["beta_sec"]) == ("XLK", 0.8, 0.3)
    assert (row["alpha"], row["sigma"], row["tot_sigma"], row["vol_z"]) == (1e-4, 0.02, 0.025, 0.3)
    assert row["price"] == panel.raw_close.at[d, "AAA"]


def test_find_events_min_price_falls_back_to_adjusted_close():
    panel = make_panel()
    dec = make_dec(panel, {(9, "AAA"): -5.0, (11, "CCC"): -5.0})
    panel.raw_close = None
    panel.close.iloc[11, TICKERS.index("CCC")] = 4.99
    out = ev.find_events(dec, panel, cooldown=0)
    assert keys(out) == [(9, "AAA")]
    assert out["price"].iloc[0] == panel.close.iloc[9]["AAA"]
    assert ev.find_events(dec, panel, cooldown=0, min_price=200.0).empty


def test_cooldown_measured_from_kept_events_per_ticker():
    panel = make_panel()
    dec = make_dec(panel, {(2, "AAA"): -4, (10, "AAA"): -4, (18, "AAA"): -4, (33, "AAA"): -4,
                           (34, "AAA"): -4, (3, "BBB"): -4, (4, "BBB"): -4})
    out = ev.find_events(dec, panel, cooldown=15)
    # 18 is kept (16 days after kept 2) although only 8 days after the suppressed 10; 33 is exactly 15 after 18.
    assert keys(out) == [(2, "AAA"), (3, "BBB"), (18, "AAA"), (34, "AAA")]
    assert keys(ev.find_events(dec, panel)) == keys(out)  # default resolves EVENT_COOLDOWN
    assert params.EVENT_COOLDOWN == 15
    assert len(ev.find_events(dec, panel, cooldown=0)) == 7


def test_side_up_mirrors_threshold():
    panel = make_panel()
    dec = make_dec(panel, {(7, "CCC"): 3.2, (7, "AAA"): -3.5, (9, "BBB"): 2.9, (11, "BBB"): 3.0})
    assert keys(ev.find_events(dec, panel, side="up", cooldown=0)) == [(7, "CCC"), (11, "BBB")]
    assert keys(ev.find_events(dec, panel, side="down", cooldown=0)) == [(7, "AAA")]
    assert keys(ev.find_events(dec, panel, z_threshold=-3.3, side="up", cooldown=0)) == []


def test_find_events_rejects_bad_arguments():
    panel = make_panel()
    dec = make_dec(panel)
    with pytest.raises(ValueError):
        ev.find_events(dec, panel, side="sideways")
    with pytest.raises(ValueError):
        ev.find_events(dec, panel, z_threshold=3.0)
    with pytest.raises(ValueError):
        ev.find_events(dec, panel, cooldown=-1)


def test_empty_events_flow_through():
    panel = make_panel()
    out = ev.find_events(make_dec(panel), panel)
    assert out.empty and list(out.columns) == ev.EVENT_COLUMNS
    fwd = ev.add_forward_returns(out, panel, horizons=(1, 5))
    assert fwd.empty and "net_abn_5" in fwd.columns
    pos = ev.position_daily_returns(out, panel, 5)
    assert pos.empty and list(pos.columns) == ["date", "event_id", "abn"]


# ---------------------------------------------------------------- add_forward_returns

def expected(panel, p, ticker, h, entry):
    c, ec = panel.close[ticker], panel.etf_close
    if entry == "next_open":
        s0, m0, k0 = panel.open[ticker].iloc[p + 1], panel.etf_open["SPY"].iloc[p + 1], \
            panel.etf_open[SECTOR[ticker]].iloc[p + 1]
    else:
        s0, m0, k0 = c.iloc[p], ec["SPY"].iloc[p], ec[SECTOR[ticker]].iloc[p]
    raw = math.log(c.iloc[p + h] / s0)
    mkt = math.log(ec["SPY"].iloc[p + h] / m0)
    sec = math.log(ec[SECTOR[ticker]].iloc[p + h] / k0)
    abn = raw - BETA_MKT[ticker] * mkt - BETA_SEC[ticker] * sec
    return {"entry_px": s0, "raw": raw, "mkt": mkt, "sec": sec, "abn": abn,
            "net_raw": raw - 2 * COST / 1e4, "net_abn": abn - 2 * COST / 1e4}


@pytest.mark.parametrize("entry", ["next_open", "close"])
def test_forward_returns_exact_values(entry):
    panel = make_panel()
    events = event_rows(panel, [(5, "AAA"), (5, "CCC"), (12, "BBB")])
    out = ev.add_forward_returns(events, panel, horizons=(1, 5, 15), entry=entry, cost_bps=COST)
    assert out[ev.EVENT_COLUMNS].equals(events)
    for i, row in out.iterrows():
        p, t = row["pos"], row["ticker"]
        assert row["entry_date"] == panel.dates[p + 1 if entry == "next_open" else p]
        for h in (1, 5, 15):
            exp = expected(panel, p, t, h, entry)
            assert row["entry_px"] == pytest.approx(exp["entry_px"], rel=1e-12)
            assert row[f"exit_date_{h}"] == panel.dates[p + h]
            assert not row[f"missing_{h}"]
            for name in ("raw", "mkt", "sec", "abn", "net_raw", "net_abn"):
                assert row[f"{name}_{h}"] == pytest.approx(exp[name], rel=1e-12, abs=1e-15), (i, h, name)


def test_forward_returns_defaults_and_bad_entry():
    panel = make_panel()
    events = event_rows(panel, [(5, "AAA")])
    out = ev.add_forward_returns(events, panel)
    assert all(f"net_abn_{h}" in out.columns for h in params.HORIZONS)
    assert out["entry_date"].iloc[0] == panel.dates[6]  # ENTRY defaults to next_open
    assert out["net_abn_5"].iloc[0] == pytest.approx(
        expected(panel, 5, "AAA", 5, "next_open")["abn"] - 2 * params.COST_BPS_PER_SIDE / 1e4)
    with pytest.raises(ValueError):
        ev.add_forward_returns(events, panel, entry="vwap")
    with pytest.raises(ValueError):
        ev.add_forward_returns(events, panel, horizons=(0,))


def test_horizon_beyond_panel_is_nan_not_missing():
    panel = make_panel(n=40)
    events = event_rows(panel, [(37, "AAA"), (39, "BBB")])
    out = ev.add_forward_returns(events, panel, horizons=(1, 5), cost_bps=COST).set_index("ticker")
    assert np.isfinite(out.at["AAA", "net_abn_1"])
    assert np.isnan(out.at["AAA", "net_abn_5"]) and pd.isna(out.at["AAA", "exit_date_5"])
    assert not out.at["AAA", "missing_5"]
    # Last panel day: no next open to trade at.
    assert pd.isna(out.at["BBB", "entry_date"]) and np.isnan(out.at["BBB", "entry_px"])
    assert out.loc["BBB", ["net_abn_1", "net_abn_5"]].isna().all()
    assert not out.loc["BBB", ["missing_1", "missing_5"]].any()
    closed = ev.add_forward_returns(events, panel, horizons=(1,), entry="close").set_index("ticker")
    assert np.isnan(closed.at["BBB", "net_abn_1"]) and closed.at["BBB", "entry_date"] == panel.dates[39]


def test_holdout_embargo_on_exit_date():
    hs = settings().holdout_start
    panel = make_panel(start=hs - pd.offsets.BDay(20), n=40)
    assert panel.dates[20] == hs
    panel.close.iloc[22, TICKERS.index("BBB")] = np.nan  # holdout-period gap must not count as "missing"
    events = event_rows(panel, [(14, "AAA"), (17, "BBB"), (19, "CCC")])
    out = ev.add_forward_returns(events, panel, horizons=(1, 5, 15), cost_bps=COST).set_index("ticker")

    assert np.isfinite(out.loc["AAA", ["net_abn_1", "net_abn_5"]].astype(float)).all()
    assert out.at["AAA", "exit_date_5"] == panel.dates[19]
    assert np.isnan(out.at["AAA", "net_abn_15"]) and pd.isna(out.at["AAA", "exit_date_15"])
    assert np.isfinite(out.at["BBB", "net_abn_1"])
    for col in ("raw_5", "mkt_5", "sec_5", "abn_5", "net_raw_5", "net_abn_5"):
        assert np.isnan(out.at["BBB", col])
    assert not out.loc[:, ["missing_1", "missing_5", "missing_15"]].to_numpy().any()
    # Entry at the t+1 open would itself be a holdout print.
    assert pd.isna(out.at["CCC", "entry_date"]) and np.isnan(out.at["CCC", "entry_px"])
    assert out.loc["CCC", ["net_abn_1", "net_abn_5", "net_abn_15"]].isna().all()

    pos = ev.position_daily_returns(events, panel, 5, cost_bps=COST)
    assert set(pos["event_id"]) == {0}
    assert pos["date"].max() < hs
    assert ev.position_daily_returns(events, panel, 15, cost_bps=COST).empty


def test_missing_flag_marks_stock_price_gaps_only():
    panel = make_panel()
    events = event_rows(panel, [(5, "AAA"), (5, "BBB"), (5, "CCC"), (20, "AAA"), (20, "BBB")])
    panel.close.iloc[10, TICKERS.index("AAA")] = np.nan       # AAA exit for h=5 (likely delisting/gap)
    panel.open.iloc[6, TICKERS.index("BBB")] = np.nan         # BBB entry open
    panel.etf_close.iloc[10, ETFS.index("XLF")] = np.nan      # CCC's sector ETF at exit: NaN but not missing
    events.loc[events["pos"].eq(20) & events["ticker"].eq("BBB"), "sector"] = None
    out = ev.add_forward_returns(events, panel, horizons=(1, 5), cost_bps=COST)
    by = {(r.pos, r.ticker): r for r in out.itertuples()}

    a, b, c = by[(5, "AAA")], by[(5, "BBB")], by[(5, "CCC")]
    assert np.isfinite(a.net_abn_1) and not a.missing_1
    assert np.isnan(a.raw_5) and np.isnan(a.mkt_5) and np.isnan(a.net_abn_5) and a.missing_5
    assert a.exit_date_5 == panel.dates[10]
    assert b.missing_1 and b.missing_5 and np.isnan(b.net_raw_1) and np.isnan(b.entry_px)
    assert np.isfinite(c.net_abn_1) and np.isnan(c.raw_5) and np.isnan(c.net_abn_5) and not c.missing_5
    assert np.isfinite(by[(20, "AAA")].net_abn_5)
    no_sector = by[(20, "BBB")]
    assert np.isnan(no_sector.net_abn_5) and np.isnan(no_sector.raw_5) and not no_sector.missing_5

    pos = ev.position_daily_returns(events, panel, 5, cost_bps=COST)
    kept = set(pos["event_id"])
    assert kept == set(out.index[np.isfinite(out["net_abn_5"])])
    assert len(kept) == 1


# ---------------------------------------------------------------- position_daily_returns

def test_position_daily_returns_exact_next_open():
    panel = make_panel()
    events = event_rows(panel, [(5, "AAA")])
    pos = ev.position_daily_returns(events, panel, 3, entry="next_open", cost_bps=COST)
    c, o, ec, eo = panel.close["AAA"], panel.open["AAA"], panel.etf_close, panel.etf_open
    bm, bs = BETA_MKT["AAA"], BETA_SEC["AAA"]
    day1 = (math.log(c.iloc[6] / o.iloc[6]) - bm * math.log(ec["SPY"].iloc[6] / eo["SPY"].iloc[6])
            - bs * math.log(ec["XLK"].iloc[6] / eo["XLK"].iloc[6]) - COST / 1e4)

    def cc(d):
        return (math.log(c.iloc[d] / c.iloc[d - 1]) - bm * math.log(ec["SPY"].iloc[d] / ec["SPY"].iloc[d - 1])
                - bs * math.log(ec["XLK"].iloc[d] / ec["XLK"].iloc[d - 1]))

    assert list(pos["date"]) == list(panel.dates[6:9])
    assert list(pos["event_id"]) == [0, 0, 0]
    np.testing.assert_allclose(pos["abn"], [day1, cc(7), cc(8) - COST / 1e4], rtol=1e-12)

    one = ev.position_daily_returns(events, panel, 1, entry="next_open", cost_bps=COST)
    assert one["abn"].iloc[0] == pytest.approx(day1 - COST / 1e4, rel=1e-12)


def test_position_daily_returns_exact_close_entry():
    panel = make_panel()
    events = event_rows(panel, [(5, "CCC")])
    pos = ev.position_daily_returns(events, panel, 3, entry="close", cost_bps=COST)
    c, ec = panel.close["CCC"], panel.etf_close
    bm, bs = BETA_MKT["CCC"], BETA_SEC["CCC"]
    want = [math.log(c.iloc[d] / c.iloc[d - 1]) - bm * math.log(ec["SPY"].iloc[d] / ec["SPY"].iloc[d - 1])
            - bs * math.log(ec["XLF"].iloc[d] / ec["XLF"].iloc[d - 1]) for d in (6, 7, 8)]
    want[0] -= COST / 1e4
    want[-1] -= COST / 1e4
    assert list(pos["date"]) == list(panel.dates[6:9])
    np.testing.assert_allclose(pos["abn"], want, rtol=1e-12)


@pytest.mark.parametrize("entry", ["next_open", "close"])
@pytest.mark.parametrize("h", [1, 5, 15])
def test_position_sum_identity(entry, h):
    panel = make_panel(n=60, seed=3)
    items = [(p, t) for p in (2, 9, 20, 33, 41, 50) for t in TICKERS]
    events = event_rows(panel, items)
    events.index = events.index * 10 + 7  # event_id must follow the index, not row order
    panel.close.iloc[23, TICKERS.index("AAA")] = np.nan     # mid-holding gap: identity must survive it
    panel.etf_close.iloc[36, ETFS.index("XLK")] = np.nan
    fwd = ev.add_forward_returns(events, panel, horizons=(h,), entry=entry, cost_bps=COST)
    pos = ev.position_daily_returns(events, panel, h, entry=entry, cost_bps=COST)

    assert pos["date"].is_monotonic_increasing and isinstance(pos.index, pd.RangeIndex)
    assert pos["abn"].notna().all()
    valid = fwd.loc[np.isfinite(fwd[f"net_abn_{h}"]), f"net_abn_{h}"]
    assert len(valid) > 0
    sums = pos.groupby("event_id")["abn"].sum()
    counts = pos.groupby("event_id").size()
    assert set(sums.index) == set(valid.index)
    assert (counts == h).all()
    np.testing.assert_allclose(sums.loc[valid.index], valid, rtol=0, atol=1e-12)


def test_position_daily_returns_requires_unique_index():
    panel = make_panel()
    events = event_rows(panel, [(5, "AAA"), (6, "BBB")])
    events.index = [1, 1]
    with pytest.raises(ValueError):
        ev.position_daily_returns(events, panel, 5)


# ---------------------------------------------------------------- no lookahead

def test_future_perturbation_leaves_earlier_outputs_unchanged():
    panel = make_panel(n=60, seed=7)
    rng = np.random.default_rng(1)
    z_points = {(int(p), t): float(rng.uniform(-6, -3)) for p in rng.integers(0, 60, 25) for t in TICKERS[:2]}
    dec = make_dec(panel, z_points)
    cut = 35
    base = ev.find_events(dec, panel, cooldown=5)
    base_fwd = ev.add_forward_returns(base, panel, horizons=(1, 5, 15), cost_bps=COST)
    base_pos = ev.position_daily_returns(base, panel, 5, cost_bps=COST)

    for frame in (dec.z, dec.sigma, dec.resid, dec.beta_mkt, dec.beta_sec):
        frame.iloc[cut:] = -9.0
    dec.sector.iloc[cut:] = "XLF"
    for frame in (panel.close, panel.open, panel.etf_close, panel.etf_open, panel.raw_close):
        frame.iloc[cut:] = frame.iloc[cut:] * rng.uniform(0.5, 1.5, size=frame.iloc[cut:].shape)
    panel.member.iloc[cut:] = False
    later = ev.find_events(dec, panel, cooldown=5)
    later_fwd = ev.add_forward_returns(later, panel, horizons=(1, 5, 15), cost_bps=COST)
    later_pos = ev.position_daily_returns(later, panel, 5, cost_bps=COST)

    early = base["pos"] < cut
    assert early.sum() >= 5
    pd.testing.assert_frame_equal(base[early], later[later["pos"] < cut])
    for h in (1, 5, 15):
        done = base_fwd["exit_date_" + str(h)] < panel.dates[cut]
        cols = [f"{n}_{h}" for n in ("exit_date", "raw", "mkt", "sec", "abn", "net_raw", "net_abn", "missing")]
        pd.testing.assert_frame_equal(base_fwd.loc[done, cols], later_fwd.loc[done, cols])
    ids = base_fwd.index[base_fwd["exit_date_5"] < panel.dates[cut]]
    pd.testing.assert_frame_equal(base_pos[base_pos["event_id"].isin(ids)].reset_index(drop=True),
                                  later_pos[later_pos["event_id"].isin(ids)].reset_index(drop=True))

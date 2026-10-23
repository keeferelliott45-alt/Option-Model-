import numpy as np
import pandas as pd
import pytest

from omodel import holdout, params
from omodel.config import MARKET_ETF, SECTOR_ETFS, settings
from omodel.data import panel as panel_mod
from omodel.data import universe, yahoo

START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31")
HOLIDAYS = pd.DatetimeIndex(["2020-01-01", "2020-01-20", "2020-02-17"])
SPY_DATES = pd.bdate_range("2019-12-02", "2020-04-30").difference(HOLIDAYS)
SPY_NAN_DAY = pd.Timestamp("2020-02-03")
CAL = SPY_DATES[(SPY_DATES >= START) & (SPY_DATES <= END) & (SPY_DATES != SPY_NAN_DAY)]
ETFS = [MARKET_ETF, *SECTOR_ETFS]


def bars(dates, close, adj=None, opens=None, split=None):
    dates = pd.DatetimeIndex(dates, name="date")
    close = np.asarray(close, dtype=float)
    df = pd.DataFrame({
        "open": close * 0.99 if opens is None else np.asarray(opens, dtype=float),
        "high": close * 1.01, "low": close * 0.98, "close": close,
        "adj_close": close if adj is None else np.asarray(adj, dtype=float),
        "volume": np.arange(len(close), dtype=float) * 1000 + 5e5,
    }, index=dates)
    if split is not None:
        df["split_factor"] = split
    return df


def sector_etfs(dates, born=None):
    born = born or {}
    out = {}
    for k, s in enumerate(SECTOR_ETFS):
        d = dates[dates >= pd.Timestamp(born.get(s, dates[0]))]
        out[s] = bars(d, np.linspace(40.0 + k, 45.0 + k, len(d)))
    return out


def make_data():
    rng = np.random.default_rng(1)
    walk = lambda n, s: s * np.exp(np.cumsum(rng.normal(0, 0.01, n)))  # noqa: E731
    n = len(SPY_DATES)
    spy_close = walk(n, 300.0)
    spy_close[SPY_DATES.get_loc(SPY_NAN_DAY)] = np.nan
    data = {"SPY": bars(SPY_DATES, spy_close, adj=spy_close * 0.98)}

    aaa_dates = SPY_DATES.append(pd.DatetimeIndex(["2020-01-04", "2020-01-20"])).sort_values()  # Saturday, holiday
    m = len(aaa_dates)
    close = walk(m, 50.0)
    div = np.where(aaa_dates < "2020-02-10", 0.90, 0.95)    # dividend ex-date moves the adjustment factor
    split = np.where(aaa_dates < "2020-02-14", 2.0, 1.0)    # 2:1 split ex 2020-02-14
    data["AAA"] = bars(aaa_dates, close, adj=close * div, opens=close * 1.01, split=split)

    bbb_dates = SPY_DATES.drop(pd.Timestamp("2020-02-20"))  # missing day: must stay NaN
    close = walk(len(bbb_dates), 20.0)
    opens = close * 0.98
    adj = close.copy()
    loc = {d: bbb_dates.get_loc(pd.Timestamp(d)) for d in ["2020-02-21", "2020-02-24", "2020-02-25"]}
    close[loc["2020-02-21"]] = 0.0
    adj[loc["2020-02-21"]] = 0.0
    opens[loc["2020-02-24"]] = -1.0
    adj[loc["2020-02-25"]] = -5.0
    data["BBB"] = bars(bbb_dates, close, adj=adj, opens=opens)

    data["DDD"] = bars(pd.bdate_range("2020-04-06", "2020-04-30"), np.full(19, 10.0))  # nothing inside [START, END]
    # XLRE "incepts" mid-window; XLC only after it (like any pre-2018 window on real data).
    data.update(sector_etfs(SPY_DATES, born={"XLRE": "2020-02-10", "XLC": "2020-04-06"}))
    data["XLK"] = bars(SPY_DATES, walk(n, 90.0))
    return data


MEMBERSHIP = pd.DataFrame({
    "ticker": ["AAA", "BBB", "CCC", "DDD", "EEE"],
    "start_date": pd.to_datetime(["2019-01-01", "2020-02-01", "2010-01-01", "2020-03-15", "2000-01-01"]),
    "end_date": pd.to_datetime([None, None, "2020-03-01", None, "2010-01-01"]),
})


def loader_for(data, calls=None):
    def load(symbol):
        if calls is not None:
            calls.append(symbol)
        df = data.get(symbol)
        return None if df is None else df.copy()
    return load


def test_panel_alignment_and_adjustments():
    data, calls = make_data(), []
    p = panel_mod.build_panel(START, END, loader=loader_for(data, calls), membership=MEMBERSHIP)
    assert calls == [*ETFS, "AAA", "BBB", "CCC", "DDD"]  # factor ETFs first, so a broken one fails fast
    pd.testing.assert_index_equal(p.dates, pd.DatetimeIndex(CAL, name="date"), check_exact=True)
    for f in (p.open, p.volume, p.raw_close, p.member, p.etf_close, p.etf_open):
        pd.testing.assert_index_equal(f.index, p.dates)
    assert p.tickers == ["AAA", "BBB"]  # CCC has no data, DDD none inside the window
    for f in (p.open, p.volume, p.raw_close, p.member):
        assert list(f.columns) == p.tickers
    assert list(p.etf_close.columns) == ETFS == list(p.etf_open.columns)
    assert p.macro is None

    a = data["AAA"].reindex(CAL)
    np.testing.assert_allclose(p.close["AAA"], a["adj_close"])
    np.testing.assert_allclose(p.open["AAA"], a["open"] * a["adj_close"] / a["close"])
    np.testing.assert_allclose(p.raw_close["AAA"], a["close"] * a["split_factor"])
    np.testing.assert_allclose(p.volume["AAA"], a["volume"])
    assert p.raw_close.loc["2020-02-13", "AAA"] == pytest.approx(2 * data["AAA"].loc["2020-02-13", "close"])

    b = p.close["BBB"]
    assert b.loc[:"2020-02-19"].notna().all()
    for frame in (p.close, p.open, p.raw_close, p.volume):
        assert np.isnan(frame.loc["2020-02-20", "BBB"])        # no forward fill
    for frame in (p.close, p.open, p.raw_close):
        assert np.isnan(frame.loc["2020-02-21", "BBB"])        # zero close
    assert np.isnan(p.open.loc["2020-02-24", "BBB"]) and b.loc["2020-02-24"] > 0
    assert np.isnan(b.loc["2020-02-25"]) and np.isnan(p.open.loc["2020-02-25", "BBB"])
    assert p.raw_close.loc["2020-02-25", "BBB"] > 0
    for frame in (p.close, p.open, p.raw_close):
        assert ((frame > 0) | frame.isna()).all().all()

    s = data["SPY"].reindex(CAL)
    np.testing.assert_allclose(p.etf_close[MARKET_ETF], s["adj_close"])
    np.testing.assert_allclose(p.etf_open[MARKET_ETF], s["open"] * s["adj_close"] / s["close"])
    xlre = p.etf_close["XLRE"]
    assert xlre.loc[:"2020-02-07"].isna().all() and xlre.loc["2020-02-10":].notna().all()
    assert p.etf_close["XLC"].isna().all()  # incepted after the window: NaN, not an error
    assert p.etf_close.drop(columns=["XLRE", "XLC"]).notna().all().all() and p.etf_open["XLK"].notna().all()

    assert p.member.dtypes.eq(bool).all()
    assert p.member["AAA"].all()
    np.testing.assert_array_equal(p.member["BBB"].to_numpy(), CAL >= "2020-02-01")


def test_panel_ignores_future_data():
    cutoff = pd.Timestamp("2020-02-18")
    base = make_data()
    perturbed = {k: v.copy() for k, v in base.items()}
    rng = np.random.default_rng(7)
    for df in perturbed.values():
        later = df.index >= cutoff
        df.loc[later, ["open", "close", "adj_close"]] *= rng.uniform(0.5, 2.0, (later.sum(), 1))
        df.loc[later, "volume"] *= 3
    perturbed["SPY"] = perturbed["SPY"].drop(pd.Timestamp("2020-03-02"))
    perturbed["CCC"] = bars(pd.bdate_range("2020-03-02", "2020-03-31"), np.full(22, 5.0))
    p1 = panel_mod.build_panel(START, END, loader=loader_for(base), membership=MEMBERSHIP)
    p2 = panel_mod.build_panel(START, END, loader=loader_for(perturbed), membership=MEMBERSHIP)
    for name in ("close", "open", "volume", "raw_close", "member", "etf_close", "etf_open"):
        f1, f2 = getattr(p1, name), getattr(p2, name)
        early = f1.loc[f1.index < cutoff]
        pd.testing.assert_frame_equal(early, f2.loc[f2.index < cutoff, f1.columns])


def test_panel_holdout_truncation(monkeypatch):
    h = settings().holdout_start
    dates = pd.bdate_range(h - pd.Timedelta(days=40), h + pd.Timedelta(days=40))
    data = {"SPY": bars(dates, np.linspace(500, 520, len(dates))),
            "AAA": bars(dates, np.linspace(50, 60, len(dates))),
            "NEW": bars(dates[dates >= h], np.full((dates >= h).sum(), 7.0)),  # lists and joins inside the holdout
            "LATE": bars(dates, np.full(len(dates), 9.0)),                     # trades before, joins inside the holdout
            **sector_etfs(dates)}
    m = pd.DataFrame({"ticker": ["AAA", "NEW", "LATE"],
                      "start_date": pd.to_datetime(["2000-01-01", h, h + pd.Timedelta(days=10)]),
                      "end_date": pd.NaT})
    start, end = h - pd.Timedelta(days=30), h + pd.Timedelta(days=30)
    calls = []
    p = panel_mod.build_panel(start, end, loader=loader_for(data, calls), membership=m)
    assert p.dates.max() == dates[dates < h].max()
    for f in (p.close, p.open, p.volume, p.raw_close, p.member, p.etf_close, p.etf_open):
        assert f.index.max() < h
    assert p.tickers == ["AAA"] and "LATE" not in calls and "NEW" not in calls  # holdout index changes stay unseen
    assert holdout.access_count() == 0
    with pytest.raises(ValueError, match="holdout"):
        panel_mod.build_panel(h, end, loader=loader_for(data), membership=m)
    assert holdout.access_count() == 0

    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    unlocked = panel_mod.build_panel(start, end, loader=loader_for(data), membership=m)
    assert unlocked.dates.max() >= h and unlocked.tickers == ["AAA", "LATE", "NEW"]
    assert holdout.access_count() > 0


def test_panel_requires_market_etf():
    data = make_data()
    del data["SPY"]
    with pytest.raises(ValueError, match=MARKET_ETF):
        panel_mod.build_panel(START, END, loader=loader_for(data), membership=MEMBERSHIP)


@pytest.mark.parametrize("mode", ["raises", "missing", "dead", "no_closes"])
def test_panel_refuses_broken_sector_etf(mode):
    data, calls = make_data(), []
    if mode == "missing":
        del data["XLK"]
    elif mode == "dead":
        data["XLK"] = data["XLK"].loc[:"2019-12-20"]  # stale cache that ends before the window
    elif mode == "no_closes":
        data["XLK"] = data["XLK"].assign(close=np.nan)
    inner = loader_for(data, calls)

    def load(symbol):
        if mode == "raises" and symbol == "XLK":
            raise ConnectionError("429 Too Many Requests")
        return inner(symbol)
    with pytest.raises((ValueError, ConnectionError), match="XLK|429"):
        panel_mod.build_panel(START, END, loader=load, membership=MEMBERSHIP)
    if mode in ("raises", "missing"):
        assert "AAA" not in calls  # failed before any stock fetch


def test_panel_rejects_empty_calendar():
    with pytest.raises(ValueError, match="empty trading calendar"):
        panel_mod.build_panel(pd.Timestamp("2015-01-01"), pd.Timestamp("2015-06-30"),
                              loader=loader_for(make_data()), membership=MEMBERSHIP)


def test_panel_default_loader_and_start(monkeypatch):
    data = make_data()
    seen = []

    def fake_many(symbols, refresh=False, max_workers=4, *, strict=False):
        seen.append((list(symbols), refresh, strict))
        return {s: data[s] for s in symbols if s in data}
    monkeypatch.setattr(yahoo, "load_many", fake_many)
    panel_mod.build_panel(START, END, refresh=True, membership=MEMBERSHIP)
    assert seen == [(ETFS, True, True), (["AAA", "BBB", "CCC", "DDD"], True, True)]

    def failing_many(symbols, refresh=False, max_workers=4, *, strict=False):
        seen.append(list(symbols))
        raise yahoo.LoadError(["XLK"])
    monkeypatch.setattr(yahoo, "load_many", failing_many)
    seen.clear()
    with pytest.raises(yahoo.LoadError, match="XLK"):
        panel_mod.build_panel(START, END, membership=MEMBERSHIP)
    assert seen == [ETFS]

    warm = settings().dev_start - pd.Timedelta(days=panel_mod.PANEL_WARMUP_DAYS)
    assert params.REGISTRY["PANEL_WARMUP_DAYS"].source == "convention"
    old = pd.bdate_range("1998-01-01", "2000-02-15")
    old_data = {"SPY": bars(old, np.linspace(100, 140, len(old))), "AAA": bars(old, np.linspace(10, 20, len(old))),
                **sector_etfs(old, born={s: "1998-12-22" for s in SECTOR_ETFS}),
                "XLRE": bars(pd.bdate_range("2015-10-08", "2015-10-30"), np.full(17, 30.0)),
                "XLC": bars(pd.bdate_range("2018-06-19", "2018-06-29"), np.full(9, 50.0))}
    old_m = MEMBERSHIP.assign(start_date=pd.Timestamp("1990-01-01"), end_date=pd.NaT).iloc[:1]
    monkeypatch.setattr(universe, "load_membership", lambda refresh=False: old_m)
    p = panel_mod.build_panel(end=pd.Timestamp("2000-01-31"), loader=loader_for(old_data))
    assert p.dates[0] == old[old >= warm][0] == pd.Timestamp("1998-07-02")
    assert p.dates[-1] == pd.Timestamp("2000-01-31") and p.tickers == ["AAA"]
    assert p.etf_close["XLB"].loc[:"1998-12-21"].isna().all() and p.etf_close["XLB"].loc["1998-12-22":].notna().all()


def test_coverage_counts_no_data_tickers(monkeypatch):
    p = panel_mod.build_panel(START, END, loader=loader_for(make_data()), membership=MEMBERSHIP)
    cov = panel_mod.coverage(p, membership=MEMBERSHIP)
    aaa, bbb = len(CAL), int((CAL >= "2020-02-01").sum())
    ccc, ddd = int((CAL < "2020-03-01").sum()), int((CAL >= "2020-03-15").sum())
    member_days = aaa + bbb + ccc + ddd
    covered = aaa + bbb - 3  # BBB: missing day, zero close, negative adj close
    assert list(cov.index) == [2020]
    row = cov.loc[2020]
    assert row["member_days"] == member_days and row["covered_days"] == covered
    assert row["coverage_pct"] == pytest.approx(100 * covered / member_days)
    assert row["members"] == 4                # EEE left the index in 2010
    assert row["members_with_no_data"] == 2   # CCC (no data) and DDD (no data inside the window)
    monkeypatch.setattr(universe, "load_membership", lambda refresh=False: MEMBERSHIP)
    pd.testing.assert_frame_equal(panel_mod.coverage(p), cov)


def reuse_fixture():
    data = make_data()
    since = lambda d: SPY_DATES[SPY_DATES >= d]  # noqa: E731
    data["REUSED"] = bars(since("2020-02-10"), np.full(len(since("2020-02-10")), 12.0))  # member since 2019, trades 2020
    data["NEARBY"] = bars(since("2020-01-06"), np.full(len(since("2020-01-06")), 13.0))  # 4 days late: tolerated
    data["TWO"] = bars(since("2020-01-21"), np.linspace(14, 15, len(since("2020-01-21"))))
    data["FIXED"] = bars(SPY_DATES, np.linspace(16, 17, len(SPY_DATES)))
    data["TRUSTED"] = bars(since("2020-03-02"), np.full(len(since("2020-03-02")), 18.0))
    extra = pd.DataFrame({
        "ticker": ["REUSED", "NEARBY", "TWO", "TWO", "FIXED", "TRUSTED"],
        "start_date": pd.to_datetime(["2019-06-01", "2020-01-02", "2019-12-16", "2020-02-18", "2019-01-01",
                                      "2019-01-01"]),
        "end_date": pd.to_datetime([None, None, "2020-02-03", None, None, None]),
    })
    return data, pd.concat([MEMBERSHIP, extra], ignore_index=True)


def test_identity_breaks_table(monkeypatch):
    data, m = reuse_fixture()
    monkeypatch.setattr(panel_mod, "IDENTITY_OVERRIDES", {"FIXED": (("2020-01-01", "2020-02-14"),), "TRUSTED": ()})
    stocks = {k: v for k, v in data.items() if k not in ETFS}
    b = panel_mod.identity_breaks(stocks, m, floor=SPY_DATES[0])
    expected = pd.DataFrame({
        "ticker": ["DDD", "FIXED", "REUSED", "TWO"],  # NEARBY is within tolerance; TRUSTED is vouched for
        "start": pd.to_datetime(["2020-03-15", "2020-01-01", "2019-06-01", "2019-12-16"]),
        "end": pd.to_datetime([None, "2020-02-14", None, "2020-02-03"]),
        "reason": ["late first bar", "override", "late first bar", "late first bar"],
    })
    pd.testing.assert_frame_equal(b[list(expected.columns)], expected)
    assert b.loc[b.ticker == "REUSED", "first_bar"].item() == pd.Timestamp("2020-02-10")
    # Without a floor, a spell older than the price source would be flagged; with it, AAA (since 2019) is fine.
    assert "AAA" in set(panel_mod.identity_breaks(stocks, m)["ticker"])
    assert params.REGISTRY["REUSE_TOLERANCE_DAYS"].source == "convention"


def test_panel_drops_prices_of_reused_symbols(monkeypatch):
    data, m = reuse_fixture()
    monkeypatch.setattr(panel_mod, "IDENTITY_OVERRIDES", {"FIXED": (("2020-01-01", "2020-02-14"),), "TRUSTED": ()})
    p = panel_mod.build_panel(START, END, loader=loader_for(data), membership=m)
    assert "REUSED" not in p.tickers  # every in-window price belongs to the spell it cannot have been a member of
    assert p.close["NEARBY"].loc["2020-01-06":].notna().all()
    for frame in (p.close, p.open, p.raw_close, p.volume):
        assert frame["TWO"].loc[:"2020-01-31"].isna().all() and frame["TWO"].loc["2020-02-04":].notna().all()
        assert frame["FIXED"].loc[:"2020-02-13"].isna().all() and frame["FIXED"].loc["2020-02-14":].notna().all()
    assert p.close["TRUSTED"].loc["2020-03-02":].notna().all()

    cov = panel_mod.coverage(p, membership=m).loc[2020]
    full = universe.membership_mask(m, p.dates)
    has = p.close.reindex(columns=full.columns).notna()
    assert cov["covered_days"] == int((full & has).to_numpy().sum())
    assert cov["members_with_no_data"] == 3  # CCC, DDD and REUSED

    # No lookahead: the same decisions are reached with only the data available up to a cutoff.
    cut = pd.Timestamp("2020-02-20")
    early = {k: (v if k in ETFS else v.loc[:cut]) for k, v in data.items() if k in ETFS or len(v.loc[:cut])}
    p_early = panel_mod.build_panel(START, cut, loader=loader_for(early), membership=m)
    assert set(p_early.tickers) <= set(p.tickers)
    for name in ("close", "open", "raw_close", "volume", "member"):
        f_early, f_full = getattr(p_early, name), getattr(p, name)
        pd.testing.assert_frame_equal(f_early, f_full.loc[f_early.index, f_early.columns])

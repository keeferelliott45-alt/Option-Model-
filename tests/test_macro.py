from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from omodel import holdout, params
from omodel.config import settings
from omodel.data import macro, store


@dataclass
class FakeResp:
    text: str
    status_code: int = 200


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    if request.node.get_closest_marker("network") is None:
        def _blocked(url, **kw):
            raise AssertionError(f"unit test tried the network: {url}")
        monkeypatch.setattr(macro.http, "get", _blocked)


def _treasury_router(monkeypatch, bodies: dict[tuple[str, int], str | FakeResp]) -> Counter:
    calls: Counter = Counter()

    def fake_get(url, **kw):
        m = re.search(r"/(\d{4})/all\?type=(\w+)&field_tdr_date_value=(\d{4})&page&_format=csv$", url)
        assert m and m.group(1) == m.group(3), url
        key = (m.group(2), int(m.group(1)))
        calls[key] += 1
        body = bodies.get(key, "")
        return body if isinstance(body, FakeResp) else FakeResp(body)

    monkeypatch.setattr(macro.http, "get", fake_get)
    return calls


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", macro.MacroSourceWarning)
        return fn(*args, **kw)


NOM, REAL = macro.NOMINAL_CURVE, macro.REAL_CURVE
NOM_2003 = ('Date,"1 Mo","3 Mo","6 Mo","2 Yr","10 Yr","20 Yr","30 Yr"\n'
            "12/31/2003,0.90,0.95,1.00,1.84,4.27,5.10,N/A\n"
            "01/02/2003,1.19,1.22,1.30,1.75,4.07,4.90,4.95\n")
NOM_2004 = ('Date,"1 Mo","3 Mo","6 Mo","1 Yr","2 Yr","5 Yr","10 Yr","20 Yr"\n'
            "01/05/2004,0.88,0.93,1.02,1.31,1.90,3.36,4.38,5.20\n"
            "01/02/2004,0.86,0.91,1.00,1.29,1.88,3.30,4.29,5.10\n")
REAL_2003 = 'Date,"5 YR","7 YR","10 YR"\n01/02/2003,1.75,2.24,2.43\n'
REAL_2004 = 'Date,"5 Yr","10 yr","20 Yr"\n01/05/2004,1.10,1.95,2.10\n'


def test_treasury_merges_years_with_different_column_sets(monkeypatch):
    calls = _treasury_router(monkeypatch, {(NOM, 2003): NOM_2003, (NOM, 2004): NOM_2004,
                                           (REAL, 2003): REAL_2003, (REAL, 2004): REAL_2004})
    with pytest.warns(macro.MacroSourceWarning, match="2002"):  # nominal 2002 body is empty here
        out = macro.load_treasury(2002, 2004)
    assert list(out.columns) == macro.TREASURY_COLUMNS
    assert (out.dtypes == float).all() and out.index.is_monotonic_increasing
    assert isinstance(out.index, pd.DatetimeIndex)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2003-01-02", "2003-12-31", "2004-01-02", "2004-01-05"]
    assert out.loc["2003-01-02", ["y3m", "y2", "y10", "y30"]].tolist() == [1.22, 1.75, 4.07, 4.95]
    assert np.isnan(out.loc["2003-12-31", "y30"])  # "N/A" text
    assert out.loc["2004-01-05", "y30"] != out.loc["2004-01-05", "y30"]  # 2004 has no 30 Yr column
    assert out.loc["2004-01-05", "y10"] == 4.38
    assert out.loc["2003-01-02", "real10"] == 2.43 and out.loc["2004-01-05", "real10"] == 1.95
    assert np.isnan(out.loc["2004-01-02", "real10"])
    assert (REAL, 2002) not in calls  # real curve only exists from 2003
    assert calls[(NOM, 2002)] == 1


def test_treasury_caches_completed_years_and_refetches_current(monkeypatch):
    monkeypatch.setattr(macro, "_current_year", lambda: 2004)
    calls = _treasury_router(monkeypatch, {(NOM, 2003): NOM_2003, (NOM, 2004): NOM_2004,
                                           (REAL, 2003): REAL_2003, (REAL, 2004): REAL_2004})
    first = macro.load_treasury(2003)
    second = macro.load_treasury(2003)
    pd.testing.assert_frame_equal(first, second)
    assert calls[(NOM, 2003)] == 1 and calls[(REAL, 2003)] == 1
    assert calls[(NOM, 2004)] == 2 and calls[(REAL, 2004)] == 2
    assert store.read_meta("treasury", f"{NOM}_2003")["complete"] is True
    assert store.read_meta("treasury", f"{NOM}_2004")["complete"] is False

    # 2004 was cached mid-year: once 2004 is over it must be refetched exactly once more.
    monkeypatch.setattr(macro, "_current_year", lambda: 2005)
    macro.load_treasury(2003, 2004)
    macro.load_treasury(2003, 2004)
    assert calls[(NOM, 2004)] == 3 and calls[(NOM, 2003)] == 1

    macro.load_treasury(2003, 2004, refresh=True)
    assert calls[(NOM, 2003)] == 2 and calls[(NOM, 2004)] == 4


def test_treasury_failed_year_warns_and_keeps_others(monkeypatch):
    monkeypatch.setattr(macro, "_current_year", lambda: 2010)
    calls = _treasury_router(monkeypatch, {(NOM, 2003): FakeResp("oops", 500), (NOM, 2004): NOM_2004,
                                           (REAL, 2004): REAL_2004, (REAL, 2003): REAL_2003})
    with pytest.warns(macro.MacroSourceWarning, match="2003"):
        out = macro.load_treasury(2003, 2004)
    assert out.loc["2004-01-05", "y10"] == 4.38  # one failed year does not trip the breaker
    assert out.loc["2003-01-02", "real10"] == 2.43 and np.isnan(out.loc["2003-01-02", "y10"])
    assert store.read("treasury", f"{NOM}_2003") is None  # failures are never cached
    with pytest.warns(macro.MacroSourceWarning):
        macro.load_treasury(2003, 2004)
    assert calls[(NOM, 2003)] == 2


def test_treasury_breaker_stops_network_after_consecutive_host_failures(monkeypatch):
    _treasury_router(monkeypatch, {(NOM, 2003): NOM_2003, (REAL, 2003): REAL_2003})
    macro.load_treasury(2003, 2003)  # cache one completed year
    gets: Counter = Counter()

    def down(url, **kw):
        gets["n"] += 1
        raise ConnectionError("connection refused")

    monkeypatch.setattr(macro.http, "get", down)
    with pytest.warns(macro.MacroSourceWarning) as rec:
        out = macro.load_treasury(1998, 2010)
    assert gets["n"] == macro._BREAKER_FAILS  # not 13 nominal + 8 real years, each with retries
    msgs = " | ".join(str(w.message) for w in rec)
    assert "no network for" in msgs
    assert out.loc["2003-01-02", "y10"] == 4.07 and out.loc["2003-01-02", "real10"] == 2.43
    assert list(out.columns) == macro.TREASURY_COLUMNS


def test_treasury_non_host_errors_do_not_trip_breaker(monkeypatch):
    calls = _treasury_router(monkeypatch, {(c, y): FakeResp("gone", 404) for c in (NOM, REAL)
                                           for y in range(2003, 2007)})
    with pytest.warns(macro.MacroSourceWarning):
        out = macro.load_treasury(2003, 2006)
    assert sum(calls.values()) == 8 and out.empty


def test_treasury_locked_never_requests_holdout_only_years(monkeypatch):
    monkeypatch.setattr(macro, "_current_year", lambda: 2026)
    cut = settings().holdout_start
    last_dev_year = (cut - pd.Timedelta(days=1)).year
    head = 'Date,"3 Mo","2 Yr","10 Yr","30 Yr"\n'
    in_hold = cut + pd.offsets.YearBegin(1) + pd.offsets.BDay(1)
    bodies = {(NOM, last_dev_year): head + f"{cut - pd.offsets.BDay(1):%m/%d/%Y},4,3.5,4,4.6\n"
                                          f"{cut:%m/%d/%Y},4.1,3.6,4.1,4.7\n",
              (NOM, in_hold.year): head + f"{in_hold:%m/%d/%Y},4.2,3.7,4.2,4.8\n"}
    calls = _treasury_router(monkeypatch, bodies)
    locked = _quiet(macro.load_treasury, last_dev_year)
    assert max(y for _, y in calls) == last_dev_year
    assert locked.index.max() == cut - pd.offsets.BDay(1)
    assert store.read("treasury", f"{NOM}_{in_hold.year}") is None  # holdout rows never cached either

    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    unlocked = _quiet(macro.load_treasury, last_dev_year)
    assert calls[(NOM, in_hold.year)] == 1 and unlocked.index.max() == in_hold
    assert holdout.access_count() == 1


def test_panel_locked_never_requests_holdout_only_treasury_years(monkeypatch):
    monkeypatch.setattr(macro, "_current_year", lambda: 2026)
    calls = _treasury_router(monkeypatch, {})

    def cboe_down(name, refresh):
        raise ConnectionError("down")

    monkeypatch.setattr(macro, "_cboe_series", cboe_down)
    monkeypatch.setattr(macro.yahoo, "load_daily", lambda ticker, refresh=False: None)
    _quiet(macro.load_macro_panel)
    assert calls and max(y for _, y in calls) == (settings().holdout_start - pd.Timedelta(days=1)).year


def test_treasury_warns_when_a_year_loses_a_mapped_column(monkeypatch):
    _treasury_router(monkeypatch, {(NOM, 2004): NOM_2004, (REAL, 2004): REAL_2004})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        macro.load_treasury(2004, 2004)
    assert not [w for w in rec if "2004" in str(w.message)]  # 30 Yr absent in 2002-2006 is expected

    respelled = 'Date,"3 Mo","2 Yr","10 Year","30 Yr"\n01/03/2008,3.1,2.9,3.9,4.4\n'
    _treasury_router(monkeypatch, {(NOM, 2008): respelled, (REAL, 2008): 'Date,"10 YR"\n01/03/2008,1.5\n'})
    with pytest.warns(macro.MacroSourceWarning, match=r"2008: no data for \['y10'\]"):
        out = macro.load_treasury(2008, 2008)
    assert out.loc["2008-01-03", "y2"] == 2.9 and np.isnan(out.loc["2008-01-03", "y10"])


def test_read_csv_recovers_minority_date_format_and_warns_on_junk():
    text = "DATE,CLOSE\n01/02/1990,1\n01/03/1990,2\n1990-01-04,3\nnot a date,4\n"
    with pytest.warns(macro.MacroSourceWarning, match="1 rows with unparseable dates"):
        out = macro._read_csv(text, "t")
    assert list(out.index.strftime("%Y-%m-%d")) == ["1990-01-02", "1990-01-03", "1990-01-04"]
    assert out["CLOSE"].tolist() == [1.0, 2.0, 3.0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        iso = macro._read_csv("DATE,X\n2020-01-03,2\n2020-01-02,1\n", "t")
    assert iso["X"].tolist() == [1.0, 2.0]


def test_price_columns_clean_nonpositive_settlements_to_nan(monkeypatch):
    # same cleaning path FX uses: non-positive prints (2020-04-20 CL=F settled at -37.63) become NaN, not ffilled here
    idx = pd.DatetimeIndex(["2020-04-17", "2020-04-20", "2020-04-21"])
    raw = pd.DataFrame({"adj_close": [18.27, -37.63, 0.0]}, index=idx)
    assert macro.PRICE_COLUMNS == ("gold", "oil")
    for col in macro.PRICE_COLUMNS:
        monkeypatch.setattr(macro.yahoo, "load_daily", lambda tic, refresh=False: raw)
        out = macro._yahoo_series(col, macro.YAHOO_SOURCES[col], refresh=False)
        assert np.isnan(out.loc["2020-04-20"]) and np.isnan(out.loc["2020-04-21"])
        assert out.loc["2020-04-17"] == 18.27 and out.index.equals(idx)


VIX_CSV = ("DATE,OPEN,HIGH,LOW,CLOSE\n"
           "01/02/1990,17.24,17.24,17.24,17.24\n"
           "01/03/1990,18.19,18.30,18.00,18.19\n")
VIX9D_CSV = ("Cboe disclaimer line\n"
             "DATE,VIX9D\n"
             "1/5/2011,15.57\n"
             "1/4/2011,16.06\n")


def _cboe_router(monkeypatch, bodies: dict[str, str]) -> Counter:
    calls: Counter = Counter()

    def fake_get(url, **kw):
        name = re.search(r"daily_prices/(\w+)_History\.csv$", url).group(1)
        calls[name] += 1
        return FakeResp(bodies[name]) if name in bodies else FakeResp("not found", 404)

    monkeypatch.setattr(macro.http, "get", fake_get)
    return calls


def test_cboe_formats_and_cache(monkeypatch):
    calls = _cboe_router(monkeypatch, {"VIX": VIX_CSV, "VIX9D": VIX9D_CSV})
    vix = macro.load_cboe_index("VIX")
    assert vix.name == "vix" and vix.dtype == float and isinstance(vix.index, pd.DatetimeIndex)
    assert vix.tolist() == [17.24, 18.19]  # CLOSE, not the first numeric column
    v9 = macro.load_cboe_index("vix9d")
    assert v9.name == "vix9d" and v9.tolist() == [16.06, 15.57] and v9.index.is_monotonic_increasing
    assert v9.index[0] == pd.Timestamp("2011-01-04")

    pd.testing.assert_series_equal(macro.load_cboe_index("VIX"), vix, check_freq=False)
    assert calls["VIX"] == 1
    macro.load_cboe_index("VIX", refresh=True)
    assert calls["VIX"] == 2

    with pytest.raises(RuntimeError, match="404"):
        macro.load_cboe_index("VIX3M")


def test_cboe_refresh_failure_falls_back_to_cache(monkeypatch):
    _cboe_router(monkeypatch, {"VIX": VIX_CSV})
    vix = macro.load_cboe_index("VIX")
    _cboe_router(monkeypatch, {})
    with pytest.warns(macro.MacroSourceWarning, match="cached"):
        again = macro.load_cboe_index("VIX", refresh=True)
    pd.testing.assert_series_equal(again, vix, check_freq=False)


def test_public_cboe_loader_applies_holdout_guard(monkeypatch):
    cut = settings().holdout_start
    days = pd.bdate_range(cut - pd.Timedelta(days=7), cut + pd.Timedelta(days=7))
    body = "DATE,CLOSE\n" + "".join(f"{d:%m/%d/%Y},{i + 10}\n" for i, d in enumerate(days))
    _cboe_router(monkeypatch, {"VIX": body})
    locked = macro.load_cboe_index("VIX")
    assert len(locked) and locked.index.max() < cut
    assert holdout.access_count() == 0

    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    assert macro.load_cboe_index("VIX").index.max() == days[-1]  # full history was cached, only the view is cut
    assert holdout.access_count() == 1


# --- panel -----------------------------------------------------------------------------------

def _bdays(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


def _series(idx: pd.DatetimeIndex, base: float) -> pd.Series:
    return pd.Series(base + np.arange(len(idx), dtype=float), index=idx)


FX_TICKERS = {tic for col, tic in macro.YAHOO_SOURCES.items() if col in macro.FX_COLUMNS}


@pytest.fixture
def sources(monkeypatch):
    """Stub every panel source; tests mutate the returned dicts before calling load_macro_panel."""
    idx = _bdays("1997-12-01", "2001-12-31")
    cboe = {"VIX": _series(idx, 10), "VIX3M": _series(idx, 20), "VIX9D": _series(idx, 30)}
    tsy = pd.DataFrame({c: _series(idx, 40 + i) for i, c in enumerate(macro.TREASURY_COLUMNS)})
    ydata = {tic: pd.DataFrame({"open": _series(idx, 70 + i), "close": _series(idx, 90),
                                "adj_close": _series(idx, 100 + i)})
             for i, tic in enumerate(macro.YAHOO_SOURCES.values())}
    state = {"cboe": cboe, "tsy": tsy, "yahoo": ydata, "calls": Counter()}

    def fake_cboe(name, refresh):
        state["calls"][("cboe", name, refresh)] += 1
        s = state["cboe"][name]
        if isinstance(s, Exception):
            raise s
        return s.rename(name.lower())

    def fake_tsy(start_year, end_year, refresh):
        state["calls"][("tsy", end_year)] += 1
        if isinstance(state["tsy"], Exception):
            raise state["tsy"]
        return state["tsy"]

    def fake_yahoo(ticker, refresh=False):
        state["calls"][("yahoo", ticker, refresh)] += 1
        d = state["yahoo"][ticker]
        if isinstance(d, Exception):
            raise d
        return d

    monkeypatch.setattr(macro, "_cboe_series", fake_cboe)
    monkeypatch.setattr(macro, "_treasury_frame", fake_tsy)
    monkeypatch.setattr(macro.yahoo, "load_daily", fake_yahoo)
    return state


def test_panel_shape_union_index_and_adjusted_close(sources):
    weekend_fx = sources["yahoo"]["EURUSD=X"]
    extra = pd.DataFrame({"open": [1.5, 1.6], "adj_close": [1.5, 1.6]},
                         index=pd.to_datetime(["1999-07-05", "1999-07-04"]))  # a Monday and a Sunday bar
    sources["yahoo"]["EURUSD=X"] = pd.concat([weekend_fx.drop(pd.Timestamp("1999-07-05")), extra])
    out = macro.load_macro_panel()
    assert list(out.columns) == macro.MACRO_COLUMNS
    assert (out.dtypes == float).all()
    assert out.index.min() == pd.Timestamp("1998-01-01")
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert (out.index.dayofweek < 5).all()
    assert 1.5 not in out["eurusd"].tolist() and 1.6 not in out["eurusd"].tolist()  # weekend snapshots dropped
    assert out.loc["1998-01-02", "dxy"] == sources["yahoo"]["DX-Y.NYB"].loc["1998-01-02", "adj_close"]
    assert out.loc["1998-01-02", "vix"] == sources["cboe"]["VIX"].loc["1998-01-02"]
    assert out.loc["1998-01-02", "real10"] == sources["tsy"].loc["1998-01-02", "real10"]
    assert out.index.max() == pd.Timestamp("2001-12-31")


def test_panel_fx_takes_london_midnight_open_on_prior_us_day(sources):
    jpy = sources["yahoo"]["JPY=X"].copy()
    jpy["open"] = 100.0
    jpy["close"] = jpy["adj_close"] = 7.0  # since 2010 Yahoo's FX close is a copy of open; never use it
    jpy.loc["1999-06-15", "open"] = 200.0  # Tue bar opens Mon ~19:00 ET: BoJ-style overnight move
    jpy.loc["1999-06-21", "open"] = 999.0  # Mon bar opens Sun evening: a weekend gap
    sources["yahoo"]["JPY=X"] = jpy
    out = macro.load_macro_panel()["usdjpy"]
    assert out.loc["1999-06-14"] == 200.0 and out.loc["1999-06-15"] == 100.0
    assert 999.0 not in out.tolist() and out.loc["1999-06-18"] == 100.0  # Friday carries Thu evening, not Sun
    assert 7.0 not in out.tolist()
    assert out.loc[: "2001-12-27"].notna().all()


def test_panel_total_return_index_is_point_in_time(sources):
    base = macro.load_macro_panel()
    hyg = sources["yahoo"]["HYG"]["adj_close"]
    expected = 100.0 * hyg / hyg.iloc[0]
    assert np.allclose(base["hyg"], expected.reindex(base.index))
    cutoff = pd.Timestamp("2000-06-15")
    for tic in ("HYG", "LQD"):  # Yahoo re-adjusts every earlier bar when a new distribution goes ex at cutoff
        d = sources["yahoo"][tic].copy()
        d.loc[d.index < cutoff, "adj_close"] *= 0.9
        sources["yahoo"][tic] = d
    readjusted = macro.load_macro_panel()
    pd.testing.assert_frame_equal(base.loc[base.index < cutoff], readjusted.loc[readjusted.index < cutoff])
    assert not np.allclose(base.loc[cutoff:, "hyg"], readjusted.loc[cutoff:, "hyg"])


def test_panel_union_of_dates_and_ffill_limit(sources):
    idx = sources["cboe"]["VIX"].index
    vix = sources["cboe"]["VIX"]
    sources["cboe"]["VIX"] = vix.drop(pd.Timestamp("1999-01-18"))  # one-day holiday gap
    dxy = sources["yahoo"]["DX-Y.NYB"]
    gap = idx[(idx >= "1999-03-01") & (idx <= "1999-03-05")]  # 5-row outage
    sources["yahoo"]["DX-Y.NYB"] = dxy.drop(gap)
    gold = sources["yahoo"]["GC=F"]
    tz = gold.copy()
    tz.index = tz.index.tz_localize("America/New_York")
    sources["yahoo"]["GC=F"] = tz
    out = macro.load_macro_panel()

    assert pd.Timestamp("1999-01-18") in out.index  # other sources still trade that day
    assert out.loc["1999-01-18", "vix"] == vix.loc["1999-01-15"]
    last = dxy.loc["1999-02-26", "adj_close"]
    assert out.loc[gap[:3], "dxy"].tolist() == [last] * 3
    assert out.loc[gap[3:], "dxy"].isna().all()  # stale beyond the limit stays NaN
    assert out.loc["1999-03-08", "dxy"] == dxy.loc["1999-03-08", "adj_close"]
    assert out.loc["1999-03-08", "gold"] == gold.loc["1999-03-08", "adj_close"]
    assert params.REGISTRY["MACRO_FFILL_LIMIT"].value == 3


def test_panel_ffill_never_runs_past_source_end(sources):
    vix = sources["cboe"]["VIX"]
    sources["cboe"]["VIX"] = vix.loc[:"2000-06-30"]
    out = macro.load_macro_panel()
    tail = out.loc["2000-06-30":, "vix"]
    assert tail.iloc[:4].tolist() == [vix.loc["2000-06-30"]] * 4
    assert tail.iloc[4:].isna().all()


def test_panel_holdout_guard(sources, monkeypatch):
    cut = settings().holdout_start
    idx = _bdays(cut - pd.Timedelta(days=30), cut + pd.Timedelta(days=30))
    sources["cboe"]["VIX"] = pd.concat([sources["cboe"]["VIX"], _series(idx, 50)])
    out = macro.load_macro_panel()
    assert out.index.max() < cut
    assert out.loc[out.index >= idx[0], "vix"].notna().all()
    assert sources["calls"][("tsy", None)] == 1

    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    unlocked = macro.load_macro_panel()
    assert unlocked.index.max() == idx[-1]
    assert holdout.access_count() == 1


def test_panel_failed_sources_become_nan_with_warning(sources):
    sources["cboe"]["VIX9D"] = ConnectionError("cboe down")
    sources["yahoo"]["LQD"] = None
    sources["yahoo"]["GC=F"] = pd.DataFrame({"volume": [1.0]}, index=pd.to_datetime(["1999-01-04"]))
    sources["yahoo"]["CL=F"] = RuntimeError("boom")
    sources["yahoo"]["EURUSD=X"] = sources["yahoo"]["EURUSD=X"].drop(columns="open")
    with pytest.warns(macro.MacroSourceWarning) as rec:
        out = macro.load_macro_panel()
    msgs = " | ".join(str(w.message) for w in rec)
    for col in ("vix9d", "lqd", "gold", "oil", "eurusd"):
        assert col in msgs
        assert out[col].isna().all()
    assert list(out.columns) == macro.MACRO_COLUMNS
    assert out[["vix", "hyg", "y10"]].notna().all().all()


def test_panel_treasury_failure_and_all_nan_source_warn(sources):
    sources["tsy"] = RuntimeError("treasury down")
    sources["yahoo"]["DX-Y.NYB"] = pd.DataFrame({"adj_close": [1.0]}, index=pd.to_datetime(["1990-01-02"]))
    with pytest.warns(macro.MacroSourceWarning) as rec:
        out = macro.load_macro_panel()
    msgs = " | ".join(str(w.message) for w in rec)
    assert "treasury" in msgs and "dxy" in msgs
    assert out[macro.TREASURY_COLUMNS + ["dxy"]].isna().all().all()
    assert out["vix"].notna().all()


def test_panel_all_sources_fail_returns_empty_frame(sources):
    for k in sources["cboe"]:
        sources["cboe"][k] = ConnectionError("down")
    for k in sources["yahoo"]:
        sources["yahoo"][k] = None
    sources["tsy"] = ConnectionError("down")
    with pytest.warns(macro.MacroSourceWarning):
        out = macro.load_macro_panel()
    assert out.empty and list(out.columns) == macro.MACRO_COLUMNS
    assert isinstance(out.index, pd.DatetimeIndex)


def test_panel_passes_refresh_through(sources):
    macro.load_macro_panel(refresh=True)
    assert sources["calls"][("cboe", "VIX", True)] == 1
    assert sources["calls"][("yahoo", "HYG", True)] == 1


def test_panel_rows_unchanged_by_future_data(sources):
    base = macro.load_macro_panel()
    cutoff = pd.Timestamp("2000-06-15")
    for name, s in sources["cboe"].items():
        sources["cboe"][name] = s.where(s.index <= cutoff, s * 7.0)
    tsy = sources["tsy"].copy()
    tsy.loc[tsy.index > cutoff] = -1.0
    sources["tsy"] = tsy
    for tic, d in sources["yahoo"].items():
        # An FX bar labelled D is the ~19:00 ET snapshot of D-1, so its information time is one day earlier.
        known = cutoff + pd.Timedelta(days=1 if tic in FX_TICKERS else 0)
        after = d.index > known
        d = d.loc[~after | (d.index > "2000-07-15")].copy()  # also open a gap
        d.loc[d.index > known] *= 7.0
        sources["yahoo"][tic] = d
    perturbed = macro.load_macro_panel()
    pd.testing.assert_frame_equal(base.loc[:cutoff], perturbed.loc[:cutoff])
    assert not base.loc[cutoff:].equals(perturbed.loc[cutoff:])


@pytest.mark.network
def test_live_cboe_vix_smoke():
    vix = macro.load_cboe_index("VIX", refresh=True)
    assert len(vix) > 8000 and vix.index.min() <= pd.Timestamp("1990-01-31")
    assert vix.between(5, 100).all()

import json

import numpy as np
import pandas as pd
import pytest
import requests

from omodel.data import store, yahoo

# 09:30 New York (EDT) bar stamps, as Yahoo returns them.
AUG26, AUG27, AUG28, AUG31, SEP01 = 1598448600, 1598535000, 1598621400, 1598880600, 1598967000
AUG31_LIVE = AUG31 + 4 * 3600  # second, later bar for the same session (Yahoo's live-bar duplicate)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def chart(ts, close, adj=None, opens=None, splits=None, tz="America/New_York", **quote):
    n = len(ts)
    q = {"open": opens or [c and c - 1 for c in close], "high": [c and c + 1 for c in close],
         "low": [c and c - 2 for c in close], "close": close, "volume": [1_000_000 + i for i in range(n)], **quote}
    ind = {"quote": [q]}
    if adj is not None:
        ind["adjclose"] = [{"adjclose": adj}]
    res = {"meta": {"symbol": "X", "exchangeTimezoneName": tz} if tz else {"symbol": "X"},
           "timestamp": ts, "indicators": ind}
    if splits:
        res["events"] = {"splits": splits}
    return {"chart": {"result": [res], "error": None}}


SPLIT_4_1 = {str(AUG31): {"date": AUG31, "numerator": 4.0, "denominator": 1.0, "splitRatio": "4:1"}}
REALISTIC = chart(
    [AUG26, AUG27, AUG28, AUG31, AUG31_LIVE, SEP01],
    close=[124.8, 125.0, None, 129.0, 129.5, 134.2],
    adj=[121.0, 121.2, None, 125.1, 125.6, 130.0],
    splits=SPLIT_4_1,
)


@pytest.fixture
def calls(monkeypatch):
    log = []

    def install(response):
        def fake_get(url, **kwargs):
            log.append((url, kwargs))
            return response(url) if callable(response) else response
        monkeypatch.setattr(yahoo.http, "get", fake_get)
        return log
    return install


def test_yahoo_symbol_mapping():
    assert yahoo.yahoo_symbol("BRK.B") == "BRK-B"
    assert yahoo.yahoo_symbol("BF/B") == "BF-B"
    assert yahoo.yahoo_symbol("AAPL") == "AAPL"
    assert yahoo.yahoo_symbol("^VIX") == "^VIX"
    assert yahoo.yahoo_symbol("CL=F") == "CL=F"
    assert yahoo.yahoo_symbol("DX-Y.NYB") == "DX-Y.NYB"


def test_fetch_daily_request_params(calls):
    log = calls(FakeResponse(payload=REALISTIC))
    yahoo.fetch_daily("BRK.B", pd.Timestamp("2020-08-26"), pd.Timestamp("2020-09-01"))
    (url, kwargs), = log
    assert url == "https://query1.finance.yahoo.com/v8/finance/chart/BRK-B"
    p = kwargs["params"]
    assert p["period1"] == 1598400000  # 2020-08-26 00:00 UTC
    assert p["period2"] == 1599004800  # 2020-09-02 00:00 UTC (end + 1 day)
    assert p["interval"] == "1d" and p["events"] == "div,split" and p["includeAdjustedClose"] == "true"
    assert "range" not in p
    assert kwargs["min_interval"] == 0.35


def test_fetch_daily_parses_realistic_chart(calls):
    calls(FakeResponse(payload=REALISTIC))
    df = yahoo.fetch_daily("AAPL", pd.Timestamp("2020-08-26"), pd.Timestamp("2020-09-01"))
    expected_dates = pd.DatetimeIndex(["2020-08-26", "2020-08-27", "2020-08-31", "2020-09-01"], name="date")
    pd.testing.assert_index_equal(df.index, expected_dates)
    assert df.index.tz is None and df.index.is_unique and df.index.is_monotonic_increasing
    assert list(df.columns[:6]) == list(yahoo.PRICE_COLUMNS)
    assert all(df[c].dtype == np.float64 for c in df.columns)
    # Duplicate session keeps the later (live) bar; the None-close day is dropped.
    assert df.loc["2020-08-31", "close"] == 129.5 and df.loc["2020-08-31", "adj_close"] == 125.6
    assert df.loc["2020-08-27", "adj_close"] == 121.2
    assert df.loc["2020-08-27", "volume"] == 1_000_001
    # 4:1 split ex-date 2020-08-31: earlier bars were back-adjusted by Yahoo, so as-traded = close * 4.
    np.testing.assert_array_equal(df["split_factor"].to_numpy(), [4.0, 4.0, 1.0, 1.0])


def test_as_traded_close_invariant_to_later_split(calls):
    calls(FakeResponse(payload=chart([AUG26, AUG27], close=[100.0, 102.0], adj=[99.0, 101.0])))
    before = yahoo.fetch_daily("X", pd.Timestamp("2020-08-26"), pd.Timestamp("2020-08-27"))
    later = chart([AUG26, AUG27, AUG31], close=[50.0, 51.0, 52.0], adj=[49.5, 50.5, 52.0],
                  splits={str(AUG31): {"date": AUG31, "numerator": 2, "denominator": 1}})
    calls(FakeResponse(payload=later))
    after = yahoo.fetch_daily("X", pd.Timestamp("2020-08-26"), pd.Timestamp("2020-08-31"))
    as_traded = (after["close"] * after["split_factor"]).loc[before.index]
    np.testing.assert_allclose(as_traded.to_numpy(), before["close"].to_numpy())


def test_reverse_split_factor(calls):
    rev = {str(AUG31): {"date": AUG31, "numerator": 1.0, "denominator": 10.0, "splitRatio": "1:10"}}
    calls(FakeResponse(payload=chart([AUG28, AUG31], close=[30.0, 31.0], splits=rev)))
    df = yahoo.fetch_daily("X", pd.Timestamp("2020-08-28"), pd.Timestamp("2020-08-31"))
    np.testing.assert_allclose(df["split_factor"].to_numpy(), [0.1, 1.0])


def test_adj_close_defaults_to_close_and_drops_nonpositive(calls):
    calls(FakeResponse(payload=chart([AUG26, AUG27, AUG28], close=[10.0, 0.0, -1.0])))
    df = yahoo.fetch_daily("X", pd.Timestamp("2020-08-26"), pd.Timestamp("2020-08-28"))
    assert list(df.index) == [pd.Timestamp("2020-08-26")]
    assert df["adj_close"].iloc[0] == 10.0


def test_timestamps_use_exchange_timezone(calls):
    late = 1598574600  # 2020-08-28 00:30 UTC = 2020-08-27 20:30 New York = 2020-08-28 09:30 Tokyo
    calls(FakeResponse(payload=chart([late], close=[10.0])))
    assert yahoo.fetch_daily("X", "2020-08-27", "2020-08-28").index[0] == pd.Timestamp("2020-08-27")
    calls(FakeResponse(payload=chart([late], close=[10.0], tz=None)))
    assert yahoo.fetch_daily("X", "2020-08-27", "2020-08-28").index[0] == pd.Timestamp("2020-08-27")
    calls(FakeResponse(payload=chart([late], close=[10.0], tz="Asia/Tokyo")))
    assert yahoo.fetch_daily("X", "2020-08-27", "2020-08-28").index[0] == pd.Timestamp("2020-08-28")


@pytest.mark.parametrize("response", [
    FakeResponse(404, {"chart": {"result": None, "error": {"code": "Not Found",
                                                           "description": "No data found, symbol may be delisted"}}}),
    FakeResponse(200, {"chart": {"result": None, "error": {"code": "Not Found", "description": "x"}}}),
    FakeResponse(400, {"chart": {"result": None, "error": {"code": "Bad Request",
                                                           "description": "Data doesn't exist"}}}),
    FakeResponse(200, {"chart": {"result": [], "error": None}}),
    FakeResponse(200, {"chart": {"result": [{"meta": {}, "indicators": {"quote": [{}]}}], "error": None}}),
])
def test_fetch_daily_no_data_returns_none(calls, response):
    calls(response)
    assert yahoo.fetch_daily("GONE", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")) is None


@pytest.mark.parametrize("status", [401, 429, 500])
def test_fetch_daily_transient_failures_raise(calls, status):
    calls(FakeResponse(status, {"chart": {"result": None, "error": {"code": "Unauthorized",
                                                                    "description": "Invalid Crumb"}}}))
    with pytest.raises(requests.HTTPError):
        yahoo.fetch_daily("AAPL", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"))


@pytest.mark.parametrize("response", [
    FakeResponse(404, None, text="<html>Not Found</html>"),           # CDN / outage page, not Yahoo's verdict
    FakeResponse(404, {"chart": {"result": None, "error": None}}),
    FakeResponse(200, None, text="<html>maintenance</html>"),
])
def test_unexplained_failures_raise_and_are_never_cached_as_missing(calls, response):
    calls(response)
    with pytest.raises((requests.HTTPError, ValueError)):
        yahoo.load_daily("AAPL")
    assert store.read_json("prices_missing", "AAPL") is None


def test_yahoo_not_found_is_cached_as_missing(calls):
    log = calls(FakeResponse(404, {"chart": {"result": None, "error": {"code": "Not Found", "description": "x"}}}))
    assert yahoo.load_daily("GONE") is None and yahoo.load_daily("GONE") is None
    assert len(log) == 1 and store.read_json("prices_missing", "GONE")["missing"] is True


def _fake_fetch(monkeypatch, result):
    log = []

    def fake(ticker, start, end):
        log.append((ticker, start, end))
        return result
    monkeypatch.setattr(yahoo, "fetch_daily", fake)
    return log


def test_load_daily_caches(monkeypatch):
    df = pd.DataFrame({c: [1.0, 2.0] for c in yahoo.COLUMNS},
                      index=pd.DatetimeIndex(["2020-01-02", "2020-01-03"], name="date"))
    log = _fake_fetch(monkeypatch, df)
    out = yahoo.load_daily("BRK.B")
    pd.testing.assert_frame_equal(out, df)
    (_, start, end), = log
    assert start == pd.Timestamp("1998-01-01") and end == pd.Timestamp.today().normalize()
    pd.testing.assert_frame_equal(yahoo.load_daily("BRK.B"), df, check_freq=False)
    assert len(log) == 1
    assert store.read_meta("prices", "BRK.B")["symbol"] == "BRK-B"
    yahoo.load_daily("BRK.B", refresh=True)
    assert len(log) == 2


def test_load_daily_missing_is_remembered(monkeypatch):
    log = _fake_fetch(monkeypatch, None)
    assert yahoo.load_daily("GONE") is None
    marker = store.read_json("prices_missing", "GONE")
    assert marker["missing"] is True and pd.Timestamp(marker["checked"]).tzinfo is not None
    assert yahoo.load_daily("GONE") is None
    assert len(log) == 1
    df = pd.DataFrame({c: [1.0] for c in yahoo.COLUMNS}, index=pd.DatetimeIndex(["2020-01-02"], name="date"))
    _fake_fetch(monkeypatch, df)
    assert yahoo.load_daily("GONE", refresh=True) is not None
    assert store.read_json("prices_missing", "GONE") is None


def test_load_many(monkeypatch, capsys):
    tickers = [f"T{i:03d}" for i in range(120)]
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"], name="date"))

    def fake(ticker, refresh=False):
        if ticker == "T007":
            raise ConnectionError("boom")
        return None if ticker.endswith("5") else df
    monkeypatch.setattr(yahoo, "load_daily", fake)
    out = yahoo.load_many(tickers + ["T000"], max_workers=3)
    assert list(out) == [t for t in tickers if not t.endswith("5") and t != "T007"]
    err = capsys.readouterr().err
    assert "50/120" in err and "100/120" in err
    assert "107/120 with data, 12 missing, 1 failed" in err and "T007" in err


def test_load_many_strict_raises_after_trying_everything(monkeypatch):
    tried = []
    df = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"], name="date"))

    def fake(ticker, refresh=False):
        tried.append(ticker)
        if ticker in ("D", "B"):
            raise ConnectionError("429")
        return None if ticker == "C" else df
    monkeypatch.setattr(yahoo, "load_daily", fake)
    with pytest.raises(yahoo.LoadError, match="B, D") as exc:
        yahoo.load_many(["A", "B", "C", "D", "E"], max_workers=2, strict=True)
    assert exc.value.failed == ["B", "D"] and sorted(tried) == ["A", "B", "C", "D", "E"]
    monkeypatch.setattr(yahoo, "load_daily", lambda t, refresh=False: None if t == "C" else df)
    assert list(yahoo.load_many(["A", "C"], strict=True)) == ["A"]  # a genuine no-data ticker is not a failure


@pytest.mark.network
def test_live_spy_and_membership():
    from omodel.data import universe
    df = yahoo.fetch_daily("SPY", pd.Timestamp("2020-08-24"), pd.Timestamp("2020-09-04"))
    assert len(df) == 10 and df.index[0] == pd.Timestamp("2020-08-24")
    assert (df["adj_close"] <= df["close"]).all()
    m = universe.load_membership(refresh=True)
    assert {"AAPL", "MSFT"} <= set(universe.tickers_between(m, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")))

import logging
import math
import os
import time

import pandas as pd
import pytest

from omodel import config, holdout
from omodel.data import cboe_options as cboe
from omodel.data import store

NOW = pd.Timestamp("2026-10-24 12:00:00", tz="UTC")
CLOSE_TS = "2026-10-22 20:45:00"  # 16:45 ET: file regenerated after the 10-22 close
CLOSE_TRADE = "2026-10-22T15:59:59"


class FakeResp:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clock_and_dev_holdout(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(cboe, "_now", lambda: NOW)
    # The test dates are real 2026 dates, i.e. inside the project holdout; move it out of the way.
    config.override(holdout_start=pd.Timestamp("2027-06-01"))


def _payload(symbol="AAPL", timestamp=CLOSE_TS, price=346.41, bid_shift=0.0, last_trade=CLOSE_TRADE):
    options = [
        {"option": f"{symbol}261023C00110000", "bid": 236.65 + bid_shift, "bid_size": 170.0, "ask": 239.3,
         "ask_size": 2.0, "iv": 0.0, "open_interest": 2.0, "volume": 0.0, "delta": 0.9999, "gamma": 0.0,
         "vega": 0.0, "theta": 0.0, "rho": 0.0085, "theo": 238.3181, "last_trade_price": 235.0,
         "last_trade_time": "2026-10-21T13:42:45"},
        {"option": f"{symbol}270115P00540000", "bid": 0.0, "bid_size": 0.0, "ask": 0.05, "ask_size": 10.0,
         "iv": 0.3216, "open_interest": 768.0, "volume": 4.0, "delta": -0.003, "gamma": 0.0002,
         "vega": 0.0166, "theta": -0.0026, "rho": 0.0023, "theo": 0.0427, "last_trade_price": 0.08,
         "last_trade_time": None},
        {"option": f"{symbol}270115C00540000", "bid": 0.0, "ask": 0.0, "iv": None, "open_interest": 0.0,
         "volume": 0.0, "last_trade_price": 0.0, "last_trade_time": None},
    ]
    data = {"symbol": symbol, "current_price": price, "iv30": 27.234, "options": options}
    if last_trade is not None:
        data["last_trade_time"] = last_trade
    return {"timestamp": timestamp, "data": data}


def _fake_http(monkeypatch, responses: dict[str, FakeResp]) -> list[str]:
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        symbol = url.rsplit("/", 1)[-1].removesuffix(".json")
        return responses.get(symbol, FakeResp(403))

    monkeypatch.setattr(cboe.http, "get", fake_get)
    return calls


def _path(date: str, key: str = "AAPL"):
    return store.path_for(f"options_snapshots/{date}", key)


@pytest.mark.parametrize("symbol, expected", [
    ("AAPL261023C00110000", ("AAPL", pd.Timestamp("2026-10-23"), "C", 110.0)),
    ("AAPL1261023C00110000", ("AAPL1", pd.Timestamp("2026-10-23"), "C", 110.0)),
    ("SPXW261120P05912500", ("SPXW", pd.Timestamp("2026-11-20"), "P", 5912.5)),
    ("BRKB270115P00412500", ("BRKB", pd.Timestamp("2027-01-15"), "P", 412.5)),
    ("F261023C00000500", ("F", pd.Timestamp("2026-10-23"), "C", 0.5)),
    ("AAPL  261023C00110000", ("AAPL", pd.Timestamp("2026-10-23"), "C", 110.0)),
])
def test_parse_occ(symbol, expected):
    assert cboe.parse_occ(symbol) == expected


@pytest.mark.parametrize("bad", ["AAPL", "261023C00110000", "AAPL261023X00110000", "AAPL261023C0011000",
                                 "AAPL261340C00110000"])
def test_parse_occ_rejects_malformed(bad):
    with pytest.raises(ValueError):
        cboe.parse_occ(bad)


def test_fetch_chain_parses_payload(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    df = cboe.fetch_chain("aapl")
    assert list(df.columns[:len(cboe.COLUMNS)]) == cboe.COLUMNS
    assert len(df) == 3
    row = df.set_index("contract").loc["AAPL261023C00110000"]
    assert (row["root"], row["cp"], row["strike"]) == ("AAPL", "C", 110.0)
    assert row["expiry"] == pd.Timestamp("2026-10-23")
    assert row["mid"] == pytest.approx((236.65 + 239.3) / 2)
    assert row["spread_pct"] == pytest.approx((239.3 - 236.65) / row["mid"])
    assert math.isnan(row["iv"])  # CBOE's iv=0 means "no iv"
    assert row["last_trade_time"] == pd.Timestamp("2026-10-21 13:42:45", tz=cboe.EXCHANGE_TZ)
    assert df["standard"].all()
    assert df.attrs["underlying"] == "AAPL"
    assert df.attrs["underlying_price"] == pytest.approx(346.41)
    assert df.attrs["iv30"] == pytest.approx(0.27234)
    assert df.attrs["asof"] == pd.Timestamp(CLOSE_TS, tz="UTC")
    assert df.attrs["quote_time"] == pd.Timestamp(CLOSE_TRADE, tz=cboe.EXCHANGE_TZ)
    assert df.attrs["session"] == pd.Timestamp("2026-10-22") and df.attrs["session_complete"]
    assert list(df["expiry"]) == sorted(df["expiry"])


def test_zero_bids_and_asks_stay_zero(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    df = cboe.fetch_chain("AAPL").set_index("contract")
    one_sided = df.loc["AAPL270115P00540000"]
    assert one_sided["bid"] == 0.0 and one_sided["mid"] == pytest.approx(0.025)
    assert one_sided["spread_pct"] == pytest.approx(2.0)
    assert pd.isna(one_sided["last_trade_time"])
    dead = df.loc["AAPL270115C00540000"]
    assert dead["bid"] == 0.0 and dead["ask"] == 0.0 and dead["mid"] == 0.0
    assert math.isnan(dead["spread_pct"])
    assert math.isnan(dead["delta"])  # missing greek -> NaN, not 0


@pytest.mark.parametrize("bid, ask", [(2.0, 1.0), (1.0, 0.0)])
def test_crossed_or_askless_quotes_have_no_spread(monkeypatch, bid, ask):
    payload = _payload()
    payload["data"]["options"][0].update(bid=bid, ask=ask)
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, payload)})
    row = cboe.fetch_chain("AAPL").set_index("contract").loc["AAPL261023C00110000"]
    assert (row["bid"], row["ask"]) == (bid, ask)
    assert math.isnan(row["spread_pct"])


def test_adjusted_contracts_are_flagged_non_standard(monkeypatch):
    payload = _payload()
    payload["data"]["options"].append({"option": "AAPL1261023C00110000", "bid": 1.0, "ask": 1.2})
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, payload)})
    df = cboe.fetch_chain("AAPL").set_index("contract")
    assert not df.loc["AAPL1261023C00110000", "standard"]
    assert df.drop(index="AAPL1261023C00110000")["standard"].all()


def test_asof_falls_back_to_fetch_time(monkeypatch):
    payload = _payload()
    del payload["timestamp"]
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, payload)})
    assert cboe.fetch_chain("AAPL").attrs["asof"] == NOW


@pytest.mark.parametrize("raw", [1761165000000, 1761165000, "1999-01-01 00:00:00", "2026-10-25 12:00:00"])
def test_implausible_asof_falls_back_to_fetch_time(monkeypatch, raw):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload(timestamp=raw))})
    assert cboe.fetch_chain("AAPL").attrs["asof"] == NOW  # never 1970, never the future


def test_quote_time_after_payload_time_is_ignored(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload(last_trade="2026-10-22T23:00:00"))})
    attrs = cboe.fetch_chain("AAPL").attrs
    assert pd.isna(attrs["quote_time"]) and not attrs["session_complete"]


def test_unparseable_contracts_are_dropped(monkeypatch):
    payload = _payload()
    payload["data"]["options"].append({"option": "GARBAGE", "bid": 1.0, "ask": 2.0})
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, payload)})
    assert len(cboe.fetch_chain("AAPL")) == 3


@pytest.mark.parametrize("given", ["BRK.B", "BRK-B", "brk/b"])
def test_class_share_fallback(monkeypatch, given):
    calls = _fake_http(monkeypatch, {"BRK.B": FakeResp(404), "BRKB": FakeResp(200, _payload("BRKB"))})
    df = cboe.fetch_chain(given)
    assert [c.rsplit("/", 1)[-1] for c in calls] == ["BRK.B.json", "BRKB.json"]
    assert df.attrs["underlying"] == "BRK.B" and df.attrs["cboe_symbol"] == "BRKB"
    assert set(df["root"]) == {"BRKB"}
    assert df["standard"].all()


def test_class_share_prefers_dotted_symbol(monkeypatch):
    calls = _fake_http(monkeypatch, {"BRK.B": FakeResp(200, _payload("BRKB"))})
    df = cboe.fetch_chain("BRK-B")
    assert len(calls) == 1
    assert df["standard"].all()  # 'BRK.B' answered with root 'BRKB': still the standard deliverable


@pytest.mark.parametrize("resp", [FakeResp(404), FakeResp(403), FakeResp(200, {"data": {"options": []}}),
                                  FakeResp(200, {"timestamp": CLOSE_TS}), FakeResp(200, [])])
def test_missing_or_empty_chain_returns_none(monkeypatch, resp):
    _fake_http(monkeypatch, {"ZZZZ": resp})
    assert cboe.fetch_chain("ZZZZ") is None


def test_server_error_raises(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(503)})
    with pytest.raises(RuntimeError):
        cboe.fetch_chain("AAPL")


def test_snapshot_is_idempotent_and_roundtrips(monkeypatch):
    calls = _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload()), "MSFT": FakeResp(200, _payload("MSFT"))})
    when = pd.Timestamp("2026-10-22")
    paths = cboe.snapshot(["AAPL", "MSFT", "aapl"], when)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    assert paths[0] == _path("2026-10-22")
    n_calls = len(calls)
    assert cboe.snapshot(["AAPL", "MSFT"], when) == []
    assert len(calls) == n_calls  # already-snapshotted tickers are not re-fetched

    df = cboe.load_snapshot("AAPL", when)
    fresh = cboe.fetch_chain("AAPL")
    pd.testing.assert_frame_equal(df, fresh)
    assert df.attrs == fresh.attrs
    assert not list(store.path_for("x", "x").parent.parent.glob("options_snapshots/2026-10-22/*.lock"))


def test_snapshot_continues_past_failures(monkeypatch, caplog):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload()), "BAD": FakeResp(500)})
    with caplog.at_level(logging.WARNING, logger=cboe.__name__):
        paths = cboe.snapshot(["BAD", "ZZZZ", "AAPL"], pd.Timestamp("2026-10-22"))
    assert paths == [_path("2026-10-22")]
    warned = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ZZZZ" in r.getMessage() for r in warned)  # not-found is visible, not INFO
    assert any("'not_found': 1" in r.getMessage() and "'failed': 1" in r.getMessage() for r in warned)
    assert any(r.levelno == logging.ERROR and "throttled" in r.getMessage() for r in warned)


def test_pre_open_run_files_chain_under_its_trading_session(monkeypatch):
    # Live shape: at 08:53 ET on 10-23 the CDN file was regenerated at 06:02 ET but holds the 10-22 close.
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-23 12:53:00", tz="UTC"))
    pre_open = _payload(timestamp="2026-10-23 10:02:33", last_trade="2026-10-22T15:59:59")
    calls = _fake_http(monkeypatch, {"AAPL": FakeResp(200, pre_open)})
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-23")) == []
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]
    assert not _path("2026-10-23").exists()
    n_calls = len(calls)
    assert cboe.snapshot(["AAPL"]) == []  # 10-22 already stored: skip
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-22")) == []
    assert len(calls) == n_calls + 1  # the dated run skips without fetching
    loaded = cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22"))
    assert loaded["last_trade_time"].max().date() <= pd.Timestamp("2026-10-22").date()
    assert loaded.attrs["quote_time"] == pd.Timestamp("2026-10-22 15:59:59", tz=cboe.EXCHANGE_TZ)
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-23")) is None


def test_existing_session_file_skips_pre_open_run(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    cboe.snapshot(["AAPL"])
    before = _path("2026-10-22").stat().st_mtime_ns
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-23 12:53:00", tz="UTC"))
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload(timestamp="2026-10-23 10:02:33", price=1.0))})
    assert cboe.snapshot(["AAPL"]) == []
    assert _path("2026-10-22").stat().st_mtime_ns == before


def test_intraday_chain_is_not_locked_in(monkeypatch):
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-22 15:00:00", tz="UTC"))
    intraday = _payload(timestamp="2026-10-22 15:00:00", last_trade="2026-10-22T10:44:59", price=1.0)
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, intraday)})
    assert cboe.snapshot(["AAPL"]) == []
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-22")) == []
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-22 21:00:00", tz="UTC"))
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]
    loaded = cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22"))
    assert loaded.attrs["underlying_price"] == pytest.approx(346.41) and loaded.attrs["session_complete"]


def test_intraday_chain_stored_only_on_request_and_marked(monkeypatch):
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-22 15:00:00", tz="UTC"))
    intraday = _payload(timestamp="2026-10-22 15:00:00", last_trade="2026-10-22T10:44:59")
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, intraday)})
    assert cboe.snapshot(["AAPL"], require_close=False) == [_path("2026-10-22")]
    attrs = cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")).attrs
    assert attrs["session_complete"] is False
    assert attrs["quote_time"] == pd.Timestamp("2026-10-22 10:44:59", tz=cboe.EXCHANGE_TZ)


def test_half_day_close_is_captured_after_the_feed_settles(monkeypatch):
    # 2026-11-27 closes at 13:00 ET (EST, UTC-5): the last print is 12:59:59, never 15:59.
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-11-27 18:30:00", tz="UTC"))
    early = _payload(timestamp="2026-11-27 18:30:00", last_trade="2026-11-27T12:59:59")
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, early)})
    assert cboe.snapshot(["AAPL"]) == []  # 13:30 ET: cannot tell a half day from a stalled feed yet
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-11-28 12:00:00", tz="UTC"))
    settled = _payload(timestamp="2026-11-28 11:02:00", last_trade="2026-11-27T12:59:59")
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, settled)})
    assert cboe.snapshot(["AAPL"]) == [_path("2026-11-27")]


def test_chain_without_underlying_trade_time_is_not_dated(monkeypatch):
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-23 12:53:00", tz="UTC"))
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload(timestamp="2026-10-23 10:02:33", last_trade=None))})
    assert cboe.snapshot(["AAPL"]) == []
    assert not _path("2026-10-23").exists()


def test_stale_underlying_is_not_backfilled(monkeypatch):
    halted = _payload(timestamp="2026-10-23 10:00:00", last_trade="2026-10-16T15:59:59")
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, halted)})
    assert cboe.snapshot(["AAPL"]) == []
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-16"), require_close=False) == []
    assert not _path("2026-10-16").exists()


def test_snapshot_dates_by_exchange_date(monkeypatch):
    # 02:00 UTC on the 23rd is 22:00 ET on the 22nd: the quotes belong to the 22nd.
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload(timestamp="2026-10-23 02:00:00"))})
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp("2026-10-23 02:05:00", tz="UTC"))
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]
    assert cboe.snapshot(["AAPL"]) == []
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is not None


def test_snapshot_refuses_quotes_from_another_date(monkeypatch):
    later = _payload(timestamp="2026-10-23 20:45:00", last_trade="2026-10-23T15:59:59")
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, later)})
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-22")) == []  # future quotes under a past date
    assert cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-26")) == []  # stale quotes under a later date
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is None
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-26")) is None


def test_later_snapshots_never_change_earlier_ones(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-22"))
    before = cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22"))
    later = _payload(timestamp="2026-10-23 20:45:00", last_trade="2026-10-23T15:59:59", price=999.0,
                     bid_shift=50.0)
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, later)})
    assert len(cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-23"))) == 1
    assert len(cboe.snapshot(["AAPL"], pd.Timestamp("2026-10-22"))) == 0
    after = cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22"))
    pd.testing.assert_frame_equal(before, after)
    assert after.attrs == before.attrs
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-23")).attrs["underlying_price"] == 999.0


def test_concurrent_run_holding_the_lock_is_not_overwritten(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    lock = store.path_for("options_snapshots/2026-10-22", "AAPL", "lock")
    lock.parent.mkdir(parents=True)
    lock.touch()
    assert cboe.snapshot(["AAPL"]) == []
    assert not _path("2026-10-22").exists()
    old = time.time() - 3600
    os.utime(lock, (old, old))  # a crashed run's lock must not block the key forever
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]
    assert not lock.exists()


def test_interrupted_write_is_invisible_and_repaired(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    cboe.snapshot(["AAPL"])
    _path("2026-10-22").with_suffix(".meta.json").unlink()  # crash between the two renames
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is None
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is not None
    assert not list(store.path_for("x", "x").parent.parent.glob("options_snapshots_staging/*/*"))


def test_load_snapshot_missing_returns_none():
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is None


def test_holdout_snapshots_need_unlock_and_are_audited(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    config.override(holdout_start=pd.Timestamp("2026-10-01"))
    assert cboe.snapshot(["AAPL"]) == [_path("2026-10-22")]  # collecting into the holdout is allowed
    with pytest.raises(holdout.HoldoutViolation):
        cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22"))
    assert holdout.access_count() == 0
    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is not None
    assert holdout.access_count() == 1


def test_pre_holdout_snapshot_loads_without_audit(monkeypatch):
    _fake_http(monkeypatch, {"AAPL": FakeResp(200, _payload())})
    cboe.snapshot(["AAPL"])
    config.override(holdout_start=pd.Timestamp("2026-10-23"))
    assert cboe.load_snapshot("AAPL", pd.Timestamp("2026-10-22")) is not None
    assert holdout.access_count() == 0


@pytest.mark.network
def test_live_fetch_chain(monkeypatch):
    monkeypatch.setattr(cboe, "_now", lambda: pd.Timestamp.now(tz="UTC"))
    df = cboe.fetch_chain("AAPL")
    assert df is not None and len(df) > 100
    assert set(cboe.COLUMNS) <= set(df.columns)
    assert df.attrs["underlying_price"] > 0
    assert pd.notna(df.attrs["quote_time"]) and df.attrs["quote_time"] <= df.attrs["asof"]

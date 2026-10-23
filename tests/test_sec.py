import copy

import numpy as np
import pandas as pd
import pytest
import requests

from omodel import config
from omodel.config import settings
from omodel.data import sec, store

ET = "America/New_York"
HOLIDAYS = pd.to_datetime(["2003-05-26", "2003-07-04", "2024-01-15", "2024-07-04", "2024-11-28", "2024-12-25",
                           "2025-01-01"])
TD = pd.bdate_range("2003-01-02", "2025-02-28").difference(HOLIDAYS)


class FakeResp:
    def __init__(self, payload=None, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return copy.deepcopy(self.payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def fake_http(monkeypatch):
    config.override(sec_user_agent="Unit Test unit@example.com")
    routes, calls = {}, []

    def get(url, *, headers=None, min_interval=0.0, **kw):
        calls.append({"url": url, "headers": headers, "min_interval": min_interval})
        return FakeResp(routes[url]) if url in routes else FakeResp(status=404)

    monkeypatch.setattr(sec.http, "get", get)
    return routes, calls


def _stamp(wall_et: str, shift: int) -> str:
    ts = pd.Timestamp(wall_et)
    for _ in range(shift):
        ts = ts.tz_localize(ET).tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _filing_date(wall_et: str, form: str) -> str:
    ts = pd.Timestamp(wall_et)
    cutoff = 22.0 if form in ("4", "144") else 17.5
    hour = ts.hour + ts.minute / 60
    return (ts.normalize() + (pd.offsets.BDay(0) if hour <= cutoff else pd.offsets.BDay(1))).strftime("%Y-%m-%d")


def _page(rows, shift):
    """rows: (accession, form, ET wall time or None, items, filing date override)."""
    return {
        "accessionNumber": [r[0] for r in rows],
        "form": [r[1] for r in rows],
        "filingDate": [r[4] or _filing_date(r[2], r[1]) for r in rows],
        "acceptanceDateTime": ["" if r[2] is None else _stamp(r[2], shift) for r in rows],
        "items": [r[3] for r in rows],
        "reportDate": ["" for _ in rows],
    }


def _realistic_rows(prefix="24"):
    rows, i = [], 0
    for day in list(pd.bdate_range("2024-01-08", periods=6)) + list(pd.bdate_range("2024-07-08", periods=6)):
        d = day.strftime("%Y-%m-%d")
        for t, form, items in (("06:01", "10-Q", ""), ("09:15", "8-K", "5.02"), ("16:30", "8-K", "2.02,9.01"),
                               ("17:45", "8-K", "8.01"), ("18:30", "4", "")):
            rows.append((f"0000000001-{prefix}-{i:06d}", form, f"{d} {t}:07", items, None))
            i += 1
    return rows


def _true_utc(wall_et: str) -> pd.Timestamp:
    return pd.Timestamp(wall_et).tz_localize(ET).tz_convert("UTC").tz_localize(None)


# --- configuration / HTTP ---------------------------------------------------------------------

def test_missing_user_agent_raises(monkeypatch):
    config.override(sec_user_agent="")
    monkeypatch.setattr(sec.http, "get", lambda *a, **k: pytest.fail("network must not be called"))
    with pytest.raises(sec.SecConfigError, match="SEC_USER_AGENT"):
        sec.ticker_to_cik()
    with pytest.raises(sec.SecConfigError):
        sec.load_submissions(320193)


def test_ticker_to_cik_variants_and_cache(fake_http):
    routes, calls = fake_http
    routes[sec.TICKERS_URL] = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
        "2": {"cik_str": 1067983, "ticker": "BRK-A", "title": "Berkshire Hathaway"},
        "3": {"cik_str": 1000, "ticker": "BF.B", "title": "Brown-Forman"},
    }
    m = sec.ticker_to_cik()
    assert m["AAPL"] == 320193
    assert m["BRK-B"] == m["BRK.B"] == m["BRK.A"] == 1067983
    assert m["BF-B"] == m["BF.B"] == 1000
    assert calls[0]["headers"] == {"User-Agent": "Unit Test unit@example.com"}
    assert calls[0]["min_interval"] == pytest.approx(0.12)
    assert store.read_json("sec", "company_tickers") is not None
    assert sec.ticker_to_cik() == m and len(calls) == 1
    sec.ticker_to_cik(refresh=True)
    assert len(calls) == 2


# --- submissions ------------------------------------------------------------------------------

def test_load_submissions_merges_files_pages_and_caches(fake_http):
    routes, calls = fake_http
    recent = _page(_realistic_rows(), shift=1)
    older_rows = [
        ("0000000001-03-000001", "8-K", None, "12", "2003-05-01"),
        ("0000000001-03-000002", "8-K", None, "5,7", "2003-05-02"),
        ("0000000001-24-000000", "10-Q", "2024-01-08 06:01:07", "", None),  # duplicate of a recent filing
    ]
    older = _page(older_rows, shift=1)
    older["acceptanceDateTime"][1] = "2003-05-02T00:00:00.000Z"  # date-only placeholder
    routes["https://data.sec.gov/submissions/CIK0000000001.json"] = {
        "cik": "1", "filings": {"recent": recent, "files": [{"name": "CIK0000000001-submissions-001.json"}]}}
    routes["https://data.sec.gov/submissions/CIK0000000001-submissions-001.json"] = older

    subs = sec.load_submissions(1)
    assert list(subs.columns) == sec.SUB_COLUMNS
    assert subs["filing_date"].dtype == "datetime64[ns]" and subs["acceptance"].dtype == "datetime64[ns]"
    assert subs["accession"].is_unique and len(subs) == 60 + 2
    assert subs["filing_date"].is_monotonic_increasing
    old = subs.set_index("accession").loc[["0000000001-03-000001", "0000000001-03-000002"]]
    assert old["acceptance"].isna().all() and list(old["items"]) == ["12", "5,7"]
    assert {c["url"] for c in calls} == set(routes)
    assert store.read_meta("sec_submissions", "1")["acceptance_shift"] == [1, 0]

    n = len(calls)
    pd.testing.assert_frame_equal(sec.load_submissions(1), subs)
    assert len(calls) == n


def test_load_submissions_missing_page_warns_and_not_cached(fake_http):
    routes, calls = fake_http
    routes["https://data.sec.gov/submissions/CIK0000000002.json"] = {
        "filings": {"recent": _page(_realistic_rows(prefix="25"), shift=1),
                    "files": [{"name": "CIK0000000002-submissions-001.json"}]}}  # older page never served -> 404

    with pytest.warns(UserWarning, match="CIK 2 older submissions page\\(s\\) missing"):
        subs = sec.load_submissions(2)
    assert len(subs) == 60  # partial frame: only the recent page
    assert store.read("sec_submissions", "2") is None  # incomplete history must not be cached

    pd.testing.assert_frame_equal(sec.load_submissions(2), subs)
    assert len(calls) == 4  # each refetch retries both pages, never serves the stale partial copy


@pytest.mark.parametrize("shift", [0, 1, 2])
def test_acceptance_normalised_to_true_utc_whatever_the_feed_convention(fake_http, shift):
    routes, _ = fake_http
    rows = _realistic_rows()
    routes["https://data.sec.gov/submissions/CIK0000000007.json"] = {
        "filings": {"recent": _page(rows, shift), "files": []}}
    subs = sec.load_submissions(7).set_index("accession")
    expected = pd.Series([_true_utc(r[2]) for r in rows], index=[r[0] for r in rows])
    np.testing.assert_array_equal(subs.loc[expected.index, "acceptance"].to_numpy(), expected.to_numpy())
    assert store.read_meta("sec_submissions", "7")["acceptance_shift"] == [shift]
    earn = sec.earnings_days(subs.reset_index(), TD)
    # 16:30 ET releases always land on the next session, whichever stamp convention the feed used
    release_days = pd.to_datetime([r[2][:10] for r in rows if r[3] == "2.02,9.01"])
    assert list(earn) == list(TD[TD.searchsorted(release_days, side="right")].unique())


def test_uninformative_page_defaults_to_latest_reading():
    raw = pd.Series(pd.to_datetime(["2024-01-08 12:00:00"]))
    assert sec._acceptance_shift(raw, pd.Series(pd.to_datetime(["2024-01-08"])), pd.Series(["8-K"])) == 0


def test_submissions_404_returns_empty(fake_http):
    subs = sec.load_submissions(999)
    assert subs.empty and list(subs.columns) == sec.SUB_COLUMNS


def test_companyfacts_404_and_cache(fake_http):
    routes, calls = fake_http
    assert sec.load_companyfacts(42) is None
    assert calls[0]["url"] == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000042.json"
    routes["https://data.sec.gov/api/xbrl/companyfacts/CIK0000000043.json"] = {"cik": 43, "facts": {"us-gaap": {}}}
    assert sec.load_companyfacts(43)["cik"] == 43
    n = len(calls)
    assert sec.load_companyfacts(43)["cik"] == 43 and len(calls) == n


def test_http_error_propagates(fake_http, monkeypatch):
    monkeypatch.setattr(sec.http, "get", lambda *a, **k: FakeResp(status=403))
    with pytest.raises(requests.HTTPError):
        sec.load_companyfacts(1)


# --- effective trading date -------------------------------------------------------------------

def _eff(acc, fd=None):
    acc = pd.Series(pd.to_datetime(acc))
    fd = pd.Series(pd.to_datetime(fd if fd is not None else [pd.NaT] * len(acc)))
    return list(sec.effective_trading_date(acc, fd, TD))


def test_utc_to_et_across_dst():
    got = _eff(["2024-07-10 21:30", "2024-01-10 13:00", "2024-01-10 20:30", "2024-07-10 20:30",
                "2024-07-10 19:59", "2024-01-10 21:00"])
    assert got == pd.to_datetime(["2024-07-11", "2024-01-10", "2024-01-10", "2024-07-11",
                                  "2024-07-10", "2024-01-11"]).tolist()


def test_weekend_holiday_early_close_and_filing_date_fallback():
    acc = ["2024-07-06 15:00", "2024-07-05 21:00", "2024-07-04 14:00", "2024-07-03 16:00", "2024-07-03 18:00",
           "2024-11-29 19:00", "2024-11-29 16:00", None, None, None, "2025-03-05 15:00"]
    fd = [None] * 7 + ["2024-07-06", "2024-07-10", "2024-07-04", None]
    assert _eff(acc, fd) == [pd.Timestamp(x) if x else pd.NaT for x in (
        "2024-07-08", "2024-07-08", "2024-07-05", "2024-07-03", "2024-07-05",
        "2024-12-02", "2024-11-29", "2024-07-08", "2024-07-10", "2024-07-05", None)]


def test_effective_date_accepts_tz_aware_and_keeps_index():
    acc = pd.Series(pd.to_datetime(["2024-01-10 13:00"]).tz_localize("UTC"), index=[17])
    out = sec.effective_trading_date(acc, pd.Series([pd.NaT], index=[17]), TD)
    assert out.index.tolist() == [17] and out.iloc[0] == pd.Timestamp("2024-01-10")


# --- news / earnings days ---------------------------------------------------------------------

def _subs(rows):
    """rows: (form, filing_date, acceptance UTC or None, items)."""
    return pd.DataFrame({
        "form": [r[0] for r in rows],
        "filing_date": pd.to_datetime([r[1] for r in rows]),
        "acceptance": pd.to_datetime([r[2] for r in rows]),
        "items": [r[3] for r in rows],
        "accession": [f"a{i}" for i in range(len(rows))],
    })


def test_earnings_days_exact_item_codes():
    subs = _subs([
        ("8-K", "2003-05-01", None, "12"),            # old Results of Operations
        ("8-K", "2003-06-02", None, "5, 12"),
        ("8-K", "2003-06-03", None, "1.02"),          # substring traps: none of these is item 12
        ("8-K", "2003-06-04", None, "1,2"),
        ("8-K", "2003-06-05", None, "11"),
        ("8-K", "2003-06-06", None, "112,121"),
        ("8-K", "2024-01-10", "2024-01-10 21:30", "2.02,9.01"),
        ("8-K", "2024-02-01", "2024-02-01 12:00", "9.01, 2.02"),
        ("8-K", "2024-03-01", "2024-03-01 15:00", "12"),         # item 12 no longer exists after 2004-08-23
        ("8-K", "2024-04-01", "2024-04-01 15:00", "1.02"),
        ("8-K", "2024-04-02", "2024-04-02 15:00", "2.021"),
        ("8-K/A", "2024-05-01", "2024-05-01 15:00", "2.02"),
        ("10-Q", "2024-05-02", "2024-05-02 15:00", ""),
    ])
    got = sec.earnings_days(subs, TD)
    assert list(got) == pd.to_datetime(["2003-05-01", "2003-06-02", "2024-01-11", "2024-02-01"]).tolist()


def test_news_days_forms_and_items():
    subs = _subs([
        ("8-K", "2024-01-10", "2024-01-10 15:00", "5.02"),
        ("8-K", "2024-01-10", "2024-01-10 16:00", "5.02,9.01"),     # same effective day: deduped
        ("8-K/A", "2024-01-12", "2024-01-12 15:00", "8.01"),
        ("10-K", "2024-01-16", "2024-01-16 15:00", ""),
        ("8-K", "2024-01-17", "2024-01-17 15:00", "5.021"),
    ])
    assert list(sec.news_days(subs, TD)) == pd.to_datetime(["2024-01-10", "2024-01-12", "2024-01-17"]).tolist()
    assert list(sec.news_days(subs, TD, items=("5.02",))) == [pd.Timestamp("2024-01-10")]
    assert list(sec.news_days(subs, TD, forms=("10-K",))) == [pd.Timestamp("2024-01-16")]
    assert len(sec.news_days(subs.iloc[:0], TD)) == 0


def test_has_item_splits_on_commas_and_whitespace():
    subs = _subs([
        ("8-K", "2024-01-01", None, "2.02 9.01"),      # space-separated, e.g. hand-typed feeds
        ("8-K", "2024-01-02", None, "2.02, 8.01"),     # comma + space
        ("8-K", "2024-01-03", None, "2.02,9.01"),
        ("8-K", "2024-01-04", None, " 5.02 "),         # padded non-match; trailing comma yields empty token
        ("8-K", "2024-01-05", None, ""),
    ])
    hit = sec._has_item(subs["items"], ("2.02",))
    assert hit.tolist() == [True, True, True, False, False]


# --- news flags -------------------------------------------------------------------------------

@pytest.fixture
def flag_world(monkeypatch):
    subs = {
        1: _subs([("8-K", "2024-07-10", "2024-07-10 15:00", "8.01"),
                  ("8-K", "2024-07-12", "2024-07-12 21:30", "2.02,9.01")]),   # Fri 17:30 ET -> Mon
        2: _subs([("10-Q", "2024-07-10", "2024-07-10 15:00", "")]),
    }
    loads = []

    def load(cik, refresh=False):
        loads.append(cik)
        return subs[cik]

    monkeypatch.setattr(sec, "ticker_to_cik", lambda refresh=False: {"AAA": 1, "BBB": 2, "CCC": 1, "BRK-B": 2})
    monkeypatch.setattr(sec, "load_submissions", load)
    return subs, loads


def test_news_flags_window_and_missing(flag_world):
    _, loads = flag_world
    td = TD[(TD >= "2024-07-01") & (TD <= "2024-07-31")]
    flags, missing = sec.news_flags(["AAA", "BBB", "ZZZ", "CCC", "brk.b"], td)
    assert missing == ["ZZZ"]
    assert flags.dtypes.eq(bool).all() and flags.index.equals(td)
    hits = flags.index[flags["AAA"]].strftime("%Y-%m-%d").tolist()
    assert hits == ["2024-07-10", "2024-07-11", "2024-07-15", "2024-07-16"]
    assert flags["CCC"].equals(flags["AAA"].rename("CCC"))
    assert not flags["BBB"].any() and not flags["ZZZ"].any() and not flags["brk.b"].any()
    assert loads == [1, 2]
    earn, _ = sec.news_flags(["AAA"], td, kind="earnings")
    assert earn.index[earn["AAA"]].strftime("%Y-%m-%d").tolist() == ["2024-07-15", "2024-07-16"]
    wide, _ = sec.news_flags(["AAA"], td, window=0)
    assert wide["AAA"].sum() == 2
    with pytest.raises(ValueError):
        sec.news_flags(["AAA"], td, kind="bogus")


def test_news_flags_ignore_future_filings(flag_world):
    subs, _ = flag_world
    td = TD[(TD >= "2024-07-01") & (TD <= "2024-09-30")]
    before, _ = sec.news_flags(["AAA"], td)
    extra = _subs([("8-K", "2024-08-01", "2024-08-01 14:00", "8.01"), ("8-K", "2024-09-03", None, "2.02")])
    subs[1] = pd.concat([subs[1], extra], ignore_index=True)
    after, _ = sec.news_flags(["AAA"], td)
    cut = td < "2024-08-01"
    pd.testing.assert_frame_equal(before[cut], after[cut])
    assert after.loc["2024-08-01", "AAA"] and not before.loc["2024-08-01", "AAA"]


def test_news_flags_progress_and_holdout(flag_world, capsys):
    start = settings().holdout_start
    td = pd.bdate_range(start - pd.Timedelta(days=20), start + pd.Timedelta(days=20))
    flags, _ = sec.news_flags(["AAA"] * 30, td)
    assert flags.index.max() < start and len(flags) > 0
    assert "25/30" in capsys.readouterr().err


def test_sec_news_window_registered():
    from omodel import params
    p = params.REGISTRY["SEC_NEWS_WINDOW"]
    assert p.value == 1 and p.source == "convention"


@pytest.mark.network
def test_live_apple_earnings_accepted_after_close():
    if not settings().sec_user_agent:
        pytest.skip("SEC_USER_AGENT not set")
    assert sec.ticker_to_cik()["AAPL"] == 320193
    subs = sec.load_submissions(320193)
    er = subs[(subs["form"] == "8-K") & subs["items"].str.contains("2.02", regex=False)
              & (subs["filing_date"] >= "2015-01-01")]
    hour = er["acceptance"].dt.tz_localize("UTC").dt.tz_convert(ET).dt.hour
    assert len(er) >= 20 and (hour == 16).mean() > 0.8  # Apple releases at ~16:30 ET

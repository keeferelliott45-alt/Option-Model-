import numpy as np
import pandas as pd
import pytest

from omodel.data import store, universe

CSV = """ticker,start_date,end_date
AAL,1996-01-02,1997-01-15
AAL,2015-03-23,2024-09-23
AAPL,1996-01-02,
BRK.B,2010-02-16,
"""


def spells(*rows):
    return pd.DataFrame({
        "ticker": [r[0] for r in rows],
        "start_date": pd.to_datetime([r[1] for r in rows]),
        "end_date": pd.to_datetime([r[2] for r in rows]),
    })


class FakeResponse:
    status_code = 200
    text = CSV

    def raise_for_status(self):
        pass


def test_load_membership_parses_and_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(universe.http, "get", lambda url, **kw: calls.append(url) or FakeResponse())
    m = universe.load_membership()
    assert calls == [universe.SOURCE_URL]
    assert list(m.columns) == ["ticker", "start_date", "end_date"]
    assert m["start_date"].dtype.kind == "M" and m["end_date"].dtype.kind == "M"
    assert list(m["ticker"]) == ["AAL", "AAL", "AAPL", "BRK.B"]
    assert pd.isna(m.loc[2, "end_date"]) and m.loc[1, "end_date"] == pd.Timestamp("2024-09-23")
    assert store.read_meta("universe", "sp500_membership")["source"] == universe.SOURCE_URL
    monkeypatch.setattr(universe.http, "get", lambda url, **kw: pytest.fail("cache should be used"))
    pd.testing.assert_frame_equal(universe.load_membership(), m)


def test_load_membership_rejects_bad_columns(monkeypatch):
    class Bad(FakeResponse):
        text = "symbol,added\nAAPL,1996-01-02\n"
    monkeypatch.setattr(universe.http, "get", lambda url, **kw: Bad())
    with pytest.raises(ValueError):
        universe.load_membership()


def test_load_membership_rejects_inverted_spells(monkeypatch):
    class Inverted(FakeResponse):
        text = CSV + "XYZ,2020-01-10,2020-01-06\n"
    monkeypatch.setattr(universe.http, "get", lambda url, **kw: Inverted())
    with pytest.raises(ValueError, match="XYZ"):
        universe.load_membership()


def test_membership_mask_ignores_inverted_and_empty_spells():
    m = spells(
        ("X", "2020-01-01", None),
        ("X", "2020-01-08", "2020-01-06"),   # inverted: must not cancel the open spell
        ("Y", "2020-01-02", "2020-01-07"),
        ("Y", "2020-01-03", "2020-01-03"),   # empty
        ("Z", "2020-01-09", "2020-01-02"),   # inverted only
    )
    mask = universe.membership_mask(m, DATES)
    assert mask["X"].loc["2020-01-01":].all()
    np.testing.assert_array_equal(mask["Y"].to_numpy(), [0, 1, 1, 1, 0, 0, 0, 0])
    assert not mask["Z"].any()
    assert universe.tickers_between(m, DATES[0], DATES[-1]) == ["X", "Y"]


MEMBERSHIP = spells(
    ("A", "2020-01-02", "2020-01-06"),   # removed on the 6th (exclusive)...
    ("A", "2020-01-08", None),           # ...re-enters on the 8th, still a member
    ("B", "2020-01-03", "2020-01-07"),
    ("C", "2019-01-01", "2020-01-01"),   # gone before the window
)
DATES = pd.bdate_range("2020-01-01", "2020-01-10")  # Jan 1, 2, 3, 6, 7, 8, 9, 10


def test_membership_mask_spell_boundaries():
    mask = universe.membership_mask(MEMBERSHIP, DATES)
    assert list(mask.columns) == ["A", "B", "C"]
    assert mask.dtypes.eq(bool).all()
    expected = pd.DataFrame({
        "A": [0, 1, 1, 0, 0, 1, 1, 1],
        "B": [0, 0, 1, 1, 0, 0, 0, 0],
        "C": [0, 0, 0, 0, 0, 0, 0, 0],
    }, index=DATES).astype(bool)
    pd.testing.assert_frame_equal(mask, expected, check_column_type=False)


def test_membership_mask_unsorted_dates_and_ticker_subset():
    shuffled = DATES[[5, 0, 3, 7, 1, 2, 6, 4]]
    mask = universe.membership_mask(MEMBERSHIP, shuffled, tickers=["B", "ZZZ", "A"])
    assert list(mask.columns) == ["B", "ZZZ", "A"]
    pd.testing.assert_index_equal(mask.index, shuffled)
    full = universe.membership_mask(MEMBERSHIP, DATES)
    pd.testing.assert_frame_equal(mask[["A", "B"]].loc[DATES], full[["A", "B"]], check_freq=False)
    assert not mask["ZZZ"].any()


def test_membership_mask_matches_bruteforce():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2019-06-01", "2021-06-01")
    rows = []
    for t in ["P", "Q", "R", "S"]:
        cuts = np.sort(rng.choice(len(days), 4, replace=False))
        rows.append((t, days[cuts[0]], days[cuts[1]]))
        rows.append((t, days[cuts[2]], None if t in "PR" else days[cuts[3]]))
    m = spells(*rows)
    mask = universe.membership_mask(m, days)
    for t in mask.columns:
        sp = m[m.ticker == t]
        brute = np.zeros(len(days), dtype=bool)
        for _, r in sp.iterrows():
            brute |= (days >= r.start_date) & ((days < r.end_date) if pd.notna(r.end_date) else True)
        np.testing.assert_array_equal(mask[t].to_numpy(), brute)


def test_tickers_between():
    m = spells(
        ("ENDS_AT_START", "2019-01-01", "2020-01-02"),  # removal date == window start -> not a member in window
        ("STARTS_AT_END", "2020-01-10", None),
        ("INSIDE", "2020-01-03", "2020-01-06"),
        ("AFTER", "2020-01-11", None),
        ("CURRENT", "2000-01-01", None),
        ("OLD", "2000-01-01", "2005-01-01"),
        ("CURRENT", "1990-01-01", "1995-01-01"),
    )
    got = universe.tickers_between(m, pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-10"))
    assert got == ["CURRENT", "INSIDE", "STARTS_AT_END"]


def test_coverage_by_year_hand_example():
    dates = pd.DatetimeIndex(["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03"])
    member = pd.DataFrame({
        "A": [True, True, True, True],
        "B": [True, True, False, False],
        "C": [False, False, True, True],
    }, index=dates)
    has_price = pd.DataFrame({
        "A": [True, False, True, True],
        "B": [False, False, True, True],   # prices only after it left the index: no member-day coverage
        "D": [True, True, True, True],     # not in member: ignored
    }, index=dates)                        # C absent: no data at all
    cov = universe.coverage_by_year(member, has_price)
    assert cov.index.name == "year" and list(cov.index) == [2019, 2020]
    assert list(cov.columns) == ["member_days", "covered_days", "coverage_pct", "members", "members_with_no_data"]
    assert cov.loc[2019].tolist() == [4, 1, 25.0, 2, 1]
    assert cov.loc[2020].tolist() == [4, 2, 50.0, 2, 1]


def test_coverage_by_year_no_members_is_nan():
    dates = pd.DatetimeIndex(["2020-01-02", "2021-01-04"])
    member = pd.DataFrame({"A": [False, True]}, index=dates)
    cov = universe.coverage_by_year(member, pd.DataFrame({"A": [True, True]}, index=dates))
    assert np.isnan(cov.loc[2020, "coverage_pct"]) and cov.loc[2020, "member_days"] == 0
    assert cov.loc[2021, "coverage_pct"] == 100.0

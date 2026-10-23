from __future__ import annotations

import io

import numpy as np
import pandas as pd

from omodel.data import http, store

SOURCE_URL = "https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv"
KIND, KEY = "universe", "sp500_membership"


def _parse(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    missing = {"ticker", "start_date", "end_date"} - set(raw.columns)
    if missing:
        raise ValueError(f"membership CSV lacks columns {sorted(missing)}")
    raw = raw.apply(lambda s: s.str.strip())
    raw = raw[raw["ticker"] != ""]
    end = raw["end_date"]
    df = pd.DataFrame({
        "ticker": raw["ticker"].astype(str),
        "start_date": pd.to_datetime(raw["start_date"], format="%Y-%m-%d"),
        "end_date": pd.to_datetime(end.where(end != ""), format="%Y-%m-%d"),
    })
    if df["start_date"].isna().any():
        raise ValueError("membership CSV has spells without a start_date")
    inverted = df["end_date"] < df["start_date"]
    if inverted.any():
        raise ValueError(f"membership CSV has spells ending before they start: {sorted(df.loc[inverted, 'ticker'])}")
    return df.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def load_membership(refresh: bool = False) -> pd.DataFrame:
    if not refresh:
        cached = store.read(KIND, KEY)
        if cached is not None:
            return cached
    resp = http.get(SOURCE_URL)
    resp.raise_for_status()
    df = _parse(resp.text)
    store.write(df, KIND, KEY, meta={"source": SOURCE_URL})
    return df


def _ns(values) -> np.ndarray:
    return np.asarray(pd.to_datetime(values), dtype="datetime64[ns]")


def membership_mask(membership: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str] | None = None) -> pd.DataFrame:
    """Member on d iff start_date <= d < end_date (end_date is the removal date; NaT = still a member)."""
    dates = pd.DatetimeIndex(dates)
    cols = sorted(membership["ticker"].unique()) if tickers is None else list(tickers)
    pos = {t: i for i, t in enumerate(cols)}
    spells = membership[membership["ticker"].isin(pos)]
    d = _ns(dates)
    order = np.argsort(d, kind="stable")
    sd = d[order]
    n = len(sd)
    start, end = _ns(spells["start_date"]), _ns(spells["end_date"])
    lo = np.searchsorted(sd, start, side="left")
    # Empty or inverted spells contribute nothing (as in tickers_between) instead of cancelling an overlapping spell.
    hi = np.maximum(lo, np.where(np.isnat(end), n, np.searchsorted(sd, end, side="left")))
    ci = spells["ticker"].map(pos).to_numpy(dtype=np.intp)
    # Difference array: +1 where a spell opens, -1 where it closes; cumsum > 0 means inside some spell.
    acc = np.zeros((n + 1, len(cols)), dtype=np.int32)
    np.add.at(acc, (lo, ci), 1)
    np.add.at(acc, (hi, ci), -1)
    inside = np.cumsum(acc[:-1], axis=0) > 0
    out = np.empty_like(inside)
    out[order] = inside
    return pd.DataFrame(out, index=dates, columns=pd.Index(cols, dtype=object))


def tickers_between(membership: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    s, e = membership["start_date"], membership["end_date"]
    hit = (s <= end) & (e.isna() | (e > start)) & (e.isna() | (e > s))
    return sorted(membership.loc[hit, "ticker"].unique())


def coverage_by_year(member: pd.DataFrame, has_price: pd.DataFrame) -> pd.DataFrame:
    m = member.eq(True)
    hp = has_price.reindex(index=m.index, columns=m.columns, fill_value=False).eq(True)
    covered = m & hp
    year = pd.Index(m.index.year, name="year")
    member_days = m.sum(axis=1).groupby(year).sum()
    covered_days = covered.sum(axis=1).groupby(year).sum()
    in_year = m.groupby(year).any()
    # "No data" = no price on any of the ticker's member days that year: those days contribute zero coverage.
    no_data = in_year & ~covered.groupby(year).any()
    out = pd.DataFrame({
        "member_days": member_days.astype("int64"),
        "covered_days": covered_days.astype("int64"),
        "coverage_pct": 100.0 * covered_days / member_days.where(member_days > 0),
        "members": in_year.sum(axis=1).astype("int64"),
        "members_with_no_data": no_data.sum(axis=1).astype("int64"),
    })
    out.index.name = "year"
    return out

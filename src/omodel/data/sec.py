from __future__ import annotations

import sys
import warnings
from typing import Any, Iterable

import numpy as np
import pandas as pd

from omodel import holdout, params
from omodel.config import settings
from omodel.data import http, store

SEC_NEWS_WINDOW = params.register(
    "SEC_NEWS_WINDOW", 1, "convention", "L2",
    "News effective the prior session can still drive day-t moves, so day t is flagged for effective dates t and t-1.")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_INTERVAL = 0.12  # SEC fair-access policy: at most 10 requests/s
ET = "America/New_York"
ITEM12_END = pd.Timestamp("2004-08-23")  # 8-K items renumbered on this date; old item 12 became 2.02
SUB_COLUMNS = ["form", "filing_date", "acceptance", "items", "accession"]

# EDGAR filing-date rule: accepted 06:00-17:30 ET -> same-day date, later -> next business day (these forms: until 22:00).
_EDGAR_OPEN_H, _CUTOFF_H, _EDGAR_CLOSE_H = 6.0, 17.5, 22.0
_LATE_SAME_DAY_FORMS = ("3", "4", "5", "3/A", "4/A", "5/A", "144", "144/A")


class SecConfigError(RuntimeError):
    pass


def _user_agent() -> str:
    ua = settings().sec_user_agent.strip()
    if not ua:
        raise SecConfigError("Set SEC_USER_AGENT='Your Name your@email' (SEC requires contact info)")
    return ua


def _get_json(url: str) -> Any | None:
    resp = http.get(url, headers={"User-Agent": _user_agent()}, min_interval=MIN_INTERVAL)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def ticker_to_cik(refresh: bool = False) -> dict[str, int]:
    """Current SEC ticker -> CIK map, with '-' and '.' spellings of class shares (BRK-B / BRK.B)."""
    raw = None if refresh else store.read_json("sec", "company_tickers")
    if raw is None:
        raw = _get_json(TICKERS_URL)
        if raw is None:
            raise RuntimeError(f"SEC ticker map not found at {TICKERS_URL}")
        store.write_json(raw, "sec", "company_tickers")
    out: dict[str, int] = {}
    for rec in raw.values() if isinstance(raw, dict) else raw:
        out.setdefault(str(rec["ticker"]).strip().upper(), int(rec["cik_str"]))
    for ticker, cik in list(out.items()):
        for variant in (ticker.replace("-", "."), ticker.replace(".", "-")):
            out.setdefault(variant, cik)
    return out


def _utc_to_et(s: pd.Series) -> pd.Series:
    return s.dt.tz_localize("UTC").dt.tz_convert(ET).dt.tz_localize(None)


def _et_to_utc(s: pd.Series) -> pd.Series:
    return (s.dt.tz_localize(ET, ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC").dt.tz_localize(None))


def _edgar_rule_fit(wall_et: pd.Series, filing_date: pd.Series, late_ok: np.ndarray) -> float:
    day = wall_et.dt.normalize()
    hour = ((wall_et - day) / pd.Timedelta(hours=1)).to_numpy()
    lag = ((filing_date - day) / pd.Timedelta(days=1)).to_numpy()
    same, later = lag == 0, (lag >= 1) & (lag <= 5)
    rule = np.where(hour <= _CUTOFF_H, same, later | (late_ok & same))
    return float(((hour >= _EDGAR_OPEN_H) & (hour <= _EDGAR_CLOSE_H) & rule).mean())


def _acceptance_shift(raw: pd.Series, filing_date: pd.Series, form: pd.Series) -> int:
    """ET-UTC offsets by which the stamp runs ahead of ET wall time: 0 = ET labelled 'Z', 1 = true UTC, 2 = shifted twice."""
    # The feed's 'Z' has not matched true UTC, so each page takes the reading that best fits EDGAR's filing-date rule;
    # ties go to 0, the latest wall-clock reading, which can only delay an effective date, never advance it.
    ok =(raw.notna() & filing_date.notna()).to_numpy()
    if not ok.any():
        return 0
    r, fd = raw[ok], filing_date[ok]
    late_ok = form[ok].isin(_LATE_SAME_DAY_FORMS).to_numpy()
    once = _utc_to_et(r)
    scores = [_edgar_rule_fit(w, fd, late_ok) for w in (r, once, _utc_to_et(once))]
    return int(np.argmax(scores))


def _true_utc(raw: pd.Series, shift: int) -> pd.Series:
    return _et_to_utc(raw) if shift == 0 else raw if shift == 1 else _utc_to_et(raw)


def _empty_subs() -> pd.DataFrame:
    return pd.DataFrame({"form": pd.Series(dtype=object), "filing_date": pd.Series(dtype="datetime64[ns]"),
                         "acceptance": pd.Series(dtype="datetime64[ns]"), "items": pd.Series(dtype=object),
                         "accession": pd.Series(dtype=object)})


def _parse_page(cols: dict) -> tuple[pd.DataFrame, int]:
    accession = list(cols.get("accessionNumber") or [])
    n = len(accession)
    if n == 0:
        return _empty_subs(), 0

    def col(name: str) -> list:
        v = cols.get(name)
        return list(v) if v is not None and len(v) == n else [None] * n

    form = pd.Series(col("form"), dtype=object).fillna("").astype(str)
    filing_date = pd.Series(pd.to_datetime(col("filingDate"), errors="coerce")).astype("datetime64[ns]")
    raw = pd.Series(pd.to_datetime(col("acceptanceDateTime"), errors="coerce", utc=True, format="ISO8601"))
    raw = raw.dt.tz_localize(None).astype("datetime64[ns]")
    raw = raw.where(raw != raw.dt.normalize())  # exact-midnight stamps are date-only placeholders, not times
    shift = _acceptance_shift(raw, filing_date, form)
    df = pd.DataFrame({
        "form": form,
        "filing_date": filing_date,
        "acceptance": _true_utc(raw, shift).astype("datetime64[ns]"),
        "items": pd.Series(col("items"), dtype=object).fillna("").astype(str),
        "accession": pd.Series(accession, dtype=object).astype(str),
    })
    return df, shift


def load_submissions(cik: int, refresh: bool = False) -> pd.DataFrame:
    """All filings for a CIK (recent page plus every older page); acceptance is true UTC, tz-naive."""
    key = str(int(cik))
    if not refresh and (cached := store.read("sec_submissions", key)) is not None:
        return cached
    doc = _get_json(f"{SUBMISSIONS_URL}CIK{int(cik):010d}.json")
    if doc is None:
        return _empty_subs()
    filings = doc.get("filings") or {}
    pages, complete = [filings.get("recent") or {}], True
    for f in filings.get("files") or []:
        page = _get_json(SUBMISSIONS_URL + f["name"])
        if page is None:
            complete = False
            warnings.warn(f"sec: CIK {key} older submissions page(s) missing; filing history is partial")
        pages.append(page or {})
    parsed = [_parse_page(p) for p in pages]
    df = pd.concat([d for d, _ in parsed], ignore_index=True).drop_duplicates("accession")
    df = df.sort_values(["filing_date", "accession"], kind="stable").reset_index(drop=True)[SUB_COLUMNS]
    if complete:
        store.write(df, "sec_submissions", key,
                    meta={"cik": int(cik), "pages": len(pages), "acceptance_shift": [k for _, k in parsed]})
    return df


def load_companyfacts(cik: int, refresh: bool = False) -> dict | None:
    key = str(int(cik))
    if not refresh and (cached := store.read_json("sec_facts", key)) is not None:
        return cached
    facts = _get_json(FACTS_URL.format(cik=int(cik)))
    if facts is not None:
        store.write_json(facts, "sec_facts", key)
    return facts


def _early_close(day: pd.Series) -> np.ndarray:
    # NYSE 13:00 closes (Jul 3, Black Friday, Dec 24); treating a full day as early only delays a few filings a day.
    m, d, wd = day.dt.month, day.dt.day, day.dt.weekday
    black_friday = (m == 11) & (wd == 4) & d.between(23, 29)
    return (((m == 7) & (d == 3)) | ((m == 12) & (d == 24)) | black_friday).to_numpy(dtype=bool)


def effective_trading_date(acceptance_utc: pd.Series, filing_date: pd.Series,
                           trading_days: pd.DatetimeIndex) -> pd.Series:
    """First trading day whose session a filing can affect (NaT when beyond trading_days)."""
    td = pd.DatetimeIndex(trading_days).as_unit("ns").sort_values()
    acc = pd.Series(pd.to_datetime(acceptance_utc))
    acc = acc.dt.tz_convert("UTC") if acc.dt.tz is not None else acc.dt.tz_localize("UTC")
    et = acc.dt.tz_convert(ET).dt.tz_localize(None)
    day = et.dt.normalize()
    hour = (et - day) / pd.Timedelta(hours=1)
    in_session = day.isin(td) & (hour < np.where(_early_close(day), 13.0, 16.0))
    target = day.where(in_session, day + pd.Timedelta(days=1))
    fd = pd.Series(pd.to_datetime(np.asarray(filing_date)), index=acc.index).dt.normalize()
    target = target.where(acc.notna(), fd).to_numpy(dtype="datetime64[ns]")
    out = np.full(len(target), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = ~np.isnat(target)
    pos = td.searchsorted(target[valid])
    hit = pos < len(td)
    out[np.flatnonzero(valid)[hit]] = td.to_numpy()[pos[hit]]
    return pd.Series(out, index=acc.index, name="effective_date")


def _has_item(items: pd.Series, codes: Iterable[str]) -> pd.Series:
    parts = items.fillna("").astype(str).str.split(r"[,\s]+").explode().str.strip()
    hit = parts.isin({str(c).strip() for c in codes}).groupby(level=0).any()
    return hit.reindex(items.index, fill_value=False).astype(bool)


def _effective_days(subs: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    eff = effective_trading_date(subs["acceptance"], subs["filing_date"], trading_days)
    return pd.DatetimeIndex(np.unique(eff.dropna().to_numpy(dtype="datetime64[ns]")))


def news_days(subs: pd.DataFrame, trading_days: pd.DatetimeIndex,
              forms: tuple[str, ...] = ("8-K", "8-K/A"), items: tuple[str, ...] | None = None) -> pd.DatetimeIndex:
    subs = subs.reset_index(drop=True)
    mask = subs["form"].isin(forms)
    if items is not None:
        mask &= _has_item(subs["items"], items)
    return _effective_days(subs[mask], trading_days)


def earnings_days(subs: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """8-K 'Results of Operations' days: item 2.02, or item 12 before the 2004-08-23 renumbering."""
    subs = subs.reset_index(drop=True)
    # Amendments re-file an earlier release, so their dates are not the earnings event.
    is_8k = subs["form"] == "8-K"
    old = _has_item(subs["items"], ("12",)) & (subs["filing_date"] < ITEM12_END)
    return _effective_days(subs[is_8k & (_has_item(subs["items"], ("2.02",)) | old)], trading_days)


def news_flags(tickers: list[str], trading_days: pd.DatetimeIndex, kind: str = "any",
               refresh: bool = False, *, window: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    """(dates x tickers) True iff a matching filing became effective on t or within `window` sessions before; holdout-guarded."""
    if kind not in ("any", "earnings"):
        raise ValueError(f"kind must be 'any' or 'earnings', got {kind!r}")
    window = params.get("SEC_NEWS_WINDOW") if window is None else window
    td = pd.DatetimeIndex(trading_days).as_unit("ns").sort_values()
    cmap = ticker_to_cik(refresh=refresh)
    flags = np.zeros((len(td), len(tickers)), dtype=bool)
    missing: list[str] = []
    positions: dict[int, np.ndarray] = {}
    for j, ticker in enumerate(tickers):
        key = str(ticker).strip().upper()
        cik = next((cmap[k] for k in (key, key.replace(".", "-"), key.replace("/", "-")) if k in cmap), None)
        if cik is None:
            missing.append(ticker)
        else:
            if cik not in positions:
                subs = load_submissions(cik, refresh=refresh)
                days = earnings_days(subs, td) if kind == "earnings" else news_days(subs, td)
                pos = td.get_indexer(days)
                positions[cik] = pos[pos >= 0]
            for lag in range(window + 1):
                p = positions[cik] + lag
                flags[p[p < len(td)], j] = True
        if (j + 1) % 25 == 0:
            print(f"sec news_flags({kind}): {j + 1}/{len(tickers)} tickers", file=sys.stderr, flush=True)
    out = pd.DataFrame(flags, index=td, columns=list(tickers))
    return holdout.guard(out, what="sec.news_flags"), missing

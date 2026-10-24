from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from omodel.data import sec, store

FLOW = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross": ["GrossProfit"],
    "ni": ["NetIncomeLoss", "ProfitLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "ebit": ["OperatingIncomeLoss"],
}
STOCK = {
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "cur_assets": ["AssetsCurrent"],
    "cur_liab": ["LiabilitiesCurrent"],
    "retained": ["RetainedEarningsAccumulatedDeficit"],
    "debt": ["LongTermDebt", "LongTermDebtNoncurrent", "DebtInstrumentCarryingAmount"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "shares": ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
}


def _entries(facts: dict, names: list[str]) -> pd.DataFrame:
    rows = []
    for ns in ("us-gaap", "dei"):
        body = (facts.get("facts") or {}).get(ns) or {}
        for rank, name in enumerate(names):
            for unit, items in ((body.get(name) or {}).get("units") or {}).items():
                if unit not in ("USD", "shares"):
                    continue
                for it in items:
                    rows.append((rank, it.get("start"), it.get("end"), it.get("val"), it.get("filed")))
    df = pd.DataFrame(rows, columns=["rank", "start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df.dropna(subset=["end", "filed", "val"])


def _quarters(df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly flow values (end, val, filed) from direct 3-month entries and year-to-date differences."""
    if df.empty:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = df.dropna(subset=["start"]).copy()
    df["days"] = (df["end"] - df["start"]).dt.days
    # First filing of each period wins: later restatements were not known at the time.
    df = df.sort_values(["rank", "filed"]).drop_duplicates(["start", "end"], keep="first")
    q = df[df["days"].between(80, 100)][["end", "val", "filed"]]
    ytd = df[df["days"].between(80, 380)].sort_values(["start", "end"])
    derived = []
    for _, g in ytd.groupby("start"):
        g = g.sort_values("end")
        prev = g.shift(1)
        step = (g["end"] - prev["end"]).dt.days
        ok = step.between(80, 100)
        derived.append(pd.DataFrame({"end": g["end"][ok], "val": (g["val"] - prev["val"])[ok],
                                     "filed": g["filed"][ok]}))
    out = pd.concat([q, *derived], ignore_index=True)
    return out.sort_values("filed").drop_duplicates("end", keep="first").sort_values("end")


def _ttm(q: pd.DataFrame) -> pd.DataFrame:
    if len(q) < 4:
        return pd.DataFrame(columns=["end", "val", "filed"])
    q = q.reset_index(drop=True)
    span = (q["end"] - q["end"].shift(3)).dt.days
    val = q["val"].rolling(4).sum()
    filed = pd.to_datetime(q["filed"].astype("int64").rolling(4).max())  # TTM is public once its last quarter is
    ok = span.between(250, 300)
    return pd.DataFrame({"end": q["end"][ok], "val": val[ok], "filed": filed[ok]})


def _stock(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["start"].isna()].sort_values(["rank", "filed"]).drop_duplicates("end", keep="first")
    return df[["end", "val", "filed"]].sort_values("end")


def company_series(facts: dict) -> dict[str, pd.DataFrame]:
    out = {k: _ttm(_quarters(_entries(facts, v))) for k, v in FLOW.items()}
    out.update({k: _stock(_entries(facts, v)) for k, v in STOCK.items()})
    return out


def _asof(series: pd.DataFrame, dates: pd.DatetimeIndex, max_age_days: int = 400) -> np.ndarray:
    """Latest value public by each date (filed + 1 day), ignoring stale periods."""
    if series.empty:
        return np.full(len(dates), np.nan)
    s = series.assign(avail=series["filed"] + pd.Timedelta(days=1)).sort_values("avail")
    s = s.groupby("avail").last()
    idx = s.index.searchsorted(dates, side="right") - 1
    vals = np.where(idx >= 0, s["val"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    ends = np.where(idx >= 0, s["end"].to_numpy()[np.clip(idx, 0, None)], np.datetime64("NaT"))
    age = (dates.to_numpy() - ends).astype("timedelta64[D]").astype(float)
    return np.where(age <= max_age_days, vals, np.nan)


def fundamental_panel(tickers: list[str], dates: pd.DatetimeIndex, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Item -> (dates x tickers) point-in-time values. Cached per run of dates."""
    key = f"fund_{dates[0].date()}_{dates[-1].date()}_{len(dates)}_{len(tickers)}"
    cached = None if refresh else store.read("factors", key)
    if cached is not None:
        return {item: g.drop(columns="item").pivot(index="date", columns="ticker", values="val")
                .reindex(index=dates, columns=tickers) for item, g in cached.groupby("item")}
    cik = sec.ticker_to_cik()
    items = list(FLOW) + list(STOCK)
    arrays = {k: np.full((len(dates), len(tickers)), np.nan) for k in items}
    for j, t in enumerate(tickers):
        c = cik.get(t.upper())
        facts = store.read_json("sec_facts", str(c)) if c else None
        if facts:
            for k, s in company_series(facts).items():
                arrays[k][:, j] = _asof(s, dates)
        if (j + 1) % 100 == 0:
            print(f"[fund] {j + 1}/{len(tickers)}", file=sys.stderr, flush=True)
    frames = {k: pd.DataFrame(a, index=dates, columns=tickers) for k, a in arrays.items()}
    long = pd.concat([f.stack().rename("val").reset_index().assign(item=k) for k, f in frames.items()])
    long.columns = ["date", "ticker", "val", "item"]
    store.write(long, "factors", key)
    return frames


def fundamental_factors(f: dict[str, pd.DataFrame], raw_price: pd.DataFrame) -> dict[str, pd.DataFrame]:
    a = f["assets"].where(f["assets"] > 0)
    mcap = (raw_price * f["shares"]).where(lambda x: x > 0)
    lag = 12  # monthly frames: 12 rows = one year earlier
    g, ni, cfo = f["gross"], f["ni"], f["cfo"]
    lt_lev = f["debt"] / a
    cur = f["cur_assets"] / f["cur_liab"].where(f["cur_liab"] > 0)
    gm = g / f["revenue"].where(f["revenue"] > 0)
    turn = f["revenue"] / a
    roa = ni / a
    def gt(x, y):  # NaN when either side is missing; plain '>' would silently score missing data as 0
        return (x > y).astype(float).where(x.notna() & (y.notna() if hasattr(y, "notna") else True))

    sh = f["shares"]
    signals = [gt(roa, 0), gt(cfo, 0), gt(roa, roa.shift(lag)), gt(cfo, ni), gt(lt_lev.shift(lag), lt_lev),
               gt(cur, cur.shift(lag)), gt(sh.shift(lag) * 1.0001, sh), gt(gm, gm.shift(lag)),
               gt(turn, turn.shift(lag))]
    avail = sum(s.notna().astype(int) for s in signals)
    f_score = sum(s.fillna(0) for s in signals).where(avail >= 6)
    wc = (f["cur_assets"] - f["cur_liab"]) / a
    z = (1.2 * wc + 1.4 * f["retained"] / a + 3.3 * f["ebit"] / a
         + 0.6 * mcap / f["liabilities"].where(f["liabilities"] > 0) + 1.0 * f["revenue"] / a)
    return {
        "gross_prof": g / a,
        "accruals": (ni - cfo) / a,
        "asset_growth": a / a.shift(lag) - 1,
        "dilution": f["shares"] / f["shares"].shift(lag) - 1,
        "earn_yield": ni / mcap,
        "sales_yield": f["revenue"] / mcap,
        "f_score": f_score,
        "leverage_crash": (f["debt"].fillna(0) - f["cash"].fillna(0)) / a,
        "zscore_crash": z,
        "mcap": mcap,
    }

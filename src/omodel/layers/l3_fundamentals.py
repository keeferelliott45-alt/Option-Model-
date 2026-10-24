from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from omodel import params

EARNINGS_CYCLE_DAYS = params.register("EARNINGS_CYCLE_DAYS", 91, "theory", "L3",
                                      "Quarterly reporting: next release estimated one quarter after the last.")
EARNINGS_BUFFER_DAYS = params.register("EARNINGS_BUFFER_DAYS", 10, "convention", "L3",
                                       "Slack around the estimated next release date.")
MAX_ND_EBITDA = params.register("MAX_ND_EBITDA", 4.0, "theory", "L3", "Leverage above this is a refinancing risk.")
MIN_INT_COVERAGE = params.register("MIN_INT_COVERAGE", 1.5, "theory", "L3", "Operating income / interest floor.")
SOFT_ND_EBITDA = params.register("SOFT_ND_EBITDA", 3.0, "theory", "L3", "Elevated leverage: soft veto.")
SOFT_GM_DROP = params.register("SOFT_GM_DROP", -0.02, "theory", "L3", "Gross margin down > 2pp YoY: soft veto.")
SOFT_PS_VS_OWN = params.register("SOFT_PS_VS_OWN", 1.3, "theory", "L3",
                                 "P/S above 1.3x its own 3-year median: expensive going in, soft veto.")

CONCEPTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross": ["GrossProfit"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "op": ["OperatingIncomeLoss"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "interest": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
    "debt": ["LongTermDebt", "LongTermDebtNoncurrent", "DebtCurrent", "LongTermDebtCurrent", "ShortTermBorrowings"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
}
FLOWS = ("revenue", "gross", "cogs", "op", "da", "interest")


def facts_long(facts: dict | None) -> pd.DataFrame:
    """companyfacts JSON -> rows (concept, start, end, val, filed); only USD/shares units."""
    rows = []
    for ns in ("us-gaap", "dei"):
        for concept, body in ((facts or {}).get("facts", {}).get(ns) or {}).items():
            for unit, items in (body.get("units") or {}).items():
                if unit not in ("USD", "shares"):
                    continue
                for it in items:
                    rows.append((concept, it.get("start"), it.get("end"), it.get("val"), it.get("filed")))
    df = pd.DataFrame(rows, columns=["concept", "start", "end", "val", "filed"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["days"] = (df["end"] - df["start"]).dt.days
    return df.dropna(subset=["end", "filed", "val"])


def _known(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    # Point-in-time: only what was filed by the as-of date; later restatements of the same period are ignored.
    k = df[df["filed"] <= asof].sort_values("filed")
    return k.drop_duplicates(["concept", "start", "end"], keep="last")


def _flow_ttm(k: pd.DataFrame, names: list[str]) -> float:
    for name in names:
        d = k[k["concept"] == name]
        q = d[d["days"].between(80, 100)].drop_duplicates("end", keep="last").sort_values("end")
        if len(q) >= 4 and (q["end"].iloc[-1] - q["end"].iloc[-4]).days < 300:
            return float(q["val"].iloc[-4:].sum())
        a = d[d["days"].between(350, 380)].sort_values("end")
        if len(a):
            return float(a["val"].iloc[-1])
    return np.nan


def _instant(k: pd.DataFrame, names: list[str], asof: pd.Timestamp) -> float:
    d = k[k["concept"].isin(names) & k["start"].isna()]
    d = d[d["end"] >= asof - pd.Timedelta(days=200)]
    if d.empty:
        return np.nan
    latest = d[d["end"] == d["end"].max()]
    return float(latest.groupby("concept")["val"].last().max())


def metrics(fl: pd.DataFrame, asof: pd.Timestamp) -> dict:
    k = _known(fl, asof)
    m = {f: _flow_ttm(k, CONCEPTS[f]) for f in FLOWS}
    if not np.isfinite(m["gross"]) and np.isfinite(m["cogs"]):
        m["gross"] = m["revenue"] - m["cogs"]
    m["debt"] = _instant(k, CONCEPTS["debt"], asof)
    m["cash"] = _instant(k, CONCEPTS["cash"], asof)
    m["shares"] = _instant(k, ["EntityCommonStockSharesOutstanding"], asof)
    with np.errstate(divide="ignore", invalid="ignore"):
        m["gm"] = m["gross"] / m["revenue"]
        ebitda = m["op"] + np.nan_to_num(m["da"])
        m["ebitda"] = ebitda
        m["nd_ebitda"] = (np.nan_to_num(m["debt"]) - np.nan_to_num(m["cash"])) / ebitda if ebitda > 0 else np.inf
        m["int_cov"] = m["op"] / m["interest"] if m["interest"] and m["interest"] > 0 else np.inf
    return m


@dataclass
class Veto:
    status: str                      # HARD | SOFT | PASS | UNKNOWN
    reasons: list[str] = field(default_factory=list)
    score: float = np.nan            # 0-100 positioning score (100 = clean)


def next_earnings_estimate(earnings_days: pd.DatetimeIndex, asof: pd.Timestamp) -> pd.Timestamp | None:
    past = earnings_days[earnings_days <= asof]
    return None if past.empty else past.max() + pd.Timedelta(days=EARNINGS_CYCLE_DAYS)


def veto(fl: pd.DataFrame | None, asof: pd.Timestamp, expiry: pd.Timestamp, earnings_days: pd.DatetimeIndex,
         bullish: bool, ps_now: float = np.nan, ps_hist: list[float] | None = None) -> Veto:
    """Thesis veto. Earnings inside the option's life kill any trade; balance-sheet traps kill bullish ones."""
    reasons, soft = [], []
    nxt = next_earnings_estimate(earnings_days, asof)
    if nxt is not None and expiry >= nxt - pd.Timedelta(days=EARNINGS_BUFFER_DAYS):
        return Veto("HARD", [f"earnings expected ~{nxt.date()} before expiry {expiry.date()}"], 0.0)
    if fl is None or fl.empty:
        return Veto("UNKNOWN", ["no SEC fundamentals"], np.nan)
    m = metrics(fl, asof)
    m_prev = metrics(fl, asof - pd.Timedelta(days=365))
    if bullish and m["nd_ebitda"] > MAX_ND_EBITDA and m["int_cov"] < MIN_INT_COVERAGE:
        return Veto("HARD", [f"net debt/EBITDA {m['nd_ebitda']:.1f}, interest cover {m['int_cov']:.1f}"], 0.0)
    if m["nd_ebitda"] > SOFT_ND_EBITDA:
        soft.append(f"leverage {m['nd_ebitda']:.1f}x EBITDA")
    gm_trend = m["gm"] - m_prev["gm"]
    if np.isfinite(gm_trend) and gm_trend < SOFT_GM_DROP:
        soft.append(f"gross margin {100 * gm_trend:+.1f}pp YoY")
    growth = m["revenue"] / m_prev["revenue"] - 1 if m_prev["revenue"] else np.nan
    if np.isfinite(growth) and growth < 0:
        soft.append(f"revenue {100 * growth:+.1f}% YoY")
    hist = [x for x in (ps_hist or []) if np.isfinite(x) and x > 0]
    if np.isfinite(ps_now) and len(hist) >= 2 and ps_now > SOFT_PS_VS_OWN * np.median(hist):
        soft.append(f"P/S {ps_now:.1f} vs own 3y median {np.median(hist):.1f}")
    if not np.isfinite(m["revenue"]):
        return Veto("UNKNOWN", ["revenue not reported in XBRL"], np.nan)
    return Veto("SOFT" if soft else "PASS", soft or ["fundamentals clean"], 100.0 - 25.0 * len(soft))


def price_to_sales(fl: pd.DataFrame, asof: pd.Timestamp, raw_price: float) -> float:
    m = metrics(fl, asof)
    if not (np.isfinite(m["shares"]) and np.isfinite(m["revenue"]) and m["revenue"] > 0):
        return np.nan
    return raw_price * m["shares"] / m["revenue"]

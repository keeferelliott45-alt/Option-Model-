from __future__ import annotations

import pandas as pd

from omodel import params
from omodel.data import http, store

SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
           "Accept": "application/json"}
MIN_MCAP = params.register("UNIVERSE_MIN_MCAP", 2e9, "convention", "universe",
                           "Full-market scan: $2B+ companies have reliably listed, liquid options.")
MIN_PRICE = params.register("UNIVERSE_MIN_PRICE", 10.0, "convention", "universe",
                            "Above $10 strike spacing is fine enough for verticals.")
MIN_VOLUME = params.register("UNIVERSE_MIN_VOLUME", 500_000, "convention", "universe",
                             "Daily share volume floor: thin stocks have thin options.")
# Broad, liquid, unleveraged ETFs: no earnings, deepest option markets. Leveraged/volatility products are excluded
# because daily rebalancing makes their return shape unlike their own history.
ETFS = ("SPY", "QQQ", "IWM", "DIA", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
        "SMH", "XBI", "KRE", "XOP", "XRT", "XHB", "ITB", "IBB", "GDX", "GDXJ", "GLD", "SLV", "USO", "UNG", "TLT",
        "IEF", "HYG", "LQD", "EEM", "EFA", "FXI", "EWZ", "EWJ", "KWEB", "ARKK", "IYR", "XME", "JETS", "TAN", "URA")


def stocks(refresh: bool = False) -> pd.DataFrame:
    """All US-listed common stocks with market cap, price and volume (Nasdaq screener, free; cached per day)."""
    key = str(pd.Timestamp.today().date())
    df = None if refresh else store.read("universe_live", key)
    if df is None:
        resp = http.get(SCREENER_URL, params={"tableonly": "true", "limit": 10000, "download": "true"},
                        headers=HEADERS, min_interval=0.5)
        resp.raise_for_status()
        rows = ((resp.json() or {}).get("data") or {}).get("rows") or []
        df = pd.DataFrame(rows)
        num = lambda s: pd.to_numeric(s.astype(str).str.replace(r"[$,%]", "", regex=True), errors="coerce")
        df = pd.DataFrame({"ticker": df["symbol"].str.strip().str.replace("/", ".", regex=False),
                           "name": df["name"], "price": num(df["lastsale"]), "mcap": num(df["marketCap"]),
                           "volume": num(df["volume"]), "sector": df.get("sector"), "country": df.get("country")})
        store.write(df, "universe_live", key, meta={"source": SCREENER_URL})
    return df


def live_universe(refresh: bool = False) -> pd.DataFrame:
    """Scanned names: liquid large/mid caps (no preferreds/warrants/units) plus the ETF list. Columns: ticker, kind, shares."""
    s = stocks(refresh)
    ok = (s["mcap"] >= MIN_MCAP) & (s["price"] >= MIN_PRICE) & (s["volume"] >= MIN_VOLUME)
    ok &= ~s["ticker"].str.contains(r"\^|\.W|\.U|\.R|\s", regex=True)
    out = s[ok].assign(kind="stock", shares=lambda d: d["mcap"] / d["price"])[["ticker", "kind", "shares"]]
    etf = pd.DataFrame({"ticker": list(ETFS), "kind": "etf", "shares": float("nan")})
    return pd.concat([out, etf], ignore_index=True).drop_duplicates("ticker").reset_index(drop=True)

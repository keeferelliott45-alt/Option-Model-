from __future__ import annotations

import sys

import pandas as pd

from omodel import params
from omodel.data import http, store

URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
FIELDS = ["symbolCode", "settlementDate", "currentShortPositionQuantity", "averageDailyVolumeQuantity",
          "daysToCoverQuantity"]
PUBLICATION_LAG_DAYS = params.register(
    "FINRA_PUBLICATION_LAG_DAYS", 12, "convention", "data",
    "FINRA publishes ~8 business days after settlement; 12 calendar days is a conservative availability date.")
_PAGE = 5000


def _post(body: dict) -> list[dict]:
    import requests

    resp = requests.post(URL, json=body, timeout=120,
                         headers={"User-Agent": http.DEFAULT_UA, "Accept": "application/json"})
    if resp.status_code == 204:
        return []
    resp.raise_for_status()
    return resp.json()


def _year(year: int, symbols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(0, len(symbols), 200):
        chunk, offset = symbols[i:i + 200], 0
        while True:
            page = _post({"limit": _PAGE, "offset": offset, "fields": FIELDS,
                          "dateRangeFilters": [{"fieldName": "settlementDate", "startDate": f"{year}-01-01",
                                                "endDate": f"{year}-12-31"}],
                          "domainFilters": [{"fieldName": "symbolCode", "values": chunk}]})
            rows += page
            if len(page) < _PAGE:
                break
            offset += _PAGE
    df = pd.DataFrame(rows, columns=FIELDS)
    df["settlementDate"] = pd.to_datetime(df["settlementDate"])
    return df.rename(columns={"symbolCode": "ticker", "settlementDate": "settlement",
                              "currentShortPositionQuantity": "short_qty", "averageDailyVolumeQuantity": "adv",
                              "daysToCoverQuantity": "days_to_cover"})


def load_short_interest(tickers: list[str], start_year: int = 2018, refresh: bool = False) -> pd.DataFrame:
    """Short interest per (ticker, settlement) with an `available` date for point-in-time use."""
    symbols = sorted({t.replace(".", " ").replace("-", " ") for t in tickers} | set(tickers))
    parts = []
    this_year = pd.Timestamp.today().year
    for year in range(start_year, this_year + 1):
        key = f"si_{year}"
        df = None if refresh or year == this_year else store.read("finra", key)
        if df is None:
            df = _year(year, symbols)
            store.write(df, "finra", key, meta={"source": URL, "symbols": len(symbols)})
            print(f"[finra] {year}: {len(df)} rows", file=sys.stderr, flush=True)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True).drop_duplicates(["ticker", "settlement"])
    out["ticker"] = out["ticker"].str.replace(" ", ".")
    out["available"] = out["settlement"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    return out.sort_values(["ticker", "settlement"]).reset_index(drop=True)

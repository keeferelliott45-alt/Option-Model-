from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Panel:
    """Daily point-in-time data. All frames share one DatetimeIndex of trading days (tz-naive)."""

    close: pd.DataFrame      # split/dividend-adjusted close, tickers as columns
    open: pd.DataFrame       # adjusted open = raw open * adj_close / raw close
    volume: pd.DataFrame     # raw share volume
    member: pd.DataFrame     # bool: was an S&P 500 member on that date
    etf_close: pd.DataFrame  # adjusted close for MARKET_ETF and SECTOR_ETFS (NaN before inception)
    etf_open: pd.DataFrame
    raw_close: pd.DataFrame | None = None  # unadjusted close, for price filters
    macro: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)


@dataclass
class Decomposition:
    """Layer 2 output. Every model quantity at date t is fit on data from t-window .. t-1 only."""

    stock_ret: pd.DataFrame  # daily log close-to-close returns
    mkt_ret: pd.Series       # MARKET_ETF log return
    sector: pd.DataFrame     # assigned sector ETF symbol per (date, ticker), object dtype
    sec_ret: pd.DataFrame    # log return of each stock's assigned sector ETF, aligned to stock_ret
    alpha: pd.DataFrame
    beta_mkt: pd.DataFrame
    beta_sec: pd.DataFrame
    resid: pd.DataFrame      # r_t - (alpha + beta_mkt*rm_t + beta_sec*rs_t) using params fit through t-1
    sigma: pd.DataFrame      # residual std over the fit window (through t-1)
    z: pd.DataFrame          # resid / sigma
    tot_sigma: pd.DataFrame  # std of total log returns over the fit window (through t-1)
    vol_z: pd.DataFrame      # log-volume z-score vs prior VOLUME_Z_WINDOW days

from __future__ import annotations

import numpy as np
import pandas as pd

from omodel.types import Decomposition, Panel


def month_ends(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).max().to_numpy())


def price_factors(panel: Panel, dec: Decomposition, earnings: dict[str, pd.DatetimeIndex] | None = None
                  ) -> dict[str, pd.DataFrame]:
    """Daily factor frames (dates x tickers); every value at t uses data through the close of t only."""
    lc = np.log(panel.close)
    r = dec.stock_ret
    out = {
        "mom_12_1": lc.shift(21) - lc.shift(252),
        "rev_1m": lc - lc.shift(21),
        "high52": panel.close / panel.close.rolling(252, min_periods=200).max(),
        "resid_mom": dec.resid.shift(21).rolling(231, min_periods=180).sum(),
        "idio_vol": dec.sigma,
        "max_ret": r.rolling(21, min_periods=15).max(),
        "beta": dec.beta_mkt,
    }
    out["idio_vol_v"] = out["idio_vol"]
    if earnings:
        pead = pd.DataFrame(np.nan, index=panel.dates, columns=panel.close.columns)
        pos = pd.Series(np.arange(len(panel.dates)), index=panel.dates)
        for t, days in earnings.items():
            if t not in pead.columns:
                continue
            d = days[days.isin(panel.dates)]
            # Earnings-day reaction = residual on the effective day, carried for 63 trading days.
            vals = dec.resid.loc[d, t]
            col = pd.Series(np.nan, index=panel.dates)
            col.loc[d] = vals.to_numpy()
            pead[t] = col.ffill(limit=63)
        out["pead"] = pead
    return out


def targets(panel: Panel, mkt: pd.Series) -> dict[str, pd.DataFrame]:
    """Forward targets at formation t with a one-day skip (start at close t+1)."""
    lc = np.log(panel.close)
    lm = np.log(mkt.reindex(panel.dates))
    ret = (lc.shift(-22) - lc.shift(-1)).sub(lm.shift(-22) - lm.shift(-1), axis=0)
    ret63 = (lc.shift(-64) - lc.shift(-1)).sub(lm.shift(-64) - lm.shift(-1), axis=0)
    daily = lc.diff()
    vol = daily.shift(-22).rolling(21, min_periods=15).std() * np.sqrt(252)
    crash = (ret63 < -0.25).astype(float).where(ret63.notna())
    return {"ret21": ret, "vol21": vol, "crash63": crash}

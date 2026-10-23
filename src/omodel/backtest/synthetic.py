from __future__ import annotations

import operator

import numpy as np
import pandas as pd

from omodel import config, holdout
from omodel.types import Panel

# Fixed properties of the fake world, not model tunables; deliberately kept out of the parameter registry.
_OVERNIGHT_FRAC = 0.35
_MKT_MU, _MKT_SD = 0.0003, 0.011
_N_BURSTS, _BURST_LEN, _BURST_MULT = 3, 20, 2.5
_CRASH_RANGE = (0.05, 0.09)
_SECTOR_BETA_RANGE, _SECTOR_NOISE_SD = (0.8, 1.2), 0.006
_STOCK_ALPHA = 0.0001
_BETA_MKT_RANGE, _BETA_SEC_RANGE, _IDIO_SD_RANGE = (0.7, 1.3), (0.5, 1.5), (0.012, 0.025)
_SHOCK_SCALE_RANGE = (0.7, 1.3)  # mean exactly 1, so E[shock] = shock_size
_VOLUME_BASE, _VOLUME_STOCK_SD, _VOLUME_DAY_SD, _VOLUME_SPIKE = 1e6, 0.3, 0.25, 3.0
_NONMEMBER_FRAC, _NONMEMBER_SPELL = 0.05, (60, 253)


def _split(rng: np.random.Generator, mean: float | np.ndarray, sd: float | np.ndarray,
           shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    # Independent overnight/intraday draws whose sum has exactly the requested close-to-close mean and sd.
    f = _OVERNIGHT_FRAC
    on = rng.normal(f * np.asarray(mean), np.sqrt(f) * np.asarray(sd), size=shape)
    intra = rng.normal((1 - f) * np.asarray(mean), np.sqrt(1 - f) * np.asarray(sd), size=shape)
    return on, intra


def _prices(p0: np.ndarray, on: np.ndarray, intra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_close = np.log(p0) + np.cumsum(on + intra, axis=0)
    return np.exp(log_close), np.exp(log_close - intra)


def _validate(shock_prob: float, reversion: float, shock_size: float, alpha: float) -> None:
    if not 0.0 <= shock_prob <= 1.0:  # also rejects NaN
        raise ValueError(f"shock_prob must be in [0, 1], got {shock_prob!r}")
    for name, v in (("reversion", reversion), ("shock_size", shock_size), ("alpha", alpha)):
        if not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")


def synthetic_panel(n_days: int = 1500, n_stocks: int = 120, n_sectors: int = 4, seed: int = 0,
                    reversion: float = 0.0, shock_prob: float = 0.003, shock_size: float = -0.08,
                    reversion_days: int = 10, start: str = "2005-01-03", *, alpha: float = _STOCK_ALPHA,
                    drift_neutral: bool = False, allow_holdout_dates: bool = False) -> tuple[Panel, pd.DataFrame]:
    """Fake market with known overnight shocks; returns (panel, truth listing every injected shock).
    Recovery is -reversion*shock (mean-reverting for either sign); shock_prob=0 or shock_size=0 gives a shock-free
    world. Unreverted shocks are permanent, so random-day idiosyncratic returns drift by alpha +
    (1-reversion)*shock_prob*shock_size per day; drift_neutral=True removes the shock part leg by leg (pass alpha=0
    too for a zero-drift null). truth.attrs: beta_mkt is the total SPY beta (SPY-only regression), while
    beta_mkt_given_sector and beta_sec are the SPY and raw-sector-ETF slopes of the L2 market+sector model.
    """
    n_days, n_stocks, n_sectors, reversion_days = map(operator.index, (n_days, n_stocks, n_sectors, reversion_days))
    if not 1 <= n_sectors <= len(config.SECTOR_ETFS):
        raise ValueError(f"n_sectors must be in 1..{len(config.SECTOR_ETFS)}")
    if n_days < 2 or n_stocks < 1 or reversion_days < 1:
        raise ValueError("need n_days >= 2, n_stocks >= 1, reversion_days >= 1")
    _validate(shock_prob, reversion, shock_size, alpha)
    dates = pd.bdate_range(start, periods=n_days, name="date")
    # Downstream code silently NaNs rows on/after the holdout start, which would quietly shrink a harness test.
    if not allow_holdout_dates and dates[-1] >= holdout.holdout_start():
        raise ValueError(f"synthetic calendar ends {dates[-1].date()}, on/after holdout start "
                         f"{holdout.holdout_start().date()}; use fewer days, an earlier start or allow_holdout_dates")
    rng = np.random.default_rng(seed)
    sectors = list(config.SECTOR_ETFS[:n_sectors])
    width = max(3, len(str(n_stocks - 1)))
    tickers = [f"S{i:0{width}d}" for i in range(n_stocks)]
    shape = (n_days, n_stocks)

    # Every draw happens unconditionally and in a fixed order, so changing reversion/shock/drift knobs under the
    # same seed leaves all other randomness identical (the counterfactual tests rely on this).
    vol_mult = np.ones(n_days)
    burst_starts = rng.integers(1, max(2, n_days - _BURST_LEN), size=_N_BURSTS)
    for s in burst_starts:
        vol_mult[s + 1:s + 1 + _BURST_LEN] = _BURST_MULT  # each crash day kicks off a high-vol burst
    crash = np.zeros(n_days)
    crash[burst_starts] = -rng.uniform(*_CRASH_RANGE, size=_N_BURSTS)
    m_on, m_id = _split(rng, _MKT_MU, _MKT_SD * vol_mult, (n_days,))
    m_on += _OVERNIGHT_FRAC * crash
    m_id += (1 - _OVERNIGHT_FRAC) * crash

    b_s = rng.uniform(*_SECTOR_BETA_RANGE, size=n_sectors)
    e_on, e_id = _split(rng, 0.0, _SECTOR_NOISE_SD, (n_days, n_sectors))

    true_sector = rng.permutation(np.arange(n_stocks) % n_sectors)
    b_m = rng.uniform(*_BETA_MKT_RANGE, size=n_stocks)
    b_x = rng.uniform(*_BETA_SEC_RANGE, size=n_stocks)
    idio_sd = rng.uniform(*_IDIO_SD_RANGE, size=n_stocks)
    p0_stock = 50.0 * np.exp(rng.normal(0.0, 0.25, size=n_stocks))
    p0_etf = 100.0 * np.exp(rng.normal(0.0, 0.05, size=1 + n_sectors))
    i_on, i_id = _split(rng, alpha, idio_sd, shape)

    is_shock = rng.random(shape) < shock_prob
    is_shock[0] = False  # day 0 has no prior close, so there is no gap to put a shock in
    shock = np.where(is_shock, shock_size * rng.uniform(*_SHOCK_SCALE_RANGE, size=shape), 0.0)
    is_shock &= shock_size != 0.0  # zero-size "shocks" are not shocks: no truth rows, no volume spikes

    log_vol = (np.log(_VOLUME_BASE) + rng.normal(0.0, _VOLUME_STOCK_SD, size=n_stocks)
               + rng.normal(0.0, _VOLUME_DAY_SD, size=shape))
    volume = np.round(np.exp(log_vol) * np.where(is_shock, _VOLUME_SPIKE, 1.0))

    member = np.ones(shape, dtype=bool)
    n_out = int(round(_NONMEMBER_FRAC * n_stocks))
    out_stocks = rng.choice(n_stocks, size=n_out, replace=False)
    spell_start = rng.integers(0, n_days, size=n_out)
    spell_len = rng.integers(*_NONMEMBER_SPELL, size=n_out)
    for j, s0, ln in zip(out_stocks, spell_start, spell_len):
        member[s0:s0 + ln, j] = False

    # Recovery is credited to the intraday legs of t+1..t+reversion_days only, never to the close[t]->open[t+1]
    # gap, so a next-open entry can capture all of it.
    recovery = np.zeros(shape)
    if reversion:
        step = -reversion * shock / reversion_days
        for lag in range(1, min(reversion_days, n_days - 1) + 1):
            recovery[lag:] += step[:-lag]

    drift_on, drift_id = np.zeros(n_days), np.zeros(n_days)
    if drift_neutral:
        # Exact expected shock (gap) and recovery (intraday) per day, including start-up days with fewer recovering
        # shocks, so each leg's unconditional idiosyncratic mean is exactly its share of alpha.
        drift_on[1:] = -shock_prob * shock_size
        n_recovering = np.clip(np.arange(n_days) - 1, 0, reversion_days)
        drift_id[:] = reversion * shock_prob * shock_size * n_recovering / reversion_days

    s_on = b_s * m_on[:, None] + e_on
    s_id = b_s * m_id[:, None] + e_id
    x_on, x_id = e_on[:, true_sector], e_id[:, true_sector]  # sector excess = sector_ret - b_s * mkt
    stock_on = b_m * m_on[:, None] + b_x * x_on + i_on + shock + drift_on[:, None]
    stock_id = b_m * m_id[:, None] + b_x * x_id + i_id + recovery + drift_id[:, None]

    close, open_ = _prices(p0_stock, stock_on, stock_id)
    etf_close, etf_open = _prices(p0_etf, np.column_stack([m_on, s_on]), np.column_stack([m_id, s_id]))

    def frame(a: np.ndarray, cols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(a, index=dates, columns=cols)

    etf_cols = [config.MARKET_ETF, *sectors]
    close_df = frame(close, tickers)
    panel = Panel(
        close=close_df,
        open=frame(open_, tickers),
        volume=frame(volume, tickers),
        member=frame(member, tickers),
        etf_close=frame(etf_close, etf_cols),
        etf_open=frame(etf_open, etf_cols),
        raw_close=close_df.copy(),
        macro=None,
    )

    rows, cols = np.nonzero(is_shock)
    truth = pd.DataFrame({"date": dates[rows], "ticker": np.asarray(tickers)[cols], "shock": shock[rows, cols]})
    truth = truth.sort_values(["date", "ticker"], ignore_index=True)
    # Plain dicts/lists (not Series) so pandas can compare attrs when truth frames are concatenated.
    truth.attrs.update(
        sector=dict(zip(tickers, np.asarray(sectors)[true_sector].tolist())),
        beta_mkt=dict(zip(tickers, b_m.tolist())),
        # Stocks load on the sector EXCESS return, so with the raw sector ETF as a regressor the SPY slope is this.
        beta_mkt_given_sector=dict(zip(tickers, (b_m - b_x * b_s[true_sector]).tolist())),
        beta_sec=dict(zip(tickers, b_x.tolist())),
        idio_sd=dict(zip(tickers, idio_sd.tolist())),
        sector_beta=dict(zip(sectors, b_s.tolist())),
        crash_dates=sorted(dates[burst_starts].unique().tolist()),
    )
    return panel, truth

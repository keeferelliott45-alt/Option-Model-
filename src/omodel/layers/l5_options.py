from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from omodel import params

MULTIPLIER = params.register("CONTRACT_MULTIPLIER", 100, "convention", "L5", "US equity options: 100 shares.")
COMMISSION = params.register("COMMISSION_PER_CONTRACT", 0.65, "convention", "L5",
                             "Typical retail per-contract commission, charged on open and close.")
MIN_OI = params.register("MIN_OPEN_INTEREST", 100, "convention", "L5", "Liquidity floor per leg.")
MIN_VOLUME = params.register("MIN_VOLUME", 10, "convention", "L5", "Liquidity floor per leg (day volume).")
MAX_SPREAD_PCT = params.register("MAX_SPREAD_PCT", 0.10, "convention", "L5",
                                 "Max (ask-bid)/mid per leg: an edge that cannot be filled is not an edge.")
DELTA_EDGES = params.register("DELTA_EDGES", (0.05, 0.20, 0.40, 0.60, 0.80), "theory", "L5",
                              "Thesis delta buckets; reported separately, never pooled.")
EV_HURDLE = params.register("EV_HURDLE", 0.05, "theory", "L5", "Min EV as a fraction of max loss to surface.")
FAR_OTM_HURDLE = params.register("FAR_OTM_EV_HURDLE", 0.15, "theory", "L5",
                                 "Higher bar for 0.05-0.20 delta: most fragile to distribution error.")
MAX_WIDTH_STEPS = params.register("MAX_SPREAD_WIDTH_STEPS", 4, "convention", "L5",
                                  "Vertical spreads span at most this many listed strikes.")
RISK_FREE = params.register("RISK_FREE_DEFAULT", 0.04, "convention", "L5",
                            "Fallback rate when no Treasury yield is passed in.")

_IV_LO, _IV_HI = 1e-4, 5.0


def _d1d2(S, K, T, r, q, sigma):
    S, K, T, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, sigma))
    vt = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vt
    return d1, d1 - vt


def bs_price(S, K, T, sigma, cp, r: float | None = None, q: float = 0.0) -> np.ndarray:
    r = RISK_FREE if r is None else r
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    call = np.asarray(cp) == "C"
    S, K, T = (np.asarray(x, dtype=float) for x in (S, K, T))
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    c = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    p = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    return np.where(call, c, p)


def bs_delta(S, K, T, sigma, cp, r: float | None = None, q: float = 0.0) -> np.ndarray:
    r = RISK_FREE if r is None else r
    d1, _ = _d1d2(S, K, T, r, q, sigma)
    dq = np.exp(-q * np.asarray(T, dtype=float))
    return np.where(np.asarray(cp) == "C", dq * norm.cdf(d1), dq * (norm.cdf(d1) - 1.0))


def implied_vol(price, S, K, T, cp, r: float | None = None, q: float = 0.0, iters: int = 100) -> np.ndarray:
    """Vectorized bisection; NaN where the price is outside no-arbitrage bounds."""
    price, S, K, T = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, S, K, T)))
    cp = np.broadcast_to(np.asarray(cp), price.shape)
    lo, hi = np.full(price.shape, _IV_LO), np.full(price.shape, _IV_HI)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = bs_price(S, K, T, mid, cp, r, q) > price
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    iv = 0.5 * (lo + hi)
    ok = (bs_price(S, K, T, _IV_LO, cp, r, q) <= price) & (price <= bs_price(S, K, T, _IV_HI, cp, r, q))
    return np.where(ok, iv, np.nan)


SMILE_DEGREE = params.register("SMILE_DEGREE", 3, "convention", "L5",
                               "Weighted cubic in log-moneyness: smooth enough to ignore wing noise, flexible "
                               "enough for skew. Flat beyond the quoted range.")


def risk_neutral_density(strikes, ivs, S: float, T: float, r: float | None = None, q: float = 0.0,
                         n_grid: int = 801, span: float = 4.0, weights=None) -> tuple[np.ndarray, np.ndarray]:
    """Breeden-Litzenberger on a liquidity-weighted smooth smile: returns (S_T grid, probability mass).

    Following noisy wing quotes exactly creates spikes in the second derivative, so the smile is a weighted
    low-order fit; quotes with tight markets dominate it.
    """
    r = RISK_FREE if r is None else r
    k, v = np.asarray(strikes, dtype=float), np.asarray(ivs, dtype=float)
    w = np.ones(len(k)) if weights is None else np.asarray(weights, dtype=float)
    ok = np.isfinite(k) & np.isfinite(v) & (v > 0) & np.isfinite(w) & (w > 0)
    if ok.sum() < SMILE_DEGREE + 2:
        raise ValueError("too few usable IVs to fit a smile")
    x, v, w = np.log(k[ok] / S), v[ok], w[ok]
    coef = np.polyfit(x, v, SMILE_DEGREE, w=np.sqrt(w))
    atm = max(float(np.polyval(coef, 0.0)), 0.05)
    half = span * atm * np.sqrt(T)
    grid = S * np.exp(np.linspace(-half, half, n_grid))
    gx = np.clip(np.log(grid / S), x.min(), x.max())
    smile = np.clip(np.polyval(coef, gx), _IV_LO, _IV_HI)
    calls = bs_price(S, grid, T, smile, "C", r, q)
    dens = np.exp(r * T) * np.gradient(np.gradient(calls, grid), grid)
    mass = np.clip(dens, 0.0, None) * np.gradient(grid)
    return grid, mass / mass.sum()


def lognormal_grid(S: float, T: float, sigma: float, mu: float = 0.0, n_grid: int = 801,
                   span: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Model distribution of S_T for a given annualised drift and vol (e.g. a realized-vol forecast)."""
    sd = sigma * np.sqrt(T)
    z = np.linspace(-span, span, n_grid)
    grid = S * np.exp((mu - 0.5 * sigma ** 2) * T + sd * z)
    w = norm.pdf(z)
    return grid, w / w.sum()


@dataclass(frozen=True)
class Leg:
    cp: str        # 'C' or 'P'
    strike: float
    qty: int       # +1 long, -1 short (per spread unit)
    bid: float
    ask: float
    delta: float = np.nan


@dataclass(frozen=True)
class Structure:
    name: str
    expiry: pd.Timestamp
    legs: tuple[Leg, ...]

    def payoff(self, s_t: np.ndarray) -> np.ndarray:
        s_t = np.asarray(s_t, dtype=float)
        out = np.zeros_like(s_t)
        for leg in self.legs:
            intrinsic = np.maximum(s_t - leg.strike, 0) if leg.cp == "C" else np.maximum(leg.strike - s_t, 0)
            out += leg.qty * intrinsic
        return out * MULTIPLIER

    def entry_cost(self) -> float:
        """Cash paid to open (negative = credit), crossing the spread: buy at ask, sell at bid."""
        px = sum(leg.qty * (leg.ask if leg.qty > 0 else leg.bid) for leg in self.legs)
        return px * MULTIPLIER + COMMISSION * sum(abs(leg.qty) for leg in self.legs)

    @property
    def short_delta(self) -> float:
        shorts = [abs(leg.delta) for leg in self.legs if leg.qty < 0]
        longs = [abs(leg.delta) for leg in self.legs if leg.qty > 0]
        return float((shorts or longs)[0])


def evaluate(structure: Structure, grid: np.ndarray, probs: np.ndarray) -> dict:
    """EV under a supplied distribution, net of crossing the spread and commissions (open + close)."""
    pay = structure.payoff(grid)
    cost = structure.entry_cost()
    close_comm = COMMISSION * sum(abs(leg.qty) for leg in structure.legs)
    pnl = pay - cost - close_comm
    max_loss = float(-pnl.min()) if pnl.min() < 0 else 0.0
    ev = float(probs @ pnl)
    return dict(name=structure.name, expiry=structure.expiry, cost=cost, ev=ev,
                max_loss=max_loss, max_gain=float(pnl.max()),
                ev_pct_risk=ev / max_loss if max_loss > 0 else np.nan,
                prob_profit=float(probs[pnl > 0].sum()), delta=structure.short_delta)


def delta_bucket(delta: float) -> str:
    d = abs(delta)
    edges = DELTA_EDGES
    if not np.isfinite(d) or d < edges[0]:
        return "<0.05"
    for lo, hi in zip(edges[:-1], edges[1:]):
        if d < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return f"{edges[-1]:.2f}+"


def liquid(rows: pd.DataFrame) -> pd.Series:
    mid = (rows["bid"] + rows["ask"]) / 2
    spread = (rows["ask"] - rows["bid"]) / mid.where(mid > 0)
    return ((rows["open_interest"] >= MIN_OI) & (rows["volume"] >= MIN_VOLUME) & (rows["bid"] > 0)
            & (spread <= MAX_SPREAD_PCT)).fillna(False)


def _leg(row, qty: int) -> Leg:
    return Leg(row.cp, float(row.strike), qty, float(row.bid), float(row.ask), float(row.delta))


def vertical_spreads(chain: pd.DataFrame, expiry: pd.Timestamp, max_steps: int | None = None) -> list[Structure]:
    """Defined-risk verticals for one expiry: credit and debit, calls and puts. Legs must pass liquidity."""
    steps = MAX_WIDTH_STEPS if max_steps is None else max_steps
    ex = chain[(chain["expiry"] == expiry) & liquid(chain)]
    out: list[Structure] = []
    for cp in ("C", "P"):
        side = ex[ex["cp"] == cp].sort_values("strike").reset_index(drop=True)
        for i in range(len(side)):
            for j in range(i + 1, min(i + 1 + steps, len(side))):
                lo, hi = side.iloc[i], side.iloc[j]
                if cp == "C":
                    out.append(Structure("bear_call_credit", expiry, (_leg(lo, -1), _leg(hi, +1))))
                    out.append(Structure("bull_call_debit", expiry, (_leg(lo, +1), _leg(hi, -1))))
                else:
                    out.append(Structure("bull_put_credit", expiry, (_leg(hi, -1), _leg(lo, +1))))
                    out.append(Structure("bear_put_debit", expiry, (_leg(hi, +1), _leg(lo, -1))))
    return out


MARKET_WEIGHT = params.register(
    "MARKET_WEIGHT", 0.5, "convention", "L5",
    "Until the forward test proves the model, trades are scored on a 50/50 blend of the model's forecast and the "
    "market-implied (Breeden-Litzenberger) distribution: a pick must survive the market's own view.")


def _candidates(chain: pd.DataFrame, steps: int) -> list[dict]:
    """Per option side: strikes plus index arrays describing every single and vertical as s*(I[a] - I[b])."""
    out = []
    c2, c4 = 2 * COMMISSION, 4 * COMMISSION
    for cp in ("C", "P"):
        side = chain[(chain["cp"] == cp) & liquid(chain)].sort_values("strike").reset_index(drop=True)
        n = len(side)
        if not n:
            continue
        K = side["strike"].to_numpy(float)
        bid, ask, dlt = (side[c].to_numpy(float) for c in ("bid", "ask", "delta"))
        mid = np.append((bid + ask) / 2, 0.0)  # index n = "no leg"
        rec = {k: [] for k in ("name", "legs", "a", "b", "s", "cost", "max_loss", "max_gain", "delta")}

        def add(name, legs, a, b, s, cost, max_loss, max_gain, delta):
            for k, v in zip(rec, (name, legs, a, b, s, cost, max_loss, max_gain, delta)):
                rec[k].append(v)

        for i in range(n):
            add("long_call" if cp == "C" else "long_put", f"+{cp}{K[i]:g}", i, n, 1.0,
                ask[i] * MULTIPLIER + c2, ask[i] * MULTIPLIER + c2, np.inf, dlt[i])
            for j in range(i + 1, min(i + 1 + steps, n)):
                w = (K[j] - K[i]) * MULTIPLIER
                if cp == "C":
                    d = (ask[i] - bid[j]) * MULTIPLIER
                    add("bull_call_debit", f"+C{K[i]:g}/-C{K[j]:g}", i, j, 1.0, d + c4, d + c4, w - d - c4, dlt[i])
                    c = (bid[i] - ask[j]) * MULTIPLIER
                    add("bear_call_credit", f"-C{K[i]:g}/+C{K[j]:g}", i, j, -1.0, -c + c4, w - c + c4, c - c4, dlt[i])
                else:
                    c = (bid[j] - ask[i]) * MULTIPLIER
                    add("bull_put_credit", f"-P{K[j]:g}/+P{K[i]:g}", i, j, 1.0, -c + c4, w - c + c4, c - c4, dlt[j])
                    d = (ask[j] - bid[i]) * MULTIPLIER
                    add("bear_put_debit", f"+P{K[j]:g}/-P{K[i]:g}", j, i, 1.0, d + c4, d + c4, w - d - c4, dlt[j])
        arr = {k: np.asarray(v) for k, v in rec.items()}
        # The market's own fair value of the structure: what you'd make if the quotes are right.
        arr["mid_value"] = arr["s"] * (mid[arr["a"]] - mid[arr["b"]]) * MULTIPLIER
        out.append({"cp": cp, "K": K, **arr})
    return out


def _moments(c: dict, grid: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean, second moment and P(profit) of each candidate's P&L under a discrete S_T distribution."""
    K = c["K"]
    intr = np.maximum(grid[None, :] - K[:, None], 0) if c["cp"] == "C" else np.maximum(K[:, None] - grid[None, :], 0)
    intr = np.vstack([intr, np.zeros(len(grid))])  # row n = "no leg"
    pnl = c["s"][:, None] * (intr[c["a"]] - intr[c["b"]]) * MULTIPLIER - c["cost"][:, None]
    return pnl @ probs, (pnl ** 2) @ probs, (pnl > 0) @ probs


def scan_expiry(chain: pd.DataFrame, grid: np.ndarray, probs: np.ndarray, max_steps: int | None = None,
                market: tuple[np.ndarray, np.ndarray] | None = None, market_weight: float | None = None) -> pd.DataFrame:
    """Every liquid single and vertical in one expiry, scored under the model and (if given) the market density.

    Fills: buy at ask, sell at bid, commissions on open and close. `ev`/`prob_profit`/`sd` are for the blend
    (1 - w) * model + w * market; `score` = ev / sd, a per-trade Sharpe ratio that penalises lottery tickets.
    """
    steps = MAX_WIDTH_STEPS if max_steps is None else max_steps
    w = MARKET_WEIGHT if market_weight is None else market_weight
    frames = []
    for c in _candidates(chain, steps):
        m1, m2, pp = _moments(c, grid, probs)
        # Market EV is anchored to the quotes (mid value - cost), exact by construction. The market density, when
        # one can be built, supplies dispersion and P(profit); otherwise the model's are used for those two only.
        d1, d2, kp = _moments(c, *market) if market is not None else (m1, m2, pp)
        k1 = c["mid_value"] - c["cost"]
        k2 = np.maximum(d2 - d1 ** 2, 0) + k1 ** 2
        mean, second, pop = (1 - w) * m1 + w * k1, (1 - w) * m2 + w * k2, (1 - w) * pp + w * kp
        frames.append(pd.DataFrame({
            "name": c["name"], "legs": c["legs"], "ev": mean, "ev_model": m1, "ev_market": k1,
            "max_loss": c["max_loss"], "max_gain": c["max_gain"], "prob_profit": pop, "pop_model": pp,
            "pop_market": kp, "sd": np.sqrt(np.maximum(second - mean ** 2, 0)), "delta": c["delta"]}))
    cols = ["name", "legs", "ev", "ev_model", "ev_market", "max_loss", "max_gain", "prob_profit", "pop_model",
            "pop_market", "sd", "delta"]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    out[cols[2:]] = out[cols[2:]].astype(float)
    out = out[(out["max_loss"] > 0) & (out["max_gain"] > 0) & np.isfinite(out["ev"])]
    out["ev_pct_risk"] = out["ev"] / out["max_loss"]
    out["score"] = out["ev"] / out["sd"].where(out["sd"] > 0)
    return out.reset_index(drop=True)


def rank(candidates: list[dict]) -> pd.DataFrame:
    """EV % of max loss must clear a per-bucket hurdle; survivors are ordered by `score` (EV / sd) when present.

    0.80+ delta is excluded (no optionality left) and < 0.05 delta is excluded (pure tail bets).
    """
    df = pd.DataFrame(candidates)
    if df.empty:
        return df
    df["bucket"] = df["delta"].map(delta_bucket)
    hurdle = np.where(df["bucket"] == f"{DELTA_EDGES[0]:.2f}-{DELTA_EDGES[1]:.2f}", FAR_OTM_HURDLE, EV_HURDLE)
    keep = (df["ev_pct_risk"] >= hurdle) & (df["bucket"] != f"{DELTA_EDGES[-1]:.2f}+") & (df["bucket"] != "<0.05")
    key = "score" if "score" in df else "ev_pct_risk"
    return df[keep].sort_values(key, ascending=False).reset_index(drop=True)

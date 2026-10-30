from __future__ import annotations

import json
import zlib
import sys

import numpy as np
import pandas as pd

from omodel import config, params
from omodel.config import MARKET_ETF, SECTOR_ETFS
from omodel.data import cboe_options, earnings_calendar, finra, macro, sec, store, universe, yahoo
from omodel.data import panel as panel_mod
from omodel.experiments.shortlist import _go_live
from omodel.factors import fundamental as fu
from omodel.forecast import engine as eng
from omodel.forecast import fit
from omodel.layers import l1_regime as l1
from omodel.layers import l3_fundamentals as l3
from omodel.layers import l5_options as l5

MIN_DTE = params.register("SCAN_MIN_DTE", 7, "convention", "scanner", "Shortest expiry scanned (calendar days).")
MAX_DTE = params.register("SCAN_MAX_DTE", 180, "convention", "scanner", "Longest expiry scanned (calendar days).")
PER_TICKER = params.register("SCAN_MAX_PER_TICKER", 2, "convention", "scanner", "Diversification: picks per stock.")
PER_SECTOR_SIDE = params.register("SCAN_MAX_PER_SECTOR_SIDE", 3, "convention", "scanner",
                                  "Diversification: same-direction picks per sector ETF.")
TOP_N = params.register("SCAN_TOP_N", 25, "convention", "scanner", "Picks shown per day.")
POST_EARN_DAYS = params.register("SCAN_POST_EARNINGS_DAYS", 22, "theory", "scanner",
                                 "Skip a stock while its last earnings day is inside the RV22 input window.")
IV_GUARD = params.register("SCAN_IV_GUARD", 1.5, "convention", "scanner",
                           "Hide contracts when model vol / market iv30 is outside [1/1.5, 1.5].")
SMILE_MAX_SPREAD = params.register("SMILE_MAX_SPREAD", 0.25, "convention", "scanner",
                                   "Quotes wider than 25% of mid are too noisy to shape the market density.")
BULLISH = {"long_call", "bull_call_debit", "bull_put_credit"}
BEARISH = {"long_put", "bear_put_debit", "bear_call_credit"}


def _log(msg: str) -> None:
    print(f"[scanner] {msg}", file=sys.stderr, flush=True)


def _latest_short_pct(tickers: list[str], shares: dict[str, float], asof: pd.Timestamp) -> dict[str, float]:
    si = finra.load_short_interest(tickers)
    si = si[si["available"] <= asof]
    last = si.sort_values("settlement").groupby("ticker").last()
    return {t: float(last.at[t, "short_qty"] / shares[t]) for t in last.index
            if t in shares and np.isfinite(shares[t]) and shares[t] > 0 and (asof - last.at[t, "settlement"]).days <= 45}


def run(refresh: bool = True) -> pd.DataFrame:
    _go_live()
    models = fit.load()
    if fit.is_stale(models):
        _log("refitting forecast models (monthly)")
        models = fit.fit_all()
    membership = universe.load_membership(refresh=refresh)
    uni = scan_universe(membership, refresh)
    members = list(uni["ticker"])
    _log(f"universe: {len(members)} names ({uni['group'].value_counts().to_dict()})")
    if refresh:
        for t in [MARKET_ETF, *SECTOR_ETFS, *members]:
            try:
                yahoo.load_daily(t, refresh=True)
            except Exception as exc:
                _log(f"{t}: price refresh failed ({exc})")
    today = pd.Timestamp.today().normalize()
    # Score the session of the latest saved chains, never today's partial bar (a run after the open would
    # otherwise look for chains that cannot exist yet).
    snap_root = config.settings().data_dir / "options_snapshots"
    snap_days = sorted(pd.Timestamp(p.name) for p in snap_root.iterdir() if p.is_dir()) if snap_root.exists() else []
    if snap_days:
        today = min(today, snap_days[-1])
    panel = panel_mod.build_panel(start=today - pd.Timedelta(days=1400), end=today, tickers=members,
                                  membership=membership)
    asof = panel.dates[-1]
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel(refresh=refresh)
    feats = eng.har_features(ret, mac["vix"], mac["vix3m"])
    X = pd.DataFrame({k: v.loc[asof] for k, v in feats.items()})
    regime_feats = l1.regime_features(mac, panel, ret)
    regime = l1.regime_label(regime_feats.loc[asof])
    _log(f"{asof.date()}: regime {regime}")

    cal = earnings_calendar.upcoming(asof, days=130, refresh_after=asof + pd.Timedelta(days=45))
    cik = sec.ticker_to_cik()
    shares = dict(zip(uni["ticker"], uni["shares"]))  # screener market cap / price; NaN for ETFs
    facts_by = _LazyFacts(cik)
    spct = _latest_short_pct(members, shares, asof)
    sector = _sectors(panel, ret)

    rows = []
    for t in members:
        if t not in X.index or X.loc[t].isna().any():
            continue
        chain = cboe_options.load_snapshot(t, asof)
        if chain is None or chain.empty:
            continue
        spot = float(chain.attrs.get("underlying_price") or panel.raw_close.at[asof, t])
        z = eng.ewma_z(ret[t].to_numpy())[-252 * eng.FHS_YEARS:]
        nxt_earn = cal.get(t.upper())
        subs = sec.load_submissions(cik[t.upper()]) if t.upper() in cik and config.settings().sec_user_agent else None
        past_earn = sec.earnings_days(subs, panel.dates) if subs is not None and len(subs) else pd.DatetimeIndex([])
        if nxt_earn is None and len(past_earn):
            nxt_earn = l3.next_earnings_estimate(past_earn, asof)
        if len(past_earn) and past_earn.max() >= panel.dates[-POST_EARN_DAYS]:
            continue  # recent earnings jump still inside the RV inputs (Stage 2 amendment 1: veto, not a model change)
        iv30 = float(chain.attrs.get("iv30") or np.nan)
        dte = (pd.to_datetime(chain["expiry"]) - asof).dt.days
        for expiry in sorted(chain.loc[dte.between(MIN_DTE, MAX_DTE), "expiry"].unique()):
            expiry = pd.Timestamp(expiry)
            if nxt_earn is not None and asof < nxt_earn <= expiry:
                continue  # Stage 2: the engine is not calibrated across earnings
            h = max(int(np.busday_count(asof.date(), expiry.date())), 1)
            vol = float(fit.har_vol(models, h, X.loc[[t]])[0])
            ex = chain[(chain["expiry"] == expiry) & (chain["iv"] > 0)]
            atm_iv = float(ex.loc[(ex["strike"] - spot).abs().idxmin(), "iv"]) if len(ex) else iv30
            if np.isfinite(atm_iv) and atm_iv > 0 and not (1 / IV_GUARD <= vol / atm_iv <= IV_GUARD):
                continue  # strong disagreement with the market = missing information, not edge
            ratio = fit.crash_ratio(models, 0.5 * np.log(float(fit.har_vol(models, 21, X.loc[[t]])[0]) ** 2),
                                    np.log(max(spct.get(t, np.nan), 1e-5)) if t in spct else np.nan)
            T = max((expiry - asof).days, 1) / 365
            ex_chain = chain[(chain["expiry"] == expiry) & (chain["bid"] > 0)]
            fwd = spot * np.exp((l5.RISK_FREE - _implied_carry(ex_chain, spot, T)) * T)
            grid, probs = eng.price_distribution(spot, vol, h, z, crash_ratio=ratio, forward=fwd,
                                                 seed=zlib.crc32(f"{t}{expiry.date()}".encode()))
            cand = l5.scan_expiry(chain[chain["expiry"] == expiry], grid, probs,
                                  market=_market_density(chain, expiry, spot, asof))
            if cand.empty:
                continue
            cand = cand.assign(ticker=t, expiry=expiry, dte=int((expiry - asof).days), spot=spot, vol_fc=vol,
                               short_pct=spct.get(t, np.nan), sector=sector.get(t))
            rows.append(cand)
    allc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    picks = _select(allc, facts_by, asof)
    _write(picks, allc, asof, regime)
    return picks


class _LazyFacts(dict):
    """SEC fundamentals parsed only for tickers that reach the veto step (parsing all ~1,600 is slow)."""

    def __init__(self, cik: dict[str, int]):
        super().__init__()
        self._cik = cik

    def get(self, t, default=None):
        if t not in self:
            c = self._cik.get(str(t).upper())
            f = store.read_json("sec_facts", str(c)) if c else None
            if f is None and c and config.settings().sec_user_agent:
                try:
                    f = sec.load_companyfacts(c)
                except Exception:
                    f = None
            self[t] = l3.facts_long(f) if f else None
        return dict.get(self, t, default)


def scan_universe(membership: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Live names with their validation group; groups that failed Stage 2 amendment 4 are dropped."""
    from omodel.data import market_universe as mu

    uni = mu.live_universe(refresh)
    current = set(membership.loc[membership["end_date"].isna(), "ticker"])
    uni["group"] = np.where(uni["kind"] == "etf", "etf", np.where(uni["ticker"].isin(current), "sp", "new"))
    path = config.settings().reports_dir / "stage2" / "groups_results.json"
    verdicts = json.loads(path.read_text()) if path.exists() else {}
    allowed = {"sp"} | {g for g, r in verdicts.items() if r.get("passed")}
    return uni[uni["group"].isin(allowed)].reset_index(drop=True)


def _market_density(chain: pd.DataFrame, expiry: pd.Timestamp, spot: float, asof: pd.Timestamp):
    """Market-implied S_T distribution from out-of-the-money IVs (Breeden-Litzenberger); None if too few quotes."""
    ex = chain[(chain["expiry"] == expiry) & (chain["iv"] > 0) & (chain["bid"] > 0)]
    spread = (ex["ask"] - ex["bid"]) / ex["mid"].where(ex["mid"] > 0)
    ex = ex.assign(spread=spread)[spread <= SMILE_MAX_SPREAD]
    otm = ex[((ex["cp"] == "C") & (ex["strike"] >= spot)) | ((ex["cp"] == "P") & (ex["strike"] < spot))]
    if len(otm) < 6:
        return None
    T = max((expiry - asof).days, 1) / 365
    q = _implied_carry(ex, spot, T)
    try:
        return l5.risk_neutral_density(otm["strike"].to_numpy(), otm["iv"].to_numpy(), spot, T, q=q,
                                       weights=1 / otm["spread"].clip(lower=0.01).to_numpy())
    except ValueError:
        return None


def _implied_carry(ex: pd.DataFrame, spot: float, T: float) -> float:
    """Dividend yield implied by put-call parity at the strike nearest spot: F = K + e^{rT}(C - P)."""
    std = ex[ex["standard"]] if "standard" in ex else ex  # adjusted (non-standard) contracts reuse strikes
    calls = std[std["cp"] == "C"].groupby("strike")["mid"].mean()
    puts = std[std["cp"] == "P"].groupby("strike")["mid"].mean()
    both = calls.index.intersection(puts.index)
    if not len(both):
        return 0.0
    near = both[np.argsort(np.abs(both - spot))[:5]]  # median over 5 strikes: one wide ATM quote is too noisy
    fwd = float(np.median(near + np.exp(l5.RISK_FREE * T) * (calls[near].to_numpy() - puts[near].to_numpy())))
    if fwd <= 0:
        return 0.0
    return float(np.clip(l5.RISK_FREE - np.log(fwd / spot) / T, -0.1, 0.1))


def _sectors(panel, ret) -> dict[str, str]:
    from omodel.layers import l2_decomposition as l2

    etf = l2.log_returns(panel.etf_close[[s for s in SECTOR_ETFS if s in panel.etf_close]])
    sec_map = l2.assign_sectors(ret, l2.log_returns(panel.etf_close[MARKET_ETF]), etf)
    last = sec_map.iloc[-1]
    return {t: s for t, s in last.items() if isinstance(s, str)}


def _select(allc: pd.DataFrame, facts_by: dict, asof: pd.Timestamp) -> pd.DataFrame:
    if allc.empty:
        return allc
    ranked = l5.rank(allc.to_dict("records"))
    if ranked.empty:
        return ranked
    vetoes = {}
    keep, per_t, per_ss = [], {}, {}
    for r in ranked.itertuples():
        bullish = r.name in BULLISH
        key = (r.ticker, bullish)
        if key not in vetoes:
            vetoes[key] = l3.veto(facts_by.get(r.ticker), asof, pd.Timestamp(r.expiry), pd.DatetimeIndex([]), bullish)
        v = vetoes[key]
        if v.status == "HARD" or (bullish and v.status == "SOFT"):
            continue  # weak companies are never bought just because an option looks cheap
        side = "bull" if bullish else "bear" if r.name in BEARISH else "neutral"
        if per_t.get(r.ticker, 0) >= PER_TICKER or per_ss.get((r.sector, side), 0) >= PER_SECTOR_SIDE:
            continue
        per_t[r.ticker] = per_t.get(r.ticker, 0) + 1
        per_ss[(r.sector, side)] = per_ss.get((r.sector, side), 0) + 1
        keep.append({**r._asdict(), "veto": v.status, "fundamentals": "; ".join(v.reasons)})
        if len(keep) >= TOP_N:
            break
    return pd.DataFrame(keep).drop(columns=["Index"], errors="ignore")


def _write(picks: pd.DataFrame, allc: pd.DataFrame, asof: pd.Timestamp, regime: str) -> None:
    out = config.settings().reports_dir / "scanner"
    out.mkdir(parents=True, exist_ok=True)
    picks.to_csv(out / f"{asof.date()}.csv", index=False)
    if len(allc):
        (config.settings().data_dir / "scanner").mkdir(parents=True, exist_ok=True)
        allc.to_parquet(config.settings().data_dir / "scanner" / f"all_{asof.date()}.parquet")
    meta = {"asof": str(asof.date()), "regime": regime, "contracts_scored": int(len(allc)),
            "tickers_scored": int(allc["ticker"].nunique()) if len(allc) else 0, "picks": int(len(picks)),
            "status": "UNVALIDATED - paper only until the Stage 3 forward test"}
    (out / f"{asof.date()}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"scored {meta['contracts_scored']} contracts on {meta['tickers_scored']} stocks; {len(picks)} picks")

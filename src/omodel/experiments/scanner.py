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
BULLISH = {"long_call", "bull_call_debit", "bull_put_credit"}
BEARISH = {"long_put", "bear_put_debit", "bear_call_credit"}


def _log(msg: str) -> None:
    print(f"[scanner] {msg}", file=sys.stderr, flush=True)


def _latest_short_pct(tickers: list[str], shares: dict[str, float], asof: pd.Timestamp) -> dict[str, float]:
    si = finra.load_short_interest(tickers)
    si = si[si["available"] <= asof]
    last = si.sort_values("settlement").groupby("ticker").last()
    return {t: float(last.at[t, "short_qty"] / shares[t]) for t in last.index
            if t in shares and shares[t] and (asof - last.at[t, "settlement"]).days <= 45}


def run(refresh: bool = True) -> pd.DataFrame:
    _go_live()
    models = fit.load()
    if fit.is_stale(models):
        _log("refitting forecast models (monthly)")
        models = fit.fit_all()
    membership = universe.load_membership(refresh=refresh)
    members = sorted(membership.loc[membership["end_date"].isna(), "ticker"].unique())
    if refresh:
        for t in [MARKET_ETF, *SECTOR_ETFS, *members]:
            try:
                yahoo.load_daily(t, refresh=True)
            except Exception as exc:
                _log(f"{t}: price refresh failed ({exc})")
    today = pd.Timestamp.today().normalize()
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
    shares, facts_by = {}, {}
    for t in members:
        c = cik.get(t.upper())
        f = store.read_json("sec_facts", str(c)) if c else None
        if f:
            facts_by[t] = l3.facts_long(f)
            s = fu.company_series(f)["shares"]
            shares[t] = float(s["val"].iloc[-1]) if len(s) else np.nan
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
        if nxt_earn is None and t in facts_by:
            subs = sec.load_submissions(cik[t.upper()]) if config.settings().sec_user_agent else None
            if subs is not None and len(subs):
                nxt_earn = l3.next_earnings_estimate(sec.earnings_days(subs, panel.dates), asof)
        dte = (pd.to_datetime(chain["expiry"]) - asof).dt.days
        for expiry in sorted(chain.loc[dte.between(MIN_DTE, MAX_DTE), "expiry"].unique()):
            expiry = pd.Timestamp(expiry)
            if nxt_earn is not None and asof < nxt_earn <= expiry:
                continue  # Stage 2: the engine is not calibrated across earnings
            h = max(int(np.busday_count(asof.date(), expiry.date())), 1)
            vol = float(fit.har_vol(models, h, X.loc[[t]])[0])
            ratio = fit.crash_ratio(models, 0.5 * np.log(float(fit.har_vol(models, 21, X.loc[[t]])[0]) ** 2),
                                    np.log(max(spct.get(t, np.nan), 1e-5)) if t in spct else np.nan)
            grid, probs = eng.price_distribution(spot, vol, h, z, crash_ratio=ratio, seed=zlib.crc32(f"{t}{expiry.date()}".encode()))
            cand = l5.scan_expiry(chain[chain["expiry"] == expiry], grid, probs)
            if cand.empty:
                continue
            cand = cand.assign(ticker=t, expiry=expiry, dte=int((expiry - asof).days), spot=spot, vol_fc=vol,
                               short_pct=spct.get(t, np.nan), sector=sector.get(t))
            rows.append(cand)
    allc = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    picks = _select(allc, facts_by, asof)
    _write(picks, allc, asof, regime)
    return picks


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
    meta = {"asof": str(asof.date()), "regime": regime, "contracts_scored": int(len(allc)),
            "tickers_scored": int(allc["ticker"].nunique()) if len(allc) else 0, "picks": int(len(picks)),
            "status": "UNVALIDATED - paper only until the Stage 3 forward test"}
    (out / f"{asof.date()}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"scored {meta['contracts_scored']} contracts on {meta['tickers_scored']} stocks; {len(picks)} picks")

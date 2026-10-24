from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from omodel import config, params
from omodel.backtest import events as ev_mod
from omodel.config import MARKET_ETF, SECTOR_ETFS
from omodel.data import cboe_options, macro, sec, universe, yahoo
from omodel.data import panel as panel_mod
from omodel.layers import l1_regime as l1
from omodel.layers import l2_decomposition as l2
from omodel.layers import l3_fundamentals as l3
from omodel.layers import l4_distribution as l4
from omodel.layers import l5_options as l5

MIN_DTE = params.register("SHORTLIST_MIN_DTE", 21, "theory", "L5", "Thesis v2 expiry window (H2c rule).")
MAX_DTE = params.register("SHORTLIST_MAX_DTE", 45, "theory", "L5", "Thesis v2 expiry window (H2c rule).")
TOP_N = params.register("SHORTLIST_TOP_N", 10, "convention", "L5", "Contracts shown per day.")
H2C_DELTA = params.register("H2C_SHORT_DELTA", 0.30, "theory", "L5", "H2c fixed rule: short call delta.")
LIVE_HORIZON_END = pd.Timestamp("2100-01-01")
BEARISH = {"bear_call_credit", "bear_put_debit"}


def _log(msg: str) -> None:
    print(f"[shortlist] {msg}", file=sys.stderr, flush=True)


def _go_live() -> None:
    # The research holdout was consumed by H2a on 2026-10-24; live scoring must see today's data.
    config.override(holdout_start=LIVE_HORIZON_END)


def _history(panel, dec, macro_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    parts = []
    dev = config.settings().data_dir / "results" / "phase1_events.parquet"
    if dev.exists():
        parts.append(pd.read_parquet(dev))
    recent = ev_mod.find_events(dec, panel)
    recent = ev_mod.add_forward_returns(recent[recent["date"] < asof], panel)
    if parts:
        recent = recent[recent["date"] > parts[0]["date"].max()]
    parts.append(recent)
    hist = pd.concat(parts, ignore_index=True)
    vix = macro_df["vix"].reindex(pd.DatetimeIndex(hist["date"])).to_numpy()
    hist["high_vix"] = vix >= l1.VIX_HIGH
    return hist


def _sec_inputs(ticker: str, days: pd.DatetimeIndex) -> tuple[pd.DataFrame | None, pd.DatetimeIndex, bool | None]:
    if not config.settings().sec_user_agent:
        return None, pd.DatetimeIndex([]), None
    cik = sec.ticker_to_cik().get(ticker.upper())
    if cik is None:
        return None, pd.DatetimeIndex([]), None
    subs = sec.load_submissions(cik, refresh=True)
    news = sec.news_days(subs, days)
    flag = bool(len(news) and news.max() >= days[-2])
    facts = sec.load_companyfacts(cik, refresh=True)
    return l3.facts_long(facts), sec.earnings_days(subs, days), flag


def _chain(ticker: str, asof: pd.Timestamp) -> pd.DataFrame | None:
    ch = cboe_options.load_snapshot(ticker, asof)
    if ch is None:
        try:
            ch = cboe_options.fetch_chain(ticker)
        except Exception as exc:  # one rate-limited ticker must not stop the scan
            _log(f"{ticker}: chain unavailable ({exc})")
            return None
    return ch


def _h2c_rule(chain: pd.DataFrame, expiry: pd.Timestamp) -> l5.Structure | None:
    calls = chain[(chain["expiry"] == expiry) & (chain["cp"] == "C")].sort_values("strike").reset_index(drop=True)
    calls = calls[l5.liquid(calls)].reset_index(drop=True)
    if len(calls) < 2:
        return None
    i = int((calls["delta"] - H2C_DELTA).abs().idxmin())
    if i + 1 >= len(calls):
        return None
    lo, hi = calls.iloc[i], calls.iloc[i + 1]
    return l5.Structure("bear_call_credit", expiry, (l5._leg(lo, -1), l5._leg(hi, +1)))


def run(refresh: bool = True, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    _go_live()
    membership = universe.load_membership(refresh=refresh)
    members = sorted(membership.loc[membership["end_date"].isna(), "ticker"].unique())
    if refresh:
        _log(f"refreshing prices for {len(members)} members")
        for t in [MARKET_ETF, *SECTOR_ETFS, *members]:
            try:
                yahoo.load_daily(t, refresh=True)
            except Exception as exc:
                _log(f"{t}: price refresh failed ({exc})")
    end = pd.Timestamp.today().normalize() if asof is None else asof
    panel = panel_mod.build_panel(start=end - pd.Timedelta(days=900), end=end, tickers=members,
                                  membership=membership)
    dec = l2.decompose(panel)
    asof = panel.dates[-1]
    macro_df = macro.load_macro_panel(refresh=refresh)
    feats = l1.regime_features(macro_df, panel, dec.stock_ret)
    regime = l1.regime_label(feats.loc[asof])
    high_vix = bool(feats.loc[asof, "vix"] >= l1.VIX_HIGH) if np.isfinite(feats.loc[asof, "vix"]) else None
    today = ev_mod.find_events(dec, panel)
    today = today[today["date"] == asof].reset_index(drop=True)
    _log(f"{asof.date()}: {len(today)} event(s); regime: {regime}")
    hist = _history(panel, dec, macro_df, asof)

    rows = []
    for _, e in today.iterrows():
        chain = _chain(e["ticker"], asof)
        if chain is None or chain.empty:
            continue
        spot = float(chain.attrs.get("underlying_price") or panel.raw_close.at[asof, e["ticker"]])
        facts, earn, news = _sec_inputs(e["ticker"], panel.dates)
        dte = (pd.to_datetime(chain["expiry"]) - asof).dt.days
        for expiry in sorted(chain.loc[dte.between(MIN_DTE, MAX_DTE), "expiry"].unique()):
            expiry = pd.Timestamp(expiry)
            n = int(np.busday_count(asof.date(), expiry.date()))
            dist = l4.conditional(hist, asof, e["ticker"], spot, float(e["tot_sigma"]), n, high_vix)
            if dist is None:
                continue
            base = l5.lognormal_grid(spot, n / 252, float(e["tot_sigma"]) * np.sqrt(252))
            h2c = _h2c_rule(chain, expiry)
            vetoes = {b: l3.veto(facts, asof, expiry, earn, b) for b in (True, False)}
            dists = {s: dist if s == "PASS" else l4.conditional(
                hist, asof, e["ticker"], spot, float(e["tot_sigma"]), n, high_vix, s)
                for s in {v.status for v in vetoes.values()} if s != "HARD"}
            for st in l5.vertical_spreads(chain, expiry):
                v = vetoes[st.name not in BEARISH]
                d = dists.get(v.status)
                if v.status == "HARD" or d is None:
                    continue
                res = l5.evaluate(st, d.grid, d.probs)
                res["ev_vs_baseline"] = res["ev"] - l5.evaluate(st, *base)["ev"]
                legs = " / ".join(f"{'+' if l.qty > 0 else '-'}{l.cp}{l.strike:g}" for l in st.legs)
                trade = (f"Trade: {st.name} {legs} exp {expiry.date()}: EV {100 * res['ev_pct_risk']:+.1f}% of "
                         f"max loss ${res['max_loss']:.0f}, P(profit) {100 * res['prob_profit']:.0f}%.")
                res.update(ticker=e["ticker"], legs=legs, veto=v.status, regime=regime,
                           h2c_rule=h2c is not None and st.legs == h2c.legs,
                           thesis=l4.thesis_sentence(e, d, regime, v.reasons, news, trade))
                rows.append(res)
    ranked = l5.rank(rows)
    out = config.settings().reports_dir / "shortlist"
    out.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out / f"{asof.date()}.csv", index=False)
    (out / f"{asof.date()}.md").write_text(_render(ranked.head(TOP_N), asof, regime, len(today)), encoding="utf-8")
    return ranked


def _render(top: pd.DataFrame, asof: pd.Timestamp, regime: str, n_events: int) -> str:
    lines = [f"# Shortlist {asof.date()}", "",
             "**UNVALIDATED — paper trading only.** Thesis v2 trade tests (H2b/H2c) are evaluated on 2027-04-30; "
             "until then EVs come from an untested distribution model. A human places or rejects every trade.", "",
             f"Regime: {regime}. Events today (z ≤ −3): {n_events}.", ""]
    if top.empty:
        lines.append("No contract cleared the liquidity, veto and EV hurdles today.")
    for i, r in enumerate(top.itertuples(), 1):
        lines += [f"## {i}. {r.ticker} — {r.name} {r.legs} (exp {pd.Timestamp(r.expiry).date()})",
                  f"EV {100 * r.ev_pct_risk:+.1f}% of max loss (${r.ev:.0f} on ${r.max_loss:.0f}), "
                  f"EV vs plain-vol baseline {r.ev_vs_baseline:+.0f}$, P(profit) {100 * r.prob_profit:.0f}%, "
                  f"delta bucket {r.bucket}, veto {r.veto}{', H2c rule trade' if r.h2c_rule else ''}.", "",
                  r.thesis, ""]
    return "\n".join(lines)

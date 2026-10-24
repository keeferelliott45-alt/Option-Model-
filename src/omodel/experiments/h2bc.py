from __future__ import annotations

import datetime as _dt
import hashlib
import json

import numpy as np
import pandas as pd

from omodel import config
from omodel.backtest import events as ev_mod
from omodel.backtest import placebo, stats
from omodel.data import cboe_options, macro, universe
from omodel.data import panel as panel_mod
from omodel.experiments import h2a
from omodel.experiments.shortlist import MAX_DTE, MIN_DTE, _go_live, _h2c_rule
from omodel.layers import l1_regime as l1
from omodel.layers import l2_decomposition as l2
from omodel.layers import l5_options as l5

EVAL_DATE = pd.Timestamp("2027-04-30")
SNAP_START = pd.Timestamp("2026-10-26")
MIN_TRADES, MIN_MONTHS = 100, 6
PREREG = config.ROOT / "docs" / "PREREGISTRATION_THESIS_V2.md"


def _marker():
    return config.settings().data_dir / "h2bc_ran.json"


def pick_expiry(chain: pd.DataFrame, t: pd.Timestamp) -> pd.Timestamp | None:
    dte = (pd.to_datetime(chain["expiry"]) - t).dt.days
    ok = chain.loc[dte.between(MIN_DTE, MAX_DTE), "expiry"].unique()
    if not len(ok):
        return None
    return pd.Timestamp(min(ok, key=lambda e: abs((pd.Timestamp(e) - t).days - 30)))


def atm_iv(chain: pd.DataFrame, expiry: pd.Timestamp, spot: float) -> float:
    ex = chain[(chain["expiry"] == expiry) & (chain["iv"] > 0)]
    if ex.empty:
        return np.nan
    k = ex.loc[(ex["strike"] - spot).abs().idxmin(), "strike"]
    return float(ex.loc[ex["strike"] == k, "iv"].mean())


def realized_vol(close: pd.Series, t: pd.Timestamp, expiry: pd.Timestamp) -> float:
    path = close[(close.index >= t) & (close.index <= expiry)].dropna()
    if len(path) < 10 or path.index[-1] < expiry - pd.Timedelta(days=4):
        return np.nan
    return float(np.log(path).diff().dropna().std() * np.sqrt(252))


def settle(st: l5.Structure, s_expiry: float) -> tuple[float, float]:
    """(P&L, max loss) in dollars at expiry; fills at bid/ask and commissions on open and close."""
    width = abs(st.legs[0].strike - st.legs[1].strike) * l5.MULTIPLIER
    close_comm = l5.COMMISSION * sum(abs(leg.qty) for leg in st.legs)
    pnl = float(st.payoff(np.array([s_expiry]))[0] - st.entry_cost() - close_comm)
    max_loss = width - (-st.entry_cost()) + close_comm
    return pnl, max_loss


def _rows(days: pd.DataFrame, panel, snap_dates: set, vix: pd.Series, group: str) -> list[dict]:
    dates = panel.dates
    out = []
    for _, r in days.iterrows():
        t, tk = pd.Timestamp(r["date"]), r["ticker"]
        row = {"group": group, "date": t, "ticker": tk, "high_vix": bool(vix.get(t, np.nan) >= l1.VIX_HIGH)}
        close = panel.raw_close[tk]
        if t in snap_dates and (ch := cboe_options.load_snapshot(tk, t)) is not None:
            ex = pick_expiry(ch, t)
            spot = ch.attrs.get("underlying_price") or close.get(t)
            if ex is not None and ex <= dates[-1]:
                row["vol_gap"] = atm_iv(ch, ex, spot) - realized_vol(close, t, ex)
        p = dates.get_loc(t)
        if p + 1 < len(dates) and (t1 := dates[p + 1]) in snap_dates and (ch := cboe_options.load_snapshot(tk, t1)):
            ex = pick_expiry(ch, t1)
            st = _h2c_rule(ch, ex) if ex is not None else None
            s_exp = close[close.index <= ex].dropna() if ex is not None else pd.Series(dtype=float)
            if st is not None and ex <= dates[-1] and len(s_exp):
                pnl, ml = settle(st, float(s_exp.iloc[-1]))
                row.update(pnl_ratio=pnl / ml, delta=st.short_delta, bucket=l5.delta_bucket(st.short_delta))
        out.append(row)
    return out


def run(today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp.today().normalize() if today is None else today
    if today < EVAL_DATE:
        raise RuntimeError(f"H2b/H2c is pre-registered for {EVAL_DATE.date()}; refusing to look early")
    if _marker().exists():
        raise RuntimeError(f"H2b/H2c already ran ({_marker()}); the pre-registration allows one evaluation")
    _go_live()
    snap_root = config.settings().data_dir / "options_snapshots"
    snap_dates = {pd.Timestamp(p.name) for p in snap_root.iterdir() if p.is_dir()} if snap_root.exists() else set()
    snap_dates = {d for d in snap_dates if SNAP_START <= d <= EVAL_DATE}
    membership = universe.load_membership()
    panel = panel_mod.build_panel(start=SNAP_START - pd.Timedelta(days=900), end=EVAL_DATE, membership=membership)
    dec = l2.decompose(panel)
    events = ev_mod.find_events(dec, panel)
    events = events[events["date"].isin(snap_dates)].reset_index(drop=True)
    ctrl = placebo.placebo_events(events, panel, dec)
    ctrl = h2a._two_sided_exclusion(ctrl, events, 15)
    ctrl = ctrl[ctrl["date"].isin(snap_dates)]
    vix = macro.load_macro_panel()["vix"]
    df = pd.DataFrame(_rows(events, panel, snap_dates, vix, "event") + _rows(ctrl, panel, snap_dates, vix, "control"))
    for c in ("vol_gap", "pnl_ratio", "bucket"):
        if c not in df:
            df[c] = np.nan
    res = _evaluate(df)
    res.update(prereg_sha256=hashlib.sha256(PREREG.read_bytes()).hexdigest(),
               run_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
               snapshot_days=len(snap_dates))
    out = config.settings().reports_dir / "h2bc"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "trades.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    if res["verdict"] != "INSUFFICIENT":
        _marker().parent.mkdir(parents=True, exist_ok=True)
        _marker().write_text(json.dumps({"run_utc": res["run_utc"], "verdict": res["verdict"]}), encoding="utf-8")
    return res


def _diff(df: pd.DataFrame, col: str) -> dict:
    d = df[df[col].notna()]
    return stats.cluster_bootstrap_diff(d[col], d["group"], stats.cluster_labels(d["date"]), "event", "control")


def _evaluate(df: pd.DataFrame) -> dict:
    ev = df[df["group"] == "event"]
    trades = ev[ev["pnl_ratio"].notna()]
    months = trades["date"].dt.to_period("M").nunique() if len(trades) else 0
    if len(trades) < MIN_TRADES or months < MIN_MONTHS:
        return {"verdict": "INSUFFICIENT", "n_trades": len(trades), "months": int(months),
                "note": "write a dated amendment with the next evaluation date before looking at any number"}
    h2b = _diff(df, "vol_gap")
    h2c = stats.summarize(trades["pnl_ratio"], trades["date"], label="H2c events")
    h2c_vs = _diff(df, "pnl_ratio")
    monthly = trades.groupby(trades["date"].dt.to_period("M"))["pnl_ratio"].sum()
    gain = monthly[monthly > 0].mean()
    worst_ok = bool(np.isfinite(gain) and -monthly.min() <= 3 * gain)
    by_bucket = trades.groupby("bucket")["pnl_ratio"].agg(["mean", "count"]).to_dict("index")
    by_vix = trades.groupby("high_vix")["pnl_ratio"].agg(["mean", "count"]).to_dict("index")
    h2b_pass = bool(h2b["diff"] > 0 and h2b["lo"] > 0)
    h2c_pass = bool(h2c["mean"] > 0 and h2c["ci_lo"] > 0 and h2c_vs["diff"] > 0 and worst_ok)
    return {"verdict": "PASS" if h2b_pass and h2c_pass else "FAIL", "h2b": h2b, "h2b_pass": h2b_pass,
            "h2c": h2c, "h2c_vs_control": h2c_vs, "worst_month_ok": worst_ok, "h2c_pass": h2c_pass,
            "by_delta_bucket": by_bucket, "by_vix": {str(k): v for k, v in by_vix.items()},
            "n_trades": len(trades), "months": int(months),
            "suspicious": bool(h2c["win_rate"] > 0.90 or stats.annualized_sharpe(monthly, periods=12) > 2)}

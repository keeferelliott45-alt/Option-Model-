from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from omodel import config
from omodel.backtest import stats
from omodel.config import MARKET_ETF
from omodel.data import finra, macro, sec, universe
from omodel.data import panel as panel_mod
from omodel.experiments import stage1
from omodel.factors import fundamental as fu
from omodel.factors import price as pf
from omodel.forecast import engine as eng
from omodel.layers import l2_decomposition as l2
from omodel.layers.l3_fundamentals import EARNINGS_CYCLE_DAYS

PREREG = config.ROOT / "docs" / "PREREGISTRATION_STAGE2.md"
START = pd.Timestamp("2003-01-01")
H, BANK_N = 21, 1000
SUBS = [(START, pd.Timestamp("2008-12-31")), (pd.Timestamp("2009-01-01"), pd.Timestamp("2016-12-31")),
        (pd.Timestamp("2017-01-01"), stage1.DEV_END)]


def _log(msg: str) -> None:
    print(f"[stage2] {msg}", file=sys.stderr, flush=True)


def _stack(frames: dict[str, pd.DataFrame], dates) -> pd.DataFrame:
    return pd.concat({k: v.reindex(dates).stack(future_stack=True) for k, v in frames.items()}, axis=1) \
        .rename_axis(["date", "ticker"]).reset_index()


def _walk_forward_har(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for year in range(START.year, df["date"].dt.year.max() + 1):
        cut = pd.Timestamp(f"{year}-01-01")
        train = df[df["date"] < cut - pd.Timedelta(days=35)]  # targets (≈1 month ahead) fully realised
        test = df["date"].dt.year == year
        if len(train) < 1000 or not test.any():
            continue
        m = eng.HAR().fit(train[list(eng.HAR_FEATURES)], train["y"])
        ok = test & df[list(eng.HAR_FEATURES)].notna().all(axis=1)
        out[ok] = m.predict_var(df.loc[ok, list(eng.HAR_FEATURES)])
    return out


def _qlike(rv2: np.ndarray, f: np.ndarray) -> np.ndarray:
    x = rv2 / f
    return x - np.log(x) - 1


def _period_test(d: pd.Series, sign: int, lags: int) -> dict:
    dev = d[d.index <= stage1.DEV_END]
    nw = stats.newey_west(dev, lags=lags)
    subs = [np.sign(dev[(dev.index >= a) & (dev.index <= b)].mean()) for a, b in SUBS]
    ho = d[(d.index >= stage1.HOLDOUT[0]) & (d.index <= stage1.HOLDOUT[1])]
    return {"mean": nw["mean"], "t": nw["t"], "months": nw["n"], "subs_right": int(sum(s == sign for s in subs)),
            "holdout_mean": float(ho.mean()) if len(ho) else np.nan}


def run() -> dict:
    config.override(holdout_start=stage1.END + pd.Timedelta(days=1))
    membership = universe.load_membership()
    panel = panel_mod.build_panel(end=stage1.END, membership=membership)
    dec = l2.decompose(panel)
    ret = dec.stock_ret
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= START]
    raw = panel.raw_close.reindex(me)
    mask = (panel.member.reindex(me).eq(True) & (raw >= 5)).stack(future_stack=True)

    _log("features")
    feats = eng.har_features(ret, mac["vix"], mac["vix3m"])
    lc = np.log(panel.close)
    frames = {**feats, "y": eng.har_target(ret, H), "r21": lc.shift(-H) - lc}
    df = _stack(frames, me)
    df = df[mask.to_numpy()].reset_index(drop=True)
    df["f_har"] = _walk_forward_har(df)
    df["f_naive"] = np.exp(2 * df["lrv22"])
    ok = df[["f_har", "f_naive", "y"]].notna().all(axis=1)
    df = df[ok].reset_index(drop=True)

    _log("A: QLIKE")
    rv2 = np.exp(2 * df["y"].to_numpy())
    df["dloss"] = _qlike(rv2, df["f_har"].to_numpy()) - _qlike(rv2, df["f_naive"].to_numpy())
    a = _period_test(df.groupby("date")["dloss"].mean(), -1, 2)
    a["passed"] = bool(a["mean"] < 0 and a["t"] <= -2 and a["subs_right"] >= 2 and a["holdout_mean"] < 0)

    _log("B/C: calibration")
    cik = sec.ticker_to_cik()
    rng = np.random.default_rng(20261024)
    pit = np.full(len(df), np.nan)
    earn_flag = np.zeros(len(df), dtype=bool)
    pos = pd.Series(np.arange(len(panel.dates)), index=panel.dates)
    for t, g in df.groupby("ticker"):
        r = ret[t]
        z = pd.Series(eng.ewma_z(r.to_numpy()), index=r.index)
        c = cik.get(t.upper())
        subs = sec.load_submissions(c) if c else None
        edays = sec.earnings_days(subs, panel.dates) if subs is not None and len(subs) else pd.DatetimeIndex([])
        for year, gy in g.groupby(g["date"].dt.year):
            hist = z[(z.index >= pd.Timestamp(f"{year - eng.FHS_YEARS}-01-01")) & (z.index < pd.Timestamp(f"{year}-01-01"))]
            if hist.notna().sum() < 250:
                continue
            ez = z.reindex(edays[(edays >= hist.index.min()) & (edays <= hist.index.max())]).to_numpy()
            b0 = eng.bank(hist.to_numpy(), H, BANK_N, rng)
            b1 = eng.bank(hist.to_numpy(), H, BANK_N, rng, earnings_z=ez) if np.isfinite(ez).any() else b0
            b0, b1 = (b / np.sqrt(np.nanmean(b ** 2) / H) for b in (b0, b1))
            for i, row in gy.iterrows():
                d = row["date"]
                past = edays[edays <= d]
                nxt = past.max() + pd.Timedelta(days=EARNINGS_CYCLE_DAYS) if len(past) else None
                p = pos[d]
                end = panel.dates[min(p + H, len(panel.dates) - 1)]
                in_win = nxt is not None and d < nxt <= end
                earn_flag[i] = in_win
                b = b1 if in_win else b0
                x = row["r21"] / np.sqrt(row["f_har"] / 252)
                if np.isfinite(x):
                    pit[i] = np.mean(b <= x)
    df["pit"], df["earn"] = pit, earn_flag

    def coverage(sub: pd.DataFrame, period) -> dict:
        s = sub[(sub["date"] >= period[0]) & (sub["date"] <= period[1])]["pit"].dropna()
        return {"n": int(len(s)), "cov50": float(((s > 0.25) & (s < 0.75)).mean()),
                "cov90": float(((s > 0.05) & (s < 0.95)).mean())}

    dev_p, ho_p = (START, stage1.DEV_END), stage1.HOLDOUT
    b = {"dev": coverage(df, dev_p), "holdout": coverage(df, ho_p)}
    b["passed"] = bool(abs(b["dev"]["cov50"] - 0.5) <= 0.03 and abs(b["dev"]["cov90"] - 0.9) <= 0.03)
    e = df[df["earn"]]
    c = {"dev": coverage(e, dev_p), "holdout": coverage(e, ho_p)}
    c["passed"] = bool(abs(c["dev"]["cov50"] - 0.5) <= 0.04 and abs(c["dev"]["cov90"] - 0.9) <= 0.04)

    _log("D: crash tail")
    d_res = _crash_test(panel, df, me, membership)

    res = {"A_har_vs_naive": a, "B_calibration": b, "C_earnings_calibration": c, "D_crash_tail": d_res,
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
           "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    out = config.settings().reports_dir / "stage2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(_render(res), encoding="utf-8")
    return res


def _crash_test(panel, df: pd.DataFrame, me: pd.DatetimeIndex, membership) -> dict:
    tickers = list(panel.close.columns)
    fund = fu.fundamental_panel(tickers, me)
    si = finra.load_short_interest(tickers)
    _, spct = stage1._finra_monthly(si, me, tickers, fund["shares"])
    tgt = pf.targets(panel, panel.etf_close[MARKET_ETF])["crash63"].reindex(me)
    x = df[["date", "ticker", "f_har"]].copy()
    x["lvol"] = 0.5 * np.log(x["f_har"])
    x["lsp"] = np.log(spct.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(x[["date", "ticker"]])).to_numpy().clip(1e-5))
    x["y"] = tgt.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(x[["date", "ticker"]])).to_numpy()
    x = x[(x["date"] >= pd.Timestamp("2018-01-01"))].dropna().reset_index(drop=True)
    score = np.full((len(x), 2), np.nan)
    for year in range(2019, x["date"].dt.year.max() + 1):
        cut = pd.Timestamp(f"{year}-01-01")
        tr = x[x["date"] < cut - pd.Timedelta(days=95)]
        te = x["date"].dt.year == year
        if tr["y"].sum() < 10 or not te.any():
            continue
        for k, cols in enumerate((["lvol"], ["lvol", "lsp"])):
            m = sm.Logit(tr["y"], sm.add_constant(tr[cols])).fit(disp=0)
            p = np.clip(m.predict(sm.add_constant(x.loc[te, cols], has_constant="add")), 1e-6, 1 - 1e-6)
            yy = x.loc[te, "y"].to_numpy()
            score[te.to_numpy(), k] = yy * np.log(p) + (1 - yy) * np.log(1 - p)
    x["gain"] = score[:, 1] - score[:, 0]
    monthly = x.dropna(subset=["gain"]).groupby("date")["gain"].mean()
    dev = monthly[monthly.index <= stage1.DEV_END]
    nw = stats.newey_west(dev, lags=4)
    return {"mean_gain": nw["mean"], "t": nw["t"], "months": nw["n"],
            "passed": bool(nw["mean"] > 0 and nw["t"] >= 2)}


def _render(r: dict) -> str:
    a, b, c, d = r["A_har_vs_naive"], r["B_calibration"], r["C_earnings_calibration"], r["D_crash_tail"]
    ok = lambda x: "**PASS**" if x else "fail"
    return f"""# Stage 2 — forecast engine

Pre-registration `docs/PREREGISTRATION_STAGE2.md` (sha256 `{r['prereg_sha256']}`), run {r['run_utc']}.

| Test | Result | Pass |
|---|---|---|
| A. HAR vs naive vol (QLIKE, lower better) | mean diff {a['mean']:+.4f}, NW t {a['t']:+.2f}, {a['months']} months, sub-periods right {a['subs_right']}/3, holdout {a['holdout_mean']:+.4f} | {ok(a['passed'])} |
| B. 21-day calibration | 50% interval {100 * b['dev']['cov50']:.1f}%, 90% interval {100 * b['dev']['cov90']:.1f}% (n={b['dev']['n']}); holdout {100 * b['holdout']['cov50']:.1f}% / {100 * b['holdout']['cov90']:.1f}% | {ok(b['passed'])} |
| C. Earnings-window calibration | 50% {100 * c['dev']['cov50']:.1f}%, 90% {100 * c['dev']['cov90']:.1f}% (n={c['dev']['n']}); holdout {100 * c['holdout']['cov50']:.1f}% / {100 * c['holdout']['cov90']:.1f}% | {ok(c['passed'])} |
| D. Short-interest crash tail (log score gain) | {d['mean_gain']:+.5f} per obs, NW t {d['t']:+.2f}, {d['months']} months | {ok(d['passed'])} |
"""

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
            b0, b1 = eng.shape(b0, H), eng.shape(b1, H)
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


def earnings_mask(ret: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """True on 8-K 2.02 effective days and the following day (earnings-jump days)."""
    cik = sec.ticker_to_cik()
    m = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    pos = pd.Series(np.arange(len(dates)), index=dates)
    for t in ret.columns:
        c = cik.get(t.upper())
        subs = sec.load_submissions(c) if c else None
        if subs is None or not len(subs):
            continue
        p = pos.reindex(sec.earnings_days(subs, dates)).dropna().astype(int).to_numpy()
        p = np.unique(np.concatenate([p, p + 1]))
        m.iloc[p[p < len(dates)], m.columns.get_loc(t)] = True
    return m


def run_a2() -> dict:
    """Amendment 1, test A2: HAR-X (earnings days excluded from RV inputs) vs HAR on non-earnings windows."""
    config.override(holdout_start=stage1.END + pd.Timedelta(days=1))
    membership = universe.load_membership()
    panel = panel_mod.build_panel(end=stage1.END, membership=membership)
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= START]
    raw = panel.raw_close.reindex(me)
    mask = (panel.member.reindex(me).eq(True) & (raw >= 5)).stack(future_stack=True).to_numpy()
    em = earnings_mask(ret, panel.dates)
    ahead = em.astype(float).shift(-H).rolling(H, min_periods=1).max().reindex(me).stack(future_stack=True).to_numpy()
    y = eng.har_target(ret, H)
    dfs = {}
    for name, r in (("har", ret), ("harx", ret.mask(em))):
        frames = {**eng.har_features(r, mac["vix"], mac["vix3m"]), "y": y}
        df = _stack(frames, me)[mask].reset_index(drop=True)
        df["f"] = _walk_forward_har(df)
        dfs[name] = df
    a, b = dfs["har"], dfs["harx"]
    keep = (ahead[mask] == 0) & a["f"].notna().to_numpy() & b["f"].notna().to_numpy() & a["y"].notna().to_numpy()
    rv2 = np.exp(2 * a["y"].to_numpy())
    d = pd.Series(_qlike(rv2, b["f"].to_numpy()) - _qlike(rv2, a["f"].to_numpy()), index=a.index)[keep]
    monthly = d.groupby(a.loc[keep, "date"]).mean()
    res = _period_test(monthly, -1, 2)
    res["passed"] = bool(res["mean"] < 0 and res["t"] <= -2 and res["subs_right"] >= 2 and res["holdout_mean"] < 0)
    res["n_obs"] = int(keep.sum())
    out = config.settings().reports_dir / "stage2"
    (out / "a2_results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    with (out / "report.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Amendment 1 — test A2 (HAR-X vs HAR, non-earnings windows)\n\n"
                 f"Mean QLIKE difference {res['mean']:+.4f} (negative = HAR-X better), NW t {res['t']:+.2f}, "
                 f"{res['months']} months, {res['n_obs']} stock-months, sub-periods right {res['subs_right']}/3, "
                 f"holdout {res['holdout_mean']:+.4f}: {'**PASS**' if res['passed'] else 'fail'}.\n")
    return res


def run_b2() -> dict:
    """Amendment 2, test B2: calibration on no-earnings windows for HAR-N (fit on no-earnings windows only)."""
    config.override(holdout_start=stage1.END + pd.Timedelta(days=1))
    membership = universe.load_membership()
    panel = panel_mod.build_panel(end=stage1.END, membership=membership)
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= START]
    raw = panel.raw_close.reindex(me)
    mask = (panel.member.reindex(me).eq(True) & (raw >= 5)).stack(future_stack=True).to_numpy()
    em = earnings_mask(ret, panel.dates)
    ahead = em.astype(float).shift(-H).rolling(H, min_periods=1).max()
    lc = np.log(panel.close)
    frames = {**eng.har_features(ret, mac["vix"], mac["vix3m"]), "y": eng.har_target(ret, H), "r21": lc.shift(-H) - lc,
              "ahead": ahead}
    df = _stack(frames, me)[mask].reset_index(drop=True)
    df["f_har"] = _walk_forward_har(df)
    quiet = df["ahead"] == 0
    df["f_harn"] = np.nan
    df.loc[quiet, "f_harn"] = _walk_forward_har(df[quiet].reset_index(drop=True)).to_numpy()
    rng = np.random.default_rng(20261024)
    pits = {"f_har": np.full(len(df), np.nan), "f_harn": np.full(len(df), np.nan)}
    for t, g in df[quiet].groupby("ticker"):
        z = pd.Series(eng.ewma_z(ret[t].to_numpy()), index=ret.index)
        for year, gy in g.groupby(g["date"].dt.year):
            hist = z[(z.index >= pd.Timestamp(f"{year - eng.FHS_YEARS}-01-01")) & (z.index < pd.Timestamp(f"{year}-01-01"))]
            if hist.notna().sum() < 250:
                continue
            b = eng.shape(eng.bank(hist.to_numpy(), H, BANK_N, rng), H)
            for col in pits:
                x = (gy["r21"] / np.sqrt(gy[col] / 252)).to_numpy()
                ok = np.isfinite(x)
                pits[col][gy.index[ok]] = (b[None, :] <= x[ok, None]).mean(axis=1)

    def cov(col, period):
        sel = (df["date"] >= period[0]) & (df["date"] <= period[1]) & quiet
        s = pd.Series(pits[col])[sel].dropna()
        return {"n": int(len(s)), "cov50": float(((s > 0.25) & (s < 0.75)).mean()),
                "cov90": float(((s > 0.05) & (s < 0.95)).mean())}

    dev_p = (START, stage1.DEV_END)
    res = {"harn_dev": cov("f_harn", dev_p), "harn_holdout": cov("f_harn", stage1.HOLDOUT),
           "har_dev": cov("f_har", dev_p), "har_holdout": cov("f_har", stage1.HOLDOUT)}
    d = res["harn_dev"]
    res["passed"] = bool(abs(d["cov50"] - 0.5) <= 0.03 and abs(d["cov90"] - 0.9) <= 0.03)
    out = config.settings().reports_dir / "stage2"
    (out / "b2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    f = lambda c: f"{100 * c['cov50']:.1f}% / {100 * c['cov90']:.1f}% (n={c['n']})"
    with (out / "report.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Amendment 2 — test B2 (no-earnings windows)\n\n"
                 f"HAR-N coverage 50%/90%: dev {f(res['harn_dev'])}, holdout {f(res['harn_holdout'])}. "
                 f"For comparison HAR: dev {f(res['har_dev'])}, holdout {f(res['har_holdout'])}. "
                 f"Verdict: {'**PASS**' if res['passed'] else 'fail'}.\n")
    return res


def run_e() -> dict:
    """Amendment 3, test E: tail payoff calibration beyond the 5th/95th percentiles (HAR-N, no-earnings windows)."""
    config.override(holdout_start=stage1.END + pd.Timedelta(days=1))
    membership = universe.load_membership()
    panel = panel_mod.build_panel(end=stage1.END, membership=membership)
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= START]
    raw = panel.raw_close.reindex(me)
    mask = (panel.member.reindex(me).eq(True) & (raw >= 5)).stack(future_stack=True).to_numpy()
    em = earnings_mask(ret, panel.dates)
    ahead = em.astype(float).shift(-H).rolling(H, min_periods=1).max()
    lc = np.log(panel.close)
    frames = {**eng.har_features(ret, mac["vix"], mac["vix3m"]), "y": eng.har_target(ret, H), "r21": lc.shift(-H) - lc,
              "ahead": ahead}
    df = _stack(frames, me)[mask].reset_index(drop=True)
    df = df[df["ahead"] == 0].reset_index(drop=True)
    df["f"] = _walk_forward_har(df)
    df = df[(df["date"] <= stage1.DEV_END) & df["f"].notna() & df["r21"].notna()]
    rng = np.random.default_rng(20261024)
    acc = {"m_up": 0.0, "r_up": 0.0, "m_dn": 0.0, "r_dn": 0.0, "n": 0}
    for t, g in df.groupby("ticker"):
        z = pd.Series(eng.ewma_z(ret[t].to_numpy()), index=ret.index)
        for year, gy in g.groupby(g["date"].dt.year):
            hist = z[(z.index >= pd.Timestamp(f"{year - eng.FHS_YEARS}-01-01")) & (z.index < pd.Timestamp(f"{year}-01-01"))]
            if hist.notna().sum() < 250:
                continue
            b = eng.shape(eng.bank(hist.to_numpy(), H, BANK_N, rng), H)
            q5, q95 = np.quantile(b, [0.05, 0.95])
            m_up, m_dn = np.maximum(b - q95, 0).mean(), np.maximum(q5 - b, 0).mean()
            sc = np.sqrt(gy["f"].to_numpy() / 252)
            x = gy["r21"].to_numpy()
            acc["m_up"] += float((m_up * sc).sum())
            acc["m_dn"] += float((m_dn * sc).sum())
            acc["r_up"] += float(np.maximum(x - q95 * sc, 0).sum())
            acc["r_dn"] += float(np.maximum(q5 * sc - x, 0).sum())
            acc["n"] += len(gy)
    res = {"n": acc["n"], "upper_ratio": acc["r_up"] / acc["m_up"], "lower_ratio": acc["r_dn"] / acc["m_dn"]}
    res["passed"] = bool(0.8 <= res["upper_ratio"] <= 1.25 and 0.8 <= res["lower_ratio"] <= 1.25)
    out = config.settings().reports_dir / "stage2"
    (out / "e_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (out / "report.md").open("a", encoding="utf-8") as fh:
        fh.write(f"\n## Amendment 3 — test E (tail payoff calibration)\n\nRealized/model expected excess beyond the "
                 f"95th percentile {res['upper_ratio']:.2f}, below the 5th {res['lower_ratio']:.2f} (n={res['n']}): "
                 f"{'**PASS**' if res['passed'] else 'fail'}.\n")
    return res


def run_groups() -> dict:
    """Amendment 4: B3 coverage and E3 tail ratios for non-S&P stocks and ETFs, engine fit on S&P rows only."""
    from omodel.data import market_universe as mu

    config.override(holdout_start=stage1.END + pd.Timedelta(days=1))
    membership = universe.load_membership()
    uni = mu.live_universe()
    sp_ever = set(membership["ticker"])
    tickers = sorted(set(uni["ticker"]) | sp_ever)
    panel = panel_mod.build_panel(end=stage1.END, tickers=tickers, membership=membership)
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= START]
    raw = panel.raw_close.reindex(me)
    member = panel.member.reindex(me).eq(True)
    group = pd.DataFrame("", index=me, columns=panel.close.columns)
    etfs = set(uni.loc[uni["kind"] == "etf", "ticker"])
    new_stocks = set(uni.loc[uni["kind"] == "stock", "ticker"]) - sp_ever
    for t in group.columns:
        group[t] = np.where(member[t], "sp", "etf" if t in etfs else "new" if t in new_stocks else "")
    keep = ((group != "") & (raw >= 5)).stack(future_stack=True).to_numpy()
    em = earnings_mask(ret, panel.dates)
    ahead = em.astype(float).shift(-H).rolling(H, min_periods=1).max()
    lc = np.log(panel.close)
    frames = {**eng.har_features(ret, mac["vix"], mac["vix3m"]), "y": eng.har_target(ret, H), "r21": lc.shift(-H) - lc,
              "ahead": ahead, "group": group}
    df = _stack(frames, me)[keep].reset_index(drop=True)
    df = df[df["ahead"] == 0].reset_index(drop=True)
    # Fit on S&P rows only (as validated), predict everyone: walk-forward by year.
    df["f"] = np.nan
    for year in range(START.year, df["date"].dt.year.max() + 1):
        cut = pd.Timestamp(f"{year}-01-01")
        train = df[(df["group"] == "sp") & (df["date"] < cut - pd.Timedelta(days=35))]
        test = (df["date"].dt.year == year) & df[list(eng.HAR_FEATURES)].notna().all(axis=1)
        if len(train) < 1000 or not test.any():
            continue
        m = eng.HAR().fit(train[list(eng.HAR_FEATURES)], train["y"])
        df.loc[test, "f"] = m.predict_var(df.loc[test, list(eng.HAR_FEATURES)])
    df = df[(df["date"] <= stage1.DEV_END) & df["f"].notna() & df["r21"].notna()]
    rng = np.random.default_rng(20261025)
    stats_by = {g: {"pit": [], "m_up": 0.0, "r_up": 0.0, "m_dn": 0.0, "r_dn": 0.0} for g in ("sp", "new", "etf")}
    for t, g in df.groupby("ticker"):
        z = pd.Series(eng.ewma_z(ret[t].to_numpy()), index=ret.index)
        for year, gy in g.groupby(g["date"].dt.year):
            hist = z[(z.index >= pd.Timestamp(f"{year - eng.FHS_YEARS}-01-01")) & (z.index < pd.Timestamp(f"{year}-01-01"))]
            if hist.notna().sum() < 250:
                continue
            b = eng.shape(eng.bank(hist.to_numpy(), H, BANK_N, rng), H)
            q5, q95 = np.quantile(b, [0.05, 0.95])
            for grp, gg in gy.groupby("group"):
                s = stats_by[grp]
                sc = np.sqrt(gg["f"].to_numpy() / 252)
                x = gg["r21"].to_numpy()
                s["pit"].extend((b[None, :] <= (x / sc)[:, None]).mean(axis=1))
                s["m_up"] += float((np.maximum(b - q95, 0).mean() * sc).sum())
                s["m_dn"] += float((np.maximum(q5 - b, 0).mean() * sc).sum())
                s["r_up"] += float(np.maximum(x - q95 * sc, 0).sum())
                s["r_dn"] += float(np.maximum(q5 * sc - x, 0).sum())
    res = {}
    for grp, s in stats_by.items():
        p = np.asarray(s["pit"])
        if not len(p):
            continue
        r = {"n": int(len(p)), "cov50": float(((p > 0.25) & (p < 0.75)).mean()),
             "cov90": float(((p > 0.05) & (p < 0.95)).mean()),
             "upper_ratio": s["r_up"] / s["m_up"], "lower_ratio": s["r_dn"] / s["m_dn"]}
        r["passed"] = bool(abs(r["cov50"] - 0.5) <= 0.03 and abs(r["cov90"] - 0.9) <= 0.03
                           and 0.8 <= r["upper_ratio"] <= 1.25 and 0.8 <= r["lower_ratio"] <= 1.25)
        res[grp] = r
    out = config.settings().reports_dir / "stage2"
    (out / "groups_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    with (out / "report.md").open("a", encoding="utf-8") as fh:
        fh.write("\n## Amendment 4 — expanded universe (B3 / E3)\n\n| Group | n | 50% | 90% | Upper tail | Lower tail | Pass |\n"
                 "|---|---|---|---|---|---|---|\n")
        for grp, r in res.items():
            fh.write(f"| {grp} | {r['n']} | {100 * r['cov50']:.1f}% | {100 * r['cov90']:.1f}% | {r['upper_ratio']:.2f} | "
                     f"{r['lower_ratio']:.2f} | {'**PASS**' if r['passed'] else 'fail'} |\n")
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

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

from omodel import config
from omodel.backtest import stats, xsection
from omodel.config import MARKET_ETF
from omodel.data import finra, macro, sec, universe
from omodel.data import panel as panel_mod
from omodel.factors import fundamental as fu
from omodel.factors import price as pf
from omodel.layers import l2_decomposition as l2

PREREG = config.ROOT / "docs" / "PREREGISTRATION_STAGE1.md"
END = pd.Timestamp("2026-10-22")
DEV_END = pd.Timestamp("2025-08-31")
HOLDOUT = (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-07-31"))
SUBS = [(pd.Timestamp("2000-01-01"), pd.Timestamp("2008-12-31")),
        (pd.Timestamp("2009-01-01"), pd.Timestamp("2016-12-31")),
        (pd.Timestamp("2017-01-01"), DEV_END)]
SUBS_FINRA = [(pd.Timestamp("2018-01-01"), pd.Timestamp("2021-12-31")), (pd.Timestamp("2022-01-01"), DEV_END)]

# (factor, target, predicted sign) exactly as pre-registered.
SPECS = [("mom_12_1", "ret21", 1), ("rev_1m", "ret21", -1), ("high52", "ret21", 1), ("resid_mom", "ret21", 1),
         ("idio_vol", "ret21", -1), ("idio_vol_v", "vol21", 1), ("max_ret", "ret21", -1), ("beta", "ret21", -1),
         ("pead", "ret21", 1), ("gross_prof", "ret21", 1), ("accruals", "ret21", -1),
         ("asset_growth", "ret21", -1), ("dilution", "ret21", -1), ("earn_yield", "ret21", 1),
         ("sales_yield", "ret21", 1), ("f_score", "ret21", 1), ("leverage_crash", "crash63", 1),
         ("zscore_crash", "crash63", -1), ("days_to_cover", "ret21", -1), ("short_pct", "ret21", -1),
         ("short_pct_crash", "crash63", 1)]
FINRA_FACTORS = {"days_to_cover", "short_pct", "short_pct_crash"}


def _log(msg: str) -> None:
    print(f"[stage1] {msg}", file=sys.stderr, flush=True)


def _finra_monthly(si: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str], shares: pd.DataFrame):
    dtc = pd.DataFrame(np.nan, index=dates, columns=tickers)
    qty = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for t, g in si.groupby("ticker"):
        if t not in dtc.columns:
            continue
        g = g.sort_values("available").groupby("available").last()
        idx = g.index.searchsorted(dates, side="right") - 1
        fresh = (idx >= 0) & ((dates - g.index[np.clip(idx, 0, None)]).days <= 45)
        dtc[t] = np.where(fresh, g["days_to_cover"].to_numpy()[np.clip(idx, 0, None)], np.nan)
        qty[t] = np.where(fresh, g["short_qty"].to_numpy()[np.clip(idx, 0, None)], np.nan)
    return dtc, qty / shares.where(shares > 0)


def _macro_tests(mac: pd.DataFrame, spy: pd.Series, me: pd.DatetimeIndex) -> list[dict]:
    lr = np.log(spy).diff()
    fwd_vol = lr.shift(-22).rolling(21, min_periods=15).std() * np.sqrt(252)
    m = mac.reindex(spy.index).ffill(limit=3)
    preds = {"vix": (m["vix"], 1), "vix_term": (m["vix"] / m["vix3m"], 1),
             "credit_20d": (np.log(m["hyg"] / m["lqd"]).diff(20), -1)}
    out = []
    for name, (x, sign) in preds.items():
        df = pd.DataFrame({"x": x.reindex(me), "y": fwd_vol.reindex(me)}).dropna()
        res = {}
        for label, part in (("dev", df[df.index <= DEV_END]), ("holdout", df[(df.index >= HOLDOUT[0]) & (df.index <= HOLDOUT[1])])):
            if len(part) < 5:
                res[label] = (np.nan, np.nan)
                continue
            fit = sm.OLS(part["y"], sm.add_constant(part["x"])).fit(cov_type="HAC", cov_kwds={"maxlags": 2})
            res[label] = (float(fit.params["x"]), float(fit.tvalues["x"]) if label == "dev" else np.nan)
        dev = df[df.index <= DEV_END]
        subs = [np.sign(np.corrcoef(p["x"], p["y"])[0, 1]) for a, b in SUBS
                if len(p := dev[(dev.index >= a) & (dev.index <= b)]) > 10]
        out.append({"factor": name, "target": "spy_vol21", "sign": sign, "coef": res["dev"][0], "t": res["dev"][1],
                    "p": float(2 * (1 - __import__("scipy").stats.norm.cdf(abs(res["dev"][1])))),
                    "subs_right_sign": int(sum(s == sign for s in subs)), "n_subs": len(subs),
                    "holdout_coef": res["holdout"][0],
                    "holdout_ok": bool(np.isfinite(res["holdout"][0]) and np.sign(res["holdout"][0]) == sign),
                    "months": int(len(dev))})
    return out


def run() -> dict:
    config.override(holdout_start=END + pd.Timedelta(days=1))  # dev/holdout split is enforced below, by date
    membership = universe.load_membership()
    panel = panel_mod.build_panel(end=END, membership=membership)
    dec = l2.decompose(panel)
    tickers = list(panel.close.columns)
    me = pf.month_ends(panel.dates)
    me = me[me >= config.settings().dev_start]

    _log("earnings days")
    cik = sec.ticker_to_cik()
    earnings = {}
    for t in tickers:
        c = cik.get(t.upper())
        if c and (subs := sec.load_submissions(c)) is not None and len(subs):
            earnings[t] = sec.earnings_days(subs, panel.dates)
    daily = pf.price_factors(panel, dec, earnings)
    fac = {k: v.reindex(me) for k, v in daily.items()}
    tgt = {k: v.reindex(me) for k, v in pf.targets(panel, panel.etf_close[MARKET_ETF]).items()}

    _log("fundamentals")
    raw = panel.raw_close.reindex(me)
    fund = fu.fundamental_panel(tickers, me)
    fac.update({k: v for k, v in fu.fundamental_factors(fund, raw).items() if k != "mcap"})
    _log("short interest")
    si = finra.load_short_interest(tickers)
    dtc, spct = _finra_monthly(si, me, tickers, fund["shares"])
    fac.update(days_to_cover=dtc, short_pct=spct, short_pct_crash=spct)

    mask = panel.member.reindex(me).eq(True) & (raw >= 5)
    rows = []
    for name, target, sign in SPECS:
        lags = 4 if target == "crash63" else 2
        ic = xsection.monthly_ic(fac[name], tgt[target], mask, me)
        subs = SUBS_FINRA if name in FINRA_FACTORS else SUBS
        ev = xsection.evaluate(ic, sign, lags, DEV_END, HOLDOUT, subs)
        rows.append({"factor": name, "target": target, "sign": sign, "n_subs": len(subs), **ev})
        _log(f"{name}: IC {ev['mean_ic']:+.4f} t {ev['t']:+.2f}")
    tab = pd.DataFrame(rows)
    bh = stats.bh_adjust(dict(zip(tab["factor"], tab["p"])))
    tab["p_adj"] = bh["p_adj"].to_numpy()
    tab["passed"] = (tab["sign_ok"] & (tab["p_adj"] <= 0.10) & tab["holdout_ok"]
                     & (tab["subs_right_sign"] >= 2))

    mac = pd.DataFrame(_macro_tests(macro.load_macro_panel(), panel.etf_close[MARKET_ETF], me))
    mbh = stats.bh_adjust(dict(zip(mac["factor"], mac["p"])))
    mac["p_adj"] = mbh["p_adj"].to_numpy()
    mac["passed"] = ((np.sign(mac["coef"]) == mac["sign"]) & (mac["t"].abs() >= 2) & (mac["p_adj"] <= 0.10)
                     & (mac["subs_right_sign"] >= 2) & mac["holdout_ok"])

    res = {"factors": tab.to_dict("records"), "macro": mac.to_dict("records"),
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
           "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
    out = config.settings().reports_dir / "stage1"
    out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "factors.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(_render(tab, mac, res), encoding="utf-8")
    return res


def _render(tab: pd.DataFrame, mac: pd.DataFrame, res: dict) -> str:
    lines = ["# Stage 1 — standalone factor tests", "",
             f"Pre-registration `docs/PREREGISTRATION_STAGE1.md` (sha256 `{res['prereg_sha256']}`), run {res['run_utc']}.",
             "Development formations 2000-01 → 2025-08; holdout 2025-10 → 2026-07 (direction only). "
             "Survivor-biased free prices: return factors that favour weak firms are understated.", "",
             "| Factor | Target | Pred. | Mean IC | NW t | BH p | Q5−Q1 | Sub-periods right | Holdout IC | Months | Pass |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        lines.append(f"| {r.factor} | {r.target} | {'+' if r.sign > 0 else '−'} | {r.mean_ic:+.4f} | {r.t:+.2f} | "
                     f"{r.p_adj:.3f} | {r.spread:+.4f} | {r.subs_right_sign}/{r.n_subs} | {r.holdout_ic:+.4f} | "
                     f"{r.months} | {'**PASS**' if r.passed else 'fail'} |")
    lines += ["", "## Macro (SPY next-month realized vol)", "",
              "| Predictor | Pred. | Coef | NW t | BH p | Sub-periods right | Holdout coef | Pass |", "|---|---|---|---|---|---|---|---|"]
    for r in mac.itertuples():
        lines.append(f"| {r.factor} | {'+' if r.sign > 0 else '−'} | {r.coef:+.4f} | {r.t:+.2f} | {r.p_adj:.3f} | "
                     f"{r.subs_right_sign}/{r.n_subs} | {r.holdout_coef:+.4f} | {'**PASS**' if r.passed else 'fail'} |")
    return "\n".join(lines) + "\n"

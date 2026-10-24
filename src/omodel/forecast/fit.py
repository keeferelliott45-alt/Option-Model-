from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from omodel import config
from omodel.config import MARKET_ETF
from omodel.data import finra, macro, universe
from omodel.data import panel as panel_mod
from omodel.experiments import stage1
from omodel.factors import fundamental as fu
from omodel.factors import price as pf
from omodel.forecast import engine as eng

HORIZONS = (5, 21, 63)
CRASH_H = 63


def _path():
    return config.settings().data_dir / "models" / "forecast_models.json"


def load() -> dict | None:
    p = _path()
    return json.loads(p.read_text()) if p.exists() else None


def is_stale(models: dict | None, max_age_days: int = 31) -> bool:
    return models is None or (pd.Timestamp.today() - pd.Timestamp(models["fitted"])).days > max_age_days


def fit_all() -> dict:
    """Fit HAR (per horizon) and the short-interest crash logistic on all realised history up to today."""
    membership = universe.load_membership()
    panel = panel_mod.build_panel(membership=membership)
    ret = np.log(panel.close).diff()
    mac = macro.load_macro_panel()
    me = pf.month_ends(panel.dates)
    me = me[me >= pd.Timestamp("2003-01-01")]
    raw = panel.raw_close.reindex(me)
    mask = (panel.member.reindex(me).eq(True) & (raw >= 5)).stack(future_stack=True).to_numpy()
    feats = eng.har_features(ret, mac["vix"], mac["vix3m"])
    out = {"fitted": str(pd.Timestamp.today().date()), "features": list(eng.HAR_FEATURES), "har": {}}
    for h in HORIZONS:
        frames = {**feats, "y": eng.har_target(ret, h)}
        df = pd.concat({k: v.reindex(me).stack(future_stack=True) for k, v in frames.items()}, axis=1)[mask]
        m = eng.HAR().fit(df[list(eng.HAR_FEATURES)], df["y"])
        out["har"][str(h)] = {"coef": m.coef.tolist(), "resid_var": m.resid_var, "n": int(df["y"].notna().sum())}

    tickers = list(panel.close.columns)
    fund = fu.fundamental_panel(tickers, me)
    si = finra.load_short_interest(tickers)
    _, spct = stage1._finra_monthly(si, me, tickers, fund["shares"])
    crash = pf.targets(panel, panel.etf_close[MARKET_ETF])["crash63"].reindex(me)
    h21 = eng.HAR()
    h21.coef, h21.resid_var = np.array(out["har"]["21"]["coef"]), out["har"]["21"]["resid_var"]
    frames = {**feats}
    X = pd.concat({k: v.reindex(me).stack(future_stack=True) for k, v in frames.items()}, axis=1)
    X["lsp"] = np.log(spct.stack(future_stack=True).reindex(X.index).clip(lower=1e-5))
    X["y"] = crash.stack(future_stack=True).reindex(X.index)
    X = X[mask]
    X = X[X.index.get_level_values(0) >= pd.Timestamp("2018-01-01")].dropna()
    X["lvol"] = 0.5 * np.log(h21.predict_var(X))
    for name, cols in (("crash_base", ["lvol"]), ("crash_si", ["lvol", "lsp"])):
        m = sm.Logit(X["y"], sm.add_constant(X[cols])).fit(disp=0)
        out[name] = {"cols": cols, "coef": m.params.tolist(), "n": int(len(X)), "events": int(X["y"].sum())}
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    return out


def har_vol(models: dict, h: int, feats: pd.DataFrame) -> np.ndarray:
    """Annualised vol forecast for any horizon: interpolate the fitted horizons in log-variance."""
    hs = np.array(HORIZONS, dtype=float)
    logs = []
    for k in HORIZONS:
        m = eng.HAR()
        m.coef, m.resid_var = np.array(models["har"][str(k)]["coef"]), models["har"][str(k)]["resid_var"]
        logs.append(np.log(m.predict_var(feats)))
    logs = np.vstack(logs)
    x = np.clip(h, hs[0], hs[-1])
    return np.sqrt(np.exp([np.interp(x, hs, logs[:, j]) for j in range(logs.shape[1])]))


def crash_ratio(models: dict, lvol: float, lsp: float) -> float:
    """P(crash | vol, short interest) / P(crash | vol): the measured short-interest tilt of the left tail."""
    def p(spec, x):
        z = spec["coef"][0] + float(np.dot(spec["coef"][1:], x))
        return 1 / (1 + np.exp(-z))
    if not np.isfinite(lsp):
        return 1.0
    return float(p(models["crash_si"], [lvol, lsp]) / p(models["crash_base"], [lvol]))

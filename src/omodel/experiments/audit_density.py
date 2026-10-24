from __future__ import annotations

import json

import numpy as np
import pandas as pd

from omodel import config
from omodel.data import cboe_options
from omodel.experiments.scanner import MAX_DTE, MIN_DTE, _market_density
from omodel.experiments.shortlist import _go_live
from omodel.layers import l5_options as l5

MAX_ABS_ERR, MAX_POS_EV = 0.03, 0.10  # fixed before the first full run


def run(asof: pd.Timestamp) -> dict:
    """Does the market-implied density reprice its own chain? Checked on every saved chain for one session."""
    _go_live()
    folder = config.settings().data_dir / "options_snapshots" / str(asof.date())
    errs, pos = [], []
    for p in sorted(folder.glob("*.parquet")):
        ch = cboe_options.load_snapshot(p.stem, asof)
        if ch is None or ch.empty:
            continue
        spot = ch.attrs["underlying_price"]
        dte = (pd.to_datetime(ch["expiry"]) - asof).dt.days
        for ex in ch.loc[dte.between(MIN_DTE, MAX_DTE), "expiry"].unique():
            ex = pd.Timestamp(ex)
            md = _market_density(ch, ex, spot, asof)
            if md is None:
                continue
            g, pr = md
            T = (ex - asof).days / 365
            q = ch[(ch["expiry"] == ex)]
            q = q[l5.liquid(q)]
            for r in q.itertuples():
                pay = np.maximum(g - r.strike, 0) if r.cp == "C" else np.maximum(r.strike - g, 0)
                v = float(pay @ pr) * np.exp(-l5.RISK_FREE * T)
                errs.append(abs(v - r.mid) / r.mid)
                pos.append(v > r.ask)
    res = {"asof": str(asof.date()), "quotes": len(errs), "median_abs_err": float(np.median(errs)),
           "share_positive_ev_at_ask": float(np.mean(pos))}
    res["passed"] = bool(res["median_abs_err"] <= MAX_ABS_ERR and res["share_positive_ev_at_ask"] <= MAX_POS_EV)
    out = config.settings().reports_dir / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"density_{asof.date()}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res

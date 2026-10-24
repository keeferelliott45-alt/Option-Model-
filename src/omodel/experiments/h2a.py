from __future__ import annotations

import datetime as _dt
import hashlib
import json

import numpy as np
import pandas as pd

from omodel import config, holdout, params
from omodel.backtest import events as ev_mod
from omodel.backtest import placebo, stats
from omodel.data import panel as panel_mod
from omodel.data import universe
from omodel.layers import l2_decomposition as l2

WINDOW_START = pd.Timestamp("2025-10-23")
WINDOW_END = pd.Timestamp("2026-10-22")
PREREG = config.ROOT / "docs" / "PREREGISTRATION_THESIS_V2.md"
H = params.PRIMARY_HORIZON
METRIC = f"net_abn_{H}"


def _marker():
    return config.settings().data_dir / "h2a_ran.json"


def _two_sided_exclusion(plc: pd.DataFrame, events: pd.DataFrame, window: int) -> pd.DataFrame:
    """Pre-registered: drop placebo days within `window` trading days of a real same-ticker event, either side."""
    by_ticker = events.groupby("ticker")["pos"].apply(np.sort).to_dict()
    keep = []
    for t, p in zip(plc["ticker"], plc["pos"]):
        ep = by_ticker.get(t)
        keep.append(ep is None or np.min(np.abs(ep - p)) > window)
    return plc[np.array(keep, dtype=bool)].reset_index(drop=True)


def run() -> dict:
    if not holdout.is_unlocked():
        raise holdout.HoldoutViolation(f"H2a reads the holdout: set {holdout.UNLOCK_ENV}={holdout.UNLOCK_VALUE}")
    if _marker().exists():
        raise RuntimeError(f"H2a already ran ({_marker()}); the pre-registration allows exactly one run")
    # Labels may use prices through WINDOW_END only; anything later is still unseen.
    config.override(holdout_start=WINDOW_END + pd.Timedelta(days=1))

    membership = universe.load_membership()
    panel = panel_mod.build_panel(start=WINDOW_START - pd.Timedelta(days=params.get("PANEL_WARMUP_DAYS")),
                                  end=WINDOW_END, membership=membership)
    dec = l2.decompose(panel)

    events = ev_mod.find_events(dec, panel)
    events = events[(events["date"] >= WINDOW_START) & (events["date"] <= WINDOW_END)].reset_index(drop=True)
    events = ev_mod.add_forward_returns(events, panel)

    plc = placebo.placebo_events(events, panel, dec, exclude_window=params.EVENT_COOLDOWN)
    plc = _two_sided_exclusion(plc, events, params.EVENT_COOLDOWN)
    plc = ev_mod.add_forward_returns(plc, panel)

    ev_sum = stats.summarize(events[METRIC], events["date"], label="events")
    plc_sum = stats.summarize(plc[METRIC], plc["date"], label="placebo")
    both = pd.concat([events[[METRIC, "date"]].assign(g="event"), plc[[METRIC, "date"]].assign(g="placebo")],
                     ignore_index=True)
    diff = stats.cluster_bootstrap_diff(both[METRIC], both["g"], stats.cluster_labels(both["date"]),
                                        high="event", low="placebo")
    passed = bool(ev_sum["mean"] < 0 and diff["diff"] < 0)
    res = {
        "test": "H2a drift replication (one-shot, holdout)",
        "window": [str(WINDOW_START.date()), str(WINDOW_END.date())],
        "metric": METRIC,
        "events": ev_sum, "placebo": plc_sum, "event_minus_placebo": diff,
        "missing_exit_share": float(events[f"missing_{H}"].mean()) if len(events) else None,
        "verdict": "PASS" if passed else "FAIL",
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "run_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    out = config.settings().reports_dir / "h2a"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(_render(res), encoding="utf-8")
    _marker().parent.mkdir(parents=True, exist_ok=True)
    _marker().write_text(json.dumps({"run_utc": res["run_utc"], "verdict": res["verdict"]}), encoding="utf-8")
    return res


def _pct(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:+.2f}%"


def _render(r: dict) -> str:
    e, p, d = r["events"], r["placebo"], r["event_minus_placebo"]
    lc = " (low-confidence: few monthly clusters)" if e.get("low_confidence") else ""
    return f"""# H2a — drift replication on the holdout

Pre-registration: `docs/PREREGISTRATION_THESIS_V2.md` (sha256 `{r['prereg_sha256']}`). Run once, {r['run_utc']}.

Window {r['window'][0]} → {r['window'][1]} (never loaded by v1). Metric: net abnormal log return, t+1 open → t+{H}
close, 10 bps round trip.

| | Mean | 95% CI | n | Effective month clusters |
|---|---|---|---|---|
| Events (z ≤ −3) | {_pct(e['mean'])} | [{_pct(e['ci_lo'])}, {_pct(e['ci_hi'])}] | {e['n']} | {e.get('n_eff_clusters', float('nan')):.1f} |
| Placebo (two-sided exclusion) | {_pct(p['mean'])} | [{_pct(p['ci_lo'])}, {_pct(p['ci_hi'])}] | {p['n']} | {p.get('n_eff_clusters', float('nan')):.1f} |
| Event − placebo | {_pct(d['diff'])} | [{_pct(d['lo'])}, {_pct(d['hi'])}] | | |

Pass rule: event mean < 0 and event − placebo < 0 (sign replication){lc}.

**Verdict: {r['verdict']}**

Missing-exit share (possible delistings): {_pct(r['missing_exit_share'])}.
"""

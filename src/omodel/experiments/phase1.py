from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import math
import pkgutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import omodel
from omodel import holdout, params
from omodel.backtest import events as ev_mod
from omodel.backtest import factor_tests, placebo, stats, walkforward
from omodel.config import settings
from omodel.data import panel as panel_mod
from omodel.data import sec, store, universe
from omodel.layers import l2_decomposition as l2
from omodel.types import Decomposition, Panel

SUBPERIODS = params.register(
    "SUBPERIODS", ((2000, 2008), (2009, 2016), (2017, 2025)), "convention", "backtest",
    "Pre-registered stability split (criterion 4): three ~8-year calendar blocks fixed before any result.")
STABILITY_MIN_SUBPERIODS = params.register(
    "STABILITY_MIN_SUBPERIODS", 2, "convention", "backtest",
    "Pre-registered: the mean must have H1's (positive) sign in at least 2 of the 3 sub-periods.")
FACTOR_N_BUCKETS = params.register(
    "FACTOR_N_BUCKETS", 3, "convention", "backtest",
    "Terciles for one-factor-at-a-time tests: two cut points, the fewest that still separate high from low.")
STALE_OPEN_RULE = params.register(
    "STALE_OPEN_RULE", "volume[t+1] == 0 or as-traded open[t+1] == close[t]", "convention", "backtest",
    "An entry print that is not a real t+1 opening trade (no volume, or a free-data open copied from the prior "
    "close, ~10-15% of post-drop S&P bars before 2008 vs ~1-2% after) is missing data, not a fill; using it would "
    "silently turn the pre-registered next-open entry into a close entry, which favours H1.")
STALE_WINDOW_SHARES = params.register(
    "STALE_WINDOW_SHARES", (0.5, 0.9), "convention", "backtest",
    "Shares of exactly-zero returns in a fit window above which the window is mostly stale prices (halt, "
    "illiquidity) and the rolling sigma is deflated, so z = resid/sigma can spike and manufacture fake events.")

PROVENANCE_PATHS = ("docs/PREREGISTRATION.md", "src", "pyproject.toml")
_H = params.PRIMARY_HORIZON
# Decisions made in code that PREREGISTRATION.md does not spell out; fixed before any real-data result.
READINGS = (
    f"Criterion 3 is judged on the placebo's abnormal {_H}d return BEFORE the fixed {2 * params.COST_BPS_PER_SIDE:g} "
    "bp round-trip cost (a constant cost would push any clean placebo below zero).",
    f"'Date-clustered' CIs resample calendar-month clusters ({stats.CLUSTER_FREQ}), "
    f"{params.BOOTSTRAP_REPS} percentile-bootstrap replications.",
    f"Walk-forward: on each fit window the threshold with the highest i.i.d. t-score among those with >= "
    f"{params.MIN_SAMPLE_FINDING} purged fit events is chosen; a window where none qualifies adds no test events. "
    f"Purged = the fit event's {_H}d exit date falls before the test window opens.",
    f"Same-ticker events within {params.EVENT_COOLDOWN} trading days of the last kept event are suppressed.",
    f"Criterion 4: at least {STABILITY_MIN_SUBPERIODS} of the {len(SUBPERIODS)} sub-periods have a positive mean; "
    "a sub-period with no events counts as not positive.",
    f"Criteria 1, 1b and 2 also require n >= {params.MIN_SAMPLE_FINDING} and >= {stats.MIN_EFFECTIVE_CLUSTERS} "
    "effective month clusters: a low-confidence estimate never passes.",
    f"Placebo: the same tickers, one draw per event, from member days with finite model inputs and as-traded price "
    f">= ${params.MIN_PRICE:g}, inside the events' date span, excluding each real event day and the "
    f"{params.EVENT_COOLDOWN} days after it. Days BEFORE an event are not excluded: that would keep only placebo "
    "windows with no future drop in them (lookahead).",
    f"Entries that are not real t+1 opens are missing data ({STALE_OPEN_RULE}), for events and placebo alike; "
    "they are counted by year and the unfiltered mean is reported as a sensitivity row.",
    "Events or placebo days whose stock or ETF price is missing at entry or exit are excluded from every mean and "
    "counted; they are never booked as losses or gains.",
    f"Placebo drift diagnostic (placebo abnormal return minus {_H} x the ticker's mean daily abnormal return over "
    "the development window) is reported only. It does not change criterion 3 or the run's validity.",
)
# Thesis-predicted sign of top-minus-bottom for each confirming signal (documentation for the report only).
FACTOR_EXPECTATION = {
    "vol_z": ("+", "high-volume drops overreact more"),
    "sec_ret": ("-", "core bet: drops on days the sector fell hardest revert more"),
    "mkt_ret": ("-", "drops on market-wide selloff days revert more"),
    "peer_z": ("-", "drops shared with sector peers (low peer z) revert more"),
    "abs_z": ("+", "larger unexplained drops revert more"),
    "no_8k": ("+", "drops with no company 8-K (price/news mismatch) revert more"),
}


def import_all() -> None:
    """Import every omodel module so the parameter registry is complete before it is reported."""
    for mod in pkgutil.walk_packages(omodel.__path__, "omodel."):
        if not mod.name.endswith("__main__"):
            importlib.import_module(mod.name)


def registry_snapshot() -> dict[str, dict[str, Any]]:
    return {p.name: {"value": p.value, "source": p.source, "layer": p.layer, "rationale": p.rationale}
            for p in sorted(params.REGISTRY.values(), key=lambda p: p.name)}


def parameters_path() -> Path:
    return settings().root / "docs" / "PARAMETERS.md"


def write_parameters_doc(path: Path | None = None) -> Path:
    import_all()
    path = parameters_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params.render_markdown(), encoding="utf-8")
    return path


def jsonable(x: Any) -> Any:
    """Recursively convert to JSON-safe builtins: NaN/inf/NaT -> None, timestamps -> ISO strings."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return jsonable(x.to_dict("records"))
    if isinstance(x, (pd.Series, pd.Index, np.ndarray)):
        return jsonable(list(x))
    if x is None or x is pd.NaT or x is pd.NA:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.datetime64):
        x = pd.Timestamp(x)
    if isinstance(x, (pd.Timestamp, _dt.datetime, _dt.date)):
        return None if pd.isna(x) else x.isoformat()
    if isinstance(x, (pd.Period, Path)):
        return str(x)
    return x


def _normalized(path: Path) -> bytes:
    # A CRLF checkout (git core.autocrlf) must hash the same as the committed LF text.
    return path.read_bytes().replace(b"\r\n", b"\n")


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(_normalized(path)).hexdigest() if path.exists() else None


def _text_sha256(lines: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _code_sha256(root: Path) -> str | None:
    files = sorted((root / "src" / "omodel").rglob("*.py"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(root).as_posix().encode("utf-8") + b"\0" + _normalized(f) + b"\0")
    return h.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=60)


def provenance(root: Path | None = None) -> dict:
    """Code fingerprint plus git HEAD and whether the pre-registration, code or pyproject are uncommitted."""
    root = settings().root if root is None else Path(root)
    out: dict[str, Any] = {"code_sha256": _code_sha256(root), "git_head": None, "git_status": "unknown",
                           "uncommitted": []}
    try:
        status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *PROVENANCE_PATHS)
        head = _git(root, "rev-parse", "--verify", "HEAD")
    except (OSError, subprocess.SubprocessError) as exc:
        out["git_status"] = f"unknown ({type(exc).__name__}: {exc})"
        return out
    if status.returncode != 0:
        out["git_status"] = f"unknown ({(status.stderr or '').strip()[:200] or 'git status failed'})"
        return out
    out["git_head"] = head.stdout.strip() if head.returncode == 0 and head.stdout.strip() else None
    out["uncommitted"] = [ln[3:] for ln in status.stdout.splitlines() if ln.strip()]
    out["git_status"] = "clean" if out["git_head"] and not out["uncommitted"] else "dirty"
    return out


def _summ(values: pd.Series, dates: pd.Series, label: str, **extra: Any) -> dict:
    return {**extra, **stats.summarize(values, dates, label=label)}


def _dev(events: pd.DataFrame) -> pd.DataFrame:
    s = settings()
    d = pd.DatetimeIndex(events["date"])
    out = events.loc[(d >= s.dev_start) & (d < s.holdout_start)].reset_index(drop=True)
    holdout.assert_clean(pd.DatetimeIndex(out["date"]), "phase1 events")
    return out


def _at(frame: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
    r = frame.index.get_indexer(pd.DatetimeIndex(events["date"]))
    c = frame.columns.get_indexer(events["ticker"])
    out = np.full(len(events), np.nan)
    ok = (r >= 0) & (c >= 0)
    out[ok] = frame.to_numpy(dtype=float)[r[ok], c[ok]]
    return out


def stale_opens(panel: Panel) -> pd.DataFrame:
    """True where that row's open is not a real opening trade (STALE_OPEN_RULE)."""
    close = panel.close
    raw = panel.raw_close.reindex_like(close) if panel.raw_close is not None else close
    raw_open = (panel.open.reindex_like(close) * raw / close).to_numpy(dtype=float)  # as-traded open
    copied = np.isclose(raw_open, raw.shift(1).to_numpy(dtype=float), rtol=1e-9, atol=0.0)
    no_trade = panel.volume.reindex_like(close).to_numpy(dtype=float) == 0
    return pd.DataFrame(copied | no_trade, index=close.index, columns=close.columns)


def _stale_entry(events: pd.DataFrame, stale: pd.DataFrame) -> np.ndarray:
    return np.nan_to_num(_at(stale.shift(-1, fill_value=False).astype(float), events)) > 0


def _flat_share(dec: Decomposition, events: pd.DataFrame, window: int | None = None) -> np.ndarray:
    """Diagnostic only: per event, the share of exactly-zero stock returns among finite returns in its fit window."""
    window = params.BETA_WINDOW if window is None else window
    a = dec.stock_ret.to_numpy(dtype=float)
    r = dec.stock_ret.index.get_indexer(pd.DatetimeIndex(events["date"]))
    c = dec.stock_ret.columns.get_indexer(events["ticker"])
    lo = np.maximum(r - window, 0)
    # Same rows the rolling fit uses: _window_sum is c[hi]-c[lo], i.e. a[lo..hi-1], so rows [t-window, t-1].
    zero = (a == 0.0).astype(float)  # exact 0.0 only: a halted/illiquid tape, never a real return
    cf = np.zeros((a.shape[0] + 1, a.shape[1]))
    cz = np.zeros((a.shape[0] + 1, a.shape[1]))
    np.cumsum(np.isfinite(a).astype(float), axis=0, out=cf[1:])
    np.cumsum(zero, axis=0, out=cz[1:])
    out = np.full(len(events), np.nan)
    ok = (r >= 0) & (c >= 0) & (r >= window)  # r < window: the fit itself has no full window either
    if ok.any():
        n = cf[r[ok], c[ok]] - cf[lo[ok], c[ok]]
        z0 = cz[r[ok], c[ok]] - cz[lo[ok], c[ok]]
        out[ok] = np.where(n > 0, z0 / np.maximum(n, 1), np.nan)
    return out


def _coverage(panel: Panel, membership: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    if membership is None:
        return None, "skipped: no membership history supplied with a pre-built panel (e.g. a synthetic run)"
    try:
        return panel_mod.coverage(panel, membership=membership), "ok"
    except Exception as exc:  # coverage is a report section, never a reason to lose the run
        return None, f"skipped: coverage failed ({exc!r})"


def _coverage_summary(cov: pd.DataFrame | None) -> dict:
    if cov is None or cov.empty:
        return {}
    dev = cov.loc[cov.index >= settings().dev_start.year]
    dev = dev if len(dev) else cov
    total = float(dev["covered_days"].sum()) / max(float(dev["member_days"].sum()), 1.0)
    return {"first_year": int(dev.index[0]), "first_pct": float(dev["coverage_pct"].iloc[0]),
            "last_year": int(dev.index[-1]), "last_pct": float(dev["coverage_pct"].iloc[-1]),
            "min_pct": float(dev["coverage_pct"].min()), "dev_pct": 100.0 * total,
            "members_with_no_data": int(dev["members_with_no_data"].max())}


def _concentration(events: pd.DataFrame, top: int = 5) -> dict:
    months = stats.cluster_labels(events["date"]).astype(str).value_counts()
    n = int(months.sum())
    return {"n_months": int(len(months)), "top_months": months.head(top).to_dict(),
            "top_share": float(months.head(top).sum() / n) if n else math.nan}


def _horizon_table(events: pd.DataFrame, kinds: tuple[str, ...], label: str) -> list[dict]:
    return [_summ(events[f"{k}_{h}"], events["date"], f"{label} {k}_{h}", horizon=int(h), kind=k)
            for h in params.HORIZONS for k in kinds]


def _lost(ev: pd.DataFrame, h: int) -> dict:
    n = len(ev)
    usable = ev[f"exit_date_{h}"].notna().to_numpy()
    gap = ev[f"missing_{h}"].to_numpy(dtype=bool)
    miss = gap & ~ev["stale_entry"].to_numpy(dtype=bool)
    other = usable & ~gap & ev[f"net_abn_{h}"].isna().to_numpy()  # ETF price gap or undefined betas
    return {"n_events": n, "missing": int(miss.sum()), "share": float(miss.mean()) if n else math.nan,
            "etf_or_beta_missing": int(other.sum()), "embargoed_or_beyond_panel": int((~usable).sum())}


def _missing_exits(primary: pd.DataFrame, plc: pd.DataFrame) -> list[dict]:
    rows = []
    for h in params.HORIZONS:
        p = _lost(plc, h)
        rows.append({"horizon": int(h), **_lost(primary, h), "placebo_missing": p["missing"],
                     "placebo_share": p["share"], "placebo_etf_or_beta_missing": p["etf_or_beta_missing"]})
    return rows


def _stale_summary(primary: pd.DataFrame, plc: pd.DataFrame, including: dict) -> dict:
    flag = primary["stale_entry"].to_numpy(dtype=bool)
    years = pd.DatetimeIndex(primary["date"]).year
    by = pd.DataFrame({"year": years, "stale": flag}).groupby("year")["stale"].agg(["size", "sum"])
    n, pn = len(primary), len(plc)
    return {"rule": STALE_OPEN_RULE, "n_events": n, "n_stale": int(flag.sum()),
            "share": float(flag.mean()) if n else math.nan,
            "placebo_n_stale": int(plc["stale_entry"].sum()) if pn else 0,
            "placebo_share": float(plc["stale_entry"].mean()) if pn else math.nan,
            "by_year": [{"year": int(y), "events": int(r["size"]), "stale": int(r["sum"]),
                         "share": float(r["sum"] / r["size"])} for y, r in by.iterrows() if r["sum"]],
            "including_stale": including}


def _drift_adjusted(plc: pd.DataFrame, dec: Decomposition, member: pd.DataFrame, dev_dates: pd.DatetimeIndex,
                    col: str, h: int) -> dict:
    daily = (dec.resid + dec.alpha).reindex(index=dev_dates)  # = r - beta_mkt*mkt - beta_sec*sec: alpha excluded
    mu = daily.where(member.reindex(index=dev_dates, columns=daily.columns).eq(True)).mean()
    adj = plc[col].to_numpy(dtype=float) - h * plc["ticker"].map(mu).to_numpy(dtype=float)
    return _summ(pd.Series(adj), plc["date"], f"placebo {col} minus {h} x ticker mean daily abnormal return")


def _news_factor(primary: pd.DataFrame, panel: Panel, news: bool, refresh: bool) -> dict:
    if not news:
        return {"status": "skipped: disabled (--no-news)"}
    if not settings().sec_user_agent.strip():
        return {"status": "skipped: SEC_USER_AGENT not set"}
    tickers = sorted(primary["ticker"].unique())
    try:
        flags, missing = sec.news_flags(tickers, panel.dates, kind="any", refresh=refresh)
    except Exception as exc:  # a secondary analysis must not sink the pre-registered test
        return {"status": f"skipped: SEC news flags failed ({exc!r})"}
    has_8k = _at(flags.astype(float), primary)
    has_8k[primary["ticker"].isin(missing).to_numpy()] = np.nan
    primary["no_8k"] = np.where(np.isnan(has_8k), np.nan, 1.0 - has_8k)
    covered = float(np.isfinite(has_8k).mean()) if len(primary) else math.nan
    return {"status": "ok", "tickers_without_cik": sorted(missing), "n_tickers": len(tickers),
            "events_covered_share": covered, "events_with_8k_share": float(np.nanmean(has_8k)) if covered else math.nan,
            "window_sessions": params.get("SEC_NEWS_WINDOW")}


def _factor_battery(primary: pd.DataFrame, metric: str, with_news: bool) -> list[dict]:
    names = ["vol_z", "sec_ret", "mkt_ret", "peer_z", "abs_z"]
    specs = [{"name": n, "kind": "bucket", "col": n, "n_buckets": FACTOR_N_BUCKETS} for n in names]
    if with_news:
        specs.append({"name": "no_8k", "kind": "binary", "col": "no_8k"})
    table = factor_tests.factor_battery(primary, specs, metric)
    rows = []
    for rec in table.to_dict("records"):
        sign, why = FACTOR_EXPECTATION.get(rec["name"], ("?", ""))
        rows.append({**rec, "thesis_sign": sign, "thesis": why})
    return rows


def _ok(s: dict) -> bool:
    return bool(np.isfinite(s.get("mean", np.nan)) and s["mean"] > 0 and np.isfinite(s.get("ci_lo", np.nan))
                and s["ci_lo"] > 0 and not s.get("low_confidence", True))


def _criteria(primary: dict, excess: dict, wf: dict, plc_gross: dict, subs: list[dict], sharpe: float) -> list[dict]:
    ci = f"{params.CI_LEVEL:.0%}"
    excess_low = (min(excess["n_high"], excess["n_low"]) < params.MIN_SAMPLE_FINDING
                  or min(excess["n_eff_high"], excess["n_eff_low"]) < stats.MIN_EFFECTIVE_CLUSTERS)
    excess_ok = bool(np.isfinite(excess["diff"]) and excess["diff"] > 0 and np.isfinite(excess["lo"])
                     and excess["lo"] > 0 and not excess_low)
    lo, hi = plc_gross["ci_lo"], plc_gross["ci_hi"]
    plc_ok = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0 <= hi)
    positive = [s for s in subs if s["n"] and np.isfinite(s["mean"]) and s["mean"] > 0]
    sharpe_ok = bool(np.isfinite(sharpe) and sharpe < params.SUSPICIOUS_SHARPE)
    floor = f"n >= {params.MIN_SAMPLE_FINDING} and >= {stats.MIN_EFFECTIVE_CLUSTERS} effective month clusters"
    h = params.PRIMARY_HORIZON
    return [
        {"id": "1", "criterion": f"Event level: mean net abnormal {h}d return", "value": primary["mean"],
         "ci": [primary["ci_lo"], primary["ci_hi"]], "n": primary["n"],
         "threshold": f"mean > 0 and {ci} CI lower bound > 0 ({floor})", "passed": _ok(primary)},
        {"id": "1b", "criterion": "Excess over placebo (events minus placebo, month clusters resampled jointly)",
         "value": excess["diff"], "ci": [excess["lo"], excess["hi"]], "n": excess["n_high"] + excess["n_low"],
         "threshold": f"difference > 0 and {ci} CI lower bound > 0 ({floor})", "passed": excess_ok},
        {"id": "2", "criterion": f"Walk-forward OUT-OF-SAMPLE mean net abnormal {h}d return", "value": wf["mean"],
         "ci": [wf["ci_lo"], wf["ci_hi"]], "n": wf["n"],
         "threshold": f"mean > 0 and {ci} CI lower bound > 0 ({floor})", "passed": _ok(wf)},
        {"id": "3", "criterion": f"Placebo: mean abnormal {h}d return on random non-event member days (before the "
                                 "fixed round-trip cost)", "value": plc_gross["mean"], "ci": [lo, hi],
         "n": plc_gross["n"], "threshold": f"{ci} CI includes 0", "passed": plc_ok},
        {"id": "4", "criterion": "Stability: sub-periods with a positive mean", "value": len(positive),
         "ci": None, "n": sum(int(s["n"]) for s in subs),
         "threshold": f">= {STABILITY_MIN_SUBPERIODS} of {len(subs)}",
         "passed": len(positive) >= STABILITY_MIN_SUBPERIODS},
        {"id": "5", "criterion": "Plausibility: calendar-time portfolio annualized Sharpe", "value": sharpe,
         "ci": None, "n": None, "threshold": f"< {params.SUSPICIOUS_SHARPE}", "passed": sharpe_ok},
    ]


def run(panel: Panel | None = None, dec: Decomposition | None = None, *, membership: pd.DataFrame | None = None,
        out_dir: Path | None = None, news: bool = True, refresh: bool = False, write: bool = True) -> dict:
    import_all()
    started = _dt.datetime.now(_dt.timezone.utc)
    cfg = settings()
    h, entry = int(params.PRIMARY_HORIZON), params.ENTRY
    metric, gross, exit_col = f"net_abn_{h}", f"abn_{h}", f"exit_date_{h}"

    if panel is None:
        membership = universe.load_membership(refresh=refresh) if membership is None else membership
        panel = panel_mod.build_panel(refresh=refresh, membership=membership)
    holdout.assert_clean(panel.dates, "phase1 panel")
    cov, cov_status = _coverage(panel, membership)
    dec = l2.decompose(panel) if dec is None else dec
    dev_dates = panel.dates[(panel.dates >= cfg.dev_start) & (panel.dates < cfg.holdout_start)]
    if not len(dev_dates):
        raise ValueError(f"panel has no trading days in the development window from {cfg.dev_start.date()}")
    stale = stale_opens(panel)
    # Every return below is computed on this panel, so a fake open is never a fill for events or placebo.
    trade = dataclasses.replace(panel, open=panel.open.mask(stale))

    base: dict[float, pd.DataFrame] = {}
    by_z: dict[float, pd.DataFrame] = {}
    primary_z = float(params.EVENT_Z_THRESHOLD)
    for z in dict.fromkeys([*map(float, params.EVENT_Z_GRID), primary_z]):
        base[z] = _dev(ev_mod.find_events(dec, panel, z_threshold=z, side="down"))
        by_z[z] = ev_mod.add_forward_returns(base[z], trade, entry=entry).assign(
            stale_entry=_stale_entry(base[z], stale))
    primary = by_z[primary_z]
    if primary.empty:
        raise ValueError("no primary events in the development window")

    primary_stat = _summ(primary[metric], primary["date"], f"primary {metric}")
    horizons = _horizon_table(primary, ("raw", "abn", "net_abn"), "primary")
    literal = ev_mod.add_forward_returns(base[primary_z], panel, entry=entry)
    including_stale = _summ(literal[metric], literal["date"], f"primary {metric}, stale opens used as fills")

    positions = ev_mod.position_daily_returns(primary, trade, h, entry=entry)
    cal = stats.calendar_time_portfolio(positions, all_dates=dev_dates)
    sharpe = stats.annualized_sharpe(cal["ret"])
    calendar = {"newey_west": stats.newey_west(cal["ret"]), "sharpe": sharpe,
                "suspicious": bool(np.isfinite(sharpe) and sharpe > params.SUSPICIOUS_SHARPE),
                "n_days": int(len(cal)), "active_days": int((cal["n_positions"] > 0).sum()),
                "mean_positions": float(cal.loc[cal["n_positions"] > 0, "n_positions"].mean())
                if (cal["n_positions"] > 0).any() else math.nan}

    splits = walkforward.make_splits(cfg.dev_start, dev_dates[-1] + pd.Timedelta(days=1))
    oos, choices = walkforward.walk_forward({f"z{z}": by_z[z] for z in params.EVENT_Z_GRID}, metric, exit_col, splits)
    wf = {"oos": _summ(oos[metric], oos["date"], f"walk-forward OOS {metric}"),
          "n_splits": len(splits), "choices": choices.to_dict("records"),
          "chosen_counts": {str(k): int(v) for k, v in choices["chosen"].fillna("none").value_counts().items()},
          "oos_start": oos["date"].min() if len(oos) else None, "oos_end": oos["date"].max() if len(oos) else None}

    plc_days = placebo.placebo_events(primary, panel, dec)
    plc = ev_mod.add_forward_returns(plc_days, trade, entry=entry).assign(stale_entry=_stale_entry(plc_days, stale))
    plc_net = _summ(plc[metric], plc["date"], f"placebo {metric}")
    plc_gross = _summ(plc[gross], plc["date"], f"placebo {gross}")
    both_dates = pd.concat([primary["date"], plc["date"]], ignore_index=True)
    excess = stats.cluster_bootstrap_diff(
        pd.concat([primary[metric], plc[metric]], ignore_index=True),
        pd.Series(["event"] * len(primary) + ["placebo"] * len(plc)),
        stats.cluster_labels(both_dates), high="event", low="placebo")

    years = pd.DatetimeIndex(primary["date"]).year
    subs = [_summ(primary.loc[(years >= a) & (years <= b), metric], primary.loc[(years >= a) & (years <= b), "date"],
                  f"{a}-{b}", period=f"{a}-{b}") for a, b in SUBPERIODS]

    close_ev = ev_mod.add_forward_returns(base[primary_z], trade, entry="close")
    bounce = _summ(close_ev[metric] - primary[metric], primary["date"], "close entry minus next-open entry")
    up = ev_mod.add_forward_returns(_dev(ev_mod.find_events(dec, panel, z_threshold=primary_z, side="up")),
                                    trade, entry=entry)
    grid = [_summ(by_z[z][metric], by_z[z]["date"], f"z <= {z}", z=z) for z in params.EVENT_Z_GRID]

    primary["peer_z"] = _at(l2.peer_z(dec, panel.member), primary)
    primary["abs_z"] = primary["z"].abs()
    news_res = _news_factor(primary, panel, news, refresh)
    factors = _factor_battery(primary, metric, news_res["status"] == "ok")

    # Diagnostic only: how stale the fit windows behind the booked events were. Never a criterion or a filter.
    flat = _flat_share(dec, primary)
    fin = np.flatnonzero(np.isfinite(flat))
    worst_idx = fin[np.argsort(-flat[fin])][:5]
    stale_windows = {
        "rule": f"share of exactly-zero returns among finite returns in the {params.BETA_WINDOW}-day fit window "
                "[t-90, t-1]",
        "n_events": len(primary),
        "mean": float(np.nanmean(flat)) if len(fin) else math.nan,
        "gt_50pct": int((flat > STALE_WINDOW_SHARES[0]).sum()),
        "gt_90pct": int((flat > STALE_WINDOW_SHARES[1]).sum()),
        "worst": [{"date": primary["date"].iloc[i], "ticker": primary["ticker"].iloc[i],
                   "flat_share": float(flat[i]), "z": float(primary["z"].iloc[i])} for i in worst_idx],
    }

    criteria = _criteria(primary_stat, excess, wf["oos"], plc_gross, subs, sharpe)
    verdict = "PASS" if all(c["passed"] for c in criteria) else "FAIL"
    prereg = cfg.root / "docs" / "PREREGISTRATION.md"
    results: dict[str, Any] = {
        "verdict": verdict,
        "run_invalid": not next(c["passed"] for c in criteria if c["id"] == "3"),
        "criteria": criteria,
        "meta": {
            "run_utc": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prereg_path": str(prereg), "prereg_sha256": _sha256(prereg),
            "provenance": provenance(cfg.root),
            "readings": list(READINGS), "readings_sha256": _text_sha256(READINGS),
            "data_first_date": panel.dates[0], "data_last_date": panel.dates[-1],
            "dev_start": dev_dates[0], "dev_end": dev_dates[-1], "holdout_start": cfg.holdout_start,
            "holdout_unlocked": holdout.is_unlocked(), "n_tickers": len(panel.tickers),
            "n_dev_days": len(dev_dates), "entry": entry, "primary_z": primary_z, "primary_horizon": h,
            "metric": metric, "cost_bps_per_side": params.COST_BPS_PER_SIDE,
            "n_events_by_z": {str(z): int(len(e)) for z, e in by_z.items()},
        },
        "params": registry_snapshot(),
        "primary": {**primary_stat, "gross": _summ(primary[gross], primary["date"], f"primary {gross}"),
                    "concentration": _concentration(primary)},
        "horizons": horizons,
        "calendar": calendar,
        "walkforward": wf,
        "placebo": {"net": plc_net, "gross": plc_gross, "excess": excess, "n_per_event": 1,
                    "drift_adjusted": _drift_adjusted(plc, dec, panel.member, dev_dates, gross, h)},
        "subperiods": subs,
        "secondary": {
            "close_entry": _horizon_table(close_ev, ("net_abn",), "close entry"),
            "bounce": bounce,
            "up": _horizon_table(up, ("abn", "net_abn"), "up z >= +3"),
            "threshold_grid": grid,
            "factors": factors,
            "stale_windows": stale_windows,
            "news": news_res,
        },
        "missing_exits": _missing_exits(primary, plc),
        "stale_entries": _stale_summary(primary, plc, including_stale),
        "coverage": {"status": cov_status, "summary": _coverage_summary(cov),
                     "table": [] if cov is None else cov.reset_index().to_dict("records")},
        "frames": {"dec": dec, "events": by_z, "primary": primary, "placebo": plc, "oos": oos, "choices": choices,
                   "calendar": cal, "coverage": cov, "up": up, "close_entry": close_ev, "stale": stale},
    }
    if write:
        results["paths"] = _write(results, out_dir)
    return results


def _write(results: dict, out_dir: Path | None) -> dict[str, str]:
    from omodel.report.markdown import render_phase1

    out = Path(out_dir) if out_dir is not None else settings().reports_dir / "phase1"
    out.mkdir(parents=True, exist_ok=True)
    frames = results["frames"]
    paths = {"report": out / "report.md", "results": out / "results.json",
             "choices": out / "walkforward_choices.csv", "coverage": out / "coverage.csv"}
    payload = jsonable({k: v for k, v in results.items() if k != "frames"})
    paths["results"].write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    paths["report"].write_text(render_phase1(payload), encoding="utf-8")
    frames["choices"].to_csv(paths["choices"], index=False)
    cov = frames["coverage"]
    if cov is None:
        cov = pd.DataFrame(columns=["member_days", "covered_days", "coverage_pct", "members",
                                    "members_with_no_data"], index=pd.Index([], name="year"))
    cov.to_csv(paths["coverage"])
    paths["parameters"] = write_parameters_doc()
    paths["events"] = store.write(frames["primary"], "results", "phase1_events",
                                  meta={"run_utc": results["meta"]["run_utc"], "metric": results["meta"]["metric"]})
    return {k: str(v) for k, v in paths.items()}

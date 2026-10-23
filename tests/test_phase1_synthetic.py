import dataclasses
import json
import re
import shutil
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from omodel import config, params
from omodel.backtest.events import EVENT_COLUMNS
from omodel.backtest.synthetic import synthetic_panel
from omodel.experiments import phase1
from omodel.types import Decomposition, Panel

START, N_DAYS, N_STOCKS = "2008-01-02", 2600, 150
DEV_START = pd.bdate_range(START, periods=401)[-1]  # ~400 sessions of warm-up for sector picks and betas
FORWARD = ("raw", "mkt", "sec", "abn", "net_raw", "net_abn", "missing", "exit_date")


@contextmanager
def dev_window(tmp, **extra):
    config.override(data_dir=tmp / "data", dev_start=DEV_START, sec_user_agent="", **extra)
    try:
        yield
    finally:
        config.reset()


def world(reversion, seed=1):
    # alpha=0: the null world has no abnormal drift anywhere, so any edge the pipeline reports is its own doing.
    return synthetic_panel(n_days=N_DAYS, n_stocks=N_STOCKS, seed=seed, reversion=reversion, start=START, alpha=0.0)


def run_world(tmp, panel, **kw):
    with dev_window(tmp):
        return phase1.run(panel=panel, news=False, write=False, **kw)


def crit(res, cid):
    return next(c for c in res["criteria"] if c["id"] == cid)


def significant_positive(s):
    return s["mean"] > 0 and s["ci_lo"] > 0


@pytest.fixture(scope="module")
def null_world(tmp_path_factory):
    panel, _ = world(0.0)
    return panel, run_world(tmp_path_factory.mktemp("null"), panel)


@pytest.fixture(scope="module")
def edge_world(tmp_path_factory):
    panel, _ = world(0.6)
    return panel, run_world(tmp_path_factory.mktemp("edge"), panel)


def test_null_world_fails_with_clean_placebo(null_world):
    _, res = null_world
    assert res["verdict"] == "FAIL"
    pr = res["primary"]
    assert pr["n"] >= params.MIN_SAMPLE_FINDING and not pr["low_confidence"]
    assert not significant_positive(pr)
    assert not significant_positive(res["walkforward"]["oos"])
    assert not crit(res, "1")["passed"] and not crit(res, "1b")["passed"] and not crit(res, "2")["passed"]
    gross = res["placebo"]["gross"]
    assert gross["n"] >= params.MIN_SAMPLE_FINDING
    assert gross["ci_lo"] <= 0 <= gross["ci_hi"]
    assert crit(res, "3")["passed"] and not res["run_invalid"]


def test_edge_world_is_found_out_of_sample_and_trips_the_plausibility_guard(edge_world):
    _, res = edge_world
    assert significant_positive(res["primary"])
    oos = res["walkforward"]["oos"]
    assert oos["mean"] > 0 and oos["ci_lo"] > 0 and oos["n"] >= params.MIN_SAMPLE_FINDING
    assert all(crit(res, c)["passed"] for c in ("1", "1b", "2", "3"))
    # A huge synthetic edge has a Sharpe no real strategy should: the run must refuse to call it a PASS.
    assert res["calendar"]["suspicious"] and not crit(res, "5")["passed"]
    assert res["verdict"] == "FAIL"


def test_moderate_edge_can_pass(tmp_path):
    # Seed 2: every sub-period with data is positive (the 2017 block holds under a year of synthetic data).
    panel, _ = world(0.25, seed=2)
    res = run_world(tmp_path, panel)
    assert [c["id"] for c in res["criteria"] if not c["passed"]] == []
    assert res["verdict"] == "PASS"
    assert 0 < res["calendar"]["sharpe"] < params.SUSPICIOUS_SHARPE


def test_results_are_internally_consistent(null_world):
    panel, res = null_world
    fr, h = res["frames"], params.PRIMARY_HORIZON
    primary = fr["primary"]
    assert primary["date"].min() >= DEV_START and primary["date"].max() <= panel.dates[-1]
    assert (primary["z"] <= params.EVENT_Z_THRESHOLD).all()
    assert res["primary"]["n"] == primary[f"net_abn_{h}"].notna().sum()
    assert res["meta"]["n_events_by_z"] == {str(float(z)): len(fr["events"][float(z)]) for z in params.EVENT_Z_GRID}
    assert set(fr["oos"]["config"]) <= {f"z{z}" for z in params.EVENT_Z_GRID}
    assert set(fr["placebo"]["ticker"]) <= set(primary["ticker"])
    assert not fr["placebo"].merge(primary[["date", "ticker"]], on=["date", "ticker"]).shape[0]
    assert [s["period"] for s in res["subperiods"]] == ["2000-2008", "2009-2016", "2017-2025"]
    assert sum(s["n"] for s in res["subperiods"]) == res["primary"]["n"]
    names = [f["name"] for f in res["secondary"]["factors"]]
    assert names == ["vol_z", "sec_ret", "mkt_ret", "peer_z", "abs_z"]
    assert res["secondary"]["news"]["status"].startswith("skipped")
    assert res["coverage"]["status"].startswith("skipped")
    assert [m["horizon"] for m in res["missing_exits"]] == list(params.HORIZONS)
    assert set(res["params"]) == set(params.REGISTRY) and "SUBPERIODS" in res["params"]
    json.dumps(phase1.jsonable({k: v for k, v in res.items() if k != "frames"}), allow_nan=False)


def _perturb_after(panel: Panel, cut: pd.Timestamp, seed: int = 99) -> Panel:
    rng = np.random.default_rng(seed)
    after = np.asarray(panel.dates > cut)
    k = int(after.sum())

    def walk(n_cols):
        return np.exp(np.cumsum(rng.normal(0.0, 0.03, (k, n_cols)), axis=0))

    def scaled(df, f):
        out = df.copy()
        out.loc[after] = df.loc[after].to_numpy() * f
        return out

    fs, fe = walk(panel.close.shape[1]), walk(panel.etf_close.shape[1])
    member = panel.member.copy()
    member.loc[after] = rng.random((k, member.shape[1])) < 0.7
    return dataclasses.replace(
        panel, close=scaled(panel.close, fs), open=scaled(panel.open, fs), raw_close=scaled(panel.raw_close, fs),
        volume=scaled(panel.volume, np.exp(rng.normal(0.0, 1.0, (k, panel.volume.shape[1])))), member=member,
        etf_close=scaled(panel.etf_close, fe), etf_open=scaled(panel.etf_open, fe))


def test_lookahead_canary_future_prices_cannot_change_the_past(null_world, tmp_path):
    panel, res = null_world
    cut = panel.dates[1800]
    shaken = run_world(tmp_path, _perturb_after(panel, cut))
    a, b = res["frames"], shaken["frames"]

    for f in dataclasses.fields(Decomposition):
        x, y = getattr(a["dec"], f.name), getattr(b["dec"], f.name)
        assert_equal = pd.testing.assert_series_equal if isinstance(x, pd.Series) else pd.testing.assert_frame_equal
        assert_equal(x.loc[:cut], y.loc[:cut], check_exact=True, obj=f"dec.{f.name}")
    assert not np.allclose(a["dec"].z.loc[cut:].iloc[1:].fillna(0), b["dec"].z.loc[cut:].iloc[1:].fillna(0))

    changed = 0
    for z in params.EVENT_Z_GRID:
        ea, eb = a["events"][float(z)], b["events"][float(z)]
        ea, eb = (e.loc[e["date"] <= cut].reset_index(drop=True) for e in (ea, eb))
        assert len(ea) > 50
        fwd = {"entry_date", "entry_px"} | {f"{k}_{h}" for k in FORWARD for h in params.HORIZONS}
        static = [c for c in ea.columns if c not in fwd]
        assert set(EVENT_COLUMNS) <= set(static)
        pd.testing.assert_frame_equal(ea[static], eb[static], check_exact=True)
        entered = (ea["entry_date"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(ea.loc[entered, ["entry_date", "entry_px"]],
                                      eb.loc[entered, ["entry_date", "entry_px"]], check_exact=True)
        for h in params.HORIZONS:
            cols = [f"{k}_{h}" for k in FORWARD]
            done = (ea[f"exit_date_{h}"] <= cut).to_numpy()
            pd.testing.assert_frame_equal(ea.loc[done, cols], eb.loc[done, cols], check_exact=True)
            crossing = ~done & ea[f"exit_date_{h}"].notna().to_numpy()
            same = np.isclose(ea.loc[crossing, f"net_abn_{h}"], eb.loc[crossing, f"net_abn_{h}"], equal_nan=True)
            changed += int((~same).sum())
    assert changed > 0  # the forward windows that do cross the cut must see the new prices

    ca, cb = a["choices"], b["choices"]
    early = (ca["test_start"] <= cut).to_numpy()
    assert early.any()
    pd.testing.assert_frame_equal(ca.loc[early, ["test_start", "chosen", "fit_n", "fit_score"]],
                                  cb.loc[early, ["test_start", "chosen", "fit_n", "fit_score"]], check_exact=True)


def test_run_builds_panel_writes_outputs_and_measures_coverage(null_world, tmp_path, monkeypatch):
    panel, _ = null_world
    membership = pd.DataFrame({"ticker": [*panel.tickers, "GONE"], "start_date": panel.dates[0],
                               "end_date": pd.NaT})
    calls = {}

    def fake_build(refresh=False, membership=None, **kw):
        calls["build"] = (refresh, membership is not None)
        return panel

    monkeypatch.setattr(phase1.universe, "load_membership", lambda refresh=False: membership)
    monkeypatch.setattr(phase1.panel_mod, "build_panel", fake_build)
    (tmp_path / "docs").mkdir()
    shutil.copy(config.ROOT / "docs" / "PREREGISTRATION.md", tmp_path / "docs" / "PREREGISTRATION.md")
    out = tmp_path / "out"
    with dev_window(tmp_path, root=tmp_path):
        res = phase1.run(out_dir=out, news=True)
        events_path = config.settings().data_dir / "results" / "phase1_events.parquet"
    assert calls["build"] == (False, True)
    assert res["secondary"]["news"]["status"] == "skipped: SEC_USER_AGENT not set"

    cov = res["coverage"]
    assert cov["status"] == "ok"
    expected = 100.0 * len(panel.tickers) / (len(panel.tickers) + 1)
    assert cov["summary"]["dev_pct"] == pytest.approx(expected)
    assert all(row["members_with_no_data"] == 1 for row in cov["table"])
    assert pd.read_csv(out / "coverage.csv")["coverage_pct"].round(6).eq(round(expected, 6)).all()

    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == res["verdict"] and "frames" not in payload
    assert payload["meta"]["prereg_sha256"] == phase1._sha256(config.ROOT / "docs" / "PREREGISTRATION.md")
    choices = pd.read_csv(out / "walkforward_choices.csv")
    assert len(choices) == len(res["walkforward"]["choices"])
    assert (tmp_path / "docs" / "PARAMETERS.md").read_text(encoding="utf-8") == params.render_markdown()
    stored = pd.read_parquet(events_path)
    assert len(stored) == len(res["frames"]["primary"])

    report = (out / "report.md").read_text(encoding="utf-8")
    for phrase in ("Verdict: FAIL", "OUT-OF-SAMPLE result (walk-forward)",
                   "pre-registered single test (no parameters fit)", "## Placebo", "95% CI",
                   "## What would have to be true for this to be a fluke", "Coverage by year",
                   "## Decision: pre-registered criteria", "in-sample, descriptive only"):
        assert phrase in report, phrase
    assert not re.search(r"nan", report, re.IGNORECASE)


def test_jsonable_handles_nan_dates_and_numpy():
    raw = {"a": np.float64("nan"), "b": [np.int64(3), np.bool_(True)], "c": pd.Timestamp("2020-01-02"),
           "d": pd.NaT, "e": (1.5, float("inf")), "f": pd.DataFrame({"x": [1.0, np.nan]}),
           "g": pd.Period("2020-01", "M"), 4: None}
    out = phase1.jsonable(raw)
    assert out == {"a": None, "b": [3, True], "c": "2020-01-02T00:00:00", "d": None, "e": [1.5, None],
                   "f": [{"x": 1.0}, {"x": None}], "g": "2020-01", "4": None}
    json.dumps(out, allow_nan=False)

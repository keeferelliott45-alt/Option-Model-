import dataclasses
import json
import re
import shutil
import subprocess
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from omodel import config, params
from omodel.backtest import events as ev_mod
from omodel.backtest import placebo, stats, walkforward
from omodel.backtest.events import EVENT_COLUMNS
from omodel.backtest.synthetic import synthetic_panel
from omodel.experiments import phase1
from omodel.report.markdown import render_phase1
from omodel.types import Decomposition, Panel

START, N_DAYS, N_STOCKS = "2008-01-02", 2600, 150
DEV_START = pd.bdate_range(START, periods=401)[-1]  # ~400 sessions of warm-up for sector picks and betas
FORWARD = ("raw", "mkt", "sec", "abn", "net_raw", "net_abn", "missing", "exit_date")
# LF-normalized sha256 of docs/PREREGISTRATION.md. Change it only together with a dated amendment written before
# any real-data result; a silent edit to the pre-registration must fail the suite.
PREREG_SHA256 = "394b7d5371623754891e33327c5d2187067c430ae24d4608743a6d9de83e5658"
NAN_RE = re.compile(r"\bnan\b", re.IGNORECASE)


def fake_git(dirty=()):
    def run(root, *args):
        if args[0] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, "0123456789abcdef0123\n", "")
        return subprocess.CompletedProcess(args, 0, "".join(f"?? {f}\n" for f in dirty), "")
    return run


@pytest.fixture(scope="module", autouse=True)
def no_git():
    # phase1 records git provenance; the suite itself must never run git.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(phase1, "_git", fake_git())
        yield


@contextmanager
def dev_window(tmp, **extra):
    config.override(data_dir=tmp / "data", dev_start=DEV_START, sec_user_agent="", **extra)
    try:
        yield
    finally:
        config.reset()


def world(reversion, seed=1):
    # alpha=0 and drift-neutral shocks: random days have exactly zero expected abnormal drift, so a placebo away
    # from zero, or an edge in the null world, can only be the pipeline's own doing.
    return synthetic_panel(n_days=N_DAYS, n_stocks=N_STOCKS, seed=seed, reversion=reversion, start=START, alpha=0.0,
                           drift_neutral=True)


def run_world(tmp, panel, **kw):
    with dev_window(tmp):
        return phase1.run(panel=panel, news=False, write=False, **kw)


def crit(res, cid):
    return next(c for c in res["criteria"] if c["id"] == cid)


def significant_positive(s):
    return s["mean"] > 0 and s["ci_lo"] > 0


@pytest.fixture(scope="module")
def null_world(tmp_path_factory, no_git):
    panel, _ = world(0.0)
    calls, real = [], walkforward.walk_forward

    def spy(events_by_config, metric_col, exit_col, splits, **kw):
        calls.append((list(events_by_config), metric_col, exit_col, splits))
        return real(events_by_config, metric_col, exit_col, splits, **kw)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(walkforward, "walk_forward", spy)
        res = run_world(tmp_path_factory.mktemp("null"), panel)
    return panel, res, calls


@pytest.fixture(scope="module")
def edge_world(tmp_path_factory, no_git):
    panel, _ = world(0.6)
    return panel, run_world(tmp_path_factory.mktemp("edge"), panel)


def test_null_world_fails_with_clean_placebo(null_world):
    _, res, _ = null_world
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
    drift = res["placebo"]["drift_adjusted"]
    assert drift["n"] == gross["n"] and drift["ci_lo"] <= 0 <= drift["ci_hi"]


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
    panel, _ = world(0.25, seed=2)
    res = run_world(tmp_path, panel)
    assert [c["id"] for c in res["criteria"] if not c["passed"]] == []
    assert res["verdict"] == "PASS"
    assert 0 < res["calendar"]["sharpe"] < params.SUSPICIOUS_SHARPE


def _forward(days: pd.DataFrame, panel: Panel, dec: Decomposition, h: int) -> pd.DataFrame:
    ev = days.copy()
    r = dec.sector.index.get_indexer(ev["date"])
    c = dec.sector.columns.get_indexer(ev["ticker"])
    ev["sector"] = dec.sector.to_numpy(dtype=object)[r, c]
    ev["beta_mkt"] = phase1._at(dec.beta_mkt, ev)
    ev["beta_sec"] = phase1._at(dec.beta_sec, ev)
    return ev_mod.add_forward_returns(ev, panel, horizons=(h,), entry=params.ENTRY)


def _mean_over(mask: np.ndarray, frame: pd.DataFrame, panel: Panel, dec: Decomposition, h: int) -> dict:
    r, c = np.nonzero(mask)
    days = pd.DataFrame({"date": frame.index[r], "ticker": np.asarray(frame.columns, dtype=object)[c]})
    fw = _forward(days, panel, dec, h)
    return stats.summarize(fw[f"abn_{h}"], fw["date"])


def test_placebo_pool_is_unbiased_in_a_drift_free_null(null_world):
    # Exact mean over EVERY day the placebo may draw (~250k ticker-days), so a few-bp bias is visible.
    panel, res, _ = null_world
    primary, dec, h, w = res["frames"]["primary"], res["frames"]["dec"], params.PRIMARY_HORIZON, params.EVENT_COOLDOWN
    pool = placebo.eligible_days(primary, panel, dec)
    clean = _mean_over(pool.to_numpy(), pool, panel, dec, h)
    assert clean["n"] > 100_000 and clean["ci_lo"] <= 0 <= clean["ci_hi"], clean
    assert abs(clean["mean"]) < 0.0015

    # Power check: also dropping days BEFORE an event (keeping only windows with no future drop) is lookahead,
    # and at this sample size that bias must be unmistakable.
    ev_pos = pool.index.get_indexer(pd.DatetimeIndex(primary["date"]))
    ev_col = pool.columns.get_indexer(primary["ticker"])
    pre = np.zeros((len(pool) + 1, pool.shape[1]), dtype=np.int64)
    np.add.at(pre, (np.clip(ev_pos - w, 0, None), ev_col), 1)
    np.add.at(pre, (ev_pos, ev_col), -1)
    biased = _mean_over(pool.to_numpy() & ~(np.cumsum(pre, axis=0)[:-1] > 0), pool, panel, dec, h)
    assert biased["ci_lo"] > 0 and biased["mean"] > clean["mean"] + 0.002, biased


def test_preregistration_contract(null_world):
    panel, res, calls = null_world
    assert phase1._sha256(config.ROOT / "docs" / "PREREGISTRATION.md") == PREREG_SHA256, (
        "docs/PREREGISTRATION.md changed: only a dated amendment written before any real-data result may change "
        "it, and PREREG_SHA256 is updated in the same commit")
    assert not hasattr(phase1, "ENTRY")  # one source of truth: the registered parameter
    assert params.ENTRY == res["meta"]["entry"] == "next_open"
    pinned = {"PRIMARY_HORIZON": 15, "HORIZONS": (1, 5, 15, 30), "COST_BPS_PER_SIDE": 5.0,
              "EVENT_Z_THRESHOLD": -3.0, "EVENT_Z_GRID": (-2.5, -3.0, -4.0), "BETA_WINDOW": 90,
              "WF_FIT_YEARS": 2, "WF_TEST_MONTHS": 6, "SUBPERIODS": ((2000, 2008), (2009, 2016), (2017, 2025)),
              "STABILITY_MIN_SUBPERIODS": 2, "SUSPICIOUS_SHARPE": 2.0, "CI_LEVEL": 0.95,
              "MIN_SAMPLE_FINDING": 30, "EVENT_COOLDOWN": 15}
    assert {k: params.get(k) for k in pinned} == pinned
    assert config.Settings().dev_start == pd.Timestamp("2000-01-03")
    assert config.Settings().holdout_start == pd.Timestamp("2025-10-23")

    h = params.PRIMARY_HORIZON
    [(names, metric, exit_col, splits)] = calls
    assert names == ["z-2.5", "z-3.0", "z-4.0"] and metric == f"net_abn_{h}" and exit_col == f"exit_date_{h}"
    assert splits and all(abs((s.fit_end - s.fit_start).days - 730) <= 3 and s.test_start == s.fit_end
                          and (s.test_end - s.test_start).days <= 184 for s in splits)

    prim = res["frames"]["primary"]
    assert (prim["z"] <= -3.0).all()
    ok = prim["entry_date"].notna().to_numpy()
    pos = prim["pos"].to_numpy()
    assert (prim["entry_date"].to_numpy()[ok] == panel.dates[pos[ok] + 1].to_numpy()).all()  # t+1 open
    done = prim[f"exit_date_{h}"].notna().to_numpy()
    assert (prim[f"exit_date_{h}"].to_numpy()[done] == panel.dates[pos[done] + h].to_numpy()).all()
    np.testing.assert_allclose(prim[f"net_abn_{h}"], prim[f"abn_{h}"] - 2 * 5.0 / 1e4)

    plc = res["frames"]["placebo"]
    gross = stats.summarize(plc[f"abn_{h}"], plc["date"])  # criterion 3 is judged before costs
    c3 = crit(res, "3")
    assert c3["value"] == gross["mean"] and c3["ci"] == [gross["ci_lo"], gross["ci_hi"]]
    assert crit(res, "4")["threshold"] == ">= 2 of 3" and crit(res, "5")["threshold"] == "< 2.0"
    assert [s["period"] for s in res["subperiods"]] == ["2000-2008", "2009-2016", "2017-2025"]


def test_results_are_internally_consistent(null_world):
    panel, res, _ = null_world
    fr, h = res["frames"], params.PRIMARY_HORIZON
    primary = fr["primary"]
    assert primary["date"].min() >= DEV_START and primary["date"].max() <= panel.dates[-1]
    assert (primary["z"] <= params.EVENT_Z_THRESHOLD).all()
    assert res["primary"]["n"] == primary[f"net_abn_{h}"].notna().sum()
    assert res["meta"]["n_events_by_z"] == {str(float(z)): len(fr["events"][float(z)]) for z in params.EVENT_Z_GRID}
    assert set(fr["oos"]["config"]) <= {f"z{z}" for z in params.EVENT_Z_GRID}
    assert set(fr["placebo"]["ticker"]) <= set(primary["ticker"])
    assert not fr["placebo"].merge(primary[["date", "ticker"]], on=["date", "ticker"]).shape[0]
    assert (fr["placebo"]["price"] >= params.MIN_PRICE).all()
    assert sum(s["n"] for s in res["subperiods"]) == res["primary"]["n"]
    names = [f["name"] for f in res["secondary"]["factors"]]
    assert names == ["vol_z", "sec_ret", "mkt_ret", "peer_z", "abs_z"]
    assert res["secondary"]["news"]["status"].startswith("skipped")
    assert res["coverage"]["status"].startswith("skipped")
    m15 = next(m for m in res["missing_exits"] if m["horizon"] == h)
    assert [m["horizon"] for m in res["missing_exits"]] == list(params.HORIZONS)
    usable = primary[f"exit_date_{h}"].notna().sum()
    assert res["primary"]["n"] + m15["missing"] + m15["etf_or_beta_missing"] + res["stale_entries"]["n_stale"] == usable
    assert res["stale_entries"]["n_stale"] == 0 and not fr["stale"].to_numpy().any()  # synthetic opens are real
    assert set(res["params"]) == set(params.REGISTRY) and {"SUBPERIODS", "STALE_OPEN_RULE"} <= set(res["params"])
    meta = res["meta"]
    assert meta["readings"] == list(phase1.READINGS) and len(meta["readings_sha256"]) == 64
    assert meta["provenance"]["git_status"] == "clean" and len(meta["provenance"]["code_sha256"]) == 64
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
    panel, res, _ = null_world
    cut = panel.dates[1800]
    shaken_panel = _perturb_after(panel, cut)
    shaken = run_world(tmp_path, shaken_panel)
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
        fwd = {"entry_date", "entry_px", "stale_entry"} | {f"{k}_{h}" for k in FORWARD for h in params.HORIZONS}
        static = [c for c in ea.columns if c not in fwd]
        assert set(EVENT_COLUMNS) <= set(static)
        pd.testing.assert_frame_equal(ea[static], eb[static], check_exact=True)
        entered = (ea["entry_date"] <= cut).to_numpy()
        pd.testing.assert_frame_equal(ea.loc[entered, ["entry_date", "entry_px", "stale_entry"]],
                                      eb.loc[entered, ["entry_date", "entry_px", "stale_entry"]], check_exact=True)
        for h in params.HORIZONS:
            cols = [f"{k}_{h}" for k in FORWARD]
            done = (ea[f"exit_date_{h}"] <= cut).to_numpy()
            pd.testing.assert_frame_equal(ea.loc[done, cols], eb.loc[done, cols], check_exact=True)
            crossing = ~done & ea[f"exit_date_{h}"].notna().to_numpy()
            same = np.isclose(ea.loc[crossing, f"net_abn_{h}"], eb.loc[crossing, f"net_abn_{h}"], equal_nan=True)
            changed += int((~same).sum())
    assert changed > 0  # the forward windows that do cross the cut must see the new prices

    # Which days the placebo may draw on or before the cut must not depend on anything after it.
    pa = placebo.eligible_days(a["primary"], panel, a["dec"])
    pb = placebo.eligible_days(b["primary"], shaken_panel, b["dec"])
    common = pa.columns.intersection(pb.columns)
    assert len(common) > 50 and a["primary"]["date"].max() > cut and b["primary"]["date"].max() > cut
    pd.testing.assert_frame_equal(pa.loc[:cut, common], pb.loc[:cut, common], check_exact=True)

    ca, cb = a["choices"], b["choices"]
    early = (ca["test_start"] <= cut).to_numpy()
    assert early.any()
    pd.testing.assert_frame_equal(ca.loc[early, ["test_start", "chosen", "fit_n", "fit_score"]],
                                  cb.loc[early, ["test_start", "chosen", "fit_n", "fit_score"]], check_exact=True)


def test_stale_opens_flags_copied_opens_and_zero_volume_through_adjustments():
    idx = pd.bdate_range("2015-01-05", periods=3)
    c = pd.DataFrame({"A": [10.0, 9.0, 9.5], "B": [20.0, 21.0, 22.0]}, index=idx)  # vendor (split-adjusted) close
    o = pd.DataFrame({"A": [10.0, 10.0, 9.4], "B": [20.0, 20.5, 21.5]}, index=idx)
    k = pd.DataFrame({"A": [0.97, 0.98, 0.98], "B": [0.99, 0.99, 0.99]}, index=idx)  # dividend adjustment
    vol = pd.DataFrame({"A": [1e6, 1e6, 1e6], "B": [1e6, 1e6, 0.0]}, index=idx)
    panel = Panel(close=c * k, open=o * k, volume=vol, member=c.notna(), etf_close=pd.DataFrame(index=idx),
                  etf_open=pd.DataFrame(index=idx), raw_close=c * 2.0)
    expected = pd.DataFrame({"A": [False, True, False], "B": [False, False, True]}, index=idx)
    pd.testing.assert_frame_equal(phase1.stale_opens(panel), expected)


def test_fake_opens_are_never_fills(null_world, tmp_path):
    panel, res, _ = null_world
    h = params.PRIMARY_HORIZON
    prim = res["frames"]["primary"]
    pick = prim.loc[prim[f"net_abn_{h}"].notna()].iloc[::7].head(40)
    opens = panel.open.copy()
    for p, t in zip(pick["pos"], pick["ticker"]):
        j = opens.columns.get_loc(t)
        opens.iat[p + 1, j] = panel.raw_close.iat[p, j]  # the t+1 open is the t close copied forward
    out = run_world(tmp_path, dataclasses.replace(panel, open=opens))

    hit = out["frames"]["primary"].set_index(["date", "ticker"]).loc[list(zip(pick["date"], pick["ticker"]))]
    assert hit["stale_entry"].all() and hit[f"net_abn_{h}"].isna().all()
    st = out["stale_entries"]
    assert st["n_stale"] == len(pick) and sum(y["stale"] for y in st["by_year"]) == len(pick)
    assert out["primary"]["n"] == res["primary"]["n"] - len(pick)
    assert st["including_stale"]["n"] == res["primary"]["n"]
    before = {m["horizon"]: m["missing"] for m in res["missing_exits"]}
    assert {m["horizon"]: m["missing"] for m in out["missing_exits"]} == before  # not survivorship gaps


def test_provenance_reads_git_state(monkeypatch, tmp_path):
    monkeypatch.setattr(phase1, "_git", fake_git())
    p = phase1.provenance(config.ROOT)
    assert p["git_status"] == "clean" and p["git_head"].startswith("0123") and len(p["code_sha256"]) == 64
    monkeypatch.setattr(phase1, "_git", fake_git(dirty=["docs/PREREGISTRATION.md", "src/omodel/cli.py"]))
    p = phase1.provenance(config.ROOT)
    assert p["git_status"] == "dirty" and p["uncommitted"] == ["docs/PREREGISTRATION.md", "src/omodel/cli.py"]

    def no_commit(root, *args):
        return subprocess.CompletedProcess(args, 0 if args[0] == "status" else 128, "", "")

    monkeypatch.setattr(phase1, "_git", no_commit)
    assert phase1.provenance(config.ROOT)["git_status"] == "dirty"

    def not_repo(root, *args):
        return subprocess.CompletedProcess(args, 128, "", "fatal: not a git repository")

    def no_git_binary(root, *args):
        raise FileNotFoundError("git")

    for fake in (not_repo, no_git_binary):
        monkeypatch.setattr(phase1, "_git", fake)
        p = phase1.provenance(tmp_path)
        assert p["git_status"].startswith("unknown") and p["git_head"] is None and p["code_sha256"] is None


def test_code_fingerprint_ignores_line_endings_but_not_content(tmp_path):
    pkg = tmp_path / "src" / "omodel"
    pkg.mkdir(parents=True)
    (pkg / "a.py").write_bytes(b"x = 1\ny = 2\n")
    first = phase1._code_sha256(tmp_path)
    (pkg / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    assert phase1._code_sha256(tmp_path) == first
    (pkg / "a.py").write_bytes(b"x = 1\ny = 3\n")
    assert phase1._code_sha256(tmp_path) != first


def test_run_builds_panel_writes_outputs_and_measures_coverage(null_world, tmp_path, monkeypatch):
    panel, _, _ = null_world
    membership = pd.DataFrame({"ticker": [*panel.tickers, "GONE"], "start_date": panel.dates[0],
                               "end_date": pd.NaT})
    calls = {}

    def fake_build(refresh=False, membership=None, **kw):
        calls["build"] = (refresh, membership is not None)
        return panel

    monkeypatch.setattr(phase1.universe, "load_membership", lambda refresh=False: membership)
    monkeypatch.setattr(phase1.panel_mod, "build_panel", fake_build)
    monkeypatch.setattr(phase1, "_git", fake_git(dirty=["docs/PREREGISTRATION.md"]))
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
    assert payload["meta"]["prereg_sha256"] == PREREG_SHA256
    assert payload["meta"]["provenance"]["git_status"] == "dirty"
    choices = pd.read_csv(out / "walkforward_choices.csv")
    assert len(choices) == len(res["walkforward"]["choices"])
    assert (tmp_path / "docs" / "PARAMETERS.md").read_text(encoding="utf-8") == params.render_markdown()
    stored = pd.read_parquet(events_path)
    assert len(stored) == len(res["frames"]["primary"])

    report = (out / "report.md").read_text(encoding="utf-8")
    for phrase in ("Verdict: FAIL", "OUT-OF-SAMPLE result (walk-forward)",
                   "pre-registered single test (no parameters fit)", "## Placebo", "95% CI",
                   "## What would have to be true for this to be a fluke", "Coverage by year",
                   "## Decision: pre-registered criteria", "in-sample, descriptive only", "PROVENANCE WARNING",
                   "## Implementation readings", "past events only", "Lost to ETF/beta gap",
                   "### Entry opens that were not real trades", "Diagnostic only (not a criterion)"):
        assert phrase in report, phrase
    assert not NAN_RE.search(report)
    primary_section = report.split("## Full-window result")[1].split("\n## ")[0]
    assert "| 30d" not in primary_section and "Horizon" not in primary_section
    secondary = report.split("## Secondary analyses (reported, not decision-driving)")[1].split("\n## ")[0]
    assert "### Other horizons and raw returns" in secondary and "| 30d" in secondary


def test_nan_guard_catches_unformatted_values(null_world):
    _, res, _ = null_world
    payload = phase1.jsonable({k: v for k, v in res.items() if k != "frames"})
    assert not NAN_RE.search(render_phase1(payload))
    assert NAN_RE.search("mean nan [x]") and NAN_RE.search("NaN%") and not NAN_RE.search("provenance")
    payload["criteria"][0]["threshold"] = float("nan")  # a raw float reaching the page
    assert NAN_RE.search(render_phase1(payload))


def test_jsonable_handles_nan_dates_and_numpy():
    raw = {"a": np.float64("nan"), "b": [np.int64(3), np.bool_(True)], "c": pd.Timestamp("2020-01-02"),
           "d": pd.NaT, "e": (1.5, float("inf")), "f": pd.DataFrame({"x": [1.0, np.nan]}),
           "g": pd.Period("2020-01", "M"), 4: None}
    out = phase1.jsonable(raw)
    assert out == {"a": None, "b": [3, True], "c": "2020-01-02T00:00:00", "d": None, "e": [1.5, None],
                   "f": [{"x": 1.0}, {"x": None}], "g": "2020-01", "4": None}
    json.dumps(out, allow_nan=False)

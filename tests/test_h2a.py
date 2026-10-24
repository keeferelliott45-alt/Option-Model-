import pandas as pd
import pytest

from omodel import config, holdout
from omodel.backtest.synthetic import synthetic_panel
from omodel.experiments import h2a


@pytest.fixture
def synthetic_holdout(monkeypatch, tmp_path):
    panel, _ = synthetic_panel(n_days=560, n_stocks=60, seed=3, start="2024-09-02", shock_prob=0.01,
                               allow_holdout_dates=True)
    monkeypatch.setattr(h2a.panel_mod, "build_panel", lambda **kw: panel)
    monkeypatch.setattr(h2a.universe, "load_membership", lambda: None)
    monkeypatch.setattr(config, "ROOT", config.ROOT)
    config.override(reports_dir=tmp_path / "reports")
    return panel


def test_refuses_without_unlock(synthetic_holdout):
    with pytest.raises(holdout.HoldoutViolation):
        h2a.run()


def test_runs_exactly_once(synthetic_holdout, monkeypatch):
    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    res = h2a.run()
    assert res["verdict"] in ("PASS", "FAIL")
    assert res["events"]["n"] > 0 and res["placebo"]["n"] > 0
    assert (config.settings().reports_dir / "h2a" / "report.md").exists()
    with pytest.raises(RuntimeError, match="exactly one run"):
        h2a.run()


def test_two_sided_exclusion():
    ev = pd.DataFrame({"ticker": ["A", "A"], "pos": [100, 200]})
    plc = pd.DataFrame({"ticker": ["A", "A", "A", "B"], "pos": [90, 150, 210, 100]})
    out = h2a._two_sided_exclusion(plc, ev, 15)
    assert list(zip(out["ticker"], out["pos"])) == [("A", 150), ("B", 100)]

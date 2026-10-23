import pandas as pd
import pytest

from omodel import holdout, params
from omodel.config import settings
from omodel.data import store


def _frame(start, periods):
    idx = pd.bdate_range(start, periods=periods)
    return pd.DataFrame({"x": range(periods)}, index=idx)


def test_guard_truncates_holdout_by_default():
    df = _frame(settings().holdout_start - pd.Timedelta(days=10), 30)
    out = holdout.guard(df)
    assert out.index.max() < settings().holdout_start
    assert len(out) < len(df)


def test_guard_unlock_logs_access(monkeypatch):
    monkeypatch.setenv(holdout.UNLOCK_ENV, holdout.UNLOCK_VALUE)
    df = _frame(settings().holdout_start - pd.Timedelta(days=10), 30)
    assert len(holdout.guard(df, what="test")) == 30
    assert holdout.access_count() == 1


def test_assert_clean_raises():
    with pytest.raises(holdout.HoldoutViolation):
        holdout.assert_clean(pd.DatetimeIndex([settings().holdout_start]))


def test_params_reject_conflicting_reregistration():
    with pytest.raises(ValueError):
        params.register("BETA_WINDOW", 61, "theory", "L2", "conflict")
    assert "BETA_WINDOW" in params.render_markdown()


def test_store_roundtrip():
    df = _frame("2020-01-01", 5)
    store.write(df, "unit", "A/B", meta={"source": "test"})
    pd.testing.assert_frame_equal(store.read("unit", "A/B"), df, check_freq=False)
    assert store.read_meta("unit", "A/B")["source"] == "test"

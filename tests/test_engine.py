import numpy as np
import pandas as pd
import pytest

from omodel.forecast import engine as eng


def test_shape_removes_historical_drift():
    rng = np.random.default_rng(0)
    trending = rng.normal(0.3, 1.0, 2000)  # a strong past uptrend in standardised units
    s = eng.shape(eng.bank(trending, 21, 5000, rng), 21)
    assert abs(s.mean()) < 1e-9
    assert np.mean(s ** 2) == pytest.approx(21)


def test_price_distribution_is_driftless_and_scaled():
    rng = np.random.default_rng(1)
    z = rng.standard_t(5, 3000) / np.sqrt(5 / 3)
    grid, probs = eng.price_distribution(100.0, 0.30, 21, z, n=20000, seed=3)
    lr = np.log(grid)
    m = probs @ (lr - np.log(100))
    sd = np.sqrt(probs @ (lr - np.log(100) - m) ** 2)
    assert abs(m) < 0.003
    assert sd == pytest.approx(0.30 * np.sqrt(21 / 252), rel=0.05)


def test_crash_ratio_moves_only_left_tail_mass():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(3000)
    g0, p0 = eng.price_distribution(100.0, 0.6, 63, z, n=20000, seed=4)
    g1, p1 = eng.price_distribution(100.0, 0.6, 63, z, n=20000, seed=4, crash_ratio=2.0)
    thr = 100 * np.exp(eng.CRASH_LOG_RET)
    assert p1[g1 < thr].sum() == pytest.approx(2 * p0[g0 < thr].sum(), rel=0.05)
    assert p1.sum() == pytest.approx(1.0)


def test_ewma_z_uses_only_past_returns():
    r = np.random.default_rng(5).normal(0, 0.01, 300)
    z1 = eng.ewma_z(r)
    r2 = r.copy()
    r2[200:] *= 10
    z2 = eng.ewma_z(r2)
    assert np.allclose(z1[:200], z2[:200], equal_nan=True)

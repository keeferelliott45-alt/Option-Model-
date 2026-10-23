import numpy as np
import pandas as pd
import pytest

from omodel import config
from omodel.backtest.synthetic import synthetic_panel

N_DAYS, N_STOCKS, N_SECTORS = 1500, 120, 4
H = 15  # PREREG primary window: open[t+1] -> close[t+15]


@pytest.fixture(scope="module")
def base():
    return synthetic_panel(N_DAYS, N_STOCKS, N_SECTORS, seed=7)


def _frames(panel):
    return {k: getattr(panel, k) for k in ("close", "open", "volume", "member", "etf_close", "etf_open", "raw_close")}


def _overnight(panel, etf=False):
    o, c = (panel.etf_open, panel.etf_close) if etf else (panel.open, panel.close)
    return np.log(o / c.shift(1))


def _intraday(panel, etf=False):
    o, c = (panel.etf_open, panel.etf_close) if etf else (panel.open, panel.close)
    return np.log(c / o)


def _positions(panel, truth):
    rows = panel.dates.get_indexer(truth["date"])
    cols = panel.close.columns.get_indexer(truth["ticker"])
    return rows, cols


def _shock_matrix(panel, truth):
    s = np.zeros(panel.close.shape)
    s[_positions(panel, truth)] = truth["shock"].to_numpy()
    return s


def _oracle_legs(panel, truth):
    # Overnight/intraday stock returns minus the factor parts, using the true generating loadings.
    tickers = panel.close.columns
    sec = pd.Series(truth.attrs["sector"]).reindex(tickers).to_numpy()
    b_m = pd.Series(truth.attrs["beta_mkt"]).reindex(tickers).to_numpy()
    b_x = pd.Series(truth.attrs["beta_sec"]).reindex(tickers).to_numpy()
    b_s = pd.Series(truth.attrs["sector_beta"]).reindex(sec).to_numpy()
    out = []
    for leg in (_overnight, _intraday):
        etf = leg(panel, etf=True).to_numpy()
        m = etf[:, [0]]
        x = etf[:, panel.etf_close.columns.get_indexer(sec)] - b_s * m
        out.append(leg(panel).to_numpy() - b_m * m - b_x * x)
    return out[0][1:], out[1][1:]


def _windows(r_on, r_id):
    # Non-overlapping open[t+1] -> close[t+H] windows: intraday legs t+1..t+H and gaps t+2..t+H.
    c_on, c_id = np.cumsum(r_on, axis=0), np.cumsum(r_id, axis=0)
    t = np.arange(0, len(r_id) - H, H)
    w = (c_id[t + H] - c_id[t]) + (c_on[t + H] - c_on[t + 1])
    return w.ravel()


def _mean_se(x):
    x = np.asarray(x).ravel()
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x))


def test_shapes_columns_and_no_nans(base):
    panel, truth = base
    tickers = [f"S{i:03d}" for i in range(N_STOCKS)]
    etfs = [config.MARKET_ETF, *config.SECTOR_ETFS[:N_SECTORS]]
    assert panel.dates.equals(pd.bdate_range("2005-01-03", periods=N_DAYS))
    for name, df in _frames(panel).items():
        assert df.index.equals(panel.dates), name
        assert list(df.columns) == (etfs if name.startswith("etf") else tickers), name
        assert not df.isna().any().any(), name
    assert panel.macro is None
    assert panel.member.dtypes.eq(bool).all()
    pd.testing.assert_frame_equal(panel.raw_close, panel.close)
    assert (panel.volume > 0).all().all() and (panel.close > 0).all().all()
    assert 30 < panel.close.iloc[0].median() < 80 and 80 < panel.etf_close.iloc[0].median() < 125
    assert 0.5e6 < panel.volume.stack().median() < 2e6
    assert panel.member.all().sum() == N_STOCKS - round(0.05 * N_STOCKS)
    assert list(truth.columns) == ["date", "ticker", "shock"]
    assert truth.sort_values(["date", "ticker"]).index.equals(truth.index)
    assert {"sector", "beta_mkt", "beta_mkt_given_sector", "beta_sec", "idio_sd", "sector_beta",
            "crash_dates"} <= set(truth.attrs)


def test_deterministic_given_seed(base):
    panel, truth = base
    again, truth2 = synthetic_panel(N_DAYS, N_STOCKS, N_SECTORS, seed=7)
    for name, df in _frames(panel).items():
        pd.testing.assert_frame_equal(df, getattr(again, name))
    pd.testing.assert_frame_equal(truth, truth2)
    assert truth.attrs == truth2.attrs
    other, _ = synthetic_panel(N_DAYS, N_STOCKS, N_SECTORS, seed=8)
    assert not np.allclose(other.close.to_numpy(), panel.close.to_numpy())


def test_truth_matches_open_gaps_exactly(base):
    panel, truth = base
    no_shock, _ = synthetic_panel(N_DAYS, N_STOCKS, N_SECTORS, seed=7, shock_size=0.0)
    diff = (_overnight(panel) - _overnight(no_shock)).to_numpy()
    np.testing.assert_allclose(np.nan_to_num(diff), _shock_matrix(panel, truth), atol=1e-10)
    np.testing.assert_allclose((_intraday(panel) - _intraday(no_shock)).to_numpy(), 0.0, atol=1e-10)


def test_truth_count_size_and_volume_spikes(base):
    panel, truth = base
    expected = 0.003 * (N_DAYS - 1) * N_STOCKS
    assert abs(len(truth) - expected) < 4 * np.sqrt(expected)
    assert truth["shock"].between(-0.08 * 1.3, -0.08 * 0.7).all()
    assert (truth["date"] > panel.dates[0]).all()
    gaps = _overnight(panel).to_numpy()
    rows, cols = _positions(panel, truth)
    assert abs((gaps[rows, cols] - truth["shock"].to_numpy()).mean()) < 0.005
    log_vol = np.log(panel.volume.to_numpy())
    mask = np.zeros(log_vol.shape, dtype=bool)
    mask[rows, cols] = True
    assert abs(log_vol[mask].mean() - log_vol[~mask].mean() - np.log(3.0)) < 0.15


@pytest.mark.parametrize("shock_size", [-0.08, 0.08])
@pytest.mark.parametrize("reversion_days", [1, 7])
@pytest.mark.parametrize("drift_neutral", [False, True])
def test_recovery_is_exactly_the_reverted_shock_over_the_next_intraday_legs(shock_size, reversion_days,
                                                                            drift_neutral):
    rev, prob = 0.5, 0.01
    kw = dict(n_days=600, n_stocks=40, n_sectors=3, seed=3, shock_prob=prob, shock_size=shock_size,
              reversion_days=reversion_days, drift_neutral=drift_neutral)
    p0, truth = synthetic_panel(reversion=0.0, **kw)
    p1, truth1 = synthetic_panel(reversion=rev, **kw)
    pd.testing.assert_frame_equal(truth, truth1)
    assert len(truth) > 100
    # nothing is recovered in any close -> next-open gap, so a next-open entry can capture all of it
    np.testing.assert_allclose(np.nan_to_num(_overnight(p1) - _overnight(p0)), 0.0, atol=1e-10)
    s = _shock_matrix(p0, truth)
    expected = np.zeros_like(s)
    for lag in range(1, reversion_days + 1):  # starts at t+1 (never day t), lasts R days, sums to -rev*shock
        expected[lag:] += -rev * s[:-lag] / reversion_days
    if drift_neutral:
        n_rec = np.clip(np.arange(len(s)) - 1, 0, reversion_days)
        expected += (rev * prob * shock_size * n_rec / reversion_days)[:, None]
    np.testing.assert_allclose((_intraday(p1) - _intraday(p0)).to_numpy(), expected, atol=1e-10)


def test_reversion_is_capturable_from_next_open():
    kw = dict(n_days=N_DAYS, n_stocks=N_STOCKS, n_sectors=N_SECTORS, seed=11, reversion_days=10)
    p0, truth = synthetic_panel(reversion=0.0, **kw)
    p1, _ = synthetic_panel(reversion=0.5, **kw)
    rows, cols = _positions(p0, truth)
    ok = rows + 10 < N_DAYS
    rows, cols, shocks = rows[ok], cols[ok], truth["shock"].to_numpy()[ok]

    def hold(panel):
        return np.log(panel.close.to_numpy()[rows + 10, cols] / panel.open.to_numpy()[rows + 1, cols])

    assert len(rows) > 300
    assert hold(p1).mean() - hold(p0).mean() == pytest.approx(0.5 * np.abs(shocks).mean(), rel=0.1)


@pytest.fixture(scope="module")
def big_kw():
    return dict(n_days=3000, n_stocks=300, n_sectors=4, seed=5, alpha=0.0)


@pytest.mark.parametrize("reversion", [0.0, 0.5])
def test_drift_neutral_world_has_zero_unconditional_abnormal_drift(big_kw, reversion):
    p, truth = synthetic_panel(reversion=reversion, drift_neutral=True, **big_kw)
    r_on, r_id = _oracle_legs(p, truth)
    for leg in (r_on, r_id):
        mean, se = _mean_se(leg)
        assert abs(mean) < 4 * se
    mean, se = _mean_se(_windows(r_on, r_id))
    assert abs(mean) < 3.5 * se and se < 5e-4


def test_default_world_drifts_with_permanent_shocks(big_kw):
    p, truth = synthetic_panel(**big_kw)
    mean, se = _mean_se(_windows(*_oracle_legs(p, truth)))
    theory = (H - 1) * 0.003 * -0.08
    assert abs(mean - theory) < 3.5 * se and mean < -5 * se


def test_alpha_and_idio_sd_are_recovered():
    p, truth = synthetic_panel(n_days=3000, n_stocks=300, seed=9, shock_prob=0.0)
    r_on, r_id = _oracle_legs(p, truth)
    for leg, share in ((r_on, 0.35), (r_id, 0.65)):
        mean, se = _mean_se(leg)
        assert abs(mean - share * 1e-4) < 4 * se
    mean, se = _mean_se(r_on + r_id)
    assert abs(mean - 1e-4) < 3 * se and se < 0.3e-4
    sd_hat = (r_on + r_id).std(axis=0, ddof=1)
    sd_true = pd.Series(truth.attrs["idio_sd"]).reindex(p.close.columns).to_numpy()
    assert np.corrcoef(sd_hat, sd_true)[0, 1] > 0.98
    np.testing.assert_allclose(sd_hat / sd_true, 1.0, atol=0.06)


def test_estimated_betas_match_configuration(base):
    panel, truth = base
    lr = np.log(panel.close).diff().iloc[1:]
    etf = np.log(panel.etf_close).diff().iloc[1:]
    mkt = etf[config.MARKET_ETF].to_numpy()
    mc = mkt - mkt.mean()
    beta_hat = (lr - lr.mean()).T.to_numpy() @ mc / (mc @ mc)
    beta_true = pd.Series(truth.attrs["beta_mkt"]).reindex(lr.columns).to_numpy()
    assert np.abs(beta_hat - beta_true).mean() < 0.06
    assert np.corrcoef(beta_hat, beta_true)[0, 1] > 0.85

    sec_hat = (etf.iloc[:, 1:] - etf.iloc[:, 1:].mean()).T.to_numpy() @ mc / (mc @ mc)
    sec_true = pd.Series(truth.attrs["sector_beta"]).reindex(etf.columns[1:]).to_numpy()
    np.testing.assert_allclose(sec_hat, sec_true, atol=0.05)
    assert set(truth.attrs["sector"].values()) == set(config.SECTOR_ETFS[:N_SECTORS])


def test_market_plus_sector_model_recovers_conditional_betas(base):
    panel, truth = base
    lr = np.log(panel.close).diff().iloc[1:]
    etf = np.log(panel.etf_close).diff().iloc[1:]
    sector = pd.Series(truth.attrs["sector"])
    hat = pd.DataFrame(index=lr.columns, columns=["mkt", "sec"], dtype=float)
    for s, names in sector.groupby(sector).groups.items():
        x = np.column_stack([np.ones(len(etf)), etf[config.MARKET_ETF], etf[s]])
        hat.loc[names] = np.linalg.lstsq(x, lr[names].to_numpy(), rcond=None)[0][1:].T
    given = pd.Series(truth.attrs["beta_mkt_given_sector"]).reindex(lr.columns)
    total = pd.Series(truth.attrs["beta_mkt"]).reindex(lr.columns)
    b_sec = pd.Series(truth.attrs["beta_sec"]).reindex(lr.columns)
    assert (hat["mkt"] - given).abs().mean() < 0.12 and np.corrcoef(hat["mkt"], given)[0, 1] > 0.9
    assert (hat["sec"] - b_sec).abs().mean() < 0.12
    assert (hat["mkt"] - total).abs().mean() > 0.5  # the total beta is NOT what this regression estimates


def test_market_regimes(base):
    panel, truth = base
    spy = np.log(panel.etf_close[config.MARKET_ETF]).diff().dropna()
    assert 0.010 < spy.std() < 0.016
    crashes = pd.DatetimeIndex(truth.attrs["crash_dates"])
    assert 1 <= len(crashes) <= 3 and (spy.reindex(crashes) < -0.02).all()
    gap = np.log(panel.etf_open[config.MARKET_ETF] / panel.etf_close[config.MARKET_ETF].shift(1)).dropna()
    assert gap.var() / spy.var() == pytest.approx(0.35, abs=0.08)


def test_vol_bursts_follow_crashes(base):
    panel, truth = base
    spy = np.log(panel.etf_close[config.MARKET_ETF]).diff().to_numpy()
    crash_pos = panel.dates.get_indexer(pd.DatetimeIndex(truth.attrs["crash_dates"]))
    burst = np.zeros(len(spy), dtype=bool)
    for c in crash_pos:
        burst[c + 1:c + 21] = True
    calm = ~burst
    calm[crash_pos] = burst[crash_pos] = False
    calm[0] = False
    assert burst.sum() >= 20
    assert spy[burst].std() / spy[calm].std() == pytest.approx(2.5, rel=0.2)
    assert spy[calm].std() == pytest.approx(0.011, rel=0.1)


def test_nonmember_spells_are_single_contiguous_runs(base):
    panel, _ = base
    out = ~panel.member.to_numpy()
    cols = np.flatnonzero(out.any(axis=0))
    assert len(cols) == round(0.05 * N_STOCKS)
    for j in cols:
        pos = np.flatnonzero(out[:, j])
        assert np.all(np.diff(pos) == 1)
        truncated = pos[-1] == N_DAYS - 1
        assert len(pos) <= 252 and (len(pos) >= 60 or truncated)


def test_holdout_dates_are_refused():
    cutoff = config.settings().holdout_start
    n_max = len(pd.bdate_range("2005-01-03", cutoff - pd.Timedelta(days=1)))
    panel, _ = synthetic_panel(n_days=n_max, n_stocks=2, n_sectors=1)
    assert panel.dates[-1] < cutoff
    with pytest.raises(ValueError, match="holdout"):
        synthetic_panel(n_days=n_max + 1, n_stocks=2, n_sectors=1)
    panel, _ = synthetic_panel(n_days=n_max + 1, n_stocks=2, n_sectors=1, allow_holdout_dates=True)
    assert panel.dates[-1] == cutoff
    full, _ = synthetic_panel(n_days=6700, n_stocks=2, n_sectors=1, start=config.settings().dev_start)
    assert full.dates[-1] < cutoff
    with pytest.raises(ValueError, match="holdout"):
        synthetic_panel(n_days=7000, n_stocks=2, n_sectors=1, start=config.settings().dev_start)
    config.override(holdout_start=pd.Timestamp("2006-01-02"))
    with pytest.raises(ValueError, match="holdout"):
        synthetic_panel(n_days=300, n_stocks=2, n_sectors=1)


@pytest.mark.parametrize("bad", [
    dict(n_sectors=0), dict(n_sectors=len(config.SECTOR_ETFS) + 1), dict(n_days=1), dict(n_stocks=0),
    dict(reversion_days=0), dict(shock_prob=1.5), dict(shock_prob=-0.1), dict(shock_prob=float("nan")),
    dict(reversion=float("nan")), dict(shock_size=float("inf")), dict(alpha=float("nan")),
])
def test_invalid_knobs_rejected(bad):
    with pytest.raises(ValueError):
        synthetic_panel(**{"n_days": 50, "n_stocks": 5, "n_sectors": 2, **bad})


@pytest.mark.parametrize("bad", [dict(n_days=50.0), dict(reversion_days=10.0), dict(n_stocks=5.5)])
def test_non_integer_counts_rejected(bad):
    with pytest.raises(TypeError):
        synthetic_panel(**{"n_days": 50, "n_stocks": 5, "n_sectors": 2, **bad})


def test_zero_size_shocks_inject_nothing():
    kw = dict(n_days=200, n_stocks=20, n_sectors=2, seed=4)
    zero_size, truth = synthetic_panel(shock_size=0.0, shock_prob=0.05, **kw)
    no_prob, _ = synthetic_panel(shock_prob=0.0, **kw)
    assert truth.empty
    for name, df in _frames(zero_size).items():
        pd.testing.assert_frame_equal(df, getattr(no_prob, name))


def test_small_and_shockless_panels_work():
    panel, truth = synthetic_panel(n_days=30, n_stocks=3, n_sectors=2, seed=1, shock_prob=0.0, reversion=0.5)
    assert truth.empty and list(truth.columns) == ["date", "ticker", "shock"]
    assert panel.close.shape == (30, 3) and not panel.close.isna().any().any()

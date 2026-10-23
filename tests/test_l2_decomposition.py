import numpy as np
import pandas as pd
import pytest

from omodel.config import MARKET_ETF
from omodel.layers import l2_decomposition as l2
from omodel.types import Decomposition, Panel

ETFS = ["XLE", "XLF", "XLK"]


def _factor_data(T=400, n_per=4, seed=0, noise=0.004, start="2015-01-01"):
    """Market + 3 sector ETFs; stock k loads on sector k % 3 with known betas."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=T)
    mkt = pd.Series(rng.normal(0.0003, 0.01, T), index=idx, name=MARKET_ETF)
    etf = pd.DataFrame({s: 0.8 * mkt.to_numpy() + rng.normal(0, 0.009, T) for s in ETFS}, index=idx)
    N = 3 * n_per
    true_sec = [ETFS[k % 3] for k in range(N)]
    b_m = rng.uniform(0.3, 1.2, N)
    b_s = rng.uniform(0.6, 1.5, N)
    alpha = rng.normal(0, 2e-4, N)
    eps = rng.normal(0, noise, (T, N))
    y = alpha + np.outer(mkt, b_m) + etf[true_sec].to_numpy() * b_s + eps
    stocks = pd.DataFrame(y, index=idx, columns=[f"S{k}" for k in range(N)])
    truth = pd.DataFrame({"sector": true_sec, "b_m": b_m, "b_s": b_s, "alpha": alpha}, index=stocks.columns)
    return stocks, mkt, etf, truth


def _const_sector(stocks, truth):
    return pd.DataFrame(np.tile(truth["sector"].to_numpy(dtype=object), (len(stocks), 1)),
                        index=stocks.index, columns=stocks.columns, dtype=object)


def _assert_same(a: pd.DataFrame, b: pd.DataFrame):
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_log_returns_series_and_frame():
    px = pd.Series([10.0, 11.0, 0.0, 12.0, np.nan, 13.0])
    r = l2.log_returns(px)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])  # log(0) and division by 0 -> NaN, never inf
    assert np.isnan(r.iloc[4]) and np.isnan(r.iloc[5])
    df = l2.log_returns(pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [5.0, 5.0, -1.0]}))
    np.testing.assert_allclose(df["a"].to_numpy()[1:], [np.log(2), np.log(2)])
    assert df["b"].iloc[1] == 0.0 and np.isnan(df["b"].iloc[2])
    assert not np.isinf(df.to_numpy()).any()


def test_known_betas_recovered():
    stocks, mkt, etf, truth = _factor_data(T=400, noise=0.001)
    fm = l2.rolling_factor_model(stocks, mkt, etf, _const_sector(stocks, truth), window=90, min_obs=60)
    assert fm["beta_mkt"].iloc[:60].isna().all().all()
    late = slice(100, None)
    for name, key in [("beta_mkt", "b_m"), ("beta_sec", "b_s")]:
        est = fm[name].iloc[late]
        assert est.notna().all().all()
        np.testing.assert_allclose(est.median().to_numpy(), truth[key].to_numpy(), atol=0.03)
        assert (est - truth[key]).abs().max().max() < 0.08
    np.testing.assert_allclose(fm["alpha"].iloc[late].median().to_numpy(), truth["alpha"].to_numpy(), atol=3e-4)
    np.testing.assert_allclose(fm["sigma"].iloc[late].median().to_numpy(), 0.001, rtol=0.1)
    assert set(fm) == {"alpha", "beta_mkt", "beta_sec", "resid", "sigma", "z", "tot_sigma", "sec_ret"}


def test_future_perturbation_leaves_past_unchanged():
    stocks, mkt, etf, _ = _factor_data(T=420, seed=1)
    t0 = 330

    def run(s, m, e):
        sec = l2.assign_sectors(s, m, e, window=120, min_obs=60)
        return sec, l2.rolling_factor_model(s, m, e, sec, window=90, min_obs=60)

    sec_a, fm_a = run(stocks, mkt, etf)
    rng = np.random.default_rng(99)
    s2, m2, e2 = stocks.copy(), mkt.copy(), etf.copy()
    s2.iloc[t0 + 1:] += rng.normal(0, 0.05, s2.iloc[t0 + 1:].shape)
    m2.iloc[t0 + 1:] = rng.normal(0, 0.03, len(m2) - t0 - 1)
    e2.iloc[t0 + 1:] = rng.normal(0, 0.03, e2.iloc[t0 + 1:].shape)
    sec_b, fm_b = run(s2, m2, e2)
    _assert_same(sec_a.iloc[:t0 + 1], sec_b.iloc[:t0 + 1])
    for k in fm_a:
        _assert_same(fm_a[k].iloc[:t0 + 1], fm_b[k].iloc[:t0 + 1])
    assert not fm_a["z"].iloc[t0 + 1:].equals(fm_b["z"].iloc[t0 + 1:])


def test_current_day_excluded_from_fit():
    stocks, mkt, etf, _ = _factor_data(T=420, seed=2)
    t0, col = 300, "S4"
    sec = l2.assign_sectors(stocks, mkt, etf, window=120, min_obs=60)
    fm_a = l2.rolling_factor_model(stocks, mkt, etf, sec, window=90, min_obs=60)
    s2 = stocks.copy()
    s2.loc[s2.index[t0], col] += 0.10
    sec_b = l2.assign_sectors(s2, mkt, etf, window=120, min_obs=60)
    fm_b = l2.rolling_factor_model(s2, mkt, etf, sec_b, window=90, min_obs=60)
    _assert_same(sec.iloc[:t0 + 1], sec_b.iloc[:t0 + 1])
    for k in ("alpha", "beta_mkt", "beta_sec", "sigma", "tot_sigma", "sec_ret"):
        _assert_same(fm_a[k].iloc[:t0 + 1], fm_b[k].iloc[:t0 + 1])
    assert fm_b["resid"].iloc[t0][col] - fm_a["resid"].iloc[t0][col] == pytest.approx(0.10, abs=1e-12)
    assert fm_b["beta_mkt"].iloc[t0 + 1][col] != fm_a["beta_mkt"].iloc[t0 + 1][col]


def test_z_is_standard_normal_on_pure_noise():
    rng = np.random.default_rng(3)
    T, N = 700, 40
    idx = pd.bdate_range("2010-01-01", periods=T)
    mkt = pd.Series(rng.normal(0, 0.01, T), index=idx)
    etf = pd.DataFrame({s: rng.normal(0, 0.012, T) for s in ETFS}, index=idx)
    stocks = pd.DataFrame(rng.normal(0, 0.02, (T, N)), index=idx, columns=[f"N{k}" for k in range(N)])
    sector = pd.DataFrame(np.array(ETFS * (N // 3 + 1), dtype=object)[:N][None, :].repeat(T, 0),
                          index=idx, columns=stocks.columns)
    z = l2.rolling_factor_model(stocks, mkt, etf, sector, window=90, min_obs=60)["z"].to_numpy().ravel()
    z = z[np.isfinite(z)]
    assert len(z) == (T - 60) * N
    assert abs(z.mean()) < 0.03
    # Out-of-sample prediction error plus estimated sigma inflate the std by ~(1 + 3/90) * t(87) ~ 1.03.
    assert 0.99 < z.std() < 1.07
    assert 0.035 < np.mean(np.abs(z) > 2) < 0.065
    assert np.mean(np.abs(z) > 3) < 0.008


def test_assignment_finds_true_sector_and_switches_after_month_end():
    stocks, mkt, etf, truth = _factor_data(T=330, seed=4, noise=0.004, start="2020-01-01")
    # S0 changes business from XLE to XLF on day 160.
    k = 160
    rng = np.random.default_rng(5)
    switched = mkt * 0.9 + rng.normal(0, 0.003, len(mkt))
    switched.iloc[:k] += etf["XLE"].iloc[:k]
    switched.iloc[k:] += etf["XLF"].iloc[k:]
    stocks["S0"] = switched
    sec = l2.assign_sectors(stocks, mkt, etf, window=60, freq="ME", min_obs=40)
    idx = sec.index
    # Jan 2020 has 23 bdays (< 40 obs); first reassignment with enough data is Fri 2020-02-28.
    assert sec.loc[:"2020-02-28"].isna().all().all()
    assert sec.loc["2020-03-02"].notna().all()
    month_start = np.r_[False, idx.month[1:] != idx.month[:-1]]
    changed = (sec != sec.shift(1)) & ~(sec.isna() & sec.shift(1).isna())
    assert changed.iloc[1:][~month_start[1:]].sum().sum() == 0
    others = [c for c in stocks.columns if c != "S0"]
    assert (sec.loc["2020-03-02":, others] == truth.loc[others, "sector"]).all().all()
    assert sec["S0"].iloc[k - 1] == "XLE"
    assert sec["S0"].iloc[-1] == "XLF"
    first_xlf = sec.index[sec["S0"].eq("XLF").to_numpy().argmax()]
    assert first_xlf > idx[k] and month_start[idx.get_loc(first_xlf)]


def test_assignment_matches_bruteforce_correlation():
    rng = np.random.default_rng(6)
    T, N = 180, 7
    idx = pd.bdate_range("2021-01-01", periods=T)
    mkt = pd.Series(rng.normal(0, 0.01, T), index=idx)
    etf = pd.DataFrame(rng.normal(0, 0.01, (T, 3)), index=idx, columns=ETFS)
    stocks = pd.DataFrame(rng.normal(0, 0.015, (T, N)) + etf.to_numpy()[:, rng.integers(0, 3, N)] * 0.5,
                          index=idx, columns=[f"S{k}" for k in range(N)])
    stocks = stocks.mask(rng.random((T, N)) < 0.1)
    etf = etf.mask(rng.random((T, 3)) < 0.05)
    stocks["S6"] = np.nan
    stocks.iloc[100:, 6] = rng.normal(0, 0.01, T - 100)
    window, min_obs = 50, 30
    sec = l2.assign_sectors(stocks, mkt, etf, window=window, freq="ME", min_obs=min_obs)
    ends = pd.Series(np.arange(T), index=idx).resample("ME").last().to_numpy()
    for p in ends[:-1]:
        win = slice(max(0, p - window + 1), p + 1)
        xs = stocks.iloc[win].sub(mkt.iloc[win], axis=0)
        es = etf.iloc[win].sub(mkt.iloc[win], axis=0)
        for c in stocks.columns:
            corrs = {s: xs[c].corr(es[s]) for s in ETFS
                     if (xs[c].notna() & es[s].notna()).sum() >= min_obs}
            got = sec[c].iloc[p + 1]
            assert (got == max(corrs, key=corrs.get)) if corrs else pd.isna(got)
        assert sec.iloc[p + 1].equals(sec.iloc[min(T - 1, ends[ends > p][0])])


def test_batched_fit_matches_lstsq():
    rng = np.random.default_rng(7)
    T, N, window, min_obs = 300, 8, 50, 30
    idx = pd.bdate_range("2012-01-02", periods=T)
    mkt = pd.Series(rng.normal(0, 0.01, T), index=idx).mask(rng.random(T) < 0.02)
    etf = pd.DataFrame(rng.normal(0, 0.01, (T, 3)), index=idx, columns=ETFS).mask(rng.random((T, 3)) < 0.03)
    y = rng.normal(0, 0.015, (T, N)) + 0.7 * mkt.fillna(0).to_numpy()[:, None]
    y[:, 0] += 0.05  # huge drift stresses cancellation in raw-sum moments
    stocks = pd.DataFrame(y, index=idx, columns=[f"S{k}" for k in range(N)]).mask(rng.random((T, N)) < 0.08)
    # Monthly-changing random assignment, some NaN: regressor history must be the current sector's.
    months = idx.to_period("M")
    pick = {m: rng.choice(np.array(ETFS + [np.nan], dtype=object), N) for m in months.unique()}
    sector = pd.DataFrame(np.stack([pick[m] for m in months]), index=idx, columns=stocks.columns, dtype=object)
    fm = l2.rolling_factor_model(stocks, mkt, etf, sector, window=window, min_obs=min_obs)
    checked = 0
    for t in range(T):
        for i, c in enumerate(stocks.columns):
            s = sector.iat[t, i]
            win = slice(max(0, t - window), t)
            r = stocks[c].iloc[win]
            tot_ok = r.notna().sum() >= min_obs
            assert fm["tot_sigma"].iat[t, i] == pytest.approx(r.std(), rel=1e-9) if tot_ok \
                else np.isnan(fm["tot_sigma"].iat[t, i])
            if pd.isna(s):
                assert np.isnan(fm["beta_mkt"].iat[t, i]) and np.isnan(fm["sec_ret"].iat[t, i])
                continue
            assert fm["sec_ret"].iat[t, i] == etf[s].iat[t] or np.isnan(etf[s].iat[t])
            X = np.column_stack([np.ones(t - win.start), mkt.iloc[win], etf[s].iloc[win]])
            ok = np.isfinite(X).all(1) & r.notna().to_numpy()
            if ok.sum() < min_obs:
                assert np.isnan(fm["beta_mkt"].iat[t, i])
                continue
            coef, ssr, *_ = np.linalg.lstsq(X[ok], r.to_numpy()[ok], rcond=None)
            np.testing.assert_allclose([fm[k].iat[t, i] for k in ("alpha", "beta_mkt", "beta_sec")], coef,
                                       rtol=1e-8, atol=1e-11)
            assert fm["sigma"].iat[t, i] == pytest.approx(np.sqrt(ssr[0] / (ok.sum() - 3)), rel=1e-8)
            pred = coef @ [1.0, mkt.iat[t], etf[s].iat[t]]
            expect = stocks[c].iat[t] - pred
            if np.isfinite(expect):
                assert fm["resid"].iat[t, i] == pytest.approx(expect, rel=1e-8, abs=1e-12)
                assert fm["z"].iat[t, i] == pytest.approx(expect / fm["sigma"].iat[t, i], rel=1e-12)
            else:
                assert np.isnan(fm["resid"].iat[t, i])
            checked += 1
    assert checked > 1000


def test_singular_and_short_windows_give_nan():
    stocks, mkt, etf, truth = _factor_data(T=200, seed=8)
    etf = etf.copy()
    etf["XLE"] = mkt.to_numpy()          # sector ETF identical to market: collinear
    etf["XLF"] = 0.0                     # constant regressor
    fm = l2.rolling_factor_model(stocks, mkt, etf, _const_sector(stocks, truth), window=90, min_obs=60)
    for s in ("XLE", "XLF"):
        cols = truth.index[truth["sector"] == s]
        for k in ("alpha", "beta_mkt", "beta_sec", "sigma", "z", "resid"):
            assert fm[k][cols].isna().all().all(), (s, k)
        assert fm["sec_ret"][cols].iloc[1:].notna().all().all()
    k_cols = truth.index[truth["sector"] == "XLK"]
    assert fm["beta_sec"][k_cols].iloc[60:].notna().all().all()
    assert fm["beta_sec"][k_cols].iloc[:60].isna().all().all()  # t=60 is the first row with 60 prior obs
    # An exact fit has zero residual variance: z is undefined, not huge.
    exact = pd.DataFrame({"E": 0.5 * mkt + 1.2 * etf["XLK"]})
    fe = l2.rolling_factor_model(exact, mkt, etf, pd.DataFrame({"E": "XLK"}, index=mkt.index, dtype=object),
                                 window=90, min_obs=60)
    assert fe["beta_mkt"]["E"].iloc[100:].sub(0.5).abs().max() < 1e-8
    assert fe["z"]["E"].isna().all()


def test_near_collinear_sector_gives_nan():
    stocks, mkt, etf, truth = _factor_data(T=200, seed=15)
    rng = np.random.default_rng(16)
    etf = etf.copy()
    etf["XLE"] = mkt.to_numpy() + rng.normal(0, 1e-7, len(mkt))  # ill-conditioned, not exactly singular
    etf["XLF"] = mkt.to_numpy() + rng.normal(0, 1e-3, len(mkt))  # highly correlated but identifiable
    fm = l2.rolling_factor_model(stocks, mkt, etf, _const_sector(stocks, truth), window=90, min_obs=60)
    bad = truth.index[truth["sector"] == "XLE"]
    good = truth.index[truth["sector"] == "XLF"]
    for k in ("alpha", "beta_mkt", "beta_sec", "sigma", "z"):
        assert fm[k][bad].isna().all().all(), k
        assert fm[k][good].iloc[60:].notna().all().all(), k


def test_string_dtype_sector_matches_object():
    stocks, mkt, etf, truth = _factor_data(T=200, seed=17)
    sec = l2.assign_sectors(stocks, mkt, etf, window=120, min_obs=60)
    assert sec.isna().any().any() and sec.notna().any().any()
    ssec = sec.astype("string")
    fm_o = l2.rolling_factor_model(stocks, mkt, etf, sec, window=90, min_obs=60)
    fm_s = l2.rolling_factor_model(stocks, mkt, etf, ssec, window=90, min_obs=60)
    for k in fm_o:
        _assert_same(fm_o[k], fm_s[k])
    member = pd.DataFrame(True, index=stocks.index, columns=stocks.columns)
    dec_o, dec_s = _dec_from(fm_o["z"], sec), _dec_from(fm_o["z"], ssec)
    _assert_same(l2.peer_z(dec_o, member), l2.peer_z(dec_s, member))


def test_volume_z_excludes_current_day():
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2019-01-01", periods=60)
    vol = pd.DataFrame({"A": rng.lognormal(14, 0.4, 60), "B": rng.lognormal(12, 0.3, 60)}, index=idx)
    vol.iloc[25, 1] = 0.0
    vz = l2.volume_z(vol, window=20)
    lv = np.log(vol["A"])
    t = 40
    prior = lv.iloc[t - 20:t]
    assert vz["A"].iat[t] == pytest.approx((lv.iat[t] - prior.mean()) / prior.std())
    assert np.isnan(vz["A"].iat[9]) and np.isfinite(vz["A"].iat[10])  # needs window // 2 prior obs
    assert np.isnan(vz["B"].iat[25])
    lb = np.log(vol["B"].where(vol["B"] > 0))
    priorb = lb.iloc[26:46]
    assert vz["B"].iat[46] == pytest.approx((lb.iat[46] - priorb.mean()) / priorb.std())
    v2 = vol.copy()
    v2.iloc[t, 0] *= 50
    vz2 = l2.volume_z(v2, window=20)
    _assert_same(vz.iloc[:t], vz2.iloc[:t])
    assert vz2["A"].iat[t] - vz["A"].iat[t] == pytest.approx(np.log(50) / prior.std())


def _dec_from(z: pd.DataFrame, sector: pd.DataFrame) -> Decomposition:
    blank = z * np.nan
    return Decomposition(stock_ret=blank, mkt_ret=pd.Series(np.nan, index=z.index), sector=sector, sec_ret=blank,
                         alpha=blank, beta_mkt=blank, beta_sec=blank, resid=blank, sigma=blank, z=z,
                         tot_sigma=blank, vol_z=blank)


def test_peer_z_leave_one_out():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    cols = list("ABCDE")
    z = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, np.nan, 2.0, 1.0, 0.0]], index=idx, columns=cols)
    sector = pd.DataFrame([["XLK", "XLK", "XLK", "XLK", "XLF"], ["XLK", "XLK", "XLK", "XLF", "XLF"]],
                          index=idx, columns=cols, dtype=object)
    member = pd.DataFrame(True, index=idx, columns=cols)
    member.loc[idx[0], "D"] = False
    dec = _dec_from(z, sector)
    p2 = l2.peer_z(dec, member, min_peers=2)
    expect2 = [[2.5, 2.0, 1.5, 2.0, np.nan], [np.nan, 0.5, np.nan, np.nan, np.nan]]
    np.testing.assert_allclose(p2.to_numpy(), expect2)
    p1 = l2.peer_z(dec, member, min_peers=1)
    expect1 = [[2.5, 2.0, 1.5, 2.0, np.nan], [2.0, 0.5, -1.0, 0.0, 1.0]]
    np.testing.assert_allclose(p1.to_numpy(), expect1)
    # Missing tickers in member count as non-members; unassigned sector gives NaN.
    sector2 = sector.copy()
    sector2.loc[idx[1], "A"] = np.nan
    p3 = l2.peer_z(_dec_from(z, sector2), member.drop(columns="C"), min_peers=1)
    assert np.isnan(p3.loc[idx[1], "A"]) and np.isnan(p3.loc[idx[1], "B"])
    assert p3.loc[idx[0], "A"] == pytest.approx(2.0)


_DEC_FRAMES = ("stock_ret", "sector", "sec_ret", "alpha", "beta_mkt", "beta_sec", "resid", "sigma", "z",
               "tot_sigma", "vol_z")


def _panel(T=500, seed=10):
    stocks, mkt, etf, truth = _factor_data(T=T, seed=seed)
    idx = stocks.index
    rng = np.random.default_rng(seed + 1)
    etf_px = np.exp(pd.concat([mkt, etf], axis=1).cumsum()) * 100
    etf_px.iloc[:150, etf_px.columns.get_loc("XLK")] = np.nan  # late inception
    close = np.exp(stocks.cumsum()) * 50
    etf_px["QQQ"] = close["S0"] * 3  # not a sector ETF: would win S0's assignment if decompose let it in
    # open, etf_open, raw_close and volume all differ from close, so any source mixup in decompose shows.
    def noisy(df: pd.DataFrame) -> pd.DataFrame:
        return df * rng.lognormal(0, 0.01, df.shape)

    volume = pd.DataFrame(rng.lognormal(13, 0.3, close.shape), index=idx, columns=close.columns)
    panel = Panel(close=close, open=noisy(close), volume=volume,
                  member=pd.DataFrame(True, index=idx, columns=close.columns),
                  etf_close=etf_px, etf_open=noisy(etf_px), raw_close=noisy(close) * 1.7)
    return panel, truth


def _cut(panel: Panel, c: int) -> Panel:
    return Panel(close=panel.close.iloc[:c], open=panel.open.iloc[:c], volume=panel.volume.iloc[:c],
                 member=panel.member.iloc[:c], etf_close=panel.etf_close.iloc[:c], etf_open=panel.etf_open.iloc[:c],
                 raw_close=panel.raw_close.iloc[:c])


def test_decompose_end_to_end():
    panel, truth = _panel()
    close, etf_px, idx = panel.close, panel.etf_close, panel.close.index
    dec = l2.decompose(panel)
    for f in _DEC_FRAMES:
        frame = getattr(dec, f)
        assert frame.shape == close.shape and frame.index.equals(idx), f
    assert dec.sector.dtypes.eq(object).all()
    assert set(dec.sector.stack().unique()) == set(ETFS)
    # Exact wiring: adjusted close (not open/raw), etf_close (not etf_open), volume, registry defaults.
    stock_ret = l2.log_returns(close)
    mkt_ret = l2.log_returns(etf_px[MARKET_ETF])
    etf_ret = l2.log_returns(etf_px[ETFS])
    sector = l2.assign_sectors(stock_ret, mkt_ret, etf_ret)
    fm = l2.rolling_factor_model(stock_ret, mkt_ret, etf_ret, sector)
    _assert_same(dec.stock_ret, stock_ret)
    pd.testing.assert_series_equal(dec.mkt_ret, mkt_ret, check_exact=True)
    _assert_same(dec.sector, sector)
    for k in fm:
        _assert_same(getattr(dec, k), fm[k])
    _assert_same(dec.vol_z, l2.volume_z(panel.volume))
    assert (dec.sector.iloc[-1] == truth["sector"]).all()
    late = dec.sector.iloc[-1]
    rs = l2.log_returns(etf_px[late.to_numpy()]).iloc[-1].to_numpy()
    np.testing.assert_allclose(dec.sec_ret.iloc[-1].to_numpy(), rs)
    assert dec.z.iloc[-50:].notna().all().all()
    assert dec.vol_z.iloc[-50:].notna().all().all()
    pz = l2.peer_z(dec, panel.member)
    assert pz.iloc[-50:].notna().all().all()  # 4 stocks per sector -> 3 peers each


def test_decompose_truncated_panel_matches_full_run():
    panel, _ = _panel(T=460, seed=12)
    idx = panel.close.index
    full = l2.decompose(panel)
    pz_full = l2.peer_z(full, panel.member)
    ends = pd.Series(np.arange(len(idx)), index=idx).resample("ME").last().to_numpy()
    ends = ends[(ends > 280) & (ends < len(idx) - 2)][:3]
    rng = np.random.default_rng(13)
    cuts = sorted({*(ends + 1), *(ends + 2), *rng.integers(200, len(idx), 4)})  # month-end, month-end+1, random
    for c in cuts:
        dec = l2.decompose(_cut(panel, c))
        for f in _DEC_FRAMES:
            _assert_same(getattr(dec, f), getattr(full, f).iloc[:c])
        pd.testing.assert_series_equal(dec.mkt_ret, full.mkt_ret.iloc[:c], check_exact=True)
        _assert_same(l2.peer_z(dec, panel.member.iloc[:c]), pz_full.iloc[:c])


def test_decompose_aligns_etf_prices_before_differencing():
    panel, _ = _panel(T=300, seed=14)
    gap = 200
    holed = Panel(close=panel.close, open=panel.open, volume=panel.volume, member=panel.member,
                  etf_close=panel.etf_close.drop(index=panel.close.index[gap]), etf_open=panel.etf_open)
    dec = l2.decompose(holed)
    full = l2.log_returns(panel.etf_close[MARKET_ETF])
    # The return into and out of the missing date is unknown; it must not become a 2-day return.
    assert dec.mkt_ret.iloc[gap - 1:gap + 3].isna().tolist() == [False, True, True, False]
    pd.testing.assert_series_equal(dec.mkt_ret.drop(index=dec.mkt_ret.index[[gap, gap + 1]]),
                                   full.drop(index=full.index[[gap, gap + 1]]), check_exact=True)
    assert dec.sec_ret.iloc[gap].isna().all() and dec.sec_ret.iloc[gap + 1].isna().all()

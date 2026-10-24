# Stage 1 pre-registration — standalone factor tests

Dated 2026-10-24. Binding from the commit that adds this file; no Stage 1 result exists at that commit.
Design: `docs/MODEL_V3_DESIGN.md`. Each factor's direction is fixed below from published research, not from data.

## Setup

- Universe: point-in-time S&P 500 members on each month-end formation date, raw price ≥ $5.
- Formation: last trading day of each month t. Factor values use only data public by the close of t
  (SEC values from their filing date + 1 day; FINRA short interest from settlement date + 12 days).
- Targets (skip one day to avoid bid-ask bounce; start = close t+1):
  - `ret21`: log return t+1 → t+22 minus SPY's over the same window.
  - `vol21`: annualised realized volatility of daily log returns t+2 … t+22.
  - `crash63`: 1 if the SPY-adjusted log return t+1 → t+64 < −25%, else 0.
- Statistic: monthly Spearman rank correlation (IC) between factor and target across stocks (≥ 50 stocks);
  mean IC with Newey-West t (lags: 2 for 21-day targets, 4 for crash63). Reported with the top-minus-bottom
  quintile spread and n.
- Windows: development = formations 2000-01 → 2025-08 (all targets end before 2025-10-23); holdout =
  formations 2025-10 → 2026-07 (targets end ≤ 2026-10-22). The holdout year was used only for the H2a event
  test; none of these variables were measured on it. FINRA factors: development from their first available
  date. Sub-periods: 2000–2008, 2009–2016, 2017–2025 (FINRA: 2018–2021, 2022–2025).

## Pass rule (all, per factor)

1. Mean IC has the predicted sign with Newey-West |t| ≥ 2 on development, **and** survives
   Benjamini–Hochberg (q = 0.10) across all factors in this file.
2. Same sign in at least 2 of 3 sub-periods (FINRA: both).
3. Same sign on the holdout (direction only; ~10 formations).

Passing factors may tilt the forecast's drift (return targets) or scale its volatility (vol targets) in the
forecast engine, with their measured out-of-sample size. Failing factors may only veto or widen uncertainty.

## Factors, targets and predicted signs

| # | Factor | Definition | Target | Sign | Source |
|---|---|---|---|---|---|
| 1 | mom_12_1 | log return t−252 → t−21 | ret21 | + | Jegadeesh & Titman 1993 |
| 2 | rev_1m | log return t−21 → t | ret21 | − | Jegadeesh 1990 |
| 3 | high52 | price / 252-day high | ret21 | + | George & Hwang 2004 |
| 4 | resid_mom | sum of Layer 2 residuals t−252 → t−21 | ret21 | + | Blitz et al. 2011 |
| 5 | idio_vol | Layer 2 residual σ (90d) | ret21 | − | Ang et al. 2006 |
| 6 | idio_vol_v | same | vol21 | + | volatility persistence |
| 7 | max_ret | max daily return over t−21 → t | ret21 | − | Bali et al. 2011 |
| 8 | beta | Layer 2 market beta | ret21 | − | Frazzini & Pedersen 2014 (BAB) |
| 9 | pead | Layer 2 residual on the last earnings day (8-K 2.02) within 63 days | ret21 | + | Bernard & Thomas 1989 |
| 10 | gross_prof | gross profit TTM / total assets | ret21 | + | Novy-Marx 2013 |
| 11 | accruals | (net income − operating cash flow) TTM / assets | ret21 | − | Sloan 1996 |
| 12 | asset_growth | total assets YoY growth | ret21 | − | Cooper et al. 2008 |
| 13 | dilution | shares outstanding YoY growth | ret21 | − | Pontiff & Woodgate 2008 |
| 14 | earn_yield | net income TTM / market cap | ret21 | + | Basu 1977 |
| 15 | sales_yield | revenue TTM / market cap | ret21 | + | Barbee et al. 1996 |
| 16 | f_score | Piotroski F-score (9 binary signals, as available) | ret21 | + | Piotroski 2000 |
| 17 | leverage_crash | (debt − cash) / assets | crash63 | + | distress literature |
| 18 | zscore_crash | Altman Z (as available) | crash63 | − | Altman 1968 |
| 19 | days_to_cover | FINRA short position / average daily volume | ret21 | − | Hong et al. 2016 |
| 20 | short_pct | FINRA short position / shares outstanding | ret21 | − | Asquith et al. 2005 |
| 21 | short_pct_crash | same | crash63 | + | Asquith et al. 2005 |

Macro (time series, not cross-section): monthly predictive regressions of SPY next-month realized vol on
VIX level (+), VIX/VIX3M (+), 20-day change in log(HYG/LQD) (−), with Newey-West t; same pass rule
(sub-periods and holdout sign), BH across these three.

## Not tested in Stage 1 (no free history)

Option-implied factors (put/call ratios, skew, IV rank, term structure), insider transactions (Form 4
parsing at scale exceeds SEC fair-access budget), Wikipedia attention, news/LLM tone. These are tested
forward on our own snapshots or deferred.

# Stage 2 pre-registration — forecast engine

Dated 2026-10-24. Binding from the commit adding this file; no Stage 2 result exists then.
Inputs follow Stage 1: no return factor passed, so **drift = 0**. Passing factors shape volatility and tails.

## Engine

1. **Volatility (HAR, pooled):** log realized vol over the next 21 trading days (t+1 → t+21) regressed on
   log RV5, log RV22, log RV66 (stock's own), log VIX and VIX/VIX3M (Stage 1 passes). Pooled OLS across stocks,
   refit every January on an expanding window of formations whose targets ended before the refit date.
   Forecast variance includes the regression's residual-variance (Jensen) correction.
2. **Shape (filtered historical simulation):** the stock's past 3 years of daily returns standardised by an
   EWMA volatility (λ = 0.94, past-only), block-bootstrapped in 5-day blocks to the horizon, scaled by the
   forecast. 1,000 draws for the calibration tests.
3. **Earnings jump:** if an estimated earnings day (last 8-K 2.02 + 91 days, ±0) falls inside the window, one
   day of the path is drawn from the stock's own past earnings-day standardised moves.
4. **Short-interest crash tail (Stage 1 pass):** a logistic model P(SPY-adjusted 63-day log return < −25%)
   on log short_pct and log forecast vol, fit walk-forward; tested here, applied only if it passes.

## Tests (development formations 2003-01 → 2025-08, monthly; holdout 2025-10 → 2026-07, direction only)

| Id | Test | Pass |
|---|---|---|
| A | HAR vs naive (forecast = RV22) on QLIKE loss | mean loss difference < 0, Diebold–Mariano Newey-West t ≤ −2 (lag 2), same sign in ≥ 2 of 3 sub-periods and on holdout |
| B | 21-day distribution calibration: share of realized raw 21-day log returns (what options pay on) inside the central 50% and 90% intervals | both within ±3 percentage points of nominal on development; holdout reported |
| C | Earnings windows only: same coverage test | both within ±4 pp |
| D | Crash tail: logistic with short_pct vs without, out-of-sample log score (FINRA era 2018 →) | improvement > 0 with DM t ≥ 2 |

Sub-periods: 2003–2008, 2009–2016, 2017–2025. Universe and price filter as Stage 1. Any Stage 2 component
that fails is replaced by its simpler alternative (A fails → RV22; C fails → no earnings jump; D fails → no
tail adjustment) and Stage 3 (the forward option test) uses whatever survives.

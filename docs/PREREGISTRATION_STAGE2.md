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

## Amendment 1 — 2026-10-24, before any HAR-X result

Reason: the first live scan showed HAR projecting just-finished earnings jumps forward (e.g. RV5 81% vs implied
28% for a stock the day after earnings), which makes options look cheap exactly when implied vol collapses.
The live scanner never holds across earnings, so the forecast should describe non-earnings volatility.

- **HAR-X:** same model, but daily returns on 8-K 2.02 effective days and the following day are set to missing
  before computing RV5/RV22/RV66 (the target is unchanged).
- **Test A2:** HAR-X vs HAR on QLIKE, evaluated only on stock-months whose 21-day target window contains no
  estimated earnings day (the scanner's use case). Same pass rule as A. Pass → the live engine uses HAR-X;
  fail → HAR stays.
- **Live guard (convention, not tested):** a contract is not shown when the engine's volatility for its
  stock is more than 1.5x or less than 0.5x the market's 30-day implied vol; such disagreement is treated as
  missing information, not as edge.

## Amendment 2 — 2026-10-24, before any HAR-N result

Reason: HAR is fit on all stock-months, so its level includes the average earnings jump; the scanner only
uses windows without earnings, where it therefore overstates volatility and makes options look cheap
(first live scan: PLTR 7-day forecast 59% vs 46% implied for that expiry). Test C already showed the mirror
image (too narrow in earnings windows).

- **HAR-N:** HAR fit only on stock-months whose 21-day target window contains no earnings effective day.
- **Test B2:** calibration (central 50% and 90% coverage, ±3 pp) on no-earnings windows only, development
  period, for HAR-N; HAR's coverage on the same windows is reported for comparison. Pass → live engine uses
  HAR-N for the (no-earnings) windows it scans; fail → no live picks are shown until a calibrated engine exists.
- The live disagreement guard compares against the at-the-money implied vol of the same expiry instead of the
  30-day index (convention).

## Amendment 3 — 2026-10-24, before any tail result

Reason: B/B2 validate only the central 50%/90%; far out-of-the-money option values depend on how far the
price goes beyond that (first live scan: a 7-day PLTR call whose model in-the-money probability matched the
market's delta, 6.8%, but whose expected payoff was ~4x the price because of the bootstrapped right tail).

- **Test E (tail payoff calibration):** on no-earnings windows (HAR-N, development), for each stock-month
  compute the model's expected excess of the 21-day log return beyond its own 95th percentile,
  E[(R − q95)+], and below its 5th percentile, E[(q5 − R)+], and the realized averages of the same
  quantities. Pass: realized/model ratio within [0.8, 1.25] for both tails.
- **Consequence:** if E fails, contracts in the 0.05–0.20 delta bucket (tail-priced) are not shown, and the
  report says so; other buckets are unaffected.

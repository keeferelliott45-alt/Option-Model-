# Phase 1 pre-registration

Written and committed **before** any Phase 1 result was computed. Changing anything below after seeing
results turns a test into a fit; if it changes, the commit history shows it.

## Hypothesis (H1)

Among point-in-time S&P 500 members, after a day whose standardized residual is z ≤ −3 — residual from
a market + sector-ETF model with betas fit on the prior 90 trading days only — the **abnormal** log return
from the **t+1 open** to the **t+15 close**, net of 10 bps round-trip cost, has a positive mean.

"Abnormal" = stock return minus (β_mkt · SPY return + β_sec · sector-ETF return) over the same
open-to-close window, using the betas fit as of the event date. Alpha is excluded.

## Data and sample

- Development window: 2000-01-03 → 2025-10-22. Events whose 15-day exit falls on/after the holdout start
  (2025-10-23) are dropped (embargo).
- Holdout: 2025-10-23 onward. Not loaded during development; code-enforced in `omodel.holdout`.
- Universe: tickers that were S&P 500 members **on the event date** (fja05680/sp500 membership history).
- Known bias: free price data omits most delisted names. Coverage of member-days is reported by year.
  This bias favours H1 (missing names skew toward drops that never recovered), so a pass is an upper
  bound and a fail is robust.

## Pass criteria — all must hold

1. **Event level:** mean net abnormal 15d return > 0, with the 95% date-clustered bootstrap CI excluding 0.
2. **Walk-forward:** threshold chosen from {−2.5, −3, −4} on each 2-year fit window only (purged),
   applied to the following 6-month test window; the concatenated out-of-sample mean is > 0 with its
   95% CI excluding 0.
3. **Placebo:** the same tickers on random non-event member days give a mean whose 95% CI includes 0.
   A placebo "edge" means a pipeline bug, and the run is invalid until it is found.
4. **Stability:** the mean has the same sign in at least 2 of 3 sub-periods
   (2000–2008, 2009–2016, 2017–2025).
5. **Plausibility:** calendar-time portfolio Sharpe below 2.0. Above that is treated as a bug hunt, not a result.

If any criterion fails: **stop and report**; later phases are not built on a false premise.

## Amendment 1 — 2026-10-23, before any real-data result was computed

Original text SHA-256: `a1495fce08f51ba18f183294df8b653f933e6c059ddd908cd6b55e70459ef359`.

Reason: free price data keeps only survivors, and survivors drift upward. That drift lifts event and
placebo returns alike, so criterion 1 alone could pass on survivor drift with no overreaction at all.
Criterion 1 is therefore tightened (never loosened):

- **1b. Excess over placebo:** the mean net abnormal 15d return of events minus that of the placebo
  (same tickers, random non-event member days), resampling month clusters jointly, is > 0 with its 95% CI
  excluding 0. Criterion 1 passes only if both 1 and 1b hold.

Nothing else changes.

## Secondary analyses (reported, not decision-driving)

- Horizons 1, 5, 30 days; raw (non-abnormal) returns.
- Entry at the event-day close instead of next open, to show how much apparent edge is untradeable bounce.
- Up-move symmetry (z ≥ +3).
- Confirming signals tested **one at a time**, with Benjamini–Hochberg correction across the battery:
  volume z-score, absence of a company 8-K filing (news-vs-price mismatch proxy), event-day sector-ETF
  move size (the thesis's core bet: correlated shocks overreact more), and peer co-movement.
- Every statistic is reported with its sample size; anything under 30 events is labelled low-confidence.

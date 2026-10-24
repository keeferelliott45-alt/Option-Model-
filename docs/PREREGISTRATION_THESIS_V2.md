# Thesis v2 pre-registration — "Trade the volatility, not the bounce"

**Dated 2026-10-24. Binding from the commit that adds this file.** No v2 result exists at that commit. v1
(`PREREGISTRATION.md`) is closed: FAIL. Changing anything below after a v2 result exists turns a test into a fit.

## Prior knowledge (disclosed)

v1 (development data 2000-01-03 → 2025-10-22, survivor-biased) found large residual drops (z ≤ −3) followed
by **continued** underperformance: −0.39% net abnormal over 15 days, walk-forward OOS −0.42%. H2a below was
suggested by that result, so it is tested only on data v1 never touched.

## Thesis

After a large company-specific shock, option prices overreact more than the stock: implied volatility (IV)
overshoots the volatility that follows, while the stock drifts on in the shock's direction. A defined-risk
**bear call credit spread** after a large drop sells that excess IV and leans with the drift. The system
outputs a ranked shortlist; a human places or rejects every trade; naked short options are never surfaced.

## Event definition (unchanged from v1)

Point-in-time S&P 500 member; standardized residual z ≤ −3 from the 90-day market + sector-ETF model fit
through t−1; raw price ≥ $5; same-ticker cooldown 15 trading days.

## H2a — drift continues (equity; one-shot replication on the v1 holdout)

- Data: 2025-10-23 → 2026-10-22, the holdout v1 never loaded. Unlocked exactly once for this test.
- Metric: net abnormal log return, t+1 open → t+15 close, 10 bps round trip, alpha excluded (as v1).
- Placebo: same tickers, random member days; excluded if within 15 trading days of a real event
  **on either side, including any day whose 15-day forward window contains a real event** (fixes the v1
  placebo contamination).
- **Pass:** event mean < 0 **and** event − placebo difference < 0. The 95% month-cluster CIs are reported,
  but with ~12 monthly clusters they are flagged low-confidence, so this is a sign replication, not a
  precise estimate. Fail → the drift leg is dropped from the thesis (H2b/H2c still run, direction-neutral).

## H2b — IV overshoots after shocks (forward test on own snapshots)

- Data: CBOE end-of-day chain snapshots collected from 2026-10-26 onward (`omodel snapshot-options`).
- For each event at close t with a snapshot at t: ATM IV of the listed expiry nearest 30 calendar days
  (21–45 DTE; strike nearest spot, call/put IV averaged). Realized vol = annualised std of daily log
  returns t+1 → expiry close.
- Control: same tickers, random non-event member days with snapshots (same exclusion rule as H2a).
- Metric: IV − realized vol (vol points).
- **Pass:** event mean > control mean, month-cluster bootstrap 95% CI on the difference excluding 0.

## H2c — the trade (forward test, executable prices)

- Entry: the t+1 end-of-day snapshot. One structure per event, fixed rule, no EV-based selection:
  bear call credit spread, expiry 21–45 DTE nearest 30 DTE, short call = listed call with delta closest
  to 0.30, long call = next listed strike above. Both legs must pass the liquidity filter
  (OI ≥ 100, volume ≥ 10, spread ≤ 10% of mid); otherwise no trade.
- Fill: sell at bid, buy at ask, $0.65/contract on open and close. Held to expiry; settled at intrinsic
  value from the expiry-day close. No early management.
- Metric: P&L / max loss per trade.
- **Pass (all):** (1) mean > 0 with month-cluster 95% CI excluding 0; (2) mean exceeds the same rule on the
  H2b control days (event − control > 0); (3) worst month loss reported and ≤ 3× the mean monthly gain;
  (4) results shown by delta bucket and by VIX regime (VIX < 20 / ≥ 20), never pooled alone.
- The Layer 5 EV under a lognormal at the pre-event 90-day realized vol is computed and reported for each
  trade, but plays no part in selection or pass/fail in this test.

## Evaluation timing and sample rules

- H2a: runs once, immediately after this file is committed.
- H2b/H2c: first and only evaluation on **2027-04-30** (≥ 6 months of snapshots), using every event whose
  expiry closed by then. If fewer than 100 trades or fewer than 6 months exist, report "insufficient" and
  set the next and only evaluation date in a dated amendment written before looking at any H2b/H2c number.
- Sharpe > 2 or win rate > 90% is treated as a probable bug (stale quotes, fill assumptions) until proven
  otherwise.
- Every parameter is in the registry (`docs/PARAMETERS.md`); all of the above are theory or convention,
  none fit to data.

## Decision

H2b and H2c pass → build Layer 4 (distribution engine) and Layer 3 veto on top, paper-trade, then small
capital per the original thesis. Either fails → stop; a further idea needs its own pre-registration.

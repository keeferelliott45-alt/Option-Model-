# Stage 2 — forecast engine

Pre-registration `docs/PREREGISTRATION_STAGE2.md` (sha256 `6a875d78dfcff5e92dacc977428dbbd7831e852bcc406658b9d1189ab50a4265`), run 2026-10-24T15:38:05+00:00.

| Test | Result | Pass |
|---|---|---|
| A. HAR vs naive vol (QLIKE, lower better) | mean diff -0.3082, NW t -2.50, 188 months, sub-periods right 2/3, holdout -0.0779 | **PASS** |
| B. 21-day calibration | 50% interval 51.1%, 90% interval 90.4% (n=75148); holdout 43.9% / 85.6% | **PASS** |
| C. Earnings-window calibration | 50% 45.5%, 90% 86.5% (n=20729); holdout 40.0% / 81.2% | fail |
| D. Short-interest crash tail (log score gain) | +0.00209 per obs, NW t +3.02, 80 months | **PASS** |
## Notes

- VIX3M (needed for the pre-registered VIX-term feature) starts in late 2007, so HAR forecasts exist from
  2008: test A covers ~188 months and the 2003–08 sub-period is thin.
- B holdout coverage (42.9% / 84.1%) is below nominal: the last year was more volatile than forecast. The
  scanner shows this as a calibration warning; it does not change the pre-registered verdict.
- C failed: around earnings the engine is too narrow. Per the pre-registration the earnings jump is dropped,
  and options spanning an estimated earnings date stay vetoed (Layer 3 HARD veto) because the engine is not
  trustworthy there.
- D passed: the short-interest crash adjustment is applied in the live engine.

## Amendment 1 — test A2 (HAR-X vs HAR, non-earnings windows)

Mean QLIKE difference -0.0059 (negative = HAR-X better), NW t -1.22, 188 months, 45355 stock-months, sub-periods right 2/3, holdout -0.0196: fail.

## Correction — 2026-10-24 (after the first live scan)

The filtered-historical-simulation shape was not re-centred, so stocks with a strong 3-year uptrend carried
their past drift into the forecast, against the pre-registered drift = 0. Bug fixed in
`forecast/engine.shape` (used by both the live engine and test B/C) and Stage 2 rerun: A and D unchanged;
B 50.9%/89.4% → 51.1%/90.4% (still PASS); C 45.3%/85.0% → 45.5%/86.5% (still fail). Verdicts unchanged.

## Amendment 2 — test B2 (no-earnings windows)

HAR-N coverage 50%/90%: dev 51.0% / 90.6% (n=48569), holdout 43.6% / 86.1% (n=3083). For comparison HAR: dev 53.9% / 92.3% (n=48569), holdout 46.3% / 88.4% (n=3083). Verdict: **PASS**.

## Amendment 3 — test E (tail payoff calibration)

Realized/model expected excess beyond the 95th percentile 0.93, below the 5th 1.22 (n=48569): **PASS**.

# Stage 2 — forecast engine

Pre-registration `docs/PREREGISTRATION_STAGE2.md` (sha256 `57a7ee741a6ddc6b257b9005f3613e5676573c69ebd6c72fe587a63557b1cc64`), run 2026-10-24T14:48:59+00:00.

| Test | Result | Pass |
|---|---|---|
| A. HAR vs naive vol (QLIKE, lower better) | mean diff -0.3082, NW t -2.50, 188 months, sub-periods right 2/3, holdout -0.0779 | **PASS** |
| B. 21-day calibration | 50% interval 50.9%, 90% interval 89.4% (n=75148); holdout 42.9% / 84.1% | **PASS** |
| C. Earnings-window calibration | 50% 45.3%, 90% 85.0% (n=20729); holdout 38.8% / 78.6% | fail |
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

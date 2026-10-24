# H2a — drift replication on the holdout

Pre-registration: `docs/PREREGISTRATION_THESIS_V2.md` (sha256 `e563771c06076e44c70a59591f533ed18dae8ddeba132c531d93e5c4b8122d86`). Run once, 2026-10-24T13:57:00+00:00.

Window 2025-10-23 → 2026-10-22 (never loaded by v1). Metric: net abnormal log return, t+1 open → t+15
close, 10 bps round trip.

| | Mean | 95% CI | n | Effective month clusters |
|---|---|---|---|---|
| Events (z ≤ −3) | -1.06% | [-1.82%, -0.30%] | 911 | 10.0 |
| Placebo (two-sided exclusion) | +0.02% | [-0.43%, +0.52%] | 707 | 10.9 |
| Event − placebo | -1.08% | [-1.81%, -0.29%] | | |

Pass rule: event mean < 0 and event − placebo < 0 (sign replication) (low-confidence: few monthly clusters).

**Verdict: PASS**

Missing-exit share (possible delistings): +0.00%.

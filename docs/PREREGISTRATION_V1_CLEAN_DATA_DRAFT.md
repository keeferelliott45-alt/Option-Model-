# Phase 1 clean-data rerun — DRAFT (superseded by thesis v2; kept for reference)

Drafted 2026-10-24. **Becomes binding only when committed with the data provider filled in and before any v2
result is computed.** v1 (`PREREGISTRATION.md`) is closed and unchanged; its result was FAIL.

## Disclosure of prior knowledge

v1 ran on survivor-only data and found large residual drops **continue** (−0.39% net 15d, OOS −0.42%).
That bias favoured reversion, so v2 is expected to fail harder. v2 exists to remove the survivorship caveat,
not to search for a pass. Any hypothesis suggested by v1 (e.g. continuation/momentum) is **not** tested here;
it would need its own pre-registration and data it was not derived from (the untouched holdout).

## What changes from v1 — and nothing else

1. **Data:** survivorship-free daily prices that include delisted securities, keyed by a permanent security
   ID (not ticker), with delisting returns. Provider: `__________` (owner's choice, see below).
2. **Delisting:** when a position's stock delists inside its window, the label uses the provider's delisting
   return (or, if absent, −30% for performance-related delistings, 0% for mergers) instead of being dropped.
   `missing_h` must be ~0%; any remainder is reported.
3. **Identity:** membership and prices joined on permanent IDs; the v1 ticker-reuse overrides are removed.
4. **Placebo fix (the cause diagnosed in v1):** placebo dates also exclude any day whose forward 15-day
   window contains a real event of the same ticker (i.e. the exclusion window becomes [t−15, t+15] on the
   event side that was missing). This only uses event dates as a sampling filter for the control group;
   events themselves are selected exactly as in v1.
5. **Coverage gate:** the run is valid only if member-day coverage is ≥ 95% in every year 2000–2025.

Hypothesis H1, event definition (z ≤ −3, 90-day market+sector model fit through t−1), next-open entry,
10 bps round trip, 15-day primary horizon, threshold grid {−2.5, −3, −4}, 2y/6m purged walk-forward,
month clusters, sub-periods, holdout (≥ 2025-10-23, still locked), and criteria 1, 1b, 2, 3, 4, 5 are
**identical to v1**.

## Provider options (owner decides; check current pricing)

| Provider | Delisted coverage | Permanent ID | Delisting returns | Notes |
|---|---|---|---|---|
| CRSP (via WRDS) | full, 1925+ | PERMNO | yes | gold standard; needs university access |
| Norgate Data | US delisted since 1950s | yes | via last price | Windows app, historical index constituents included |
| Sharadar SEP (Nasdaq Data Link) | delisted included | permaticker | no (last price) | API, CSV bulk |
| EODHD | delisted tickers | ticker+exchange | no | cheapest; identity weaker |

Preference order for rigour: CRSP > Norgate > Sharadar > EODHD.

## Implementation steps (no result is computed until step 5)

1. Add `data/<provider>.py` returning the same frame shape as `yahoo.load_daily` plus `delist_ret`, `permno`;
   inject it via `panel.build_panel(loader=..., membership=...)`.
2. Add delisting-return handling in `backtest/events.py` (+ tests: delist mid-window, delist on exit day).
3. Change placebo exclusion per item 4 (+ test that no placebo window contains a same-ticker event).
4. Run coverage only (`panel.coverage`) and verify the ≥ 95% gate. **Commit this file with the provider
   filled in.**
5. Run `python -m omodel phase1` once. Report as v1 did. FAIL → project ends. PASS → Phase 2 per HANDOFF.

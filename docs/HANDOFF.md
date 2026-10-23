# Handoff — Options Overreaction Model

For any agent continuing this build. Source of truth for intent: the owner's thesis (summarised in
`README.md` once written) and `docs/PREREGISTRATION.md`. Branch: `feat/phase0-1-pipeline`.

## Non-negotiable rules (from the thesis)

1. Build order is gated: Phase N+1 starts only after Phase N **passes** on out-of-sample data. If Phase 1
   fails, STOP and report — do not build later layers on a false premise.
2. No lookahead. Anything used at date t is known by the close of t; model params at t are fit on t−1 and
   earlier. Every new module needs a test that perturbs future data and asserts past outputs are unchanged.
3. Holdout = dates on/after **2025-10-23**, enforced by `omodel.holdout`. Never set
   `OMODEL_UNLOCK_HOLDOUT` until the design is frozen; it is used exactly once, for the final check.
4. Walk-forward only (2y fit / 6m test, purged). Never report an in-sample number as a result.
5. Costs always included; entry at next open, never mid or event-day close.
6. Every statistic ships with n and a CI; n < 30 is "low-confidence", not a finding. Sharpe > 2 = bug hunt.
7. Every tunable is registered via `params.register(...)` with source `theory|convention|data-fit`.
8. LLMs only classify/summarise text; every number comes from data or computation.
9. Never edit `docs/PREREGISTRATION.md` after results exist. Tightening before results = dated amendment.

## Setup

```bash
cd Option-Model-
python -m venv .venv --system-site-packages
.venv/Scripts/python.exe -m pip install -e . --no-deps
.venv/Scripts/python.exe -m pytest -q
```

Env: `SEC_USER_AGENT="Name email"` (SEC requires contact info). Data cache: `data/` (gitignored, re-fetchable).

## Status (2026-10-23)

| Area | File(s) | State |
|---|---|---|
| Foundation | `config.py`, `params.py`, `holdout.py`, `types.py`, `data/http.py`, `data/store.py` | done, tested |
| Prices + point-in-time S&P 500 membership + panel | `data/yahoo.py`, `data/universe.py`, `data/panel.py` | done, reviewed, fixed |
| Macro (Treasury, CBOE VIX family, credit/FX/commodities) | `data/macro.py` | done, reviewed, fixed |
| SEC 8-K news/earnings dates, companyfacts fetch | `data/sec.py` | done, reviewed, fixed |
| CBOE live chains + daily snapshot archiver | `data/cboe_options.py` | done, reviewed, fixed |
| Layer 2 rolling market+sector residual model | `layers/l2_decomposition.py` | done, reviewed, fixed |
| Event study + forward returns + calendar-time positions | `backtest/events.py` | done, reviewed, fixed; boundary + lookahead tests added |
| Cluster bootstrap, Newey-West, BH, factor tests | `backtest/stats.py`, `backtest/factor_tests.py` | done, reviewed, fixed |
| Walk-forward + placebo | `backtest/walkforward.py`, `backtest/placebo.py` | done, reviewed, fixed |
| Synthetic market generator | `backtest/synthetic.py` | done, reviewed, fixed |
| Phase 1 experiment, CLI, report, end-to-end synthetic tests | `experiments/phase1.py`, `report/markdown.py`, `cli.py`, `tests/test_phase1_synthetic.py` | done, tested; run on real data 2026-10-23 — **FAIL** (see Phase 1 outcome below) |

Data cached locally (measured 2026-10-23): 758 price files (718 of the 1,130 dev-window ticker requests have data; 412 known-missing markers — mostly delisted, i.e. survivorship bias, reported by `panel.coverage`), 686 SEC submission caches, the membership history, and the macro panel (its cache predates the non-positive-price cleaning fix: refetch with `--refresh` before any Phase 2 use). Cache verified compatible with the current loader.

## Phase 1 outcome (2026-10-23): FAIL — STOP. Do not build Phases 2+.

`python -m omodel phase1 --no-news` on the cached real panel (news factor skipped: no SEC_USER_AGENT set).
Run at 2026-10-23T14:56Z from a clean tree at `10d6b33`; pre-registration SHA-256 matches the committed
file; holdout untouched. Full numbers: `reports/phase1/{report.md,results.json}`.

| id | Criterion | Value | 95% CI | n | Result |
|---|---|---|---|---|---|
| 1 | Event mean net abnormal 15d | −0.39% | [−0.63%, −0.20%] | 16,055 | FAIL (sign negative) |
| 1b | Excess over placebo | −0.14% | [−0.37%, +0.06%] | 31,942 | FAIL |
| 2 | Walk-forward OOS | −0.42% | [−0.69%, −0.18%] | 11,920 | FAIL |
| 3 | Placebo CI includes 0 | −0.15% | [−0.27%, −0.04%] | 15,887 | FAIL → run_invalid = true |
| 4 | ≥ 2 of 3 sub-periods positive | 0 of 3 | — | 16,055 | FAIL |
| 5 | Calendar Sharpe < 2 | −0.97 | — | 6,491 days | pass |

Criterion-3 diagnosis (the "invalid until it is found" condition is satisfied): NOT a pipeline bug. The
pre-registered placebo pool keeps days BEFORE events (excluding them would be lookahead), so 12.66% of
placebo windows contain a same-ticker event day; those windows average −5.87% (the event day itself
averages −5.98% abnormal) and account for the entire placebo mean: removing them leaves +0.68%
[+0.58%, +0.79%], and the mixture reproduces −0.155% to within 5%. Ruled out empirically: terminal
delisting stretches (0% of the drift), single-year concentration (none; mild negative drift spread
across 2006–2025), selloff-window concentration (none), near-data-end events (0.34% of events, ~1% of
the primary mean). The synthetic null-world placebo is clean in the test suite, so the machinery can
produce a clean placebo when the world has no events.

Substantive reading: in this survivor-biased sample, large idiosyncratic drops do NOT revert. The net
mean is significantly BELOW zero (−0.39%, CI excludes 0); walk-forward OOS matches in-sample (−0.42%,
48 splits; the deepest-drop threshold z=−4 chosen in 29 of 48 windows, still negative OOS); costs
explain only 0.10 pp of the −0.39%; up-move events (z ≥ +3) are symmetrically negative (−0.49% net);
the calendar-time portfolio loses significantly (Newey-West t = −5.0, 37 positions/day average). The
pre-registered excess over placebo is −0.14% (CI includes 0); versus crash-free windows only — a
lookahead-style filter, diagnostic only — the gap is about −0.97 pp. The data point the opposite way
from H1, never towards it.

Per thesis rule 1 and the pre-registration: STOP. Later phases are not built on this premise. Rule 9:
docs/PREREGISTRATION.md must not be edited now. Any redesign (e.g. excluding pre-event windows from the
placebo, or survivorship-free price data via the documented loader hook in README) requires a NEW dated
pre-registration written and committed before the next result — the owner's call. Secondary notes: 8-K
factor skipped (--no-news); stale-open entries 4.0% (13–16% in 2001–2005; including them the mean is
−0.40%, unchanged); stale fit-windows: 1 event > 50% zero returns (PARA 2023-06-14); close-entry bounce
≈ 0; factor battery: peer_z is the only borderline BH discovery (p_adj 0.0997, thesis-predicted sign) —
reported, not decided on; dev-window member-day coverage 70.2% (2000: 47.0%, 2025: 96.9%).

## Session log — 2026-10-23 (review-gap closure, integrity fix, real-data run)

Commits this session, in order: `0ff10f8` (finish Phase 1 integration), `8f0d0ba` (track src/omodel/data),
`10d6b33` (review fixes + tests), `c437557` (document the Phase 1 outcome and run artifacts).

1. Verified the finished Phase 1 integration against the pre-registration, then committed it (0ff10f8).
   Suite at that point: 314 passed + 1 environmental failure (the CLI dirty-tree guard, fixed by committing).
2. Adversarial review of `data/sec.py`, `backtest/walkforward.py`, `backtest/placebo.py` (lenses: lookahead,
   spec-vs-code, real-data robustness, test adequacy): no lookahead found in any of them. Confirmed correct:
   walk-forward purge (`exit_ < test_start`), anchored non-drifting splits, contiguous non-overlapping test
   windows, deterministic tie-breaks, legitimate reuse of realized test labels as later fit labels; placebo
   cooldown arithmetic `[e, e+window]` inclusive, deliberate non-exclusion of pre-event days, unbiased reuse,
   month-cluster pairing with `cluster_bootstrap_diff`.
3. Fixes applied from the reviews (10d6b33): `_has_item` splits on commas AND whitespace ("2.02 9.01" used to
   silently not match); partial SEC submissions pagination now warns and is not cached; a non-positive price
   guard for gold/oil in `macro.py` (the 2020-04-20 CL=F −37.63 settlement is now cleaned to NaN — refetch the
   macro cache with `--refresh` before any Phase 2 use); walk-forward t-score guard hardened to `np.isfinite`;
   placebo eligibility additionally requires finite `mkt_ret`/`sec_ret` (defence-in-depth); `types.py` volume
   comment corrected to "split-adjusted".
4. Deferred items decided: stale-sigma → diagnostic only (`phase1._flat_share` + the report's "Stale-price
   windows (diagnostic)" section; on real data 1 of 16,782 events exceeds the 50% zero-return share — PARA
   2023-06-14 — negligible); panel volume → data already correct, doc fixed; CL=F → cleaned (above).
5. Tests added: `test_events.py` exit-on-holdout-start embargo boundary (exit landing ON the holdout start is
   embargoed, one session earlier is kept; entry falling on the holdout start is embargoed) and a model-column
   lookahead canary (beta_mkt, beta_sec, alpha, tot_sigma, vol_z bit-identical after future prices are
   scrambled); `test_walkforward.py` equal consecutive fit-window lengths, test_end boundary flows to the next
   split, realized-test-event reuse as a later fit label; `test_placebo.py` reuse uniformity (floor/ceil),
   n_per_event=0 empty frame, finite-z-but-NaN-return day ineligible; SEC item splitting + missing-page
   warning; macro non-positive settlements; stale-window unit test. Suite after all changes: **327 passed,
   4 deselected** (network smoke tests), clean tree.
6. Repo-integrity fix (8f0d0ba): the unanchored `data/` ignore rule shadowed `src/omodel/data/` — the entire
   data-source package had never been tracked, so the CLI's tamper-evidence check was blind to half the code.
   Anchored to `/data/` and committed the 9 package files. Git identity configured repo-locally
   (keeferelliott45-alt / GitHub noreply email) so commits work without global config.
7. Real-data Phase 1 run (`python -m omodel phase1 --no-news`; SEC_USER_AGENT unset, so the secondary 8-K
   factor is skipped): **FAIL**, `run_invalid: true` — details and diagnosis in the section above.
8. Placebo diagnosis executed per criterion 3's own rule (read-only recomputation from the cached panel and
   `data/results/phase1_events.parquet`): the −0.155% placebo mean is fully accounted for by event-day
   contamination inherent in the pre-registered pool definition (12.66% of windows contain a same-ticker event
   day averaging −5.87%); mixture accounting reproduces the mean to within 5%; alternative explanations
   (terminal delisting stretches, year concentration, selloff concentration, near-data-end events) quantified
   and excluded.
9. Per thesis rule 1: STOPPED. No Phase 2+ code was written. `docs/PREREGISTRATION.md` untouched (SHA verified
   at run time and absent from every commit).

Environment notes for the next agent: `SEC_USER_AGENT` is NOT set (set it before any run that needs the 8-K
factor; SEC submissions for 686 tickers are already cached, so no fetching is needed); the Phase 1 run works
fully offline from the cache (~20 s end to end); `data/results/phase1_events.parquet` holds the event list;
`reports/phase1/` holds the failing run's artifacts (committed at c437557).

## Next steps, in order

### 1. Finish Phase 1 integration — DONE (commit 0ff10f8)
- `run(panel=None, dec=None, *, out_dir=None, news=True, write=True) -> dict` executing
  `docs/PREREGISTRATION.md` exactly, including Amendment 1 (events − placebo difference must be > 0, CI excl. 0).
- Steps: build panel → decompose → events for each z in `EVENT_Z_GRID` with next-open forward returns →
  primary summary (`net_abn_15`) + horizon table → calendar-time portfolio (Newey-West, Sharpe, suspicious flag)
  → walk-forward OOS → placebo + event−placebo diff (`stats.cluster_bootstrap_diff`) → sub-periods
  2000–08 / 2009–16 / 2017–25 → secondary (close-entry bounce check, up-moves, factor battery: vol_z, sec_ret,
  mkt_ret, peer_z, |z|, no-8-K flag) → missing-exit counts → criteria table + verdict.
- Writes `reports/phase1/{report.md,results.json,walkforward_choices.csv,coverage.csv}`, `docs/PARAMETERS.md`
  (generated by `params.render_markdown()`), and includes the SHA-256 of the pre-registration.
- `tests/test_phase1_synthetic.py`: null synthetic world → verdict FAIL and placebo CI includes 0;
  injected-reversion world → detected OOS; lookahead canary (scrambling prices after date D leaves every
  pre-D decomposition value and event unchanged).
- CLI: `python -m omodel {fetch-prices,fetch-macro,fetch-sec,snapshot-options,phase1,params}`.

### 2. Close review gaps — DONE (commits 8f0d0ba, 10d6b33: reviews clean, deferred items decided, events tests added)
- Adversarially review `data/sec.py`, `backtest/walkforward.py`, `backtest/placebo.py` (lenses: lookahead,
  correctness vs spec, real-data robustness, test adequacy).
- `events.py`: add a test for the exit-on-holdout-start boundary and a lookahead test covering the event
  model columns (beta_mkt, beta_sec, alpha, tot_sigma, vol_z).
- Deferred items to decide on: stale/zero-return stretches deflating sigma (l2); panel volume is
  split-adjusted while types.py says raw (prices); CL=F negative print 2020-04-20 (macro).

### 3. Run Phase 1 on real data — DONE 2026-10-23: FAIL (see Phase 1 outcome)
`SEC_USER_AGENT=... .venv/Scripts/python.exe -m omodel phase1`. Then audit before believing it: placebo
clean? Sharpe < 2? Does the effect survive the close-vs-open entry comparison? Is it concentrated in
low-coverage years? Report: what was tested, OOS result, n, CI, and what would have to be true for it to
be a fluke. **If the verdict is FAIL, stop here and report.**

### 4. Phase 2 — Layer 1 regime filter (BLOCKED: Phase 1 failed)
`layers/l1_regime.py`: daily point-in-time features from `macro.load_macro_panel()` — VIX level, VIX/VIX3M
(>1 = backwardation), 5d Δ2Y/Δ10Y, 2s10s, 20d Δlog(HYG/LQD), 20d ΔDXY, |5d| gold/oil moves, breadth
(% members above 50dma), 60d average pairwise correlation (variance-ratio formula), and a market-implied
tension index (equal-weight expanding z-scores, EWM decay, half-life registered). Test each factor alone with
`factor_tests.factor_battery` on Phase 1 events (BH-corrected); keep only factors with standalone OOS signal.
Pass = regime-filtered walk-forward OOS beats unfiltered Phase 1 OOS with a CI on the difference.

### 5. Phase 3 — Layer 3 fundamental veto
`layers/l3_fundamentals.py` from `sec.load_companyfacts` (point-in-time by `filed` date): TTM revenue, gross
margin and its 4-quarter trend, EBITDA proxy, net debt/EBITDA, interest coverage, P/S vs same-sector peers and
own 3y history. Veto: HARD (leverage + going-concern-type event; earnings inside the option horizon via
`sec.earnings_days`), SOFT (expensive, margins already falling), PASS. Thresholds registered as theory. Pass =
veto improves OOS win rate vs Phase 2, tested, not assumed.

### 6. Phase 4 — Layer 5 options EV (math is buildable now; validation needs historical chains)
`layers/l5_options.py`: Black-Scholes price/IV/delta; Breeden-Litzenberger risk-neutral density from a
smoothed IV smile; EV = Σ P_model(S_T)·payoff − premium at the ask (sell legs at bid); defined-risk structures
only (long call/put, debit and credit verticals); liquidity filter (min OI, volume, max spread % of mid);
delta buckets 0.05–0.20 / 0.20–0.40 / 0.40–0.60 / 0.60–0.80 / 0.80+ reported separately, higher EV hurdle
for 0.05–0.20, 0.80+ excluded. No free historical chains exist: run `snapshot-options` daily to build them,
or plug in a paid source (ORATS, Polygon, ThetaData, CBOE DataShop). Phase 4 cannot pass on synthetic BS prices.

### 7. Phase 5 — Layer 4 distribution engine + paper trading
`layers/l4_distribution.py`: empirical distribution of forward returns after similar past events (only events
whose exit < as-of date), standardised by pre-event vol, partially pooled stock→universe (prior strength
registered), regime and fundamental adjustments, rescaled to current vol; output discrete {bucket: prob}.
Report EV under this distribution **minus** EV under the unconditional distribution, so variance-risk-premium
"edge" is not mistaken for overreaction edge. One-sentence thesis per contract where every number traces to a
layer. Daily shortlist → paper trade only. The system never places orders.

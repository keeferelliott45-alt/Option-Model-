# omodel: options overreaction research system

omodel looks for liquid optionable stocks whose large moves are **not** explained by market or sector
factors ("overreactions", especially during correlated sector or macro shocks). The end goal is a ranked
shortlist of option contracts where a defensible, data-derived return distribution disagrees with the market's
price. It is a research tool. **It never places or submits a trade**; a human decides every trade.

What it refuses to do, by design:

- Predict direction from price action alone, with no causal story.
- Trust an in-sample backtest. Every layer must survive walk-forward, out-of-sample testing.
- Recommend undefined-risk structures (naked short options).
- Treat any single signal, including the overreaction score, as enough on its own.
- Claim precision it does not have. Every statistic comes with its sample size and a confidence interval, and
  anything under 30 events is labelled low-confidence, not a finding.

A backtest that looks too good is treated as a bug: a calendar-time Sharpe above 2 fails the run.

## Phase plan and status

| Phase | Builds | Proves | Status |
|---|---|---|---|
| 0 | Point-in-time price, sector-ETF, macro, SEC and option-snapshot pipeline | Nothing yet: infrastructure | **Built** |
| 1 | Layer 2 decomposition (rolling market + sector residuals) and an event study | Whether overreaction is real and exploitable at all | **Ran 2026-10-23: FAIL** — no post-drop reversion; stopped per [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (see docs/HANDOFF.md and reports/phase1/report.md) |
| 2 | Layer 1 macro/regime context as a filter on Phase 1 | Whether regime context improves the signal | Not started: gated on a Phase 1 PASS |
| 3 | Layer 3 fundamental veto | Whether the veto improves win rate over Phase 2 | Gated |
| 4 | Layer 5 options EV on real historical chains, by delta bucket | Whether the equity edge survives spreads and IV | Gated |
| 5 | Layer 4 distribution engine, full pipeline paper-traded | End to end on live data, no capital | Gated |
| 6 | Small real capital | Whether live fills match the backtest | Gated |

Phase 1 is fixed in advance by [docs/PREREGISTRATION.md](docs/PREREGISTRATION.md) (hypothesis, sample and five
pass criteria, plus Amendment 1). If any criterion fails, the project stops and reports.

## Setup (Windows)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
$env:SEC_USER_AGENT = "Your Name you@example.com"   # or: setx SEC_USER_AGENT "Your Name you@example.com"
```

SEC EDGAR requires a contact string in the User-Agent. Without `SEC_USER_AGENT`, `fetch-sec` refuses to run and
Phase 1 skips the 8-K news signal (and says so in the report). Data is cached under `data/` (override with
`OMODEL_DATA_DIR`). It is git-ignored and can always be fetched again.

## Commands

```powershell
omodel fetch-prices                     # Yahoo daily bars: SPY, 11 sector ETFs, every dev-window S&P 500 member
omodel fetch-prices --tickers AAPL MSFT # just these symbols
omodel fetch-macro                      # rates, VIX term structure, credit, dollar, gold, oil, FX
omodel fetch-sec                        # SEC filing histories (needs SEC_USER_AGENT)
omodel snapshot-options --tickers AAPL  # store today's CBOE delayed chains; run after ~16:30 ET
omodel phase1                           # run the pre-registered study (add --no-news to skip SEC, --refresh to refetch)
omodel params                           # regenerate docs/PARAMETERS.md from the parameter registry
.venv\Scripts\python -m pytest -q       # tests; live-network smoke tests are deselected by default
```

`python -m omodel <command>` works the same way. Every fetch is cached and rerunnable: failed tickers are not
cached, so a second run retries only those.

## Reading the Phase 1 report

`omodel phase1` writes `reports/phase1/`:

- `report.md` is the document to read. In order, it shows:
  - **Verdict and criteria table**: PASS only if every pre-registered criterion passes.
  - **OUT-OF-SAMPLE result (walk-forward)**: the only result where anything was chosen. A threshold is picked
    on each 2-year fit window and applied to the next 6 months.
  - **Full-window result, pre-registered single test (no parameters fit)**.
  - **Placebo**: the same tickers on random days. It should be zero. If it is not, the run is invalid until
    the cause is found.
  - **Calendar-time Sharpe**: the plausibility check. Above 2.0 means hunt for a bug.
  - **Sub-periods** and **secondary analyses**: close-entry bounce, up-moves, the threshold grid (in-sample,
    descriptive only) and the one-factor-at-a-time signal battery with Benjamini-Hochberg correction. None of
    these decide anything.
  - **Coverage and survivorship numbers**, and **"What would have to be true for this to be a fluke"**.
- `results.json` holds every number, machine-readable (NaN is written as null). `walkforward_choices.csv`
  and `coverage.csv` hold the tables, and `data/results/phase1_events.parquet` holds the event list.

Every tunable parameter is registered in code with its provenance (theory, convention or data-fit). See
[docs/PARAMETERS.md](docs/PARAMETERS.md), which is generated, never hand-edited.

## The holdout rule

Everything dated on or after **2025-10-23** is the holdout. The date is fixed in `omodel.config`; it never
rolls with today's date. Code enforces it (`omodel.holdout`): loaders drop holdout rows, analyses raise
`HoldoutViolation`, and forward returns whose exit would land in the holdout are embargoed.
`OMODEL_UNLOCK_HOLDOUT=I_HAVE_FROZEN_THE_DESIGN` exists **only for the single final check** after the design is
frozen. Every unlocked access is logged to `data/holdout_access.log`. Note: every option snapshot collected from
now on falls inside this holdout, so options-layer research will need its own fixed, forward-only holdout
before Phase 4.

## Known limitation: survivorship

Free Yahoo data mostly lacks companies that were later delisted, acquired or went bankrupt. The S&P 500
membership history (fja05680/sp500) is point-in-time, but many past members have no prices. Those missing
names lean toward drops that never recovered, so the bias **favours** the overreaction hypothesis: a PASS is an
upper bound and a FAIL is robust. Phase 1 measures member-day coverage by year and prints it in the report.
Missing exit prices are counted separately, and Amendment 1 adds an excess-over-placebo test against survivor
drift.

To plug in a survivorship-free provider (CRSP, Norgate, EODHD and similar), pass a loader and a membership
table to the panel builder. No other code changes:

```python
from omodel.data import panel
from omodel.experiments import phase1

def my_loader(symbol):   # called for SPY, the sector ETFs and every stock
    ...                  # -> DataFrame indexed by date: close, open, volume (+ adj_close, split_factor)

membership = ...         # DataFrame: ticker, start_date, end_date (NaT = still a member)
p = panel.build_panel(loader=my_loader, membership=membership)
phase1.run(panel=p, membership=membership)
```

Use identifiers that are never reused (permnos, or ticker plus date range). Recycled tickers attribute one
company's prices to another; `data.panel.IDENTITY_OVERRIDES` patches the known Yahoo cases.

## Layout

`src/omodel/`: `data/` (sources, cache, point-in-time panel), `layers/l2_decomposition.py` (residual model),
`backtest/` (events, placebo, walk-forward, cluster-bootstrap statistics, factor tests, synthetic worlds),
`experiments/phase1.py`, `report/markdown.py`, `cli.py`. `tests/test_phase1_synthetic.py` is the most important
test. It runs the whole study on synthetic markets with known truth: a null world must FAIL with a clean
placebo, an injected edge must be found out-of-sample, and changing prices after a date must leave every
earlier output bit-for-bit unchanged.

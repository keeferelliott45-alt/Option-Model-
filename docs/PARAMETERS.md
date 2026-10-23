# Parameter registry

34 parameters, 0 fit to data. Data-fit parameters are only ever chosen inside walk-forward fit windows and get the most out-of-sample scrutiny.

| Parameter | Value | Source | Layer | Rationale |
|---|---|---|---|---|
| `MACRO_FFILL_LIMIT` | `3` | convention | L1 | Bridge holiday-calendar mismatches between sources only; longer gaps stay NaN so stale macro is never used. |
| `MACRO_START_YEAR` | `1998` | convention | L1 | Two years of warm-up before dev_start (2000) so trailing macro features are defined from day one. |
| `BETA_MIN_OBS` | `60` | convention | L2 | Minimum valid days inside the beta window before a fit is trusted. |
| `BETA_WINDOW` | `90` | theory | L2 | Thesis: rolling 60-90d betas; upper end for estimate stability. |
| `EVENT_COOLDOWN` | `15` | convention | L2 | Suppress same-ticker events inside the primary horizon to avoid double counting. |
| `EVENT_Z_GRID` | `(-2.5, -3.0, -4.0)` | theory | L2 | Only thresholds the walk-forward selector may choose between. |
| `EVENT_Z_THRESHOLD` | `-3.0` | theory | L2 | Pre-registered: 3-sigma residual drop = 'large unexplained move'. |
| `MIN_PRICE` | `5.0` | convention | L2 | Exclude sub-$5 prints where spreads and tick size dominate returns. |
| `PRIMARY_HORIZON` | `15` | theory | L2 | Pre-registered primary horizon (thesis worked example uses 15d). |
| `SECTOR_REASSIGN_FREQ` | `'ME'` | convention | L2 | Sector assignment refreshed month-end using only prior data. |
| `SECTOR_WINDOW` | `252` | convention | L2 | One trading year of history to pick each stock's sector ETF. |
| `SEC_NEWS_WINDOW` | `1` | convention | L2 | News effective the prior session can still drive day-t moves, so day t is flagged for effective dates t and t-1. |
| `VOLUME_Z_WINDOW` | `20` | theory | L2 | Thesis: volume vs 20-day average. |
| `HORIZONS` | `(1, 5, 15, 30)` | theory | L2/L4 | Thesis: forward returns at 1, 5, 15, 30 trading days. |
| `BOOTSTRAP_REPS` | `2000` | convention | backtest | Cluster-bootstrap replications for confidence intervals. |
| `CI_LEVEL` | `0.95` | convention | backtest | Two-sided confidence level. |
| `CLUSTER_FREQ` | `'M'` | convention | backtest | Calendar-month date clusters: wider than the 15d primary horizon, so overlapping forward windows and crash-clustered events mostly share a cluster. |
| `COST_BPS_PER_SIDE` | `5.0` | convention | backtest | Conservative half-spread + slippage for S&P 500 names, per side. |
| `ENTRY` | `'next_open'` | convention | backtest | Signal is known at close t; first tradable price is the t+1 open. |
| `EXPANDING_MIN_OBS` | `252` | convention | backtest | Minimum history before an expanding-window quantile is defined. |
| `FACTOR_N_BUCKETS` | `3` | convention | backtest | Terciles for one-factor-at-a-time tests: two cut points, the fewest that still separate high from low. |
| `FDR_Q` | `0.1` | convention | backtest | Benjamini-Hochberg false-discovery rate across one-factor-at-a-time tests. |
| `MIN_EFFECTIVE_CLUSTERS` | `30` | convention | backtest | Kish effective number of date clusters, (sum c_g)^2 / sum c_g^2, below which cluster-robust CIs and p-values are anti-conservative (few-cluster guidance of ~20-50, Cameron & Miller 2015); such results are flagged low-confidence, matching the 30-instance rule. |
| `MIN_SAMPLE_FINDING` | `30` | theory | backtest | Thesis: under ~30 instances is low-confidence, not a finding. |
| `RANDOM_SEED` | `20261023` | convention | backtest | Fixed seed for reproducibility. |
| `STABILITY_MIN_SUBPERIODS` | `2` | convention | backtest | Pre-registered: the mean must have H1's (positive) sign in at least 2 of the 3 sub-periods. |
| `STALE_OPEN_RULE` | `'volume[t+1] == 0 or as-traded open[t+1] == close[t]'` | convention | backtest | An entry print that is not a real t+1 opening trade (no volume, or a free-data open copied from the prior close, ~10-15% of post-drop S&P bars before 2008 vs ~1-2% after) is missing data, not a fill; using it would silently turn the pre-registered next-open entry into a close entry, which favours H1. |
| `STALE_WINDOW_SHARES` | `(0.5, 0.9)` | convention | backtest | Shares of exactly-zero returns in a fit window above which the window is mostly stale prices (halt, illiquidity) and the rolling sigma is deflated, so z = resid/sigma can spike and manufacture fake events. |
| `SUBPERIODS` | `((2000, 2008), (2009, 2016), (2017, 2025))` | convention | backtest | Pre-registered stability split (criterion 4): three ~8-year calendar blocks fixed before any result. |
| `SUSPICIOUS_SHARPE` | `2.0` | theory | backtest | Thesis: Sharpe above ~2-3 is treated as a probable bug until proven otherwise. |
| `WF_FIT_YEARS` | `2` | theory | backtest | Thesis: fit on rolling ~2 years. |
| `WF_TEST_MONTHS` | `6` | theory | backtest | Thesis: test on next 3-6 months. |
| `PANEL_WARMUP_DAYS` | `550` | convention | data | Calendar days loaded before dev_start: SECTOR_WINDOW + BETA_WINDOW = 342 trading days (~495 calendar days) of warm-up plus a holiday buffer. |
| `REUSE_TOLERANCE_DAYS` | `7` | convention | data | A member cannot predate its first trade: a price history starting more than ~5 trading days after a membership spell begins marks the symbol as reused by another company, and that spell's prices are dropped. |

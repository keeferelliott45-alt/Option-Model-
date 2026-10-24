# Parameter registry

82 parameters, 0 fit to data. Data-fit parameters are only ever chosen inside walk-forward fit windows and get the most out-of-sample scrutiny.

| Parameter | Value | Source | Layer | Rationale |
|---|---|---|---|---|
| `BREADTH_MA` | `50` | theory | L1 | Thesis: % of members above 50-day MA. |
| `CORR_WINDOW` | `60` | convention | L1 | Days for average pairwise correlation. |
| `HIGH_CORR` | `0.35` | convention | L1 | Label threshold: 'market trading as one'. |
| `HIGH_TENSION` | `0.5` | convention | L1 | Label threshold on the tension z-index. |
| `MACRO_FFILL_LIMIT` | `3` | convention | L1 | Bridge holiday-calendar mismatches between sources only; longer gaps stay NaN so stale macro is never used. |
| `MACRO_START_YEAR` | `1998` | convention | L1 | Two years of warm-up before dev_start (2000) so trailing macro features are defined from day one. |
| `REGIME_Z_MIN_OBS` | `252` | convention | L1 | History before an expanding z-score exists. |
| `TENSION_HALFLIFE` | `10` | theory | L1 | Days for old tension to fade unless refreshed (thesis decay function). |
| `VIX_HIGH` | `20.0` | theory | L1 | Conventional calm/stressed VIX boundary (H2c split). |
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
| `EARNINGS_BUFFER_DAYS` | `10` | convention | L3 | Slack around the estimated next release date. |
| `EARNINGS_CYCLE_DAYS` | `91` | theory | L3 | Quarterly reporting: next release estimated one quarter after the last. |
| `MAX_ND_EBITDA` | `4.0` | theory | L3 | Leverage above this is a refinancing risk. |
| `MIN_INT_COVERAGE` | `1.5` | theory | L3 | Operating income / interest floor. |
| `SOFT_GM_DROP` | `-0.02` | theory | L3 | Gross margin down > 2pp YoY: soft veto. |
| `SOFT_ND_EBITDA` | `3.0` | theory | L3 | Elevated leverage: soft veto. |
| `SOFT_PS_VS_OWN` | `1.3` | theory | L3 | P/S above 1.3x its own 3-year median: expensive going in, soft veto. |
| `L4_BINS` | `201` | convention | L4 | Discrete S_T buckets in the output. |
| `L4_MIN_HISTORY` | `100` | theory | L4 | Fewer comparable past events than this and no distribution is stated. |
| `L4_PRIOR_STRENGTH` | `30` | theory | L4 | Pseudo-events: stock- and regime-specific history gets weight n/(n+30). |
| `L4_UNCERTAIN_WIDEN` | `1.25` | theory | L4 | SOFT/UNKNOWN fundamentals widen the distribution; never shift its center. |
| `COMMISSION_PER_CONTRACT` | `0.65` | convention | L5 | Typical retail per-contract commission, charged on open and close. |
| `CONTRACT_MULTIPLIER` | `100` | convention | L5 | US equity options: 100 shares. |
| `DELTA_EDGES` | `(0.05, 0.2, 0.4, 0.6, 0.8)` | theory | L5 | Thesis delta buckets; reported separately, never pooled. |
| `EV_HURDLE` | `0.05` | theory | L5 | Min EV as a fraction of max loss to surface. |
| `FAR_OTM_EV_HURDLE` | `0.15` | theory | L5 | Higher bar for 0.05-0.20 delta: most fragile to distribution error. |
| `H2C_SHORT_DELTA` | `0.3` | theory | L5 | H2c fixed rule: short call delta. |
| `MARKET_WEIGHT` | `0.5` | convention | L5 | Until the forward test proves the model, trades are scored on a 50/50 blend of the model's forecast and the market-implied (Breeden-Litzenberger) distribution: a pick must survive the market's own view. |
| `MAX_SPREAD_PCT` | `0.1` | convention | L5 | Max (ask-bid)/mid per leg: an edge that cannot be filled is not an edge. |
| `MAX_SPREAD_WIDTH_STEPS` | `4` | convention | L5 | Vertical spreads span at most this many listed strikes. |
| `MIN_OPEN_INTEREST` | `100` | convention | L5 | Liquidity floor per leg. |
| `MIN_VOLUME` | `10` | convention | L5 | Liquidity floor per leg (day volume). |
| `RISK_FREE_DEFAULT` | `0.04` | convention | L5 | Fallback rate when no Treasury yield is passed in. |
| `SHORTLIST_MAX_DTE` | `45` | theory | L5 | Thesis v2 expiry window (H2c rule). |
| `SHORTLIST_MIN_DTE` | `21` | theory | L5 | Thesis v2 expiry window (H2c rule). |
| `SHORTLIST_TOP_N` | `10` | convention | L5 | Contracts shown per day. |
| `SMILE_DEGREE` | `3` | convention | L5 | Weighted cubic in log-moneyness: smooth enough to ignore wing noise, flexible enough for skew. Flat beyond the quoted range. |
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
| `FINRA_PUBLICATION_LAG_DAYS` | `12` | convention | data | FINRA publishes ~8 business days after settlement; 12 calendar days is a conservative availability date. |
| `PANEL_WARMUP_DAYS` | `550` | convention | data | Calendar days loaded before dev_start: SECTOR_WINDOW + BETA_WINDOW = 342 trading days (~495 calendar days) of warm-up plus a holiday buffer. |
| `REUSE_TOLERANCE_DAYS` | `7` | convention | data | A member cannot predate its first trade: a price history starting more than ~5 trading days after a membership spell begins marks the symbol as reused by another company, and that spell's prices are dropped. |
| `CRASH_LOG_RET` | `-0.25` | theory | forecast | Stage 1/2 crash definition over 63 days; scaled by sqrt(h/63) for other horizons. |
| `EWMA_LAMBDA` | `0.94` | theory | forecast | RiskMetrics decay for standardising returns. |
| `FHS_BLOCK` | `5` | convention | forecast | Bootstrap block length (keeps vol clustering). |
| `FHS_YEARS` | `3` | convention | forecast | Years of own history for the return shape. |
| `SIM_PATHS` | `20000` | convention | forecast | Simulated paths per live forecast. |
| `SCAN_IV_GUARD` | `1.5` | convention | scanner | Hide contracts when model vol / market iv30 is outside [1/1.5, 1.5]. |
| `SCAN_MAX_DTE` | `180` | convention | scanner | Longest expiry scanned (calendar days). |
| `SCAN_MAX_PER_SECTOR_SIDE` | `3` | convention | scanner | Diversification: same-direction picks per sector ETF. |
| `SCAN_MAX_PER_TICKER` | `2` | convention | scanner | Diversification: picks per stock. |
| `SCAN_MIN_DTE` | `7` | convention | scanner | Shortest expiry scanned (calendar days). |
| `SCAN_POST_EARNINGS_DAYS` | `22` | theory | scanner | Skip a stock while its last earnings day is inside the RV22 input window. |
| `SCAN_TOP_N` | `25` | convention | scanner | Picks shown per day. |
| `SMILE_MAX_SPREAD` | `0.25` | convention | scanner | Quotes wider than 25% of mid are too noisy to shape the market density. |

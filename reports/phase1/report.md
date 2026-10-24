# Phase 1 report: is 'overreaction' a real, exploitable pattern?

**Verdict: FAIL** (1 of 6 pre-registered criteria passed).
Failed: criterion 1, 1b, 2, 3, 4. Per the pre-registration: stop and report; later phases are not built on this premise.
**The placebo criterion failed, so this run is INVALID until the cause (a pipeline bug, or a drift shared by every day such as survivor drift) is found.**

Run 2026-10-24T13:38:11Z; data 1998-07-02 to 2025-10-22; 695 tickers, 6,491 development trading days; PREREGISTRATION.md sha256 `394b7d5371623754891e33327c5d2187067c430ae24d4608743a6d9de83e5658`.
Code: git `4d34828f08b2` (clean); code sha256 `409b652ece286bbf4ff6393091cd526baebc094ad45248dd4283f9292b71ac99`.

## Decision: pre-registered criteria

All must pass (PREREGISTRATION.md, including Amendment 1: criterion 1 needs both 1 and 1b).

| # | Criterion | Value (95% CI, n) | Threshold | Result |
|---|---|---|---|---|
| 1 | Event level: mean net abnormal 15d return | -0.39% [-0.63%, -0.20%], n=16,055 | mean > 0 and 95% CI lower bound > 0 (n >= 30 and >= 30 effective month clusters) | FAIL |
| 1b | Excess over placebo (events minus placebo, month clusters resampled jointly) | -0.14% [-0.37%, +0.06%], n=31,942 | difference > 0 and 95% CI lower bound > 0 (n >= 30 and >= 30 effective month clusters) | FAIL |
| 2 | Walk-forward OUT-OF-SAMPLE mean net abnormal 15d return | -0.42% [-0.69%, -0.18%], n=11,920 | mean > 0 and 95% CI lower bound > 0 (n >= 30 and >= 30 effective month clusters) | FAIL |
| 3 | Placebo: mean abnormal 15d return on random non-event member days (before the fixed round-trip cost) | -0.15% [-0.27%, -0.04%], n=15,887 | 95% CI includes 0 | FAIL |
| 4 | Stability: sub-periods with a positive mean | 0 positive | >= 2 of 3 | FAIL |
| 5 | Plausibility: calendar-time portfolio annualized Sharpe | -0.97 | < 2.0 | PASS |

## What was tested

- **Hypothesis H1** (docs/PREREGISTRATION.md): after a day whose standardized residual is z <= -3.0, the abnormal log return from the **t+1 open to the t+15 close**, net of costs, has a positive mean.
- **Residual model:** stock return = alpha + beta_mkt * SPY + beta_sec * sector ETF, betas fit on the prior 90 trading days only (min 60 obs); the sector ETF is re-picked each month-end from the prior year of data. z = residual / residual std of the fit window.
- **Events:** point-in-time S&P 500 members on the event date, as-traded price >= $5.0, same-ticker events within 15 days suppressed.
- **Return:** stock minus (beta_mkt * SPY + beta_sec * sector ETF) over the same open-to-close window with the betas as of the event date (alpha excluded), minus 5.0 bp per side (10.0 bp round trip).
- **Statistics:** means with 95% CIs from a calendar-month clustered bootstrap (2,000 reps); anything under 30 events or 30 effective months is low-confidence, not a finding.
- **Development window** 2000-01-03 to 2025-10-22; the holdout from 2025-10-23 was not touched.

## Implementation readings

Choices made in code that PREREGISTRATION.md does not spell out. They were fixed before any real-data result and belong verbatim in a dated amendment before the first real run (readings sha256 `2213957ff2e78ec2e18480b55444f1eb313d9ff59924083bc3a35494724d95a9`).

1. Criterion 3 is judged on the placebo's abnormal 15d return BEFORE the fixed 10 bp round-trip cost (a constant cost would push any clean placebo below zero).
2. 'Date-clustered' CIs resample calendar-month clusters (M), 2000 percentile-bootstrap replications.
3. Walk-forward: on each fit window the threshold with the highest i.i.d. t-score among those with >= 30 purged fit events is chosen; a window where none qualifies adds no test events. Purged = the fit event's 15d exit date falls before the test window opens.
4. Same-ticker events within 15 trading days of the last kept event are suppressed.
5. Criterion 4: at least 2 of the 3 sub-periods have a positive mean; a sub-period with no events counts as not positive.
6. Criteria 1, 1b and 2 also require n >= 30 and >= 30 effective month clusters: a low-confidence estimate never passes.
7. Placebo: the same tickers, one draw per event, from member days with finite model inputs and as-traded price >= $5, inside the events' date span, excluding each real event day and the 15 days after it. Days BEFORE an event are not excluded: that would keep only placebo windows with no future drop in them (lookahead).
8. Entries that are not real t+1 opens are missing data (volume[t+1] == 0 or as-traded open[t+1] == close[t]), for events and placebo alike; they are counted by year and the unfiltered mean is reported as a sensitivity row.
9. Events or placebo days whose stock or ETF price is missing at entry or exit are excluded from every mean and counted; they are never booked as losses or gains.
10. Placebo drift diagnostic (placebo abnormal return minus 15 x the ticker's mean daily abnormal return over the development window) is reported only. It does not change criterion 3 or the run's validity.

## OUT-OF-SAMPLE result (walk-forward)

The only result in which anything was chosen. On each 2-year fit window the threshold with the best t-score among [-2.5, -3.0, -4.0] was picked, using only events whose 15-day exit fell before the test window (purged), then applied unchanged to the next 6 months. The test windows are concatenated.

- **OUT-OF-SAMPLE net abnormal return:** mean -0.42%, 95% CI [-0.69%, -0.18%]; n=11,920 events in 286 months (effective 164); median -0.12%; share positive 48.9% [48.0%, 49.8%]
- Test period 2002-01-03 to 2025-10-01, 48 splits; thresholds chosen: z-4.0: 29, z-3.0: 11, z-2.5: 8.

<details><summary>Walk-forward choices per test window</summary>

| Test start | Test end | Chosen | Fit n | Fit t-score | Test n |
|---|---|---|---|---|---|
| 2002-01-03 | 2002-07-03 | z-2.5 | 1,013 | 0.17 | 287 |
| 2002-07-03 | 2003-01-03 | z-2.5 | 967 | -0.02 | 365 |
| 2003-01-03 | 2003-07-03 | z-2.5 | 1,133 | -0.21 | 214 |
| 2003-07-03 | 2004-01-03 | z-2.5 | 1,130 | 0.01 | 211 |
| 2004-01-03 | 2004-07-03 | z-4.0 | 335 | 0.55 | 53 |
| 2004-07-03 | 2005-01-03 | z-3.0 | 632 | 0.94 | 174 |
| 2005-01-03 | 2005-07-03 | z-2.5 | 896 | 1.30 | 238 |
| 2005-07-03 | 2006-01-03 | z-4.0 | 271 | 0.67 | 86 |
| 2006-01-03 | 2006-07-03 | z-4.0 | 301 | 0.88 | 84 |
| 2006-07-03 | 2007-01-03 | z-2.5 | 1,044 | 1.16 | 282 |
| 2007-01-03 | 2007-07-03 | z-4.0 | 335 | -0.14 | 83 |
| 2007-07-03 | 2008-01-03 | z-4.0 | 336 | -1.71 | 157 |
| 2008-01-03 | 2008-07-03 | z-4.0 | 405 | -2.20 | 124 |
| 2008-07-03 | 2009-01-03 | z-4.0 | 436 | -0.89 | 289 |
| 2009-01-03 | 2009-07-03 | z-4.0 | 647 | -1.00 | 61 |
| 2009-07-03 | 2010-01-03 | z-4.0 | 630 | -0.70 | 74 |
| 2010-01-03 | 2010-07-03 | z-4.0 | 541 | -0.84 | 107 |
| 2010-07-03 | 2011-01-03 | z-4.0 | 524 | -1.30 | 139 |
| 2011-01-03 | 2011-07-03 | z-3.0 | 780 | -1.62 | 260 |
| 2011-07-03 | 2012-01-03 | z-2.5 | 1,361 | -1.99 | 526 |
| 2012-01-03 | 2012-07-03 | z-2.5 | 1,678 | -1.50 | 427 |
| 2012-07-03 | 2013-01-03 | z-4.0 | 580 | -1.66 | 162 |
| 2013-01-03 | 2013-07-03 | z-4.0 | 609 | -1.44 | 150 |
| 2013-07-03 | 2014-01-03 | z-4.0 | 616 | -1.21 | 136 |
| 2014-01-03 | 2014-07-03 | z-4.0 | 571 | -0.66 | 141 |
| 2014-07-03 | 2015-01-03 | z-4.0 | 577 | -1.34 | 203 |
| 2015-01-03 | 2015-07-03 | z-4.0 | 619 | -0.73 | 151 |
| 2015-07-03 | 2016-01-03 | z-4.0 | 620 | -0.52 | 232 |
| 2016-01-03 | 2016-07-03 | z-4.0 | 714 | -0.00 | 187 |
| 2016-07-03 | 2017-01-03 | z-4.0 | 745 | 1.14 | 260 |
| 2017-01-03 | 2017-07-03 | z-3.0 | 1,467 | 0.23 | 306 |
| 2017-07-03 | 2018-01-03 | z-3.0 | 1,483 | -0.43 | 447 |
| 2018-01-03 | 2018-07-03 | z-3.0 | 1,512 | -1.00 | 415 |
| 2018-07-03 | 2019-01-03 | z-4.0 | 899 | -2.42 | 279 |
| 2019-01-03 | 2019-07-03 | z-3.0 | 1,635 | -1.04 | 361 |
| 2019-07-03 | 2020-01-03 | z-4.0 | 926 | -0.25 | 220 |
| 2020-01-03 | 2020-07-03 | z-3.0 | 1,679 | -0.87 | 679 |
| 2020-07-03 | 2021-01-03 | z-4.0 | 1,150 | -4.94 | 165 |
| 2021-01-03 | 2021-07-03 | z-4.0 | 1,033 | -5.22 | 125 |
| 2021-07-03 | 2022-01-03 | z-4.0 | 973 | -5.55 | 237 |
| 2022-01-03 | 2022-07-03 | z-4.0 | 989 | -5.47 | 248 |
| 2022-07-03 | 2023-01-03 | z-4.0 | 761 | -1.44 | 247 |
| 2023-01-03 | 2023-07-03 | z-4.0 | 849 | -0.46 | 225 |
| 2023-07-03 | 2024-01-03 | z-4.0 | 941 | -1.32 | 275 |
| 2024-01-03 | 2024-07-03 | z-3.0 | 1,808 | -0.65 | 458 |
| 2024-07-03 | 2025-01-03 | z-3.0 | 1,751 | -1.91 | 516 |
| 2025-01-03 | 2025-07-03 | z-3.0 | 1,857 | -4.21 | 594 |
| 2025-07-03 | 2025-10-23 | z-3.0 | 2,010 | -1.95 | 260 |

</details>

## Full-window result: pre-registered single test (no parameters fit)

Every event in the development window at the pre-registered threshold and horizon; nothing was chosen. Only the 15-day net abnormal return decides; other horizons and raw returns are secondary.

- **Net abnormal 15d return:** mean -0.39%, 95% CI [-0.63%, -0.20%]; n=16,055 events in 310 months (effective 204); median -0.09%; share positive 49.2% [48.4%, 50.0%]
- Before costs (abnormal 15d): mean -0.29%, 95% CI [-0.53%, -0.10%]; n=16,055 events in 310 months (effective 204); median +0.01%; share positive 50.1% [49.3%, 50.8%]

## Placebo

Same tickers, one random draw per event from member days with finite model inputs and the same price filter, excluding each real event day and the days inside its own forward window (past events only, so the placebo never conditions on its own future), run through the identical return code.

- **Placebo abnormal 15d return (before the fixed cost):** mean -0.15%, 95% CI [-0.27%, -0.04%]; n=15,887 events in 310 months (effective 277); median +0.08%; share positive 51.0% [50.2%, 51.7%]
- Placebo net of costs: mean -0.25%, 95% CI [-0.37%, -0.14%]; n=15,887 events in 310 months (effective 277); median -0.02%; share positive 49.7% [48.9%, 50.5%]
- **Events minus placebo (criterion 1b):** -0.14%, 95% CI [-0.37%, +0.06%]; events n=16,055 (effective months 204), placebo n=15,887 (effective months 277).
- Diagnostic only (not a criterion): placebo minus 15 x each ticker's mean daily abnormal return over the window: mean -0.01%, 95% CI [-0.12%, +0.11%]; n=15,887 events in 310 months (effective 277); median +0.16%; share positive 51.6% [50.8%, 52.3%]. Near zero here while the raw placebo is not points to drift shared by every day of these tickers (e.g. survivors), not to lookahead.

A placebo mean away from zero means a pipeline bug or a drift shared by every day (e.g. survivors); the pre-registration treats the run as invalid until the cause is found. The diagnostic above does not restore validity; only an amendment written before any result could.

## Calendar-time portfolio

Equal weight across open positions each day, flat (0) on days without one, costs charged on entry and exit days.

- Mean daily return -3.03 bp (Newey-West se +0.60 bp, t -5.03, 15 lags), n=6,491 trading days, 6,489 with positions (mean 37.1 open).
- Annualized Sharpe **-0.97**.

## Stability by sub-period

| Period | Net abnormal mean [95% CI] (n) | Effective months |
|---|---|---|
| 2000-2008 | -0.41% [-0.79%, -0.03%] (n=3,546) | 68 |
| 2009-2016 | -0.22% [-0.42%, +0.01%] (n=4,622) | 76 |
| 2017-2025 | -0.49% [-0.89%, -0.14%] (n=7,887) | 78 |

## Secondary analyses (reported, not decision-driving)

### Other horizons and raw returns

Mean [95% CI] (n); raw = unhedged stock return, abnormal = factor-hedged. Only the 15d net abnormal cell repeats the pre-registered test.

| Horizon | Raw | Abnormal | Net abnormal |
|---|---|---|---|
| 1d | -0.12% [-0.18%, -0.06%] (n=16,105) | -0.07% [-0.12%, -0.02%] (n=16,105) | -0.17% [-0.22%, -0.12%] (n=16,105) |
| 5d | -0.15% [-0.67%, +0.22%] (n=16,096) | -0.19% [-0.37%, -0.06%] (n=16,096) | -0.29% [-0.47%, -0.16%] (n=16,096) |
| 15d | +0.17% [-0.62%, +0.86%] (n=16,055) | -0.29% [-0.53%, -0.10%] (n=16,055) | -0.39% [-0.63%, -0.20%] (n=16,055) |
| 30d | +0.61% [-0.54%, +1.59%] (n=16,000) | -0.42% [-0.71%, -0.15%] (n=16,000) | -0.52% [-0.81%, -0.25%] (n=16,000) |

### Entry at the event-day close (how much is untradeable bounce)

| Horizon | Net abnormal, close entry |
|---|---|
| 1d | -0.18% [-0.23%, -0.13%] (n=16,776) |
| 5d | -0.30% [-0.47%, -0.17%] (n=16,767) |
| 15d | -0.40% [-0.63%, -0.21%] (n=16,726) |
| 30d | -0.54% [-0.82%, -0.27%] (n=16,669) |

Close entry minus next-open entry, per event (15d): mean +0.00%, 95% CI [-0.03%, +0.03%]; n=16,055 events in 310 months (effective 204); median +0.02%; share positive 51.3% [50.5%, 52.1%]

### Up-moves (z >= +3), long-side returns

An overreaction story predicts a negative mean here (gains that fade).

| Horizon | Abnormal | Net abnormal (long) |
|---|---|---|
| 1d | -0.00% [-0.05%, +0.04%] (n=15,667) | -0.10% [-0.15%, -0.06%] (n=15,667) |
| 5d | -0.15% [-0.28%, -0.04%] (n=15,650) | -0.25% [-0.38%, -0.14%] (n=15,650) |
| 15d | -0.39% [-0.65%, -0.16%] (n=15,626) | -0.49% [-0.75%, -0.26%] (n=15,626) |
| 30d | -0.50% [-0.80%, -0.21%] (n=15,584) | -0.60% [-0.90%, -0.31%] (n=15,584) |

### Threshold grid: in-sample, descriptive only

Shown for context. Picking the best row here would be in-sample selection; the walk-forward section is the only place a threshold is chosen.

| Threshold | Net abnormal 15d |
|---|---|
| z <= -2.5 | -0.36% [-0.53%, -0.20%] (n=24,128) |
| z <= -3.0 | -0.39% [-0.63%, -0.20%] (n=16,055) |
| z <= -4.0 | -0.48% [-0.80%, -0.22%] (n=8,583) |

### Confirming signals, one at a time

Events split into expanding-window terciles (cut points from earlier dates only; binary flags split False/True). Difference = top minus bottom group mean net abnormal 15d return, month-clustered bootstrap CI, Benjamini-Hochberg across the battery. A low-confidence row is never a discovery.

| Signal | Thesis predicts | n | Top - bottom [95% CI] | p | BH p | Discovery | Low-conf. |
|---|---|---|---|---|---|---|---|
| vol_z | + (high-volume drops overreact more) | 15,830 | -0.14% [-0.53%, +0.25%] | 0.496 | 0.496 | no | no |
| sec_ret | - (core bet: drops on days the sector fell hardest revert more) | 15,832 | +0.53% [+0.04%, +1.08%] | 0.052 | 0.156 | no | no |
| mkt_ret | - (drops on market-wide selloff days revert more) | 15,832 | +0.37% [-0.14%, +0.97%] | 0.193 | 0.288 | no | no |
| peer_z | - (drops shared with sector peers (low peer z) revert more) | 15,832 | -0.35% [-0.65%, -0.06%] | 0.020 | 0.120 | no | no |
| abs_z | + (larger unexplained drops revert more) | 15,832 | -0.21% [-0.52%, +0.12%] | 0.200 | 0.288 | no | no |
| no_8k | + (drops with no company 8-K (price/news mismatch) revert more) | 16,055 | +0.17% [-0.13%, +0.44%] | 0.240 | 0.288 | no | no |

SEC 8-K news proxy: ok; 100.0% of events covered, 44.5% of covered events had an 8-K effective on the event day or the 1 session(s) before; 0 of 596 tickers have no CIK in SEC's current map

### Stale-price windows (diagnostic)

Rule: share of exactly-zero returns among finite returns in the 90-day fit window [t-90, t-1]. Halted or illiquid stretches make stock returns exactly zero, deflating the rolling sigma so z = resid/sigma can spike and manufacture a fake event. Diagnostic only: no criterion or event selection uses it.

- Mean flat share over 16,782 events: 0.8%; events above 50%: 1; above 90%: 0.

| Date | Ticker | Flat share | z |
|---|---|---|---|
| 2023-06-14 | PARA | 53.3% | -4.92 |
| 2019-07-22 | AMCR | 50.0% | -3.71 |
| 2023-05-08 | PARA | 50.0% | -4.39 |
| 2022-12-14 | PARA | 50.0% | -3.01 |
| 2022-06-22 | PARA | 45.6% | -6.38 |

## Data coverage and survivorship

Free price data covers **47.0%** of S&P 500 member-days in 2000, **96.9%** in 2025 and 70.2% over the development window (worst year 47.0%). Up to 302 members a year have no price at all. Missing names are mostly delisted, acquired or failed companies, so the sample over-represents survivors; this favours H1, which makes a PASS an upper bound and a FAIL robust.

<details><summary>Coverage by year</summary>

| Year | Member-days | Covered | Coverage % | Members | Members with no data |
|---|---|---|---|---|---|
| 1998 | 62,366 | 27,504 | 44.1 | 512 | 291 |
| 1999 | 123,695 | 56,179 | 45.4 | 530 | 298 |
| 2000 | 124,001 | 58,241 | 47.0 | 545 | 302 |
| 2001 | 122,786 | 59,796 | 48.7 | 522 | 277 |
| 2002 | 124,665 | 62,117 | 49.8 | 516 | 265 |
| 2003 | 124,487 | 63,822 | 51.3 | 503 | 247 |
| 2004 | 124,677 | 64,883 | 52.0 | 513 | 253 |
| 2005 | 124,961 | 66,040 | 52.8 | 511 | 241 |
| 2006 | 124,746 | 69,071 | 55.4 | 527 | 245 |
| 2007 | 124,744 | 71,852 | 57.6 | 533 | 235 |
| 2008 | 125,841 | 75,651 | 60.1 | 531 | 216 |
| 2009 | 125,690 | 78,423 | 62.4 | 524 | 202 |
| 2010 | 125,537 | 81,127 | 64.6 | 513 | 188 |
| 2011 | 125,242 | 82,606 | 66.0 | 515 | 179 |
| 2012 | 124,242 | 84,881 | 68.3 | 514 | 167 |
| 2013 | 125,241 | 88,233 | 70.5 | 515 | 156 |
| 2014 | 125,534 | 89,770 | 71.5 | 513 | 150 |
| 2015 | 126,147 | 92,440 | 73.3 | 529 | 150 |
| 2016 | 127,356 | 97,261 | 76.4 | 533 | 134 |
| 2017 | 126,995 | 100,525 | 79.2 | 533 | 119 |
| 2018 | 126,917 | 102,684 | 80.9 | 529 | 107 |
| 2019 | 127,260 | 106,508 | 83.7 | 533 | 90 |
| 2020 | 127,767 | 111,358 | 87.2 | 525 | 72 |
| 2021 | 127,260 | 113,373 | 89.1 | 526 | 63 |
| 2022 | 126,437 | 115,988 | 91.7 | 527 | 52 |
| 2023 | 125,750 | 118,483 | 94.2 | 522 | 34 |
| 2024 | 126,757 | 121,319 | 95.7 | 523 | 26 |
| 2025 | 101,606 | 98,475 | 96.9 | 516 | 18 |

</details>

Events whose stock price disappears before the exit are excluded, not counted as losses. A higher share than on placebo days is a survivorship warning sign. Events lost to a sector-ETF or SPY price gap (or undefined betas) are excluded and counted separately.

| Horizon | Events | Missing stock price | Lost to ETF/beta gap | Embargoed / beyond data | Placebo missing stock price |
|---|---|---|---|---|---|
| 1d | 16,782 | 0 (0.0%) | 0 | 6 | 0.0% |
| 5d | 16,782 | 0 (0.0%) | 0 | 15 | 0.0% |
| 15d | 16,782 | 0 (0.0%) | 0 | 56 | 0.0% |
| 30d | 16,782 | 0 (0.0%) | 0 | 113 | 0.0% |

### Entry opens that were not real trades

Rule: volume[t+1] == 0 or as-traded open[t+1] == close[t]. Such an open is a free-data artifact (often the prior close copied forward), so using it would quietly turn the next-open entry into a close entry. These entries are treated as missing for events and placebo alike.

- Primary events affected: 671 of 16,782 (4.0%); placebo days: 843 (5.0%). By year (stale/events): 2000: 38/356, 2001: 43/343, 2002: 64/491, 2003: 49/307, 2004: 47/353, 2005: 48/362, 2006: 35/394, 2007: 28/518, 2008: 17/791, 2009: 6/295, 2010: 13/518, 2011: 17/629, 2012: 9/565, 2013: 13/554, 2014: 15/651, 2015: 22/722, 2016: 27/810, 2017: 46/794, 2018: 18/953, 2019: 17/781, 2020: 13/1005, 2021: 10/768, 2022: 19/983, 2023: 26/919, 2024: 14/992, 2025: 17/928.
- Sensitivity, stale opens used as fills (literal reading, not decision-driving): mean -0.40%, 95% CI [-0.63%, -0.21%]; n=16,726 events in 310 months (effective 209); median -0.09%; share positive 49.2% [48.5%, 50.0%]

## What would have to be true for this to be a fluke

- **Survivorship.** Coverage is 47% of member-days in 2000 and 70% over the window. A positive result is a fluke if the uncovered member-days hide drops that never recovered; the placebo comparison removes drift shared by all survivor days, not selection that is specific to post-drop names.
- **Disappearing exits.** 0.0% of events lose their price before the 15-day exit versus 0.0% of placebo days; if those were mostly collapses, the reported mean is too high.
- **Few episodes.** The events fall in 310 calendar months (effective 204); the 5 busiest months hold 6% of them (2020-03: 393, 2025-04: 184, 2008-10: 182, 2018-10: 160, 2020-11: 138). If the mean comes from a handful of crash months, it describes those episodes, not a repeatable pattern.
- **Costs.** The mean is already -0.29% before costs, so cost assumptions cannot explain a negative result.
- **Fake opens.** 4.0% of events had a t+1 open that was not a real trade and were dropped; using them as fills gives -0.40% instead of -0.39%. A large gap means early-year data quality, not the effect, drives part of the result.
- **Untradeable bounce.** Entering at the event-day close instead of the next open changes the mean by +0.00% [-0.03%, +0.03%]; that part could not have been captured.
- **Selection.** Out-of-sample (walk-forward) mean -0.42% [-0.69%, -0.18%] versus the full-window -0.39% [-0.63%, -0.20%]. An out-of-sample mean well below the full-window one would mean the threshold choice, not the effect, drives the in-sample number.
- **Chance.** This report computes about 42 summary statistics; with no effect anywhere roughly 2.1 would exclude zero at 95% by luck. Only the pre-registered criteria decide, and the factor battery is BH-corrected.
- **Pipeline.** Random member days give -0.15% [-0.27%, -0.04%] before costs. A non-zero placebo would mean lookahead or a shared drift that inflates the event mean too.
- **For a FAIL to be a fluke** (a real effect missed): a positive true mean would have to lie above the whole 95% CI [-0.63%, -0.20%]. Chance alone puts it there about 2.5% of the time, unless the CI is too narrow (few independent months, or dependence the month clusters miss).

## Parameters

34 registered parameters, 0 fit to data. Full list with provenance: docs/PARAMETERS.md.

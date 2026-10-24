# Model v3 design — general options scanner

Drafted 2026-10-24. Replaces the crash-only scope. Goal: every trading day, scan every liquid option on every
S&P 500 stock (all expiries) and rank the contracts with the best expected value per dollar at risk, showing
probability of profit, after vetting the company, its events, the economy and investor positioning.
Free data only. A human places every trade; the system never trades.

## Principles

1. **Two different questions, kept separate.** (a) *What will the stock do?* — a forecast distribution of
   the price at expiry. (b) *Is this contract mispriced against that forecast?* — pure math (Layer 5).
   Every factor below feeds (a) or vetoes a trade; none touches (b).
2. **Factors earn their place.** Each factor is tested on its own (does it improve the forecast out of
   sample?). Untested factors may veto or widen uncertainty, but never shift the forecast's direction.
3. **Point-in-time only.** Every input is what was public at the time (filing dates, release dates).
4. **Uncertainty is priced in.** Less confidence → wider distribution → lower EV automatically, plus a
   higher bar for far out-of-the-money contracts.

## The forecast engine (what the stock will do)

Price at expiry = today's price × exp(drift + volatility × shape + event jumps), simulated 20,000 times:

| Piece | Method | Source |
|---|---|---|
| Base volatility | HAR model: blend of 1-, 5-, 22-, 66-day realized volatility (strong, well-documented forecaster) | prices |
| Return shape (fat tails, skew) | Filtered historical simulation: the stock's own past standardized daily returns, block-bootstrapped | prices |
| Earnings jump | If earnings fall before expiry: add a draw from this stock's own past earnings-day moves | Nasdaq calendar + SEC 8-K 2.02 history |
| Macro event days | FOMC / CPI / jobs days inside the window add that stock's average event-day variance | Fed calendar + Nasdaq economic calendar |
| Regime scaling | Volatility scaled by VIX level/term structure, credit and correlation state | Layer 1 |
| Drift | Zero by default; only factors that pass validation may tilt it, with their measured size | factor tests |

## Factor families (what can tilt, widen or veto)

### A. Company quality — "is this company worth touching?" (SEC XBRL, point-in-time)
Profitability (gross/operating margin, return on capital), growth and acceleration, margin trend, balance
sheet (net debt/EBITDA, interest cover, current ratio, cash runway for loss-makers), earnings quality
(operating cash flow vs net income, accruals), share dilution, Piotroski F-score, Altman Z distress score.
**Use:** low quality/distress vetoes bullish trades; strong quality vetoes bearish ones unless other
evidence overrides. Tested as predictors of 1–6 month returns and of crash risk.

### B. Valuation — "is it cheap for what it is?"
P/S, EV/EBITDA, P/E, free-cash-flow yield vs sector peers and vs its own 3-year history, combined with
quality (cheap *and* healthy, not just cheap).

### C. Company events — "what happens before expiry?"
Earnings date (Nasdaq calendar, verified free), this stock's historical earnings-move size vs the move the
options imply (front-vs-back implied vol), dividends/ex-dates (early-exercise risk on short calls), material
8-Ks (M&A, guidance, executive changes, litigation). **Use:** earnings inside expiry is either avoided or
traded deliberately, never ignored.

### D. Economy events and macro
Fed meetings (Fed site, verified), CPI/jobs/GDP release days (Nasdaq economic calendar), rates level and
trend, 2s10s curve, credit spreads, dollar, oil, VIX term structure, breadth, cross-stock correlation,
geopolitical tension index (Layer 1), sector rotation (sector ETF relative strength).

### E. Investor positioning and sentiment — "what does the crowd think?"
- From our own option snapshots: put/call volume and open-interest ratios, skew (cost of crash protection),
  implied-vol rank, term-structure slope, unusual volume vs open interest, implied vs realized vol.
- Short interest and days-to-cover (FINRA API, verified free, history back to ~2018).
- Insider buying/selling (SEC Form 4).
- Attention: Wikipedia page views (free, documented attention proxy).
- Optional, costs money: news and earnings-call tone via an LLM (classification only, never numbers).

### F. Price behaviour (documented market anomalies)
12-1 month momentum, 1-month reversal, distance from 52-week high, residual (stock-specific) momentum,
idiosyncratic volatility, post-earnings drift, the post-shock drift already confirmed (H2a).

### G. Sector and peers
Sector momentum, peer moves on the same day, sector valuation, the Layer 2 "sector vs company" split.

## From forecast to picks (Layer 5, built)

For every liquid contract and every defined-risk structure (long calls/puts, debit and credit verticals,
iron condors for non-directional premium selling): expected value under the forecast at real bid/ask with
commissions, max loss, probability of profit, delta bucket. Ranking = EV per dollar at risk, after vetoes,
with a higher bar for the 0.05–0.20 delta bucket. Portfolio check: no stacking of correlated bets
(e.g. five bearish tech trades); position size capped as a fraction of a stated account.

## Validation plan

| Stage | What is tested | Data | When |
|---|---|---|---|
| 1 | Each factor alone: does it predict the stock's return, volatility or crash risk out of sample? | 25 years of free prices + SEC filings (+ FINRA from 2018) | now |
| 2 | Volatility forecast vs realized; vs implied vol where snapshots exist | prices; snapshots | now / forward |
| 3 | Full scanner picks, fixed rules, real bid/ask | our own daily snapshots | forward, first look after ≥ 6 months |

Each stage gets a dated pre-registration before its first result. Factors that fail Stage 1 are dropped from
the drift and kept only as vetoes or uncertainty wideners if justified by theory.

## Honest limits

- Options are, on average, slightly overpriced; most real edge is expected in **selling** defined-risk premium
  where implied volatility exceeds the forecast. Truly cheap options to buy will be rare.
- "Highest probability" and "most undervalued" trade off; the ranking optimises EV per dollar at risk and
  shows probability alongside.
- Free data has no historical option prices, so Stage 3 cannot be proven before saved chains accumulate.
- Many factors = overfitting risk. The validation stages exist to stop that; do not skip them.

## Run schedule

One local run before the US open (~14:45 Berlin): save the previous session's chains, refresh data, score,
write the shortlist and dashboard data. The PC only needs to be on during the day.

# Stage 1 — standalone factor tests

Pre-registration `docs/PREREGISTRATION_STAGE1.md` (sha256 `eb9f5661809dbf1f0c9774407d3c75e4745a8a480102fcb0de6f1274df63f1ae`), run 2026-10-24T14:39:50+00:00.
Development formations 2000-01 → 2025-08; holdout 2025-10 → 2026-07 (direction only). Survivor-biased free prices: return factors that favour weak firms are understated.

| Factor | Target | Pred. | Mean IC | NW t | BH p | Q5−Q1 | Sub-periods right | Holdout IC | Months | Pass |
|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | ret21 | + | -0.0043 | -0.35 | 0.843 | +0.0011 | 1/3 | -0.0015 | 308 | fail |
| rev_1m | ret21 | − | -0.0167 | -1.98 | 0.252 | -0.0007 | 3/3 | +0.0242 | 308 | fail |
| high52 | ret21 | + | -0.0131 | -1.02 | 0.776 | +0.0014 | 0/3 | -0.0251 | 308 | fail |
| resid_mom | ret21 | + | -0.0034 | -0.57 | 0.776 | +0.0001 | 1/3 | +0.0534 | 308 | fail |
| idio_vol | ret21 | − | +0.0023 | +0.20 | 0.853 | -0.0043 | 2/3 | -0.0237 | 308 | fail |
| idio_vol_v | vol21 | + | +0.5794 | +69.67 | 0.000 | +0.2130 | 3/3 | +0.6529 | 308 | **PASS** |
| max_ret | ret21 | − | -0.0020 | -0.19 | 0.853 | -0.0037 | 1/3 | +0.0117 | 308 | fail |
| beta | ret21 | − | -0.0017 | -0.28 | 0.853 | -0.0006 | 1/3 | -0.0277 | 308 | fail |
| pead | ret21 | + | -0.0028 | -0.68 | 0.776 | -0.0007 | 1/3 | -0.0240 | 265 | fail |
| gross_prof | ret21 | + | +0.0063 | +0.55 | 0.776 | -0.0004 | 1/3 | -0.0394 | 184 | fail |
| accruals | ret21 | − | -0.0038 | -0.59 | 0.776 | -0.0008 | 1/3 | -0.0088 | 185 | fail |
| asset_growth | ret21 | − | +0.0068 | +0.84 | 0.776 | +0.0010 | 1/3 | -0.0231 | 182 | fail |
| dilution | ret21 | − | -0.0129 | -1.49 | 0.509 | -0.0014 | 2/3 | -0.0383 | 182 | fail |
| earn_yield | ret21 | + | +0.0054 | +0.54 | 0.776 | +0.0001 | 2/3 | +0.0699 | 185 | fail |
| sales_yield | ret21 | + | +0.0048 | +0.36 | 0.843 | +0.0003 | 1/3 | +0.0631 | 185 | fail |
| f_score | ret21 | + | -0.0034 | -0.55 | 0.776 | -0.0009 | 1/3 | -0.0217 | 181 | fail |
| leverage_crash | crash63 | + | -0.0197 | -3.23 | 0.009 | -0.0102 | 0/3 | -0.0481 | 194 | fail |
| zscore_crash | crash63 | − | -0.0176 | -1.31 | 0.568 | -0.0070 | 2/3 | +0.1051 | 184 | fail |
| days_to_cover | ret21 | − | -0.0127 | -1.46 | 0.509 | -0.0038 | 1/2 | +0.0011 | 92 | fail |
| short_pct | ret21 | − | -0.0119 | -0.86 | 0.776 | -0.0042 | 1/2 | +0.0017 | 92 | fail |
| short_pct_crash | crash63 | + | +0.1035 | +13.09 | 0.000 | +0.0577 | 2/2 | +0.0728 | 92 | **PASS** |

## Macro (SPY next-month realized vol)

| Predictor | Pred. | Coef | NW t | BH p | Sub-periods right | Holdout coef | Pass |
|---|---|---|---|---|---|---|---|
| vix | + | +0.0094 | +8.73 | 0.000 | 3/3 | +0.0013 | **PASS** |
| vix_term | + | +0.7564 | +3.58 | 0.001 | 2/2 | +0.1223 | **PASS** |
| credit_20d | − | -1.1168 | -1.80 | 0.072 | 3/3 | -3.1879 | fail |

## Run notes

- First run had a bug in `f_score`: missing SEC data was scored as 0 instead of missing (308 months instead
  of ~181). Fixed in `factors/fundamental.py` and the whole stage rerun; only `f_score` changed
  (IC −0.0049 → −0.0034, still fail). No other code or parameter was changed after seeing results.
- Fundamental factors start ~2009 (XBRL era); FINRA factors start 2018.
- Survivor bias: delisted firms are missing, which hides crashes of weak/levered firms. This likely explains
  `leverage_crash` coming out with the wrong sign and understates distress factors.

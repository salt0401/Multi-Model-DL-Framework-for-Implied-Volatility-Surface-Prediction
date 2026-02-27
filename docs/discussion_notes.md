# SSVI Model Discussion Notes

## 3. Open / Unresolved Issues

### 3.2 2022-2026 data quality issues (prs_dataset_full)

The `prs_dataset_full` dataset extends from 2014-2021 to 2014-2026. The 2014-2021 portion is bit-for-bit identical to `prs_dataset_no_fat(clean)`. The new 2022-2026 data has several quality concerns:

1. **tau distribution shift**: Median tau dropped from 0.156 to 0.071. Sub-7-day options went from 5.2% to 22.3% of data (weekly/0DTE options emerged).
2. **Expiry dates explosion**: From ~17 per year (monthly + quarterly) to 58-86 per year (weeklies).
3. **Risk-free rate hardcoded**: All 226,150 rows in 2022-2026 have `r = 0.015`, while older data has 6 different values varying by year.
4. **Extreme IV values**: 13 rows with IV > 100% (10 from 2024/8/5 carry-trade crash, 3 from 2025/4 tariff shock).
5. **Near-zero total_var**: 2 rows with unreasonably low IV for 1-day expiry.
6. **Price precision difference**: Older data has 68.7% of prices with >5 decimal places (includes theoretical prices), new data has all prices rounded to <=1 decimal (pure market quotes).
7. **Underlying price format**: Older data uses integers, newer data uses floats.

**Impact**: These differences mean training on the full dataset would require:

- Proper time-varying risk-free rate handling
- Short-expiry option filtering or special treatment
- Extreme value handling (IV > 100% clipping or removal)
- Possible domain adaptation for the different data regimes

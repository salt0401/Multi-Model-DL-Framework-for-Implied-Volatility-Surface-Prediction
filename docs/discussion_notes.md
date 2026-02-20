# SSVI Model Discussion Notes

> Last updated: 2026-02-20

---

## 1. Resolved Issues

### 1.2 eta/gamma bounded parameterization
- **Problem**: `eta = exp(raw_eta)` caused eta explosion during training (1.0 -> 31.18 in 10 epochs).
- **Fix**: `eta = 2 * sigmoid(raw_eta)` in (0, 2), `gamma = sigmoid(raw_gamma)` in (0, 1).
- **File**: `src/model.py` lines 40-41
- **Reasoning**: Gatheral & Jacquier (2014) no-butterfly-arbitrage condition requires `eta * (1 + |rho|) < 2`. Bounded parameterization prevents violation.
- **Note**: The joint constraint `eta * (1 + |rho|) < 2` is not strictly enforced yet. Currently eta < 2 independently and rho < 0 independently. This is sufficient for now but could be tightened.

### 1.5 config.ini dataset reversion (commit 262471d)
- **Problem**: Commit `262471d` changed dataset from `prs_dataset_no_fat(clean)` to `prs_dataset_full` and shifted date ranges to 2022-2026, causing IV smile plots to look abnormal.
- **Fix**: Reverted config.ini to original settings:
  - `prs_dataset = prs_dataset_no_fat(clean)`
  - `train_end_date = 20201231`
  - `test_start_date = 20210101`, `test_end_date = 20211231`
- **Note**: `condition_dim` remains at 13 (diffusion model setting, does not affect base model).

### 1.6 Prediction zigzag pattern (was §3.1)
- **Observation**: Predicted IV lines in regime-colored plots showed extreme zigzag/sawtooth patterns.
- **Root cause**: **Plotting artifact**, not overfitting. The plotting scripts binned yATM into 3 coarse regime categories, then connected predictions sorted by logm as a line. Within each bin, yATM still varied significantly (e.g., 0.0003 to 0.003 in "Low Vol"), causing the model's correct yATM-dependent predictions to zigzag when connected.
- **Verification** (`scripts/plot_smooth_iv_check.py`, 5 epochs):
  1. **Fixed yATM synthetic test**: Locking yATM to exact values and sweeping logm produced smooth V-shaped curves — model learned a proper surface.
  2. **Narrow yATM band test**: Filtering real data to tight yATM ranges dramatically reduced jaggedness.
- **Conclusion**: The model correctly learns `f(tau, logm, yATM)` as a smooth 3D surface. Future plots must use fixed yATM values (one curve per yATM level) instead of mixing different yATM values in the same line.
- **Deleted files**: Old zigzag plotting scripts (`plot_iv_smiles.py`, `plot_regime_predictions.py`) and all debug PNG images removed during 2026-02-20 cleanup.

---

## 2. Explained Phenomena

### 2.1 IV smile band structure (multiple parallel bands in observed data)
- **Observation**: Validation data (2019.8 - 2020.12) shows 3-4 parallel bands of IV values at the same tau.
- **Explanation**: These correspond to different volatility regimes within the validation period:
  - Very Calm (2019.8-9): IV ~ 0.15
  - Calm (2019.11-12): IV ~ 0.16
  - Normal (2020.2-3 pre-COVID): IV ~ 0.19
  - Elevated (2020.5-9 post-crash): IV ~ 0.27
  - Crisis (2020.5 peak): IV ~ 0.28
- **Conclusion**: This is normal market behavior, not a data quality issue. Per-date y_atm (fix 1.4) provides the model with regime information.
- **Visualizations**: (debug images deleted 2026-02-20; regenerate with `scripts/plot_smooth_iv_check.py`)

---

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

### 3.3 Gatheral-Jacquier joint constraint
- Current: `eta < 2` and `rho < 0` independently.
- Ideal: `eta * (1 + |rho|) < 2` jointly enforced.
- Status: Not yet implemented. Current bounds are sufficient in practice but not theoretically tight.

### 3.5 Additive vs multiplicative architecture (pending decision)
- **Current state**: `output = SSVI_prior(logm, yATM) + yATM * SmileNN(tau, logm)` (additive, in uncommitted changes).
- **Original**: `output = SSVI_prior(logm, yATM) * SmileNN(tau, logm)` (multiplicative).
- **Additive pros**: Derivatives have no cross-terms (simpler gradient flow for butterfly/calendar losses). yATM scaling keeps NN correction proportional to vol level. SSVI prior always present as baseline.
- **Multiplicative pros**: NN can scale the entire SSVI shape (e.g., flatten or steepen). May better capture situations where the correction is proportional to SSVI output rather than yATM alone.
- **Question**: With rho/eta/butterfly fixes already applied, is the multiplicative architecture now stable enough? Or does the additive form's simpler gradient flow provide meaningful training benefits?
- **Status**: Pending user decision. Both architectures are valid; choice depends on empirical performance comparison.

### 3.6 Model training duration
- Only 10 epochs have been tested so far.
- The model captures ~79% of regime spread at 10 epochs.
- Longer training (50-100+ epochs) has not been tested after all the fixes.
- Status: Pending user decision.

---

## 4. Key Files Reference

| File | Purpose |
|------|---------|
| `src/model.py` | Model architecture (SSVI, SmileNN, MultiModel, losses) |
| `src/dataset.py` | Data processing, per-date y_atm computation |
| `src/config.ini` | Training configuration |
| `src/train.py` | Training loop |
| `src/utils.py` | Config loading, seed setting utilities |
| `scripts/plot_smooth_iv_check.py` | Fixed-yATM smooth surface verification |
| `scripts/inspect_ssvi_params.py` | SSVI parameter inspection |
| `scripts/diagnose_rho_gradient.py` | Per-loss rho gradient analysis (§3.3) |
| `scripts/train_diagnose.py` | Training with per-epoch parameter tracking |
| `scripts/plot_training_curves.py` | Training loss curve visualization |

## 5. Cleanup Log

| Date | Action |
|------|--------|
| 2026-02-20 | Deleted 9 debug PNG images from project root (zigzag investigation artifacts) |
| 2026-02-20 | Deleted `scripts/plot_iv_smiles.py`, `scripts/plot_regime_predictions.py` (superseded by `plot_smooth_iv_check.py`) |
| 2026-02-20 | Moved §3.1 zigzag issue to §1.6 Resolved |

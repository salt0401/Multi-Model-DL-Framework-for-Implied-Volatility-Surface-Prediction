# SSVI Model Discussion Notes


---

### ~~1.7 Model 3 architecture comparison~~ → Resolved (2026-02-21)
- **Problem**: GRU baseline may not be optimal for seq_len=20 crisis adjustment task
- **Research**: Evaluated 7+ architectures (TFT, xLSTM, Mamba, Neural SDE, PatchTST, iTransformer, Foundation Models). Selected TFT and xLSTM for implementation.
- **Training**: All three models trained on identical data (245K sequences, chronological split, GPU float64)
- **Result**: xLSTM (mLSTM) wins — 4.27% RMSE improvement over GRU, fewest parameters (39K). TFT also beats baseline (+1.7%) with excellent interpretability.
- **Files**: `model3_research/xlstm_adjustment.py`, `model3_research/tft_adjustment.py`, `model3_research/train_models.py`
- **Full analysis**: `model3_research/README.md`, `model3_research/full_research_report.md`

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

### 3.7 Model 3 overfitting (train-val gap 2.8–6.9x)
- **Problem**: All three Model 3 architectures show significant train-val gap. TFT is worst (6.9x) despite best train loss. Gap means model capacity is wasted memorizing training noise instead of learning generalizable patterns. Regularization could push the val loss floor lower.
- **Research directions**:
  1. **Parameter Drift Analysis** (original idea) — Track |θ(t) - θ*| after val loss bottoms out. Classify parameters as stable (signal) vs drifting (noise memorizers). Apply targeted L1/L2 regularization based on parameter distribution statistics.
  2. **Cautious Weight Decay (CWD)** — ICLR 2026 (arXiv:2510.12402). One-line modification to AdamW: only apply decay when optimizer update and parameter sign align. Zero new hyperparameters.
  3. **Constrained Parameter Regularization (CPR)** — NeurIPS 2024. Per-parameter-matrix adaptive regularization.
- **Status**: Research phase. Literature review complete. Implementation pending.
- **Files**: `model3_research/overfitting_research/README.md`, `model3_research/overfitting_research/cwd_notes.md`, `model3_research/parameter_dynamic_analysis_and_regularization.txt`



### ~~3.4 Adjustment model data leakage~~ → Resolved (2026-02-20)
- **Problem**: `train_adjustment.py` used `torch.utils.data.random_split` for train/val split. This caused three types of temporal data leakage:
  1. **Market feature leakage**: VIX/return info from future dates could appear in train while past dates were in val.
  2. **Sequence overlap leakage**: Adjacent sliding windows share 19/20 days; random split puts near-identical sequences on both sides.
  3. **Cross-option date leakage**: Different options from different dates randomly mixed across train/val.
- **Fix (2 parts)**:
  1. `dataset.py:prepare_adjustment_data` now returns per-sequence dates. `train_adjustment.py` splits chronologically (first 80% of dates → train, last 20% → val).
  2. `adjustment.py:fit_kde_weights` now only fits KDE on train targets. Val weights are computed via `eval_kde_weights` using the train-fitted KDE, preventing val target distribution leakage.
- **Files**: `src/dataset.py`, `src/train_adjustment.py`, `src/adjustment.py`
- **Note**: Models 1 (eSSVI+NN), 4 (HyperIV), 5 (DDPM) already used chronological split. Only Model 3 (Adjustment) had this bug.

### ~~3.5 Additive vs multiplicative architecture~~ → Resolved (2026-02-20)
- **Decision**: **Additive architecture confirmed** via 8-epoch A/B experiment.
- **Architecture**: `output = SSVI_prior(logm, yATM) + yATM * SmileNN(tau, logm)`
- **Experiment** (`scripts/compare_architectures.py`, 8 epochs each, GPU, same seed):
  - **Additive**: Stable convergence (train 0.080→0.069), zero butterfly violations, grad norm 0.05-0.13, smooth SSVI param trajectory.
  - **Multiplicative**: Exploded at epoch 2 (train 0.105→0.816→14.8→96.8), massive butterfly violations (6.55), grad norm pinned at clip ceiling (1.0), SSVI params oscillating.
- **Root cause of multiplicative instability**: Product-rule cross-terms in d²w/dk² (specifically `2·dSSVI/dk·dNN/dk`) create feedback loops between butterfly loss and SSVI parameters. Gradient clipping cannot prevent the underlying loss landscape from having sharp ridges.
- **Full results**: `logs/architecture_comparison.json`

### ~~3.6 Model training duration~~ → Resolved (2026-02-20)
- **Conclusion**: Longer training is **not needed and counterproductive**.
- **Evidence** (from 8-epoch A/B experiment, `logs/architecture_comparison.json`):
  - Train loss convergence rate: -8.1% (ep1→2), -3.0% (ep2→3), -0.26% (ep5→6) — diminishing returns
  - Val loss **increases** from epoch 1 (0.117) to epoch 8 (0.183) — overfitting
  - Val/Train gap grows from 1.46x (ep1) to 2.64x (ep8)
  - MAPE ~7% is the structural floor of the eSSVI+NN additive architecture (MAPE accounts for 97% of the loss; constraints are ~0%)
- **Root cause of rapid convergence**: SSVI prior provides ~97% of the loss structure. The NN correction, scaled by yATM (median ~0.005), is inherently small. The constraints are satisfied by the SSVI prior alone.
- **Path to further improvement**: Not more epochs, but better models (HyperIV already achieves MAPE 20% on the full surface).

---

## 4. Key Files Reference

| File | Purpose |
|------|---------|
| `model1_research/model.py` | Model architecture (eSSVI, SmileNN, MultiModel, losses) |
| `src/dataset.py` | Data preprocessing, dataloaders |
| `model1_research/train.py` | Training loop |
| `src/config.ini` | Training configuration |
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
| 2026-02-20 | Resolved §3.5: Additive architecture confirmed via 8-epoch A/B experiment (multiplicative explodes at ep2) |
| 2026-02-20 | Resolved §3.6: Longer training not needed (val loss overfits from ep1, MAPE 7% is structural floor) |
| 2026-02-20 | Resolved §3.4: Fixed adjustment model data leakage (random_split → chronological, KDE fit train-only) |
| 2026-02-21 | Resolved §1.7: Model 3 architecture comparison complete (xLSTM > TFT > GRU). Full results in `model3_research/` |
| 2026-02-21 | Added §3.7: Model 3 overfitting research (Parameter Drift Analysis, CWD, CPR) |

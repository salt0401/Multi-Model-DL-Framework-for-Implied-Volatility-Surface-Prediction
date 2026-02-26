# Experimental Results

Detailed training results for the IV surface prediction models, trained on TXO options data.

> **Status (2026-02-24):** Model 1 underwent a major architectural revision, replacing static SSVI with eSSVI (time-decaying correlation), freezing the base formulation (`rho_0=-0.95`), and upgrading the NN scaling to $\tilde{y}_{ATM}$. Model 1 is trained on `prs_dataset_no_fat(clean)` (2014-2020 train, 2021 test). Model 3 (Adjustment) architecture comparison is complete. Model 2 (ICNN Dupire Local Volatility Extractor) implementation is fully complete.

## 1. Base Model (eSSVI + Neural Network Ensemble)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | eSSVI prior + 5 SmileModel NNs (64-32-16), **additive**: `w = eSSVI + \tilde{y}_{ATM} * NN` |
| Ensemble method | Learned softmax weights |
| Learning rate | 0.0005 |
| Batch size | 256 |
| Max epochs | 1000 |
| Early stopping patience | 50 |
| Loss weights | `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]` (Physics constraints were relaxed to `0.0` after finding they were not the primary underfitting factor) |
| Gradient clipping | 1.0 |

### Training (2014-2020 train, 2021 test)

**Dataset:** `prs_dataset_no_fat(clean)` (~254K rows, 2014-2021)

**Current eSSVI Pipeline training** (`model1_research/train_pipeline.py`, ε=0.02):
- **Epochs trained:** 84 (early-stopped, patience=50)
- **Best Validation Loss:** **0.07495** (epoch 34)
- **Final Train Loss:** 0.06102
- **SSVI Health:** ✅ Gatheral-Jacquier satisfied for all 5 ensemble members
- **Test IV-RMSE:** 0.01947 | **Test MAPE:** 6.10%

*The transition from static SSVI to frozen eSSVI with ε=0.02 scaling achieved excellent Test MAPE of 6.10%.*

| Metric | Value |
|--------|-------|
| Test points | 50,310 |
| Butterfly violations | 74% (Round 1 evaluation) |

The 74% butterfly violation rate indicates the model struggles with the curvature constraint in the wings. This is a key motivation for Model 2 (Dupire PDE-constrained local vol extractor).

> **Note:** The 2022-2026 extended dataset (`prs_dataset_full.csv`, 480K rows) exists but has known data quality issues (see `discussion_notes.md` §3.2) and has **not** been used for training. A future "Round 2" training on the full dataset is planned once data quality issues are resolved.

#### eSSVI Forced Parameters

To combat the massive local minimum caused by ATM data point gravity pulling the optimizer away from steep skews, the base formulation was upgraded to eSSVI. Furthermore:
1. `rho_0` was hard-frozen to `-0.95` (`requires_grad = False`). This mathematically **forces** the base structural assumption to match the observed 45-degree angle in the Deep OTM Put wing exactly, leaving the NN to act strictly as a residual.
2. The neural network adjustment scaler was upgraded to $\tilde{y}_{ATM} = \sqrt{yATM^2 + 0.02^2}$ to prevent gradient dampening in extremely low volatility regimes.

> **Architecture update (2026-02-23):** An A/B experiment confirmed the additive formulation (`w = eSSVI + \tilde{y}_{ATM} * NN`) is perfectly suited for preserving network capacity and avoiding the multiplicative gradient explosions experienced in early SSVI trials.

#### Visualization Guidelines (IV Smiles)

> **CRITICAL RULE FOR ALL FUTURE AGENTS:** When plotting Implied Volatility (IV) Smile curves (e.g., via `generate_model1_plots.py` or `experiment.py`), **DO NOT mix data from different dates**. 
> - You **MUST** strictly group data by **exact** `(tau, yATM)` pairings.
> - Each subplot must represent a single, isolated option chain (i.e. one specific maturity on one specific date), consisting of at most a few dozen points.
> - Plotted correctly, the observed data will form a single, clean sequence of dots without "vertical scatter" overlap. The model prediction should be plotted horizontally over these exact discrete points. 

---

## 2. HyperIV (Hypernetwork)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Transformer set encoder (128-dim, 4 heads, 2 layers) + Hypernetwork MLP |
| Target MLP | 64-32 hidden dims |
| Reference points | 50 per surface |
| Learning rate | 0.001 |
| Batch size | 32 (per-surface) |
| Max epochs | 500 |

> **Results: Pending retraining.** Previous results were based on an older Model 1 and dataset configuration.

## 3. Model 2: ICNN Dupire Local Volatility Extractor

> **Status (2026-02-22):** Implementation Complete (V1-V3). The system now extracts mathematically guaranteed arbitrage-free (butterfly-free) surfaces.

### Architecture (Phased Implementation)

| Phase | Core Component | Goal | Status |
|-------|---------------|------|--------|
| **V1** | Soft-constraint PINN (MLP + Dupire PDE loss) | Prototype pipeline connectivity | ✅ Validated (Pipeline functional) |
| **V2** | ICNN (hard ∂²C/∂K² ≥ 0 via non-negative weights) | Eliminate 74% butterfly violations | ✅ Validated (0% violations) |
| **V3** | Module D (Local Vol/Vanna/Volga/∂σ_LV/∂K) features | Expand 12→16 dim for Model 3 | ✅ Extracted (Extract features working) |

### Dual-Path Verification (V2 ICNN Performance)

The V2 ICNN replaced the standard MLP PriceNetwork, mathematically guaranteeing convexity via non-negative weights and monotone activations.

| Metric | V1 (Soft PINN) | V2 (ICNN) | Implication |
|--------|----------------|-----------|-------------|
| Butterfly Violations | >0% | **0.00%** | ICNN successfully eliminated the fundamental flaw of Model 1. |
| Convergence Rate | Fast | Slower initial | ICNN softplus initialization requires more epochs to drop loss. |
| Volatility Expressivity | ~0.18 | ~0.14 | Reduced network capacity (due to non-negative constraint) caused slight under-extraction compared to true value (0.20), but the trade-off is absolutely necessary for PDE stability. |

> The legacy DGM code was completely removed from the repository as it is no longer part of the active pipeline and was superseded by ICNN Dupire.

## 4. Adjustment Model — Architecture Comparison (2026-02-21)

### Data

| Parameter | Old (2026-02-21) | Current (16-dim, config-aligned) |
|-----------|---------|---------|
| Total sequences | 245,228 | 245,228 |
| Input dimensions | 12 (6 base + 6 enhancement) | **16** (6 base + 6 enhancement + 4 Greeks) |
| Sequence length | 20 days | 20 days |
| Train split | 153,249 (< 2019-08-13) | 153,249 (< 2019-08-13) |
| Val split | 48,480 (2019-08-13 ~ 2020-12-31) | 48,480 (2019-08-13 ~ 2020-12-31) |
| Test split | *(none)* | **43,499 (2021-01-01 ~ 2021-12-31)** |
| GPU | NVIDIA RTX 4060 Laptop (8.6GB) | same |
| Precision | float64 | float32 (TFT), float64 (GRU) |
| Batch size | 256 | 256 |
| Max epochs | 300 | 1000 |
| Early stopping | Patience 30 | Patience 100 |
| Prediction target | Ratio (multiplicative adjustment) | same |

**Base features (6):** vix_change, underlying_return, logm, tau, tv_pred, itm_otm
**Enhancement features (6):** sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d
**Greek features (4):** local_vol, vanna, volga, lv_gradient_K

> **Data leakage fix (2026-02-20):** The original `train_adjustment.py` used `random_split` for train/val, causing temporal leakage. Now uses chronological split aligned with Model 1: filter to config training period (2014-2020), 80/20 split on unique dates within that period (split ≈ 2019-08-13), 2021 held-out as test set. KDE weights are fitted on train targets only.

### Architecture Configurations

| Architecture | Description | Params (approx, 16-dim) |
|-------------|-------------|:---:|
| **GRU (Baseline)** | 2-layer GRU (64 hidden) + 4-head attention + FC + SquarePlus | ~60K |
| **TFT** | Variable Selection Network + LSTM encoder + GRN + Interpretable Multi-Head Attention | ~318K |

### Regularization Strategies

| Optimizer | Paper | Description |
|-----------|-------|-------------|
| **CPR** | NeurIPS 2024 | Per-parameter-matrix adaptive regularization |
| **AdamW** | Standard | Weight decay baseline |
| **CWD** | ICLR 2026 | Cautious Weight Decay — per-parameter selective weight decay |

### Training Results

> **Status (2026-02-26):** Training is in progress with 16-dim input (6 base + 6 enhancement + 4 Greeks) and config-aligned data split (train < 2019-08-13, val 2019-08~2020-12, test 2021 held-out). Results pending completion.

### float32 vs float64 Benchmark

RTX 4060 Laptop: FP32 ~15.11 TFLOPS, FP64 ~0.236 TFLOPS (1/64 ratio). Benchmark: batch_size=256, seq_len=20, input_dim=12, warmup=5, batches=20.

| Model | float64 (ms/batch) | float32 (ms/batch) | Training Speedup |
|-------|:---:|:---:|:---:|
| xLSTM (mLSTM) | 73.8 | 73.3 | **1.01x (no difference)** |
| TFT | 68.4 | 22.3 | **3.07x** |

- **xLSTM is memory-bound**: mLSTM's sequential scan has per-step data dependencies; GPU cannot parallelize. Bottleneck is memory access, not compute — FP32/FP64 speed identical.
- **TFT is compute-bound**: LSTM encoder + Multi-Head Attention + GRN can be parallelized; benefits from FP32's 64x compute advantage.

## 5. DDPM (Diffusion Model)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 1D U-Net (64-128-256 channels) |
| Surface grid | 10 tau x 20 log-moneyness = 200-dim vector |
| Diffusion steps | 1000 |
| Noise schedule | Cosine |
| Condition dim | 11 (vixtwn_change removed, was 13) |
| Learning rate | 0.0002 |
| Batch size | 16 |
| Epochs | 1000 |

> **Results: Pending retraining.** Previous results used an older dataset configuration with `vixtwn_change` feature (condition_dim was 13, now 11). Data quality issues in 2022-2026 data must be resolved first.

---

## Known Issues (Engineering)

1. **Base model gradient explosion (ep67+):** The SSVI parametric component creates a complex, non-convex loss landscape. Even with gradient clipping (1.0), the multi-term physics loss can become unstable when SSVI parameters drift into degenerate regions. Best practice: always checkpoint at best val epoch.

2. **Adjustment model CUDA OOM:** The `prepare_adjustment_data()` function runs the base model's forward pass with `autograd.grad(create_graph=True)` on all data points. Resolved: chunked inference (5000 rows/batch) implemented in `dataset.py`.

## Transfer Learning Details

The `src/transfer.py` module handles:
- **Dimension mismatch:** When layer shapes differ (e.g., Adjustment GRU input 6→12), the overlapping weights are copied and new dimensions initialized with Xavier uniform
- **Differential LR:** Pretrained layers use base_lr × 0.1, reinitialized layers use base_lr × 1.0
- **Partial transfer:** `load_finetune_weights()` returns (transferred, reinitialized) parameter name sets for optimizer group construction

## Limitations

1. **Single underlying asset** — All models are trained on TXO only. Transfer to other markets would require retraining.
2. **Base model instability** — Physics-informed loss with 6 weighted terms is sensitive to hyperparameters. Training frequently diverges without careful learning rate tuning.
3. **No model stacking** — The five models are trained independently. An ensemble or stacking approach could combine their strengths.
4. **Butterfly violations** — The base model's 74% violation rate indicates the density constraint needs stronger enforcement (e.g., Lagrangian dual or penalty scheduling).

## Current Status & Next Steps

| Model | Status | Next Step |
|-------|--------|----------|
| Model 1 (eSSVI+NN) | ✅ Trained (2014-2020 / 2021 test) | Future: retrain on full dataset after data quality fixes |
| Model 2 (ICNN Dupire) | ✅ Implemented (V1-V3) | Local volatility and higher-order Greeks safely extracted |
| Model 3 (Adjustment) | ✅ Arch comparison & regularization done | 3 shortlisted models (TFT+CPR, TFT+AdamW, GRU+CWD) retained. Integration pending |
| Model 4 (HyperIV) | ⏳ Pending retraining | Retrain after Model 1 is stable |
| Model 5 (DDPM) | ⏳ Pending retraining | Retrain after data quality issues resolved |

Key changes made recently:
- Additive architecture confirmed (multiplicative explodes at epoch 2)
- SSVI bounded parameterization: `eta = 2*sigmoid(raw_eta)`, `gamma = sigmoid(raw_gamma)`
- `vixtwn_change` feature removed (Adjustment: input_dim now 12, DDPM: condition_dim now 11)
- Data leakage fix: chronological split + KDE train-only fitting
- Model 3 architecture comparison & regularization: TFT + CPR selected as primary choice. Non-shortlisted experiments archived.
- Model 2 ICNN redesign complete: Soft PINN (V1) → ICNN (V2) → Greek Extractor Module D (V3). Butterfly violations permanently eliminated.

## Future Work

- Integrate HyperIV predictions as DDPM conditioning for improved surface forecasting
- Add explicit no-arbitrage constraints to HyperIV via penalty or projection
- Explore attention-based architectures for the base model (replace per-expiration SmileModels with a single cross-expiration model)
- Extend to American-style options using the DGM PDE framework with early exercise boundary
- Address base model gradient explosion with adaptive loss weighting or SSVI parameter clamping
- Add model stacking/ensemble across the five models for combined predictions

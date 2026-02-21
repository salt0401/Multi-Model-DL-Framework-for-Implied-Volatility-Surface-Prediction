# Experimental Results

Detailed training results for the IV surface prediction models, trained on TXO options data.

> **Status (2026-02-21):** Model 1 (SSVI+NN) has been retrained and validated. Model 3 (Adjustment) architecture comparison is complete — three architectures trained and compared. Models 2, 4, 5 require retraining. See "Retraining Plan" at the bottom.

## 1. Base Model (SSVI + Neural Network Ensemble)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | SSVI prior + 5 SmileModel NNs (64-32-16), **additive**: `w = SSVI + yATM * NN` |
| Ensemble method | Learned softmax weights |
| Learning rate | 0.001 |
| Batch size | 256 |
| Max epochs | 2000 |
| Early stopping patience | 50 |
| Loss weights | `[1, 1, 10, 10, 10, 10]` (data, SSVI, calendar, butterfly, density, smoothness) |
| Gradient clipping | 1.0 |

### Round 1 Training (2014-2020 train, 2021 test)

- **Epochs trained:** 76 / 2000 (early stopped)
- **Best epoch:** 26 (val loss = 2.6624)
- Initial instability: first 3 epochs had losses in the millions due to physics loss terms (calendar/butterfly constraints) calibrating against random weights
- Converged by epoch ~20, then gradually overfit
- A second instability spike at epoch 62 (train loss = 38.4) and epoch 74 (train loss = 59,351) triggered early stopping

| Metric | Value |
|--------|-------|
| TV-RMSE | 0.0134 |
| MAPE | 44.1% |
| IV-RMSE | 0.209 |
| Butterfly violations | 74% |

The high MAPE (44.1%) is driven by near-ATM options where true total variance is very small — even a small absolute error produces a large percentage error. The 74% butterfly violation rate indicates the model struggles with the curvature constraint in the wings.

### Round 2 Training (2014-2024 train, 2025-2026 test)

- **Epochs trained:** 105 / 2000 (early stopped)
- **Best epoch:** 55 (val loss = 1.914)
- Gradient explosion after epoch 67: train loss spiked from ~2.37 to 1029. The best model (ep55) was safely checkpointed before the instability.

| Metric | R1 (2021 test) | R2 (2025-26 test) | Change |
|--------|----------------|-------------------|--------|
| TV-RMSE | 0.0134 | **0.0120** | -10.4% |
| MAPE | 44.1% | **33.0%** | -25.2% |
| IV-RMSE | 0.209 | 0.219 | +4.8% |

**Key observations:**
- TV-RMSE improved (more training data helps generalization)
- MAPE improved dramatically (44.1% → 33.0%) — the extended dataset reduced near-ATM prediction errors
- IV-RMSE increased slightly — the 2025-2026 test period has more short-maturity options where `IV = sqrt(TV/tau)` amplifies errors

#### Arbitrage Violations (2025-2026 Test)

| Violation | Rate |
|-----------|------|
| Calendar | 53.3% (105/197) |
| Butterfly | 83.7% (77,256/92,270) |

Butterfly violations increased from 74% to 84%. The more complex 2025-2026 market conditions (higher vol, more skew) make the constraint harder to satisfy.

#### SSVI Learned Parameters

All 5 ensemble members satisfy the Gatheral-Jacquier no-arbitrage constraint `eta*(1+|rho|) < 2`:

| Member | rho | eta | gamma | GJ value |
|--------|-----|-----|-------|----------|
| 0 | -0.315 | 1.060 | 0.533 | 1.394 |
| 1 | -0.309 | 1.069 | 0.538 | 1.400 |
| 2 | -0.311 | 1.068 | 0.537 | 1.400 |
| 3 | -0.309 | 1.074 | 0.540 | 1.406 |
| 4 | -0.306 | 1.073 | 0.540 | 1.402 |

#### Loss Component Breakdown (Final Epoch)

| Component | Weight | Value |
|-----------|--------|-------|
| RMSE | 1 | 0.0015 |
| MAPE | 1 | 0.063 |
| Calendar | 10 | ~0 |
| Butterfly | 10 | 0 |
| Linear (density) | 10 | 3e-6 |
| Upper bound | 10 | 0 |

> **Architecture update (2026-02-20):** An A/B experiment confirmed the additive formulation (`w = SSVI + yATM * NN`) is strictly superior to the original multiplicative formulation (`w = SSVI * NN`). The multiplicative version explodes at epoch 2 due to product-rule cross-terms in the butterfly constraint derivatives. Full results in `logs/architecture_comparison.json`.

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

## 3. DGM (Deep Galerkin Method PDE Solver)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 3 S-layers, 64 hidden dim |
| Domain | sigma: [0.05, 1.0], t: [0.02, 2.0], S: [0.5, 1.5] |
| Loss weights | PDE: 1.0, BC: 1.0, TC: 1.0 |
| Collocation points | Interior: 5000, Boundary: 500, Terminal: 500 |
| Resample every | 100 epochs |
| Epochs | 5000 |

> **Results: Pending retraining.** DGM is domain-independent (doesn't depend on Model 1), but will be retrained for consistency with the latest codebase.

## 4. Adjustment Model — Architecture Comparison (2026-02-21)

### Data

| Parameter | Value |
|-----------|-------|
| Total sequences | 245,228 |
| Input dimensions | 12 (6 base + 6 enhancement) |
| Sequence length | 20 days |
| Train split | 179,615 sequences (dates < 2020-06-05) |
| Val split | 65,613 sequences (dates >= 2020-06-05) |
| GPU | NVIDIA RTX 4060 Laptop (8.6GB) |
| Precision | float64 |
| Batch size | 256 |
| Max epochs | 300 |
| Early stopping | Patience 30 |
| Prediction target | Ratio (multiplicative adjustment) |

**Base features (6):** vix_change, underlying_return, logm, tau, tv_pred, itm_otm
**Enhancement features (6):** sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d

> **Data leakage fix (2026-02-20):** The original `train_adjustment.py` used `random_split` for train/val, causing temporal leakage. Now uses chronological split (first 80% dates → train, last 20% → val). KDE weights are fitted on train targets only.

### Architecture Configurations

| Architecture | Description | Parameters |
|-------------|-------------|:---:|
| **GRU (Baseline)** | 2-layer GRU (64 hidden) + 4-head attention + FC + SquarePlus | 58,689 |
| **xLSTM (mLSTM)** | Matrix LSTM with QKV retrieval + 4-head attention + FC + SquarePlus | 39,133 |
| **TFT** | Variable Selection Network + LSTM encoder + GRN + Interpretable Multi-Head Attention | 265,281 |

### 3-Way Model Comparison

| Metric | GRU (Baseline) | xLSTM (mLSTM) | TFT | Winner |
|--------|:---:|:---:|:---:|:---:|
| **Val Loss (best)** | 0.1639 | **0.1544** | 0.1581 | xLSTM |
| **Val RMSE** | 0.1477 | **0.1414** | 0.1452 | xLSTM |
| **Val MAPE** | 9.43% | **9.01%** | 9.12% | xLSTM |
| Parameters | 58,689 | **39,133** | 265,281 | xLSTM (fewest) |
| Best Epoch | 70 | 138 | 112 | GRU (fastest convergence) |
| Training Time | **41.4 min** | 311.5 min | 207.1 min | GRU |
| Interpretability | Low | Low | **Excellent** | TFT |

#### Improvement over GRU Baseline

| Model | RMSE Improvement | MAPE Improvement |
|-------|:---:|:---:|
| xLSTM | **-4.27%** (0.1477 → 0.1414) | **-4.45%** (9.43% → 9.01%) |
| TFT | **-1.69%** (0.1477 → 0.1452) | **-3.29%** (9.43% → 9.12%) |

### Overfitting Analysis (Train-Val Gap)

| Model | Best Train Loss | Best Val Loss | Gap Ratio |
|-------|:---:|:---:|:---:|
| GRU | ~0.055 | 0.1639 | ~3.0x |
| xLSTM | ~0.055 | 0.1544 | ~2.8x |
| TFT | ~0.023 | 0.1581 | ~6.9x |

All three models show significant overfitting. Early stopping selects the correct epoch, but the large train-val gap indicates that regularization could push the val loss floor lower. This is the subject of ongoing research — see `model3_research/overfitting_research/`.

**Current overfitting research directions:**
1. **Parameter Drift Analysis** — Track which parameters drift after val loss bottoms out, classify as stable (signal) vs drifting (noise), apply targeted regularization
2. **Cautious Weight Decay (CWD)** — ICLR 2026, one-line modification to AdamW for per-parameter selective weight decay
3. **Constrained Parameter Regularization (CPR)** — NeurIPS 2024, per-parameter-matrix adaptive regularization

### TFT Feature Importance (Variable Selection Network)

| Feature | Importance | Category |
|---------|:---:|----------|
| tv_pred | 21.4% | Model 1 prediction (most critical) |
| tau | 15.9% | Term structure |
| iv_term_slope | 12.7% | Enhancement: IV structure |
| rv_20d | 11.7% | Enhancement: realized vol |
| iv_skew | 10.3% | Enhancement: tail risk |
| vrp_20d | 7.0% | Enhancement: vol risk premium |
| futures_basis_pct | 6.0% | Enhancement: basis |
| itm_otm | 5.0% | Moneyness indicator |
| vix_change | 4.8% | Market volatility |
| sp500_return | 2.2% | US market proxy |
| logm | 1.9% | Log moneyness |
| underlying_return | 1.0% | TAIEX return |

Enhancement features contribute **47.7%** of total importance. Temporal attention gives last timestep 20.3% weight, confirming recency matters most.

### float32 vs float64 Benchmark

RTX 4060 Laptop: FP32 ~15.11 TFLOPS, FP64 ~0.236 TFLOPS (1/64 ratio). Benchmark: batch_size=256, seq_len=20, input_dim=12, warmup=5, batches=20.

| Model | float64 (ms/batch) | float32 (ms/batch) | Training Speedup |
|-------|:---:|:---:|:---:|
| xLSTM (mLSTM) | 73.8 | 73.3 | **1.01x (no difference)** |
| TFT | 68.4 | 22.3 | **3.07x** |

- **xLSTM is memory-bound**: mLSTM's sequential scan has per-step data dependencies; GPU cannot parallelize. Bottleneck is memory access, not compute — FP32/FP64 speed identical.
- **TFT is compute-bound**: LSTM encoder + Multi-Head Attention + GRN can be parallelized; benefits from FP32's 64x compute advantage. Estimated full training: 207 min (FP64) → ~68 min (FP32).

### Key Findings

1. **xLSTM is the best model**: 4.3% RMSE improvement over GRU baseline, with fewest parameters (39K)
2. **TFT also beats baseline**: 1.7% RMSE improvement, but 6.8x more parameters than xLSTM
3. **GRU is fastest to train**: 41.4 min vs 207.1 (TFT) and 311.5 (xLSTM) — cuDNN optimized kernel
4. **All three models overfit**: train-val gap 2.8–6.9x, regularization research in progress
5. **Enhancement features contribute 47.7%** of TFT importance — validates the 6 new features
6. **tv_pred is the most important feature** (21.4%) — Model 1's output is the key input
7. **float32 only helps TFT** (3x speedup); xLSTM is memory-bound (no benefit)

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

> **Results: Pending retraining.** Previous results used an older dataset configuration with `vixtwn_change` feature (condition_dim 13→11).

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
4. **Butterfly violations** — The base model's 84% violation rate indicates the density constraint needs stronger enforcement (e.g., Lagrangian dual or penalty scheduling).

## Retraining Plan

Remaining models need retraining:

1. **Model 4 (HyperIV)** — Independent of Model 1, can train in parallel
2. **Model 2 (DGM)** — Independent of Model 1, can train in parallel
3. ~~**Model 3 (Adjustment)**~~ — **Architecture comparison complete (2026-02-21)**. xLSTM selected. Regularization research in progress before production integration.
4. **Model 5 (DDPM)** — Independent but benefits from updated conditioning features

Key changes since last training:
- `vixtwn_change` feature removed (Adjustment: 13→12 input dims, DDPM: 13→11 condition dims)
- Model 1 retrained with additive architecture and fixed SSVI constraints
- Data leakage fix applied to Adjustment model's train/val split
- Model 3 architecture comparison: xLSTM > TFT > GRU (see Section 4)

## Future Work

- Integrate HyperIV predictions as DDPM conditioning for improved surface forecasting
- Add explicit no-arbitrage constraints to HyperIV via penalty or projection
- Explore attention-based architectures for the base model (replace per-expiration SmileModels with a single cross-expiration model)
- Extend to American-style options using the DGM PDE framework with early exercise boundary
- Address base model gradient explosion with adaptive loss weighting or SSVI parameter clamping
- Add model stacking/ensemble across the five models for combined predictions

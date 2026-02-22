# Experimental Results

Detailed training results for the IV surface prediction models, trained on TXO options data.

> **Status (2026-02-22):** Model 1 (SSVI+NN) is trained on `prs_dataset_no_fat(clean)` (2014-2020 train, 2021 test). Model 3 (Adjustment) architecture comparison is complete — three architectures trained and compared; overfitting regularization research in progress. Model 2 (ICNN Dupire Local Volatility Extractor) implementation is fully complete (V1-V3) and functionally eliminates 100% of butterfly violations. Models 4, 5 await retraining.

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

### Training (2014-2020 train, 2021 test)

**Dataset:** `prs_dataset_no_fat(clean)` (~254K rows, 2014-2021)

**Standalone training** (`src/train.py`):
- **Epochs trained:** 20 / 2000 (early stopped)
- **Best epoch:** 17 (val loss = 1.940)
- Initial instability: first 8 epochs had losses in the millions due to physics loss terms (calendar/butterfly constraints) calibrating against random weights
- Converged by epoch ~9, then gradually overfit

**Pipeline training** (`src/train_pipeline.py`, final run):
- **Epochs trained:** 3 (Stage 1), 231 (Stage 2)
- **Stage 1 best val loss:** 0.117 (epoch 1)
- **Stage 2 (Adjustment):** RMSE 0.1373, MAPE 9.22%, input_dim=13

| Metric | Value |
|--------|-------|
| Test points | 50,310 |
| Butterfly violations | 74% (Round 1 evaluation) |

The 74% butterfly violation rate indicates the model struggles with the curvature constraint in the wings. This is a key motivation for Model 2 (Dupire PDE-constrained local vol extractor).

> **Note:** The 2022-2026 extended dataset (`prs_dataset_full.csv`, 480K rows) exists but has known data quality issues (see `discussion_notes.md` §3.2) and has **not** been used for training. A future "Round 2" training on the full dataset is planned once data quality issues are resolved.

#### SSVI Learned Parameters

All 5 ensemble members satisfy the Gatheral-Jacquier no-arbitrage constraint `eta*(1+|rho|) < 2`:

| Member | rho | eta | gamma | GJ value |
|--------|-----|-----|-------|----------|
| 0 | -0.315 | 1.060 | 0.533 | 1.394 |
| 1 | -0.309 | 1.069 | 0.538 | 1.400 |
| 2 | -0.311 | 1.068 | 0.537 | 1.400 |
| 3 | -0.309 | 1.074 | 0.540 | 1.406 |
| 4 | -0.306 | 1.073 | 0.540 | 1.402 |

> **Architecture update (2026-02-20):** An A/B experiment confirmed the additive formulation (`w = SSVI + yATM * NN`) is strictly superior to the original multiplicative formulation (`w = SSVI * NN`). The multiplicative version explodes at epoch 2 due to product-rule cross-terms in the butterfly constraint derivatives. Full results in `logs/architecture_comparison.json`.

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
| **V3** | Module D (Vanna/Volga/∂σ_LV/∂K) features | Expand 12→15 dim for Model 3 | ✅ Extracted (Extract features working) |

### Dual-Path Verification (V2 ICNN Performance)

The V2 ICNN replaced the standard MLP PriceNetwork, mathematically guaranteeing convexity via non-negative weights and monotone activations.

| Metric | V1 (Soft PINN) | V2 (ICNN) | Implication |
|--------|----------------|-----------|-------------|
| Butterfly Violations | >0% | **0.00%** | ICNN successfully eliminated the fundamental flaw of Model 1. |
| Convergence Rate | Fast | Slower initial | ICNN softplus initialization requires more epochs to drop loss. |
| Volatility Expressivity | ~0.18 | ~0.14 | Reduced network capacity (due to non-negative constraint) caused slight under-extraction compared to true value (0.20), but the trade-off is absolutely necessary for PDE stability. |

> The legacy DGM code (`src/dgm.py`, `src/train_dgm.py`) is retained for reference but is no longer part of the active pipeline.

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

> **Data leakage fix (2026-02-20):** The original `train_adjustment.py` used `random_split` for train/val, causing temporal leakage. Now uses chronological split (first 80% dates → train, last 20% → val, split at 2020-06-05). KDE weights are fitted on train targets only.

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

**Current overfitting research directions (in progress, 2026-02-22):**
1. **AdamW baseline** — Standard weight decay as baseline reference
2. **Cautious Weight Decay (CWD)** — ICLR 2026, one-line modification to AdamW for per-parameter selective weight decay
3. **Constrained Parameter Regularization (CPR)** — NeurIPS 2024, per-parameter-matrix adaptive regularization
4. **Elastic Weight Consolidation (EWC)** — Penalizes deviation from important parameter values

Experiments were conducted on both GRU and TFT architectures. The final results successfully established a new higher validation floor:
- **GRU**: Cautious Weight Decay (CWD) improved validation loss from `0.1639` to `0.1582` (-3.4%).
- **TFT**: Constrained Parameter Regularization (CPR) alongside `float32` training reached a new lowest validation loss of **`0.1521`** (improving upon the previous best xLSTM score of `0.1544`). TFT + CPR is now the recommended architecture for Model 3.
- Full results and loss curves are documented in `model3_research/regularization_results.md`.

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

1. **TFT + CPR is the best model**: Applying CPR to TFT achieved the lowest overall RMSE (0.1404) and MAPE (8.89%), beating the unregularized xLSTM.
2. **xLSTM is highly parameter-efficient**: 39K parameters vs TFT's 265K, while still beating the GRU baseline by 4.3% in RMSE.
3. **GRU is fastest to train**: 41.4 min vs 175.3 (TFT fp32) and 311.5 (xLSTM) — cuDNN optimized kernel
4. **All three models overfit**: train-val gap 2.8–6.9x without regularization. Adding CPR or CWD is mandatory.
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
| Model 1 (SSVI+NN) | ✅ Trained (2014-2020 / 2021 test) | Future: retrain on full dataset after data quality fixes |
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

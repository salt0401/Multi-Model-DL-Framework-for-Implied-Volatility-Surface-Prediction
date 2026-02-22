# System Architecture

## Pipeline Overview

```
                         Raw TXO Data (FinMind API + historical CSV)
                         + Enhancement Features (VIX, S&P, institutional)
                                        |
                                 DataProcessor
                                /      |       \
                          Train      Val       Test
                       (2014-2020)  (20%)    (2021)
                            |
        +------------------+------------------+------------------+
        |                  |                  |                  |
    Phase 1            Phase 2            Phase 3           Phase 4/5
    Base Model         ICNN Dupire        Adjustment         HyperIV / DDPM
    (SSVI+NN)          Local Vol          (TFT+Attn)       (independent)
        |                  |                  |                  |
    MultiModel.pt    DupireICNN.pt      AdjModel.pt      HyperIV.pt / Diffusion.pt
        |                  |                  |                  |
        +--------+---------+--------+--------+                  |
                 |                                               |
            test.py                                   train_hyperiv.py
            experiment.py                             train_diffusion.py
                 |                                               |
         IV Surface Predictions                       Surface Forecasts
```

All training scripts support `--finetune <path>` for transfer learning from existing checkpoints.
`src/transfer.py` handles weight loading with dimension mismatch support and differential learning rates.

## Data Pipeline

### DataProcessor (`dataset.py`)

Handles all data loading, feature engineering, and splitting:

```
Raw CSV (prs_dataset_no_fat(clean).csv, ~254K rows, 2014-2021)
  Sources: original 2014-2021 CSV (pre-processed from PKL)
  Note: prs_dataset_full.csv (480K rows, 2014-2026) exists but has
        known data quality issues — not used for training yet
    |
    v
Column Handling
    - Rename Chinese columns to English (交易日期→date, 履約價→strike_price, etc.)
    - Parse dates, compute tau (time-to-expiry)
    - Filter: remove deep OTM/ITM, very short tau
    |
    v
Feature Engineering
    - Compute log-moneyness: ln(K/S)
    - Black-Scholes implied volatility (Newton-Raphson)
    - Total variance: IV^2 * tau
    - ATM total variance (yATM) via interpolation at logm=0
    - Season/year dummies for structural break features
    |
    v
Enhancement Features (dataset/enhancement/daily_features.csv)
    Created by: scripts/build_features.py
    - Realized volatility (20-day)
    - IV term slope, IV skew
    - Variance risk premium (VRP)
    - S&P 500 return, futures basis %
    - Institutional net buy/sell ratio
    |
    v
Beta-Tau Estimation
    - Linear regression of SSVI betas on tau
    - Merge beta predictions back into main dataset
    |
    v
Chronological Split (no leakage)
    - Train: 2014-01-01 to 2020-12-31
    - Test:  2021-01-01 to 2021-12-31
    - Val:   last 20% of training dates (chronological)
    |
    v
PyTorch DataLoaders
    - TensorDataset with float64
    - Batch size from config.ini
```

**Key design decision:** The train/test split is strictly chronological — no future data leaks into training. Validation split is also chronological (last 20% of training dates), not random.

### Input Features

| Feature | Symbol | Shape | Description |
|---------|--------|-------|-------------|
| Log-moneyness | `logm` | (N,) | ln(K/S), centered at 0 |
| ATM total variance | `yATM` | (N,) | Total variance at logm=0, per expiration |
| Time to expiry | `tau` | (N,) | Years until expiration |
| Underlying price | `S` | (N,) | Normalized TAIEX index level |

## Model Architectures

### Phase 1: Base Model (Additive Architecture)

> **Architecture decision (2026-02-20):** Additive formulation confirmed via A/B experiment. Multiplicative (`SSVI * NN`) explodes at epoch 2 due to product-rule cross-terms in butterfly constraint derivatives. See `logs/architecture_comparison.json`.

```
            (tau, logm, yATM)
                    |
         +----------+----------+
         |                     |
     SSVIModel              SmileModel x5 ensemble
     (logm, yATM)           (tau, logm)
         |                     |
     output_Prior          output_NN
         |                     |
         |              yATM * output_NN
         |                     |
         +------( + )----------+
                 |
          output = SSVI + yATM * NN   (additive)
                 |
          SoftmaxModel
          (learned weights, input: logm, tau, yATM)
                 |
          Weighted sum of 5 ensemble predictions
                 |
          WeightedSumLoss
          = w1*RMSE + w2*MAPE + w3*calendar
            + w4*butterfly + w5*density + w6*upperbound
```

**SingleModel forward (`model.py`):**
```python
output = output_Prior + yATM * output_NN
grad_ttm1 = grad_ttm1_prior + yATM * grad_ttm1_NN
grad_logm1 = grad_logm1_prior + yATM * grad_logm1_NN
grad_logm2 = grad_logm2_prior + yATM * grad_logm2_NN
```
No cross-terms in derivatives (unlike multiplicative product rule).

**SmileModel details:**
- Input: tau, logm (each scalar) -> detach + requires_grad_(True)
- 3 hidden layers: 64 -> 32 -> 16, with custom bilinear input + LayerNorm
- Output: 1 scalar (total variance correction)
- Computes 1st derivative via `autograd.grad(..., create_graph=True)`
- Computes 2nd derivative via `autograd.grad(..., retain_graph=True)`
- Returns 4-tuple: `(TV, grad_ttm, grad_logm1, grad_logm2)`

**Loss components (weights `[1, 1, 10, 10, 10, 10]`):**
1. **RMSE** — Root mean squared error of total variance predictions
2. **MAPE** — Mean absolute percentage error (with ε=0.005 stability)
3. **Calendar** — penalizes `dw/dtau < 0` (variance must increase with time)
4. **Butterfly** — penalizes negative probability density (convexity constraint)
5. **Density** — penalizes `|d2w/dk2|` on synthetic wing data
6. **Upper bound** — penalizes `w > 2|k|` on synthetic wing data (Lee's bound)

### Phase 2: ICNN Dupire Local Volatility Extractor

> **Redesign (2026-02-22):** The original DGM (fixed-sigma BS PDE solver) has been replaced with a Dupire PDE-constrained local volatility extractor. The old DGM code (`src/dgm.py`) is retained for reference but is no longer part of the active pipeline. See `model2_research/action_plan.md` for full design rationale.

**Problem:** Model 1's 74% butterfly violation rate means direct application of the Dupire formula produces negative denominators (imaginary local vol) at most test points. Model 2 solves this by learning a self-consistent (call price, local vol) pair that satisfies the Dupire PDE.

**Dual-Pipeline Strategy:**
```
Path α (primary):  Model 1 → ICNN (hard ∂²C/∂K² ≥ 0) → Module D → Model 3/5
Path β (compare):  Model 1 → Module A (soft correction) → GNO → Module D → Model 3/5
```

**Architecture (V1 → V2 → V3):**

```
V1 — Prototype (Soft-constraint PINN):
    (A) Price Network: π_θ(k, τ)    (B) Local Vol Network: σ_LV_φ(k, τ)
        3 residual blocks, 64 neurons     3 residual blocks, 64 neurons
        Input: Model 1 call price prior   Output: σ²_LV(K,T) > 0 (softplus)
        Output: corrected C(K,T)
                    │                         │
                    └─── Dupire PDE ───────────┘
                    ∂C/∂T = ½ σ²_LV K² ∂²C/∂K²

V2 — ICNN (Hard convexity guarantee):
    Replace Price Network MLP with Input-Convex NN:
    - K→C(K) path: all weight matrices ≥ 0 (softplus(W))
    - Activations: monotone increasing (ReLU/Softplus)
    - Guarantees ∂²C/∂K² ≥ 0 → butterfly violation = 0%
    - Reference: Amos et al. (2017), ARBITER Legendre conjugate head

V3 — Module D + GNO exploration:
    - Module D: Vanna, Volga, ∂σ_LV/∂K from clean local vol → Model 3 (15-dim)
    - GNO: offline-trained global mapping, 100x inference speedup
```

**Loss Function (5 terms):**
```
L = λ_fit    · L_fit      # fit Model 1's call prices
  + λ_dup    · L_dupire    # Dupire PDE residual: ∂C/∂T = ½σ²_LV K² ∂²C/∂K²
  + λ_arb    · L_arb       # no-arbitrage inequalities (calendar + butterfly + delta)
  + λ_ini    · L_initial   # boundary: τ=0 payoff = max(S-K, 0)
  + λ_smooth · L_smooth    # local vol smoothness (Sobolev penalty)
```

**Input/Output:**
- Input: Model 1's total variance surface w(τ, logm) + derivatives
- Output: local vol surface σ_LV(K,T), risk-neutral density q(K,T) = ∂²C/∂K², PDE residual map

**Hardware notes:** ICNN dual-network ~18K params total (3 layers × 64 neurons each). Mixed precision: NN forward in float32, Dupire PDE operator in float64. Must `detach()` + GC after each batch to prevent VRAM fragmentation.

### Phase 3: Adjustment Model — Architecture Comparison

Three architectures were evaluated (2026-02-21). All share the same input/output interface and prediction target.

**Input features (12 dims):**
- Base (6): vix_change, underlying_return, logm, tau, tv_pred, itm_otm
- Enhancement (6): sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d

**Data preparation:** The `tv_pred` feature is computed by running the trained base model on all data points with **chunked inference** (5000 rows per batch) to avoid GPU OOM.

#### Architecture A: GRU + Attention (Baseline)

```
   (batch, seq_len=20, 12 features)
              |
         GRU (2 layers, 64 hidden)         [cuDNN fused kernel]
              |
   (batch, seq_len, 64) hidden states
              |
    TemporalAttention (4 heads)
              |
   (batch, 64) context vector
              |
         Linear(64, 1) → SquarePlus
              |
    Adjustment ratio                        [58,689 params]
```

#### Architecture B: xLSTM (mLSTM) + Attention

```
   (batch, seq_len=20, 12 features)
              |
         mLSTM (matrix memory, QKV)         [sequential scan, memory-bound]
         - Exponential gating
         - Matrix memory C (d×d)
         - Covariance update: C_t = f_t * C_{t-1} + v_t * k_t^T
              |
   (batch, seq_len, 64) hidden states
              |
    TemporalAttention (4 heads)
              |
   (batch, 64) context vector
              |
         Linear(64, 1) → SquarePlus
              |
    Adjustment ratio                        [39,133 params]
```

#### Architecture C: Temporal Fusion Transformer (TFT) (Winner)

```
   (batch, seq_len=20, 12 features)
              |
    Variable Selection Network (VSN)
    - Softmax gating per timestep
    - Learns feature importance dynamically
              |
    Gated Residual Networks (GRN)
    - ELU + skip connections + gate
              |
    LSTM Encoder (2 layers, 64 hidden)
              |
    Interpretable Multi-Head Attention (4 heads)
    - Shared V weights (interpretable)
              |
    Position-wise Feed-Forward
              |
    SquarePlus → Adjustment ratio           [265,281 params]
```

#### Comparison Results

| Metric | GRU | xLSTM | TFT (fp32) |
|--------|:---:|:---:|:---:|
| Val RMSE | 0.1477 | 0.1414 | **0.1404** (CPR) |
| Params | 59K | **39K** | 265K |
| Training | **41 min** | 312 min | 175 min |

**Selection: TFT with CPR** — Achieved the best overall accuracy (RMSE 0.1404) after employing Constrained Parameter Regularization (CPR) to solve its overfitting issues. Furthermore, it provides excellent interpretability via the Variable Selection Network, and trains reasonably fast (~3 hours) when using `float32` precision.

**Three prediction modes:**
1. `ratio` — multiplicative: `adjusted_IV = base_IV * ratio`
2. `residual` — additive: `adjusted_IV = base_IV + residual`
3. `direct` — replace: `adjusted_IV = prediction`

> **Data leakage fix (2026-02-20):** The original `train_adjustment.py` used `random_split`, causing temporal leakage. Now uses chronological split + KDE fitted on train targets only. See `discussion_notes.md` §3.4.

> **Overfitting research (2026-02-22):** All three architectures showed significant train-val gap (2.8–6.9x). Regularization experiments (AdamW, CWD, CPR) successfully addressed this: CWD improved the GRU baseline (val loss `0.1582` vs `0.1639`), and CPR significantly boosted TFT performance reaching a record low val loss of **`0.1521`** (RMSE 0.1404), officially beating xLSTM. TFT + CPR is the final selected model. See `model3_research/regularization_results.md` for details.

### Phase 4: HyperIV

```
   Reference options set               Query point
   (batch, n_ref=50, 3)               (tau, logm)
   [tau, logm, total_var]                  |
         |                                 |
   SetEmbeddingNetwork                     |
   Linear(3, 128)                          |
   TransformerEncoder                      |
   (2 layers, 4 heads)                     |
   Mean pooling                            |
         |                                 |
   context (batch, 128)                    |
         |                                 |
   HyperNetwork                            |
   MLP: 128 -> [generate                   |
   weights & biases for                    |
   TargetMLP layers]                       |
         |                                 |
   TargetMLP weights                       |
   (dynamically generated)                 |
         |                                 |
         +----------+---------------------+
                    |
              TargetMLP
              (2, 64, 32, 1)
              applied with generated weights
                    |
              Total variance prediction
```

**Key insight:** The hypernetwork generates *different* weights for each surface in the batch. This means every day gets a specialized predictor rather than sharing a single model.

### Phase 5: DDPM

```
   Current surface + noise          Market conditions
   x_t (batch, 1, 200)             (batch, 11)
         |                              |
   1D U-Net                        FiLM conditioning
   Encoder:                        (gamma * feature + beta)
     Conv1d(1, 64, k=3)               |
     + FiLM + ResBlock                 |
     Conv1d(64, 128, k=3, s=2)        |
     + FiLM + ResBlock                 |
     Conv1d(128, 256, k=3, s=2)       |
     + FiLM + ResBlock                 |
   Bottleneck:                         |
     Conv1d(256, 256)                  |
   Decoder:                            |
     ConvT(256, 128, s=2)             |
     + FiLM + ResBlock + skip          |
     ConvT(128, 64, s=2)              |
     + FiLM + ResBlock + skip          |
     Conv1d(64, 1, k=1)               |
         |                              |
   epsilon_theta                  Sinusoidal time embedding
   (predicted noise)              t -> (embed_dim,)
         |                              |
         +--------- condition ----------+
                    |
              x_{t-1} = denoise(x_t, epsilon_theta, t)
              (repeat 1000 steps for generation)
```

**Noise schedule:** Cosine schedule (`beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}`), 1000 timesteps.

**FiLM conditioning:** Each U-Net block receives `condition = time_embed + market_features` and applies Feature-wise Linear Modulation: `gamma * x + beta` where gamma and beta are projected from the condition vector.

**Condition features (11 dims):** Base 4 (VIX level, VIX change, underlying return, realized vol) + 7 enhancement features (S&P 500 return, IV term slope, IV skew, VRP, futures basis, realized vol 20d, institutional net ratio).

## Transfer Learning (`transfer.py`)

All five training scripts support `--finetune <path>` to initialize from a pretrained checkpoint. This is implemented in `src/transfer.py` with three utilities:

### `load_finetune_weights(model, checkpoint_path, device)`
- Loads state dict from checkpoint
- For matching-shape parameters: copies weights directly
- For dimension-mismatched parameters (e.g., GRU input 6→13): initializes with Xavier uniform, then copies overlapping weights from the pretrained model
- Returns `(transferred_params, reinitialized_params)` name sets

### `setup_finetune_optimizer(model, transferred, reinitialized, base_lr, new_lr)`
- Creates optimizer with **differential learning rates**:
  - Pretrained (transferred) layers: `base_lr` (typically lr × 0.1)
  - New/reinitialized layers: `new_lr` (full learning rate)
- This prevents catastrophic forgetting of pretrained features while allowing new dimensions to learn quickly

### `freeze_transferred(model, transferred_names)`
- Optional: completely freezes pretrained parameters (sets `requires_grad=False`)
- Not used in current training but available for fine-tuning experiments

### Dimension Mismatch Handling
When the model architecture changes (e.g., adding enhancement features), layers that change size are handled gracefully:
- The overlapping portion of weights is copied from the pretrained model
- New dimensions are initialized with Xavier uniform random values
- This allows models to benefit from prior training even when input/output dimensions change

## Testing Strategy

### Overview

215 tests across 10 test files, all running in ~4 seconds on CPU.

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Float64 everywhere** | `conftest.py` sets `torch.set_default_dtype(torch.float64)` session-wide |
| **Tiny architectures** | `hidden_sizes=[5,5,5]`, `ensemble_num=2` for sub-second tests |
| **No file I/O** | Dataset tests inject mock DataFrames; model tests use synthetic tensors |
| **Autograd-safe** | Never wrap SmileModel forward passes in `torch.no_grad()` |
| **Regression coverage** | 18 tests named after bug IDs (M1-M5, X1-X3, T1-T2, E1) |

### Test Distribution

```
test_model.py              52 tests   Base model classes & losses
test_utils.py              22 tests   Utilities, metrics, early stopping
test_adjustment.py         21 tests   GRU+Attention model
test_diffusion.py          19 tests   DDPM components
test_model_regression.py   18 tests   Bug regression tests
test_dgm.py                17 tests   DGM PDE solver
test_structural_break.py   16 tests   Change-point detection
test_hyperiv.py            15 tests   HyperIV hypernetwork
test_dataset.py            14 tests   DataProcessor pipeline
test_train_integration.py  10 tests   End-to-end training loops
                          ---
                          215 total
```

### What Tests Verify

- **Forward pass shapes and dtypes** — every model class is tested for correct output dimensions
- **Gradient flow** — backward pass completes without error, gradients are non-zero
- **Loss computation** — each loss component returns a scalar with grad_fn
- **Bug regressions** — specific bug fixes are locked in with dedicated tests
- **Training loop** — `train_one_epoch` and `validate` functions produce decreasing loss
- **Data pipeline** — chronological splitting, no label leakage, correct feature engineering

## Training Performance Summary

> **Status (2026-02-22):** Model 1 results are current (trained on 2014-2020 / 2021 test). Model 3 architecture comparison is complete, overfitting regularization in progress. Model 2 direction confirmed (ICNN Dupire), V1 in development. Models 4, 5 await retraining.

| Model | Status | Key Test Metric | Notes |
|-------|--------|----------------|-------|
| Base (SSVI+NN) | **Current** | Val loss 0.117 (pipeline 3ep), Butterfly 74% | prs_dataset_no_fat(clean), 2014-2020 |
| Adjustment (xLSTM) | **Arch. comparison done** | Val RMSE 0.1414, MAPE 9.01% | 39K params, highly efficient |
| Adjustment (TFT+CPR) | **Arch. comparison done** | Val RMSE 0.1404, MAPE 8.89% | 265K params, **Best Model**, excellent interpretability |
| Adjustment (GRU) | **Arch. comparison done** | Val RMSE 0.1477, MAPE 9.43% | 59K params, baseline reference |
| HyperIV | Pending retrain | — | Independent model |
| ICNN Dupire | **V1 in development** | — | ICNN local vol extractor, see `model2_research/action_plan.md` |
| DDPM | Pending retrain | — | condition_dim=11 |

### Known Issues
- **Base model gradient explosion:** SSVI parameter optimization becomes unstable after extended training. Best model is saved before instability. Aggressive early stopping (patience=50) is recommended.
- **Adjustment GPU OOM:** Data preparation runs base model autograd on all ~254K rows. Now mitigated with chunked inference (5000 rows/batch) in `dataset.py:prepare_adjustment_data()`.

## File Index

| File | Purpose |
|------|---------|
| `src/config.ini` | All hyperparameters, paths, data splits |
| `src/model.py` | SSVIModel, SmileModel, SingleModel, MultiModel, SoftmaxModel, losses |
| `src/dataset.py` | DataProcessor: loading, features, per-date yATM, splits, loaders |
| `src/train.py` | Base model training loop |
| `src/experiment.py` | Base model evaluation + plots |
| `src/test.py` | Base model evaluation + arbitrage violation checks |
| `src/dgm.py` | DGM PDE solver model + sampler **(legacy, retained for reference)** |
| `src/train_dgm.py` | DGM training script **(legacy)** |
| `src/dupire_pinn.py` | Dupire PINN local vol extractor **(planned, V1)** |
| `src/dupire_icnn.py` | ICNN Dupire with hard convexity **(planned, V2)** |
| `src/adjustment.py` | AdjustmentModel (GRU+Attention) |
| `src/train_adjustment.py` | Adjustment training (chronological split) |
| `src/hyperiv.py` | HyperIV (Transformer + Hypernetwork) |
| `src/train_hyperiv.py` | HyperIV training script |
| `src/diffusion.py` | DDPM (1D U-Net + FiLM conditioning) |
| `src/train_diffusion.py` | DDPM training script |
| `src/transfer.py` | Transfer learning utilities |
| `src/structural_break.py` | Change-point detection (PELT algorithm) |
| `src/utils.py` | Config loading, metrics, early stopping, seed |
| `scripts/download_data.py` | Data download pipeline (FinMind + yfinance) |
| `scripts/build_features.py` | Enhancement feature computation |
| `scripts/compare_architectures.py` | Additive vs multiplicative A/B test |
| `scripts/plot_smooth_iv_check.py` | Fixed-yATM smooth surface verification |
| `scripts/plot_training_curves.py` | Training loss curve visualization |
| `scripts/inspect_ssvi_params.py` | SSVI parameter inspection |
| `scripts/diagnose_rho_gradient.py` | Per-loss rho gradient analysis |
| `scripts/train_diagnose.py` | Training with per-epoch parameter tracking |
| `model3_research/xlstm_adjustment.py` | xLSTM (mLSTM) adjustment model |
| `model3_research/tft_adjustment.py` | TFT adjustment model |
| `model3_research/optimizers.py` | Custom optimizers (AdamCPR, CautiousAdamW) |
| `model3_research/scripts/train_models.py` | Unified training script for architecture comparison & regularization |
| `model3_research/scripts/run_tft_experiments.py` | Launcher script for running all TFT models |
| `model3_research/scripts/plot_loss_curves.py` | Train/val loss curve visualization |
| `model3_research/scripts/plot_regularization_results.py` | Evaluation and plotting for the regularization research |
| `model3_research/scripts/benchmark_dtype.py` | float32 vs float64 speed benchmark |

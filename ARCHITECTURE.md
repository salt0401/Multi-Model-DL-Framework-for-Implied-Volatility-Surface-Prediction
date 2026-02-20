# System Architecture

## Pipeline Overview

```
                         Raw TXO Data (FinMind API + historical CSV)
                         + Enhancement Features (VIX, S&P, institutional)
                                        |
                                 DataProcessor
                                /      |       \
                          Train      Val       Test
                       (2014-2024)  (20%)    (2025-2026)
                            |
        +------------------+------------------+------------------+
        |                  |                  |                  |
    Phase 1            Phase 2            Phase 3           Phase 4/5
    Base Model         DGM PDE           Adjustment         HyperIV / DDPM
    (SSVI+NN)          Solver            (GRU+Attn)         (independent)
        |                  |                  |                  |
    MultiModel.pt      DGMModel.pt      AdjModel.pt      HyperIV.pt / Diffusion.pt
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
Raw CSV (prs_dataset_full.csv, 480K rows, 2014-2026)
  Sources: original 2014-2021 CSV + FinMind API (2022-2026)
  Created by: scripts/download_data.py
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
    - Train: 2014-01-01 to 2024-12-31
    - Test:  2025-01-01 to 2026-12-31
    - Val:   20% random split of train (within-period)
    |
    v
PyTorch DataLoaders
    - TensorDataset with float64
    - Batch size from config.ini
```

**Key design decision:** The train/test split is strictly chronological — no future data leaks into training. Validation is sampled randomly within the training period.

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

### Phase 2: DGM PDE Solver

```
        (S, t, sigma)  — 3-dim input
              |
         Linear(3, 64)
              |
         SLayer x3
         (LSTM-like gating:
          Z=update, G=forget,
          R=reset, H=candidate
          + LayerNorm)
              |
         Linear(64, 1)
              |
         u(S, t, sigma)
         (learned PDE solution)
```

**SLayer computation:**
```
z = sigmoid(W_z * x + U_z * s_prev)      # update gate
g = sigmoid(W_g * x + U_g * s_prev)      # forget gate
r = sigmoid(W_r * x + U_r * s_prev)      # reset gate
h = tanh(W_h * x + U_h * (s_prev * r))   # candidate
s_new = LayerNorm((1 - g) * h + z * s_prev)
```

**Loss = lambda_pde * L_pde + lambda_bc * L_bc + lambda_tc * L_tc**

- `L_pde`: PDE residual of backward Kolmogorov equation at interior points
- `L_bc`: boundary conditions at extreme S values
- `L_tc`: terminal condition `u(S, T, sigma) = payoff(S)` at expiration

Collocation points are re-sampled every 100 epochs via `DGMSampler`.

### Phase 3: Adjustment Model

```
   Sequence of daily IV features
   (batch, seq_len=20, 13 features)
              |
         GRU (2 layers, 64 hidden)
              |
   (batch, seq_len, 64) hidden states
              |
    TemporalAttention (4 heads)
    Q=K=V from hidden states
    Scaled dot-product attention
              |
   (batch, 64) context vector
              |
         Linear(64, output_dim)
              |
         SquarePlus activation
         (ensures positive output)
              |
    Adjustment ratio or residual
```

**Input features (12 dims):**
- Base (6): vix_change, underlying_return, logm, tau, tv_pred, itm_otm
- Enhancement (6): sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d

**Data preparation:** The `tv_pred` feature is computed by running the trained base model on all data points. Since `SmileModel.forward()` uses `autograd.grad(create_graph=True)` for second-order derivatives, this consumes significant memory. The implementation uses **chunked inference** (5000 rows per batch) to avoid GPU OOM.

**Three prediction modes:**
1. `ratio` — multiplicative: `adjusted_IV = base_IV * ratio`
2. `residual` — additive: `adjusted_IV = base_IV + residual`
3. `direct` — replace: `adjusted_IV = prediction`

> **Data leakage fix (2026-02-20):** The original `train_adjustment.py` used `random_split`, causing temporal leakage. Now uses chronological split + KDE fitted on train targets only. See `discussion_notes.md` §3.4.

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
   x_t (batch, 1, 200)             (batch, 13)
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

Results from the extended dataset experiment (train 2014-2024, test 2025-2026, transfer learning from Round 1 checkpoints). See `EXPERIMENT.md` Section 6 for full details.

| Model | Epochs | Key Test Metric | Training Time | Device |
|-------|--------|----------------|---------------|--------|
| Base (SSVI+NN) | 105/2000 | TV-RMSE 0.0120, MAPE 33.0% | ~9h | GPU |
| HyperIV | 69/500 | TV-RMSE 0.0056, MAPE 20.0% | ~30min | GPU |
| DGM | 5000/5000 | PDE residual 2.0e-5 | ~25min | GPU |
| DDPM | 1000/1000 | Surface RMSE 0.0072 (test) | ~8h | GPU |
| Adjustment | 1000/1000 | RMSE 52.20, MAPE 70.15% | ~46h | CPU |

### Known Issues
- **Base model gradient explosion:** SSVI parameter optimization becomes unstable after ~60 epochs on extended data. Best model is saved before instability. Aggressive early stopping (patience=50) is recommended.
- **Adjustment GPU OOM:** Data preparation runs base model autograd on all 463K rows. Now mitigated with chunked inference (5000 rows/batch) in `dataset.py:prepare_adjustment_data()`.

## File Index

| File | Purpose |
|------|---------|
| `src/config.ini` | All hyperparameters, paths, data splits |
| `src/model.py` | SSVIModel, SmileModel, SingleModel, MultiModel, SoftmaxModel, losses |
| `src/dataset.py` | DataProcessor: loading, features, per-date yATM, splits, loaders |
| `src/train.py` | Base model training loop |
| `src/experiment.py` | Base model evaluation + plots |
| `src/test.py` | Base model evaluation + arbitrage violation checks |
| `src/dgm.py` | DGM PDE solver model + sampler |
| `src/train_dgm.py` | DGM training script |
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

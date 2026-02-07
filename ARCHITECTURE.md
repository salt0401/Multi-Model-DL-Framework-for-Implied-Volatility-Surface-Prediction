# System Architecture

## Pipeline Overview

```
                              Raw TXO Data
                                  |
                           DataProcessor
                          /      |       \
                    Train      Val       Test
                   (2014-20)  (20%)    (2021)
                      |
      +---------------+---------------+---------------+
      |               |               |               |
  Phase 1         Phase 2         Phase 3         Phase 4/5
  Base Model      DGM PDE         Adjustment      HyperIV / DDPM
  (SSVI+NN)       Solver          (GRU+Attn)      (independent)
      |               |               |               |
  MultiModel.pt   DGMModel.pt   AdjModel.pt     HyperIV.pt / Diffusion.pt
      |               |               |               |
      +-------+-------+-------+-------+               |
              |                                        |
         test.py                              train_hyperiv.py
         experiment.py                        train_diffusion.py
              |                                        |
      IV Surface Predictions                  Surface Forecasts
```

## Data Pipeline

### DataProcessor (`dataset.py`)

Handles all data loading, feature engineering, and splitting:

```
Raw CSV/PKL (2009-2023 TXO options)
    |
    v
Column Handling
    - Rename Chinese columns to English
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
Beta-Tau Estimation
    - Linear regression of SSVI betas on tau
    - Merge beta predictions back into main dataset
    |
    v
Chronological Split (no leakage)
    - Train: 2014-01-01 to 2020-12-31
    - Test:  2021-01-01 to 2021-12-31
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

### Phase 1: Base Model

```
                    logm (1-dim)
                         |
              +----------+----------+
              |          |          |
          SmileModel  SmileModel  ... (x5 ensemble)
              |          |          |
          [64]-[32]-[16]-[1]     (3-layer MLP, LayerNorm)
              |          |          |
              |     autograd.grad   |
              |     (1st & 2nd      |
              |      derivatives)   |
              |          |          |
          (TV, dTV/dk, d2TV/dk2, BSModel_baseline)
              |          |          |
              +-----+----+----------+
                    |
             SoftmaxModel
             (learned weights,
              input: logm, tau, yATM)
                    |
             Weighted sum of
             ensemble predictions
                    |
             WeightedSumLoss
             = w1*data + w2*ssvi + w3*calendar
               + w4*butterfly + w5*density + w6*smooth
```

**SmileModel details:**
- Input: logm (scalar) -> detach + requires_grad_(True)
- 3 hidden layers: 64 -> 32 -> 16, each with LayerNorm
- Output: 1 scalar (total variance)
- Computes 1st derivative via `autograd.grad(..., create_graph=True)`
- Computes 2nd derivative via `autograd.grad(..., retain_graph=True)`
- Returns 4-tuple: `(TV, grad1, grad2, BS_baseline)`

**Loss components:**
1. **Data loss** — MSE between predicted and observed total variance
2. **SSVI loss** — MSE between NN output and SSVI parametric prediction
3. **Calendar loss** — penalizes `dw/dtau < 0` (variance must increase with time)
4. **Butterfly loss** — penalizes negative probability density (convexity constraint)
5. **Density loss** — penalizes `|d2w/dk2|` (smoothness of the density)
6. **Smoothness loss** — L2 regularization on curvature

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
   (batch, seq_len=20, features)
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

**Three prediction modes:**
1. `ratio` — multiplicative: `adjusted_IV = base_IV * ratio`
2. `residual` — additive: `adjusted_IV = base_IV + residual`
3. `direct` — replace: `adjusted_IV = prediction`

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
   x_t (batch, 1, 200)             (batch, 4)
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

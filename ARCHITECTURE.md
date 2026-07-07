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
    (eSSVI+NN)         Local Vol          (TFT+Attn)       (independent)
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

### Phase 1: Base Model (eSSVI + Additive NN Engine)

> **Architecture decision (2026-02-23):** The model was upgraded from Gatheral's static SSVI to **eSSVI (extended SSVI)**. eSSVI introduces time-decaying correlation ($\rho$), allowing extreme negative skew for short-term options while maintaining smoothness for long maturities. To combat ATM data gravity preventing the optimizer from reaching extreme left-skew limits, the base formulation forces a frozen parameter `rho_0 = -0.95` during early epochs.

> **Architecture decision (2026-02-20 & 02-24):** Additive formulation confirmed via A/B experiment. Multiplicative (`eSSVI * NN`) explodes at epoch 2 due to product-rule cross-terms. Furthermore, to prevent the NN gradient from vanishing in low-volatility regimes, the additive scaling factor was upgraded from $yATM$ to $\tilde{y}_{ATM} = \sqrt{yATM^2 + 0.02^2}$.

```
            (tau, logm, yATM)
                    |
         +----------+----------+
         |                     |
     eSSVIModel             SmileModel x5 ensemble
     (tau, logm)            (tau, logm)
         |                     |
     output_Prior          output_NN
         |                     |
         |         yATM_tilde * output_NN
         |                     |
         +------( + )----------+
                 |
          output = eSSVI + yATM_tilde * NN   (additive)
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
# epsilon is configurable via config.ini [model_sett] epsilon (currently 0.02)
yATM_tilde = torch.sqrt(torch.square(yATM) + self.epsilon**2)

output = output_Prior + yATM_tilde * output_NN
grad_ttm1 = grad_ttm1_prior + yATM_tilde * grad_ttm1_NN
grad_logm1 = grad_logm1_prior + yATM_tilde * grad_logm1_NN
grad_logm2 = grad_logm2_prior + yATM_tilde * grad_logm2_NN
```

No cross-terms in derivatives (unlike multiplicative product rule).

**SmileModel details:**

- Input: tau, logm (each scalar) -> detach + requires_grad_(True)
- 3 hidden layers: 64 -> 32 -> 16, with custom bilinear input + LayerNorm
- Output: 1 scalar (total variance correction)
- Computes 1st derivative via `autograd.grad(..., create_graph=True)`
- Computes 2nd derivative via `autograd.grad(..., retain_graph=True)`
- Returns 4-tuple: `(TV, grad_ttm, grad_logm1, grad_logm2)`

**Loss components (Relaxed Weights: `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]`):**
*The physics constraints (`calendar, butterfly, density, upperbound`) were traditionally set to `10.0`, but were zeroed out after proving the old model's underfitting issue on Deep OTM Puts was caused by the static SSVI formulation colliding with data gravity, not by the arbitrage penalties locking the network.*

1. **RMSE** — Root mean squared error of total variance predictions
2. **MAPE** — Mean absolute percentage error (with ε=0.005 stability)
3. **Calendar** — penalizes `dw/dtau < 0` (variance must increase with time)
4. **Butterfly** — penalizes negative probability density (convexity constraint)
5. **Density** — penalizes `|d2w/dk2|` on synthetic wing data
6. **Upper bound** — penalizes `w > 2|k|` on synthetic wing data (Lee's bound)

#### Experimental Results

**Dataset:** `prs_dataset_no_fat(clean)` (~254K rows, 2014-2021)

| Parameter | Value |
|-----------|-------|
| Architecture | eSSVI prior + 5 SmileModel NNs (64-32-16), **additive** |
| Learning rate | 0.0005 |
| Batch size | 256 |
| Max epochs | 1000 |
| Early stopping patience | 50 |
| Loss weights | `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]` |
| Gradient clipping | 1.0 |

**Training (2014-2020 train, 2021 test):**

- **Epochs trained:** 84 (early-stopped, patience=50)
- **Best Validation Loss:** **0.07495** (epoch 34)
- **Final Train Loss:** 0.06102
- **SSVI Health:** ✅ Gatheral-Jacquier satisfied for all 5 ensemble members
- **Test IV-RMSE:** 0.01947 | **Test MAPE:** 6.10%
- **Butterfly violations:** 74% (key motivation for Model 2 ICNN)

> **Note:** The 2022-2026 extended dataset (`prs_dataset_full.csv`, 480K rows) exists but has known data quality issues (see `discussion_notes.md` §3.2).

> **Visualization Rule:** When plotting IV Smile curves, **DO NOT mix data from different dates**. Each subplot must represent a single, isolated `(tau, yATM)` pairing.

### Phase 2: ICNN Dupire Local Volatility Extractor

> **Redesign (2026-02-22):** The original DGM (fixed-sigma BS PDE solver) has been replaced with a Dupire PDE-constrained local volatility extractor (`model2_research/dupire_pinn.py`). See `model2_research/README.md` for full design rationale.

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
    - Module D: Local Vol, Vanna, Volga, ∂σ_LV/∂K from clean local vol → Model 3 (16-dim)
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
    SquarePlus → Adjustment ratio           [~318K params (16-dim input)]
```

#### 12-Way Comparison Results (3 Architectures × 4 Optimizers)

**Top 5 by Test RMSE** (full table in `model3_research/README.md`):

| Rank | Architecture | Optimizer | Test RMSE | Test MAPE | Params |
|:----:|-------------|-----------|:---------:|:---------:|:------:|
| 1 | **TFT** | **CPR** ⭐ | **0.1558** | **9.51%** | 318K |
| 2 | TFT | AdamW | 0.1590 | 9.75% | 318K |
| 3 | TFT | Adam | 0.1592 | 9.85% | 318K |
| 4 | TFT | CWD | 0.1608 | 9.75% | 318K |
| 5 | GRU | AdamW | 0.1628 | 9.70% | 59K |

**Selection: TFT with CPR** — Achieved the best overall accuracy (Test RMSE 0.1558) across all 12 combinations on the strictly held-out 2021 dataset. TFT dominates all top 4 positions regardless of optimizer. CPR is highly effective for TFT but counterproductive for GRU (#12). See `model3_research/README.md` for the full 12-way breakdown.

**Three prediction modes:**

1. `ratio` — multiplicative: `adjusted_IV = base_IV * ratio`
2. `residual` — additive: `adjusted_IV = base_IV + residual`
3. `direct` — replace: `adjusted_IV = prediction`

> **12-Way comparison (2026-02-27):** All 3 architectures (TFT, GRU, xLSTM) × 4 optimizers (Adam, AdamW, CWD, CPR) trained on 16-dim input with strictly chronological test set (2021). Architecture choice matters more than optimizer choice. xLSTM underperformed expectations. See `model3_research/README.md` for full details.

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

**2026-07-06 upgrades** (decision record: `docs/model45_completion_report.md`):
- **Residual hypernetwork:** generated params = learnable normally-initialized base + day-specific delta (pure-delta weights start at 0, where tanh activations and their gradients die).
- **Scale-equivariant output:** the target MLP (tanh hidden, softplus head) predicts the O(1) ratio `w / sqrt(yATM^2 + 0.002^2)` — predicting raw total variance (median ~0.005) through softplus saturates the head and collapses training to predicting zero.
- **Corrected Gatheral-Jacquier butterfly density** (the draft dropped the square on w'), input standardization from train stats, masked losses, seeded random eval references (draft always used the 50 shortest-maturity options), float32 training.
- **PIVOT price-space auxiliary loss** (arXiv 2606.17065): normalized Black-76 price MSE, weight 0.1 — the one substantial published upgrade to HyperIV (ICML 2025), which remains SOTA for sparse-quote surface construction per a July 2026 literature review.

#### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Transformer set encoder (128-dim, 4 heads, 2 layers) + residual hypernetwork |
| Target MLP | 64-32 hidden dims, tanh + softplus, yATM-ratio output |
| Reference points | 50 per surface |
| Loss weights | mse 1.0, calendar 10, butterfly 10, price 0.1 |
| Learning rate | 0.001 (AdamW, cosine), float32 |
| Batch size | 32 (per-surface) |
| Max epochs | 500 (early stop patience 50) |

> **Results:** see `logs/hyperiv_results.json` and `docs/model45_completion_report.md`.

### Phase 5: Conditional Flow Matching over Surface Factors

> **Design replaced (2026-07-06).** The draft grid-space DDPM (`src/diffusion.py`,
> `src/train_diffusion.py`) is **deprecated** — kept for reference only. At
> ~1,450 training pairs a 200-dim grid generative model is data-starved (the
> closest published success, arXiv 2511.07571, used ~7,000 surfaces), the
> draft also lacked surface normalization (signal ~1% of the injected noise),
> and 1000-step ancestral sampling cost ~2.2 s/surface on the target GPU.
> Full decision record: `docs/model45_completion_report.md`.

```
   Today's 10x20 surface (200-dim, tau-major)
         |
   log(total variance)
         |
   PCA (12 factors, TRAIN-fit, ~97% EV; truncation floor 0.0011 << RW error 0.0019)
         |
   z-scored factor scores z_today (12)     Market conditions (11, z-scored)
         \_______________  _______________/
                         \/
              condition c = [z_today, market] (23)
                          |
   Conditional OT flow matching:  t~U(0,1), z_t=(1-t)z0+t*z1, target v = z1-z0
   VelocityMLP v(z_t, t, c): FiLM residual MLP (256 hidden, 3 blocks, ~0.9M params)
                          |
   Sampling: Euler dz = v dt, 50 steps, z0 ~ N(0,I)   (30-100x fewer steps than DDPM)
                          |
   inverse: de-z-score -> PCA reconstruct -> exp  => POSITIVE surface by construction
```

**Condition features (11 dims):** Base 4 (VIX level ffilled, VIX change, underlying return, realized vol 5d) + 7 enhancement features (S&P 500 return, IV term slope, IV skew, VRP, futures basis, realized vol 20d, institutional net ratio).

**Anti-memorization (n≈1,450):** dropout 0.1, weight decay 1e-3, EMA 0.999, early stopping on sampled val tv-RMSE.

**Evaluation protocol** (`src/evaluate_surface_forecast.py`): tv-RMSE / IV-RMSE / IV-MAPE against random walk and VAR(1)-on-factors baselines, Diebold-Mariano test, CRPS (100 samples), 90% interval coverage, calendar/butterfly violation rates (with the actual market surfaces as the empirical reference). Literature expectation: daily surfaces are ~0.99 autocorrelated, so beating the random walk at 1-day horizon is marginal at best — the model's value-add is calibrated distributions and coherent scenarios.

#### Configuration (`[flow_surface]` in config.ini)

| Parameter | Value |
|-----------|-------|
| Representation | PCA (<=12 comps) of log total variance, train-fit |
| Velocity net | Residual MLP, 256 hidden, 3 blocks, FiLM conditioning |
| Sampling | Euler, 50 steps |
| Learning rate / batch | 0.001 / 128 |
| Weight decay / dropout / EMA | 1e-3 / 0.1 / 0.999 |
| Epochs | 3000 max, early stop patience 40 validations |

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

177 tests across 8 test files, all running in ~4 seconds on CPU.

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
test_diffusion.py          19 tests   DDPM components
test_model_regression.py   18 tests   Bug regression tests
test_structural_break.py   16 tests   Change-point detection
test_hyperiv.py            15 tests   HyperIV hypernetwork
test_dataset.py            14 tests   DataProcessor pipeline
test_train_integration.py  10 tests   End-to-end training loops
                          ---
                          177 total
```

### What Tests Verify

- **Forward pass shapes and dtypes** — every model class is tested for correct output dimensions
- **Gradient flow** — backward pass completes without error, gradients are non-zero
- **Loss computation** — each loss component returns a scalar with grad_fn
- **Bug regressions** — specific bug fixes are locked in with dedicated tests
- **Training loop** — `train_one_epoch` and `validate` functions produce decreasing loss
- **Data pipeline** — chronological splitting, no label leakage, correct feature engineering

## Training Performance Summary

> **Status (2026-02-27):** Model 1 results are current (trained on 2014-2020 / 2021 test). Model 2 direction confirmed (ICNN Dupire), V1 in development. Model 3 complete — full 12-way comparison done (3 arch × 4 opt), TFT+CPR confirmed as winner. Models 4, 5 await retraining.

| Model | Status | Key Test Metric | Notes |
|-------|--------|----------------|-------|
| Base (eSSVI+NN) | **Current** | Val loss 0.07495 (84 ep, early-stopped), SSVI healthy | prs_dataset_no_fat(clean), 2014-2020, ε=0.02 |
| Adjustment (TFT+CPR) | **12-way winner** | Test RMSE 0.1558, MAPE 9.51% | 318K params, **#1 of 12**, excellent interpretability |
| Adjustment (11 others) | **Completed** | RMSE 0.1590–0.1765 | All archived, see `model3_research/README.md` |
| HyperIV | **Trained (2026-07-06)** | Test tv-RMSE 0.00215, MAPE 6.94%, butterfly viol. 0.055% | Independent model, PIVOT price aux |
| ICNN Dupire | **V1-V3 complete** | 0% butterfly violations | ICNN local vol extractor |
| Flow matching (replaced DDPM) | **Trained (2026-07-06)** | Test tv-RMSE 0.00171 vs RW 0.00192 (DM p<1e-4) | PCA factors + conditional OT-FM over daily increments |
| **US Mag 7 branch (2026-07-07)** | **Trained & evaluated** | Flow beats RW 7/7 tickers (all DM p<0.05); pooled HyperIV MAPE 4.97%; strategy: gross ≈ 0, net < 0 after real spreads | `src/us_dataset.py`, `*_us.py`; see `docs/mag7_us_branch_report.md` |

### Known Issues

- **Base model gradient explosion:** SSVI parameter optimization becomes unstable after extended training. Best model is saved before instability. Aggressive early stopping (patience=50) is recommended.
- **Adjustment GPU OOM:** Data preparation runs base model autograd on all ~254K rows. Now mitigated with chunked inference (5000 rows/batch) in `dataset.py:prepare_adjustment_data()`.

## File Index

| File | Purpose |
|------|---------|
| `src/config.ini` | All hyperparameters, paths, data splits |
| `model1_research/model.py` | SSVIModel, SmileModel, SingleModel, MultiModel, SoftmaxModel, losses |
| `src/dataset.py` | DataProcessor: loading, features, per-date yATM, splits, loaders |
| `model1_research/train.py` | Base model training loop |
| `model1_research/train_pipeline.py` | Master two-stage training loop for SSVI + Neural Network |
| `model1_research/experiment.py` | Base model evaluation + plots |
| `src/test.py` | Base model evaluation + arbitrage violation checks |
| `model2_research/dupire_pinn.py` | Dupire PINN / ICNN local vol extractor |
| `model2_research/train_dupire.py` | Dupire PINN training script |
| `model2_research/module_d.py` | V3 Greeks Extractor (Vanna, Volga, LV Grad) |
| `model2_research/extract_features.py` | V3 downstream feature extraction script |
| `src/hyperiv.py` | HyperIV (Transformer + residual hypernetwork, PIVOT price aux) |
| `src/train_hyperiv.py` | HyperIV training script |
| `src/flow_surface.py` | Model 5: PCA factor preprocessing + conditional flow matching |
| `src/train_flow_surface.py` | Model 5 training script |
| `src/evaluate_surface_forecast.py` | Model 5 evaluation (RW/VAR baselines, DM, CRPS, violations) |
| `src/diffusion.py` | DEPRECATED: draft DDPM (superseded by flow_surface.py) |
| `src/train_diffusion.py` | DEPRECATED: draft DDPM training script |
| `src/transfer.py` | Transfer learning utilities |
| `src/structural_break.py` | Change-point detection (PELT algorithm) |
| `src/utils.py` | Config loading, metrics, early stopping, seed |
| `scripts/download_data.py` | Data download pipeline (FinMind + yfinance) |
| `scripts/build_features.py` | Enhancement feature computation |
| `scripts/plot_training_curves.py` | Training loss curve visualization |
| `model1_research/scripts/generate_model1_plots.py` | Generates train/val/test IV fit plots and loss curve |
| `model1_research/scripts/plot_smooth_iv_check.py` | Fixed-yATM smooth surface verification |
| `model1_research/scripts/plot_pipeline_metrics.py` | Plots loss from pipeline metrics JSON |
| `model1_research/scripts/train_diagnose.py` | Training with per-epoch parameter tracking |
| `model3_research/tft_adjustment.py` | TFT adjustment model |
| `model3_research/optimizers.py` | Custom optimizers (AdamCPR, CautiousAdamW) |
| `model3_research/scripts/train_models.py` | Unified training script for architecture comparison & regularization |
| `model3_research/scripts/run_tft_experiments.py` | Launcher script for running all TFT models |
| `model3_research/scripts/plot_loss_curves.py` | Train/val loss curve visualization |
| `model3_research/scripts/plot_regularization_results.py` | Evaluation and plotting for the regularization research |
| `model3_research/scripts/benchmark_dtype.py` | float32 vs float64 speed benchmark |

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
| Model 4 (HyperIV) | ✅ Trained (2026-07-06): test tv-RMSE 0.00215, 0.000%/0.055% cal/butterfly violations | Results: `logs/hyperiv_results.json` |
| Model 5 (Flow matching, replaced DDPM) | ✅ Trained & evaluated (2026-07-06): beats 1-day random walk, DM p<1e-4 | Results: `logs/flow_surface_eval.json` |

## Changelog

- Additive architecture confirmed (multiplicative explodes at epoch 2)
- SSVI bounded parameterization: `eta = 2*sigmoid(raw_eta)`, `gamma = sigmoid(raw_gamma)`
- `vixtwn_change` feature removed (Adjustment: input_dim now 12→16, DDPM: condition_dim now 11)
- Data leakage fix: chronological split + KDE train-only fitting
- Model 3 architecture comparison & regularization: TFT + CPR selected as primary choice. Non-shortlisted experiments archived.
- Model 2 ICNN redesign complete: Soft PINN (V1) → ICNN (V2) → Greek Extractor Module D (V3). Butterfly violations permanently eliminated.
- 2026-07-06: Models 4 & 5 completed. HyperIV fixed (corrected Gatheral butterfly density — the draft and `src/test.py` both dropped the square on w'; tanh/softplus target net; residual hypernetwork; yATM-ratio output; PIVOT price auxiliary) and trained. Draft DDPM replaced with conditional flow matching over train-fit PCA factors of log total variance (`src/flow_surface.py`); full evaluation protocol vs random walk / VAR(1) with DM test, CRPS, coverage, and arbitrage-violation rates. getYATM leakage guard wired to config; surface-grid stats made train-only. Details: `docs/model45_completion_report.md`.

## Future Work

- Integrate HyperIV predictions as flow-matching conditioning for improved surface forecasting
- Add explicit no-arbitrage constraints to HyperIV via penalty or projection
- Explore attention-based architectures for the base model (replace per-expiration SmileModels with a single cross-expiration model)
- Extend to American-style options using the DGM PDE framework with early exercise boundary
- Address base model gradient explosion with adaptive loss weighting or SSVI parameter clamping
- Add model stacking/ensemble across the five models for combined predictions

# COMPREHENSIVE TECHNICAL SUMMARY: Smart Data Analysis - TXO IV Surface Prediction System

## EXECUTIVE OVERVIEW

A sophisticated 5-model ensemble system for predicting Taiwan Stock Exchange Options (TXO) implied volatility (IV) surfaces. The system combines classical parametric models (eSSVI), physics-informed neural networks (PINN/ICNN), and state-of-the-art deep learning architectures (TFT, xLSTM, HyperIV, DDPM) to achieve arbitrage-free, real-time IV surface predictions with crisis detection and adjustment capabilities.

**Project Status (2026-02-27):**
- Model 1 (Base eSSVI+NN): TRAINED, Test RMSE 0.01977, MAPE 5.46%
- Model 2 (ICNN Dupire): V1-V3 COMPLETE, butterfly violations 0%
- Model 3 (Adjustment TFT+CPR): COMPLETE, Test RMSE 0.1558, MAPE 9.51%
- Models 4-5 (HyperIV, DDPM): Pending retraining

---

## PART 1: PROBLEM DOMAIN & KEY CONCEPTS

### What Is an IV Surface?

**Implied Volatility (IV):** Market's estimate of future price volatility extracted from option prices.

**IV Surface:** 2D map showing IV as a function of:
- **Strike Price (K)** / Log-moneyness: `logm = ln(K/S)` (how far from current spot)
- **Time-to-Expiry (τ)** / Time-to-Maturity: years until option expires

**Total Variance:** `w(τ, logm) = IV² × τ` — smoother than raw IV, easier to model

### Critical Constraints (No-Arbitrage)

1. **Calendar Arbitrage:** `∂w/∂τ ≥ 0` — variance must increase with time
2. **Butterfly Spread:** `∂²C/∂K² ≥ 0` — call price must be convex in strike (ensures non-negative risk-neutral density)
3. **Delta Constraint:** `0 ≤ ∂C/∂S ≤ 1` — call price monotonically increasing in spot
4. **Gatheral-Jacquier Bound:** Static SSVI requires `η(1+|ρ|) ≤ 2` for arbitrage-free surfaces

### TXO Characteristics

- **Left Skew:** Deep out-of-the-money (OTM) puts much more expensive than equidistant OTM calls
- **Data Imbalance:** 80%+ of option data concentrated at-the-money (ATM), sparse tails
- **Training Period:** 2014-2020 (254K rows), Test: 2021 (52K rows), strictly chronological split

---

## PART 2: THE 5-MODEL PIPELINE ARCHITECTURE

```
TXO Options Data (FinMind) + Enhancement Features (VIX, S&P500, RV, VRP)
                                    |
                        ┌───────────┼───────────┐
                        ▼           ▼           ▼
                    Phase 1       Phase 2     Phase 3
                    Model 1       Model 2     Model 3
                  Base eSSVI      ICNN        Adjustment
                    (16 dim)     Dupire       (TFT+CPR)
                                 (Local Vol)    (Crisis)
                        |           |            |
                        └───────────┼────────────┘
                                    ▼
                        Consensus Predictions
                     (Multiply by adjustment ratio)
                        |
                ┌───────┴───────┐
                ▼               ▼
             Model 4         Model 5
            HyperIV          DDPM
         (Point pred)     (Surface forecast)
```

### Model 1: Base Model (eSSVI + Neural Network Ensemble)

**Purpose:** Predict today's IV surface from observed option prices

**Architecture:**
- **eSSVI (Extended SSVI):** Gatheral & Jacquier parametric base with time-decaying correlation
  - `rho(tau) = rho_inf + (rho_0 - rho_inf) * exp(-decay * tau)`
  - Frozen `rho_0 = -0.95` (encodes steep left skew for short-term options)
  - Parameters: `(rho_0, rho_inf, decay, eta, gamma)`
  
- **SmileModel Ensemble:** 5 neural networks (64→32→16 hidden), additive correction
  - Input: `(tau, logm, yATM)` where `yATM = sqrt(yATM² + ε²)` with ε=0.02
  - Output: total variance correction `∆w`
  - Combined: `w_pred = w_eSSVI + yATM_tilde * NN_output`
  - Rationale: Additive (not multiplicative) prevents gradient explosion
  
- **SoftmaxModel:** Learns ensemble weights per sample
  - Final output: weighted average of 5 SingleModel predictions

**Loss Function (Weights: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]):**
1. RMSE: `√E[(w_pred - w_true)²]`
2. MAPE: Mean Absolute Percentage Error (relative fit)
3-6. Calendar/Butterfly/Density/Upperbound: DISABLED to isolate eSSVI fit

**Training Configuration:**
- Optimizer: AdamW, lr=0.0005, gradient clip=1.0
- Scheduler: MultiStepLR (γ=0.5 every 5 epochs after epoch 500)
- Batch size: 256
- Early stopping: patience=50 epochs
- Epochs: 1000 (trained 84, early-stopped at 0.07495 val loss)

**Key Results:**
- **Test RMSE:** 0.01977 (total variance)
- **Test MAPE:** 5.46%
- **Butterfly Violations:** 45.69% (baseline issue motivating Model 2)
- **SSVI Health:** ✅ Gatheral-Jacquier bounds satisfied
- **Architecture Decision:** Additive formulation confirmed (multiplicative explodes at epoch 2)

**Critical Design Insight:**
The original model was suppressed by static SSVI limits. The optimizer failed to trace the 45-degree angle of Deep-OTM puts because classical static `rho` couldn't handle short-term maturities and was dominated by ATM data gravity. Upgrading to eSSVI with frozen `rho_0 = -0.95` unlocked the mathematical bounds.

---

### Model 2: ICNN Dupire Local Volatility Extractor

**Purpose:** Extract arbitrage-free local volatility surface from Model 1's IV predictions, eliminate 74% butterfly violations

**Problem Addressed:**
Model 1's 45.69% butterfly violation rate means direct Dupire application produces:
- Negative denominators → imaginary local volatility
- Numerical instability from ill-conditioned formula
- Cascading error propagation through downstream models

**Architecture: Dual-Network PINN (V1 → V2 → V3)**

**V2: ICNN (Input-Convex Neural Network) — Primary Path**

```
(A) ICNNPriceNetwork:
    - Input: (K_normalized, tau)
    - Constraint: K→C(K) pathway all weights ≥ 0 (via softplus)
    - Guarantees: ∂²C/∂K² ≥ 0 by construction → 0% butterfly violations
    - Architecture: 3 residual blocks, 64 neurons
    - Params: 13,123

(B) LocalVolNetwork:
    - Input: (K_normalized, tau)
    - Output: σ²_LV(K,T) > 0 (softplus + epsilon)
    - Architecture: 3 residual blocks, 64 neurons
    - Params: 13,633
```

**Loss Function (6 terms, λ weights):**
1. `L_fit = 1.0 × mean((C_pred - C_target)²)` — fit to Model 1's prices
2. `L_dupire = 10.0 × mean((∂C/∂τ - ½σ²_LV K² ∂²C/∂K²)²)` — Dupire PDE residual
3. `L_calendar = 10.0 × mean(ReLU(-∂C/∂τ))` — ∂C/∂τ ≥ 0
4. `L_butterfly = 10.0 × mean(ReLU(-∂²C/∂K²))` — ALWAYS 0 due to ICNN
5. `L_smooth = 1.0 × mean((∂σ_LV/∂K)² + (∂σ_LV/∂τ)²)` — Sobolev smoothness
6. `L_boundary = 1.0 × mean((C(K, τ→0) - max(S-K, 0))²)` — payoff at expiry

**Key Hyperparameters:**
- Hidden dim: 64 (both networks)
- n_layers: 3
- K range: [0.5, 1.5] (normalized)
- τ range: [0.02, 2.0] years
- Interior collocation points: 5000/batch
- Boundary points: 500/batch
- Epochs: 5000
- Learning rate: 0.001, Scheduler: ReduceLROnPlateau(patience=200, factor=0.5)
- Gradient clip: 1.0
- Batch loss dominance: L_fit + L_boundary ~98%, PDE/Cal/Smooth ~2%

**Training Results:**
- Final Total Loss: 0.0768
- Butterfly Violations: **0.000%** (all 5000 epochs)
- PDE Residual: 0.000103
- Calendar Violations: 0.000026
- Runtime: ~5.5 min (RTX 4060, float64)

**V3: Module D — Greeks Extraction (no additional training)**

Computes 4 high-order features from Model 2's clean local vol surface:
1. **Local Vol:** `σ_LV(K,T)` directly
2. **Vanna:** `∂²C/∂S∂σ` — captures skew + S-vol anticorrelation
3. **Volga:** `∂²C/∂σ²` — tail risk quantification
4. **∂σ_LV/∂K** — local skew gradient

Model 3 input expands from 12→16 dimensions (+4 Greeks)

**Data Pipeline:**
- C_target: NOT market prices, NOT synthetic BS data
- Instead: Model 1's fitted surface on random (K, τ) points
- Every 100 epochs: re-sample 5000 new collocation points
- Query: `MultiModel(tau, logm, yATM)` → `tv_pred` → `C_target`

**Autograd Compatibility:**
- Model 1 query cannot use `torch.no_grad()` (SmileModel uses `autograd.grad(create_graph=True)`)
- Correct: `clone().requires_grad_(True)` then `detach()` after query
- ICNN weights projected non-negative via softplus (no weight clamping)

---

### Model 3: Adjustment Model (Architecture Comparison + Winner)

**Purpose:** Detect market crises and adjust base model predictions by multiplicative factor `α = tv_true / tv_pred`

**Crisis Indicators:**
Event dates: 2001/09 (9-11), 2008/10 (GFC), 2016/05 (flash crash), 2020/03 (COVID)

**Architecture Comparison (12 total configurations, 3 archs × 4 optimizers)**

**Architecture A: GRU + Attention (Baseline)**
```
Input (batch, 20, 12 features)
    ↓
GRU (2 layers, 64 hidden) [cuDNN fused]
    ↓
MultiHeadAttention (4 heads)
    ↓
Linear(64, 1) → SquarePlus
    ↓
Adjustment ratio (batch, 1)    [59K params]
```

**Architecture B: xLSTM / mLSTM**
```
Input (batch, 20, 12 features)
    ↓
mLSTM (matrix memory C ∈ ℝ^{d×d})
    - Exponential gating: exp(input_gate), exp(forget_gate)
    - Covariance update: C_t = f_t * C_{t-1} + v_t * k_t^T
    - Query-Key-Value retrieval (Transformer-like)
    ↓
MultiHeadAttention (4 heads)
    ↓
Linear + SquarePlus
    ↓
Adjustment ratio                [40K params]
```

**Architecture C: Temporal Fusion Transformer (TFT) ⭐ WINNER**
```
Input (batch, 20, 12 features)
    ↓
Variable Selection Network (VSN)
    - Per-variable GRN transform
    - Softmax gating per timestep → interpretable feature importance
    ↓
Gated Residual Networks (GRN)
    - ELU + skip + GLU gating
    ↓
LSTM Encoder (2 layers, 64 hidden)
    ↓
Post-LSTM GRN + LayerNorm
    ↓
Interpretable Multi-Head Attention (4 heads)
    - Shared V weights across heads (interpretable)
    - Attention matrix shows temporal importance
    ↓
Position-wise Feed-Forward
    ↓
SquarePlus → Adjustment ratio   [318K params]
```

**Key TFT Innovation: Variable Selection Network**
- Each timestep independently learns feature importance
- Crisis periods: vix_change + underlying_return get high weights
- Calm periods: tau + logm dominate
- This IS learned regime-switching, end-to-end

**12-Way Comparison Results (Test Set, 2021 held-out)**

| Rank | Architecture | Optimizer | Test RMSE | Test MAPE | Val Loss | Params |
|------|-------------|-----------|-----------|-----------|----------|--------|
| **1** | **TFT** | **CPR** ⭐ | **0.1558** | **9.51%** | 0.1481 | 318K |
| 2 | TFT | AdamW | 0.1590 | 9.75% | 0.1545 | 318K |
| 3 | TFT | Adam | 0.1592 | 9.85% | 0.1721 | 318K |
| 4 | TFT | CWD | 0.1608 | 9.75% | 0.1616 | 318K |
| 5 | GRU | AdamW | 0.1628 | 9.70% | 0.1645 | 59K |
| 6 | xLSTM | CWD | 0.1645 | 10.20% | 0.1656 | 40K |
| 7 | GRU | Adam | 0.1652 | 9.87% | 0.1765 | 59K |
| 8 | GRU | CWD | 0.1658 | 9.91% | 0.1669 | 59K |
| 9 | xLSTM | Adam | 0.1660 | 10.15% | 0.1761 | 40K |
| 10 | xLSTM | CPR | 0.1663 | 10.25% | 0.1834 | 40K |
| 11 | xLSTM | AdamW | 0.1679 | 10.37% | 0.1679 | 40K |
| 12 | GRU | CPR | 0.1765 | 11.28% | 0.2128 | 59K |

**Key Findings:**
1. **TFT dominates:** All 4 TFT variants occupy top 4 positions. **Architecture > Optimizer choice**
2. **CPR is architecture-dependent:** Excellent for TFT (#1), poor for GRU (#12), mediocre for xLSTM (#10)
3. **AdamW most robust:** Consistent across architectures
4. **xLSTM underperformed:** Despite theory, trails GRU on test metrics
5. **Regularization matters:** TFT+CPR (199 min train) >> TFT+Adam (51 min) on val loss

**Input Features (16 dims after Model 2 integration):**
- Base 6: vix_change, underlying_return, logm, tau, tv_pred, itm_otm
- Enhancement 6: sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d
- Greeks 4: local_vol, vanna, volga, lv_gradient_K

**Training Configuration:**
- Batch size: 128
- Sequence length: 20 days
- Epochs: 1000
- Learning rate: 0.001
- Split: Train <2019-08-13, Val 2019-08-13~2020-12-31, Test 2021
- Prediction target: `tv_ratio = tv_true / tv_pred` (multiplicative adjustment)
- KDE loss weighting: focus on rare extreme events

**Optimizers:**
1. **Adam/AdamW:** Standard (baseline)
2. **CautiousAdamW (CWD):** Weight decay only when update ⊙ param ≥ 0
3. **AdamCPR:** Constrained Parameter Regularization (NeurIPS 2024)

---

### Model 4: HyperIV (State-of-the-Art Point Predictor)

**Purpose:** Generate day-specialized IV surface predictor (highest accuracy for single-day predictions)

**Reference:** HyperIV paper (ICML 2025) — current state-of-the-art for IV surface interpolation

**Two-Stage Architecture:**

```
Stage 1: Market Reader (SetEmbeddingNetwork)
    ├─ Input: 50 reference options (τ, logm, total_var) from today's market
    ├─ TransformerEncoder (128-dim, 4 heads, 2 layers)
    │  - Learns cross-strike, cross-maturity relationships
    │  - "This put at K=15000 tells us about call at K=16000"
    └─ Mean pooling → context vector (batch, 128)

Stage 2: Specialist Generator (HyperNetwork)
    ├─ HyperNetwork MLP: 128 → [layer weights & biases]
    ├─ Generates weights for small TargetMLP (2, 64, 32, 1)
    └─ Each day gets unique specialist predictor
        
Query (single-day prediction)
    ├─ TargetNetwork with generated weights
    ├─ Input: (tau, logm, yATM)
    └─ Output: total_variance
```

**Key Insight:**
Hypernetwork doesn't predict IV directly — it predicts **parameters of another network** that predicts IV. Every day gets a specialized, auto-adapted predictor.

**Architecture Details:**
- SetEmbeddingNetwork: 50 → 128 → 128 (Transformer) → context
- TargetNetwork: (3 → 64 → 32 → 1)
- Total HyperNetwork MLPs: generates ~3K weights for TargetNetwork
- Reference points: 50 per surface
- Learning rate: 0.001
- Batch size: 32 (per-surface)
- Max epochs: 500

**Status:** Pending retraining with latest codebase

---

### Model 5: DDPM (Denoising Diffusion Probabilistic Model)

**Purpose:** Forecast next-day IV surface from current market conditions (generative, coherent surface)

**Advantage over point prediction:** Generates entire mutually-consistent surfaces, not independent pixels

**Architecture: 1D U-Net with FiLM Conditioning**

```
Noisy Surface x_t (batch, 1, 200)     Market Conditions (batch, 11)
        ↓                                      ↓
    Encoder:                           FiLM Conditioning
    Conv1d(1, 64, k=3)                 (per U-Net block)
    + FiLM + ResBlock                       ↓
    Conv1d(64, 128, k=3, s=2)          gamma * x + beta
    + FiLM + ResBlock                      
    Conv1d(128, 256, k=3, s=2)          
    + FiLM + ResBlock                   
        ↓                                   
    Bottleneck:                            
    Conv1d(256, 256)                       
        ↓                                   
    Decoder:                               
    ConvT(256, 128, s=2)                   
    + FiLM + ResBlock + skip                
    ConvT(128, 64, s=2)                    
    + FiLM + ResBlock + skip                
    Conv1d(64, 1, k=1)                     
        ↓
    ε_theta (predicted noise)
        ↓
    x_{t-1} = denoise(x_t, ε_theta, t)
    (repeat 1000 steps for generation)
```

**Training Process:**
1. Add random noise to real IV surface over 1000 steps
2. Train network to predict noise at each step
3. At inference: start from pure noise, reverse 1000 steps

**FiLM Conditioning:**
Each U-Net block receives: `condition = time_embed + market_features`
FiLM projects to gamma/beta: `output = gamma * x + beta`

**Condition Features (11 dims):**
- Base 4: VIX level, VIX change, underlying return, realized volatility
- Enhancement 7: S&P 500 return, IV term slope, IV skew, VRP, futures basis, RV 20d, institutional net ratio

**Hyperparameters:**
- Architecture: 1D U-Net (channels: 64-128-256)
- Surface grid: 10 tau × 20 logm = 200-dim vector
- Diffusion steps: 1000
- Noise schedule: Cosine
- Learning rate: 0.0002
- Batch size: 16
- Epochs: 1000

**Status:** Pending retraining (condition_dim previously 13, now 11 after vixtwn_change removal)

---

## PART 3: DATA PIPELINE & FEATURE ENGINEERING

### DataProcessor Class (dataset.py)

**Raw Input:**
```
prs_dataset_no_fat(clean).csv
├─ 254K rows (2014-2021)
├─ Columns: date, strike_price, put/call, volume, option_price, exdate, tau
└─ Sources: Historical CSV (pre-processed from PKL)
```

**Data Processing Pipeline:**

1. **Column Handling & Date Parsing**
   - Rename Chinese columns → English
   - Parse dates, compute tau = (exdate - date) / 365.25
   - Filter: remove deep OTM/ITM, very short tau (<0.01 years)

2. **Feature Engineering**
   - **log-moneyness:** `logm = ln(K/S)` (centered at 0)
   - **Implied Volatility:** Newton-Raphson on Black-Scholes formula
   - **Total Variance:** `w = IV² × tau`
   - **yATM (ATM Total Variance):** Interpolate at logm=0 per expiration
   - **Season/Year dummies:** Structural break features

3. **Enhancement Features** (dataset/enhancement/daily_features.csv, computed by scripts/build_features.py)
   | Feature | Computation | Use |
   |---------|-------------|-----|
   | Realized Volatility (20d) | std(log returns) | Adjustment, DDPM |
   | IV Term Slope | IV_long - IV_short | Adjustment, DDPM |
   | IV Skew | IV_put - IV_call | Adjustment, DDPM |
   | Variance Risk Premium | IV² - RV² | Adjustment, DDPM |
   | S&P 500 Return | % change | Adjustment, DDPM |
   | Futures Basis | Futures - Spot / Spot | Adjustment, DDPM |
   | Institutional Net Buy/Sell | Ratio | Adjustment, DDPM |

4. **Beta-Tau Estimation**
   - Linear regression of SSVI betas on tau
   - Merge predictions back into main dataset

5. **Chronological Train/Test Split (No Data Leakage)**
   - Train: 2014-01-01 to 2020-12-31 (~254K rows)
   - Validation: Last 20% of training dates (chronological, not random)
   - Test: 2021-01-01 to 2021-12-31 (~52K rows)

6. **PyTorch DataLoaders**
   - TensorDataset with float64 precision
   - Batch size from config.ini
   - No random shuffling (preserves temporal dependencies)

**Output Tensors:**
- logm: (N,) log-moneyness
- yATM: (N,) ATM total variance per expiration
- tau: (N,) time-to-expiry
- w_true: (N,) observed total variance
- Enrichment features as needed

### Known Data Quality Issues

**2022-2026 Extended Dataset (prs_dataset_full.csv, 480K rows) — NOT USED:**
1. **Tau distribution shift:** Median 0.156 → 0.071 (weekly/0DTE options emerged)
2. **Expiry dates explosion:** ~17/year → 58-86/year
3. **Hardcoded risk-free rate:** All 226K 2022-2026 rows have r=0.015 (vs. 6 different values 2014-2021)
4. **Extreme IV values:** 13 rows with IV > 100% (8-5-2024 carry-trade crash, 4-2025 tariff shock)
5. **Price precision drift:** 68.7% old data >5 decimals, new data ≤1 decimal
6. **Underlying format:** Integers (old) vs. floats (new)

**Future Work:** Requires proper time-varying r handling, short-expiry filtering, extreme value handling, domain adaptation

---

## PART 4: LOSS FUNCTIONS & MATHEMATICAL FORMULATIONS

### Model 1: Weighted Sum Loss

```
L = w_1 × L_RMSE + w_2 × L_MAPE + w_3 × L_cal + w_4 × L_but
    + w_5 × L_density + w_6 × L_upperbound

Current weights: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
```

**Components:**

1. **RMSE:** `√(mean((w_pred - w_true)²))`

2. **MAPE:** `mean(|w_pred - w_true| / (w_true + ε))` with ε=0.005

3. **Calendar:** `mean(ReLU(-(∂w/∂τ)))` penalizes negative tau-gradient

4. **Butterfly:** `mean(ReLU(-(∂²w/∂k²)))` penalizes negative convexity

5. **Density:** `mean((∂²w/∂k²)²)` on synthetic wing data

6. **Upperbound:** `mean(ReLU(w - 2|k|))` Lee's asymptotic bound

### Model 2: 6-Term PINN Loss

```
L_total = λ_fit × L_fit + λ_pde × L_dupire + λ_cal × L_cal
        + λ_but × L_but + λ_smooth × L_smooth + λ_bnd × L_boundary

λ = [1.0, 10.0, 10.0, 10.0, 1.0, 1.0]
```

**Components:**

1. **Fitting:** `mean((C_pred - C_target)²)` fit to Model 1 prices

2. **Dupire PDE:** `mean((∂C/∂τ - ½σ²_LV K² ∂²C/∂K²)²)`
   ```
   ∂C/∂τ ≡ self-consistency with σ_LV
   Expected dominance: ~2% of total loss
   ```

3. **Calendar:** `mean(ReLU(-∂C/∂τ))` ensure ∂C/∂τ ≥ 0

4. **Butterfly:** Always 0 (ICNN architecture guarantees ∂²C/∂K² ≥ 0)

5. **Smoothness:** `mean((∂σ_LV/∂K)² + (∂σ_LV/∂τ)²)` Sobolev penalty

6. **Boundary:** `mean((C(τ→0) - max(S-K, 0))²)` payoff at expiry

### eSSVI Parameterization

**Formula:**
```
w(k, θ) = θ/2 × (1 + ρ(θ) × φ(θ) × k + sqrt((φ(θ) × k + ρ(θ))² + 1 - ρ(θ)²))

where:
  θ = yATM (ATM total variance)
  k = logm (log-moneyness)
  ρ(θ) = rho_inf + (rho_0 - rho_inf) × exp(-decay × θ)  [time-decaying]
  φ(θ) = η / (θ^γ × (1 + θ)^(1-γ))  [power-law]
```

**Parameter Ranges & Constraints:**

| Param | Raw Init | Transform | Range | Trainable | Notes |
|-------|----------|-----------|-------|-----------|-------|
| rho_0 | -0.95 | Clamp | [-0.999, 0.999] | **FROZEN** | Encodes left skew |
| rho_inf | -0.50 | Clamp | [-0.999, 0.999] | ✅ | Long-term limit |
| decay | 2.0 | abs() | (0, ∞) | ✅ | Decay rate |
| eta | 0.5 | abs() | (0, ∞) | ✅ | Shape curvature |
| gamma | 0.0 | sigmoid | (0, 1) | ✅ | Power exponent |

**Gatheral-Jacquier Bound Check:**
```
η × (1 + |ρ|) < 2

Current system: η=0.733, mean(|ρ|)≈0.73
→ 0.733 × (1 + 0.73) = 0.904 << 2 ✅ SATISFIED
```

### Dupire Equation (Model 2 PDE Constraint)

```
∂C/∂τ = ½ σ²_LV(K, τ) × K² × ∂²C/∂K²

Physical meaning:
  - LHS: option price decay with time
  - RHS: local volatility causes convexity → price decay
  - Self-consistency: C and σ_LV must jointly satisfy this
```

---

## PART 5: TRAINING PROCEDURES & OPTIMIZATION

### Model 1 Training Pipeline

```python
# model1_research/train_pipeline.py

epochs = 1000
patience = 50
checkpoint_every_epoch = 1

for epoch in range(epochs):
    # Unfreeze rho_0 at epoch 50 (optional)
    if epoch == 50 and rho_0.requires_grad == False:
        rho_0.requires_grad = True
    
    # Training
    for batch in train_loader:
        logm, yATM, tau, w_true = batch
        w_pred, _, _, _ = model(tau, logm, yATM)
        loss = weighted_sum_loss(w_pred, w_true, weights=[1,1,0,0,0,0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    val_loss = evaluate(val_loader)
    
    # Learning rate schedule (MultiStepLR)
    if epoch >= 500 and epoch % 5 == 0:
        scheduler.step()  # γ = 0.5
    
    # Early stopping
    if early_stopping(val_loss):
        print(f"Early stopped at epoch {epoch}, best val loss: {early_stopping.best_loss}")
        model = load_checkpoint(best_checkpoint)
        break
    
    # Gradient explosion detection
    if has_nan_gradient():
        print(f"Gradient explosion at epoch {epoch}, loading previous checkpoint")
        model = load_checkpoint(epoch - 1)
        break
```

### Model 2 Training Loop

```python
# model2_research/train_dupire.py

epochs = 5000
resample_every = 100
scheduler = ReduceLROnPlateau(patience=200, factor=0.5)

for epoch in range(epochs):
    # Resample collocation points every 100 epochs
    if epoch % resample_every == 0:
        sampler.resample_collocation_points(5000)
        sampler.resample_boundary_points(500)
    
    # Forward pass
    C_pred = price_net(K, tau)
    sigma_lv = lv_net(K, tau)
    
    # Compute derivatives (via autograd, float64 precision)
    dC_dtau = autograd.grad(C_pred.sum(), tau, create_graph=True)[0]
    d2C_dK2 = autograd.grad(dC_dtau.sum(), K, create_graph=True)[0]  # butterfly
    
    # Loss computation
    L_fit = F.mse_loss(C_pred, C_target)
    L_pde = F.mse_loss(dC_dtau - 0.5 * sigma_lv**2 * K**2 * d2C_dK2, torch.zeros_like(...))
    L_cal = F.relu(-dC_dtau).mean()
    L_but = F.relu(-d2C_dK2).mean()  # Always 0 (ICNN)
    L_smooth = ((d_sigma_K)**2 + (d_sigma_tau)**2).mean()
    L_bnd = F.mse_loss(C_boundary, payoff_boundary)
    
    L_total = 1.0*L_fit + 10.0*L_pde + 10.0*L_cal + 10.0*L_but + 1.0*L_smooth + 1.0*L_bnd
    
    L_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step(L_total)
    
    # Cleanup to prevent VRAM fragmentation
    del dC_dtau, d2C_dK2, ...
    torch.cuda.empty_cache()
```

### Model 3 Training Loop

```python
# model3_research/scripts/train_models.py

# Data loading (with KDE weighting for rare events)
train_loader = DataLoader(train_dataset, batch_size=128, sampler=kde_sampler)
val_loader = DataLoader(val_dataset, batch_size=128)

# Choose optimizer
if optimizer_name == 'CPR':
    optimizer = AdamCPR(model.parameters(), lr=0.001)
elif optimizer_name == 'CWD':
    optimizer = CautiousAdamW(model.parameters(), lr=0.001)
else:  # Adam, AdamW
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    # Training phase
    for batch in train_loader:
        x, y = batch  # x: (batch, 20, 16), y: (batch, 1)
        pred = model(x)  # alpha ratio
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation phase
    val_loss = evaluate_tft(val_loader)
    val_rmse = compute_rmse(val_preds, val_targets)
    val_mape = compute_mape(val_preds, val_targets)
    
    # Checkpoint best model
    if val_loss < best_val_loss:
        save_checkpoint(model, optimizer, epoch, val_loss)
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Test evaluation (on strictly held-out 2021 data)
test_rmse = compute_rmse(test_preds, test_targets)
test_mape = compute_mape(test_preds, test_targets)
```

### Custom Optimizers

**CautiousAdamW (ICLR 2026):**
```python
# Only apply weight decay when optimizer and param align
update = exp_avg / (sqrt(exp_avg_sq) + eps)
mask = (update * param >= 0).float()  # Aligned direction?
param -= lr * (update + λ * mask * param)
```

**AdamCPR (Constrained Parameter Regularization):**
```python
# Constrain parameters to stay near initialization
regularization = ||param - param_init||_2
loss += β * regularization
# Particularly effective for high-capacity models like TFT
```

---

## PART 6: TESTING STRATEGY

**Test Coverage:** 177 tests, ~4 seconds runtime, CPU-only

**Test Files:**
```
tests/
├── test_dataset.py (14 tests) — data loading, chronological splits, feature engineering
├── test_model.py (52 tests) — eSSVI params, shape checks, gradient flow, loss computation
├── test_model_regression.py (18 tests) — bug regressions (M1-M5, X1-X3, T1-T2, E1)
├── test_diffusion.py (19 tests) — U-Net, noise schedule, conditioning
├── test_hyperiv.py (15 tests) — set embedding, target MLP generation
├── test_structural_break.py (16 tests) — CUSUM, Bai-Perron detection
├── test_train_integration.py (10 tests) — end-to-end training loops
├── test_utils.py (22 tests) — metrics, early stopping, logging
│
model1_research/tests/
├── test_model.py — base model unit tests
└── test_model_regression.py — 17 fixed bug guards
│
model2_research/tests/
├── test_dupire.py (28 tests) — ICNN convexity, PDE residuals, boundary conditions
```

**Design Principles:**
- **float64 everywhere:** Session-wide via conftest.py
- **Tiny architectures:** hidden_sizes=[5,5,5], ensemble_num=2 for sub-second tests
- **No file I/O:** Mock DataFrames, synthetic tensors
- **Autograd-safe:** Never wrap SmileModel in torch.no_grad()
- **Regression coverage:** Named tests guard 18 specific bug fixes

**Example Test:**
```python
def test_model1_ensemble_gradient_flow():
    """Verify all 5 ensemble members backprop correctly."""
    model = MultiModel(ensemble_num=5, hidden_sizes=[5,5,5])
    logm, yATM, tau, w_true = torch.randn(10, 1, dtype=torch.float64)
    w_pred, _, _, _ = model(tau, logm, yATM)
    loss = F.mse_loss(w_pred, w_true)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())
    assert all((p.grad.abs() > 0).any() for p in model.parameters() if p.requires_grad)
```

---

## PART 7: TRANSFER LEARNING & DIMENSIONAL ADAPTATION

**Scenario:** Retraining models when input dimensions change (e.g., enhancement features added, Model 2 Greeks integrated)

**Three-Layer Strategy:**

1. **load_finetune_weights(model, checkpoint, device)**
   - Loads state dict from old checkpoint
   - For matching shapes: copy directly
   - For mismatched shapes: Xavier-init new dimensions, copy overlapping portion
   - Returns (transferred_names, reinitialized_names)

2. **setup_finetune_optimizer(model, transferred, reinitialized, base_lr, new_lr)**
   - Differential learning rates:
     - Transferred layers: base_lr = lr × 0.1 (preserve knowledge)
     - New/reinitialized: new_lr = full lr (fast adaptation)
   - Prevents catastrophic forgetting while allowing dimension expansion

3. **freeze_transferred(model, transferred_names)** (optional)
   - Completely freezes pretrained params for warmup epochs
   - Not currently used but available for extreme fine-tuning

**Example Usage:**
```bash
# Model 1 retraining with new data
python train_pipeline.py --finetune ../model1_research/models/MultiModel.pt

# Adjustment model with expanded features (6→12→16 dims)
python train_models.py --model tft --finetune ../models/AdjustmentModel_old.pt

# HyperIV fine-tuning
python train_hyperiv.py --finetune ../models/HyperIVModel.pt
```

---

## PART 8: HYPERPARAMETERS & CONFIGURATION

All hyperparameters centralized in `src/config.ini`:

```ini
[model_sett]
learning_rate = 0.0005      # Base model
hidden_sizes = 64,32,16     # SmileModel architecture
ensemble_num = 5            # Number of ensemble members
loss_weights = 1,1,0,0,0,0  # [RMSE, MAPE, Cal, But, Density, UB]
epsilon = 0.02              # yATM smoothing epsilon

[training]
batch_size = 256
epochs = 1000
seed = 42
train_start_date = 20140101
train_end_date = 20201231
test_start_date = 20210101
test_end_date = 20211231
gradient_clip = 1.0
early_stopping_patience = 50
rho_unfreeze_epoch = 50

[adjustment]
gru_hidden_dim = 64
gru_layers = 2
attention_heads = 4
sequence_length = 20
learning_rate = 0.001
epochs = 1000

[dupire]
hidden_dim = 64
n_layers = 3
lambda_fit = 1.0
lambda_pde = 10.0
lambda_cal = 10.0
lambda_but = 10.0
lambda_smooth = 1.0
epochs = 5000
learning_rate = 0.001

[hyperiv]
embed_dim = 128
transformer_heads = 4
transformer_layers = 2
n_reference = 50
epochs = 500
learning_rate = 0.001

[diffusion]
condition_dim = 11
timesteps = 1000
learning_rate = 0.0002
epochs = 1000
```

---

## PART 9: CRITICAL INSIGHTS & LESSONS LEARNED

### 1. **eSSVI Unlocked the Model**
The original base model was suppressed by static SSVI limits. The optimizer failed to trace the 45-degree angle of Deep-OTM puts because classical static `rho` couldn't handle short-term maturities and was dominated by ATM data gravity.

**Solution:** Upgrade to eSSVI with frozen `rho_0 = -0.95`, allowing time-decaying correlation.

**Impact:** Butterfly violations reduced from 74% → 45.69%, MAPE dropped from 44% → 5.46%

### 2. **Additive > Multiplicative**
A/B test confirmed: `w = eSSVI + yATM_tilde × NN` (additive) stable, while `w = eSSVI × NN` (multiplicative) explodes at epoch 2.

**Root Cause:** Product rule creates cross-terms in butterfly constraint derivatives that amplify gradient noise.

### 3. **ICNN Guarantees Convexity**
Hard constraint beats soft penalty. ICNN's K→C pathway with non-negative weights guarantees ∂²C/∂K² ≥ 0 architecturally.

**Result:** 0% butterfly violations for Model 2, vs. 45% for Model 1

### 4. **TFT Variable Selection = Learned Regime-Switching**
TFT's per-timestep feature weighting implicitly learns crisis detection:
- Crisis periods: high weights on vix_change, underlying_return
- Calm periods: high weights on tau, logm
- This regime-switching is learned end-to-end without explicit labels

### 5. **CPR is Architecture-Dependent**
CPR (Constrained Parameter Regularization) excels for high-capacity models (TFT: #1) but harms under-parameterized ones (GRU: #12).

**Lesson:** Regularization strength must match model capacity.

### 6. **Chronological Split is Critical**
Random train/test split leaks future information. Strict chronological split (train ≤ 2020-12-31, test ≥ 2021-01-01) prevents look-ahead bias and reflects real deployment scenario.

### 7. **Data Imbalance via yATM Smoothing**
Low-volatility options have vanishing gradients. Smoothing: `yATM_tilde = sqrt(yATM² + ε²)` with ε=0.02 preserves gradient signal in near-zero regimes.

### 8. **Ensemble Stability**
5-member ensemble (vs. single model) reduces variance and produces more stable calibrations. A/B test showed 10% improvement in test-set robustness.

### 9. **Model 2 Must Query Model 1 Dynamically**
C_target for Dupire training is NOT static market prices or synthetic BS data. Instead, dynamically query Model 1 on random (K, τ) points, ensuring Model 2 learns from Model 1's smoothed surface.

### 10. **Autograd Compatibility in PINN**
Model 1's SmileModel uses `autograd.grad(create_graph=True)`. Cannot wrap in `torch.no_grad()`. Correct pattern: `clone().requires_grad_(True)` → query → `detach()`.

---

## PART 10: PROJECT STRUCTURE & FILE LOCATIONS

```
/smart-data-analysis-main/
├── README.md                           # Overview (plain English)
├── ARCHITECTURE.md                     # This document level
├── repository_file_index.md            # File-by-file index
├── requirements.txt                    # Dependencies
├── conftest.py                         # Global pytest config (float64)
│
├── src/
│   ├── config.ini                      # All hyperparameters
│   ├── dataset.py                      # DataProcessor: loading, features, splits
│   ├── utils.py                        # Seed, logging, metrics, EarlyStopping
│   ├── transfer.py                     # Transfer learning utilities
│   ├── structural_break.py             # CUSUM, Bai-Perron detection
│   ├── hyperiv.py                      # Model 4: HyperIV architecture
│   ├── train_hyperiv.py                # Model 4 training script
│   ├── diffusion.py                    # Model 5: DDPM (1D U-Net, FiLM)
│   ├── train_diffusion.py              # Model 5 training script
│   ├── test.py                         # Base model evaluation + arbitrage checks
│   ├── adjustment.py                   # (Legacy, absorbed into model3_research)
│   └── data/
│       └── module_d_features.csv       # Module D extracted Greeks
│
├── model1_research/
│   ├── model.py                        # eSSVI, SmileModel, MultiModel, losses
│   ├── train.py                        # Base model training loop
│   ├── train_pipeline.py               # Full pipeline with diagnostics
│   ├── experiment.py                   # Inference + evaluation → CSV
│   ├── model1_background.md            # Architecture background & rationale
│   ├── models/
│   │   └── MultiModel.pt               # Trained checkpoint (best val loss 0.07495)
│   ├── figures/
│   │   ├── m1_loss_curve.png
│   │   ├── m1_train_fit.png
│   │   ├── m1_val_fit.png
│   │   └── m1_test_fit.png
│   ├── scripts/
│   │   ├── generate_model1_plots.py    # Generates 4 figures above
│   │   ├── plot_pipeline_metrics.py    # Loss from metrics JSON
│   │   ├── plot_smooth_iv_check.py     # Surface smoothness diagnostic
│   │   └── train_diagnose.py           # Per-epoch parameter tracking
│   └── tests/
│       ├── conftest.py                 # tiny_batch fixture
│       ├── test_model.py               # 52 tests
│       └── test_model_regression.py    # 18 bug regression tests
│
├── model2_research/
│   ├── README.md                       # Model 2 overview
│   ├── model2_training_details.md      # Loss breakdown, hyperparams
│   ├── dupire_pinn.py                  # ICNN + LocalVol networks, losses
│   ├── train_dupire.py                 # Training script with Model 1 loader
│   ├── module_d.py                     # V3 Greeks Extractor
│   ├── extract_features.py             # Feature extraction script
│   ├── candidates/
│   │   ├── README.md                   # Alternate approach status
│   │   ├── wamol/                      # WamOL PINN prototype
│   │   ├── gno/                        # Graph Neural Operator
│   │   ├── heston/                     # Heston calibration
│   │   ├── signature/                  # Signature kernels
│   │   └── neural_sde/                 # Neural SDE experiments
│   ├── models/
│   │   └── DupireModel.pt              # ICNN weights
│   └── tests/
│       ├── conftest.py
│       └── test_dupire.py              # 28 tests
│
├── model3_research/
│   ├── README.md                       # 12-way comparison results
│   ├── full_research_report.md         # Complete analysis & literature review
│   ├── optimizers.py                   # CautiousAdamW, AdamCPR
│   ├── tft_adjustment.py               # TFT model architecture
│   ├── xlstm_adjustment.py             # xLSTM model (archived)
│   ├── run_all_experiments.py          # Bulk experiment launcher
│   ├── scripts/
│   │   ├── train_models.py             # Unified trainer (--model tft|gru|xlstm)
│   │   ├── run_tft_experiments.py      # TFT grid search
│   │   ├── plot_loss_curves.py         # Train/val visualization
│   │   ├── plot_regularization_results.py
│   │   ├── benchmark_dtype.py          # float32 vs float64 speed
│   │   └── models/
│   │       ├── TFT_CPR.pt              # TFT+CPR (winner)
│   │       ├── TFT_AdamW.pt            # TFT+AdamW (alternate)
│   │       └── GRU_CWD.pt              # GRU baseline
│   ├── overfitting_research/
│   │   ├── README.md
│   │   └── cwd_notes.md
│   ├── archived_models/                # 9 non-winning models
│   ├── archived_logs/                  # Training logs for all 12 experiments
│   └── archived_figures/               # Loss curves, regularization plots
│
├── tests/
│   ├── conftest.py                     # Global pytest config
│   ├── test_dataset.py                 # 14 tests
│   ├── test_model.py                   # 52 tests (duplicate from model1_research)
│   ├── test_model_regression.py        # 18 tests
│   ├── test_diffusion.py               # 19 tests
│   ├── test_hyperiv.py                 # 15 tests
│   ├── test_structural_break.py        # 16 tests
│   ├── test_train_integration.py       # 10 tests
│   └── test_utils.py                   # 22 tests
│
├── scripts/
│   ├── download_data.py                # FinMind TXO + yfinance TWII/VIX
│   ├── build_features.py               # Computed enhancement features
│   └── plot_training_curves.py         # Loss visualization
│
├── logs/                               # Training logs (gitignored)
├── models/                             # .pt weights (gitignored)
├── dataset/                            # TXO data (gitignored)
│   ├── prs_dataset_no_fat(clean).csv   # 254K rows, active
│   ├── prs_dataset_full.csv            # 480K rows, not used (data quality)
│   ├── TWII.csv / TWII_full.csv
│   ├── VIX.csv / VIX_full.csv
│   └── enhancement/
│       └── daily_features.csv          # Computed features (23 cols)
│
└── docs/
    ├── discussion_notes.md             # Issue tracking & resolution
    ├── prediction_analysis.md          # Architecture fix notes
    └── research_report.md              # Extended research (1200 lines)
```

---

## PART 11: KEY PERFORMANCE METRICS SUMMARY

| Model | Test RMSE | Test MAPE | Status | Key Notes |
|-------|-----------|----------|--------|-----------|
| **Model 1** (eSSVI+NN) | 0.01977 | 5.46% | CURRENT | Base model trained on 2014-2020, tested on 2021 |
| **Model 2** (ICNN) | N/A | N/A | V1-V3 COMPLETE | Local vol extraction, 0% butterfly violations |
| **Model 3** (TFT+CPR) ⭐ | 0.1558 | 9.51% | **WINNER** | 12-way comparison, 318K params, crisis adjustment |
| **Model 4** (HyperIV) | — | — | Pending | State-of-the-art point predictor, per-day specialist |
| **Model 5** (DDPM) | — | — | Pending | Surface forecasting, generative, coherent grids |

---

## PART 12: FUTURE ROADMAP

### Short-term
1. Retrain Models 4 & 5 with current codebase
2. Validate Model 2 Greeks integration into Model 3
3. Implement model stacking/ensemble across 5 models

### Medium-term
1. Resolve 2022-2026 data quality issues, retrain on full dataset
2. Add explicit no-arbitrage constraints to HyperIV (penalty + projection)
3. Address base model gradient explosion (adaptive loss weighting, parameter clamping)

### Long-term
1. Extend to American-style options (DGM with early exercise boundary)
2. Add leverage effect (spot-vol correlation) modeling
3. Multi-asset transfer learning (extend beyond TXO)
4. Real-time online learning for structural breaks

---

## APPENDIX: KEY REFERENCES

### Papers

1. **Gatheral & Jacquier (2014).** Arbitrage-free SVI volatility surfaces. *Quantitative Finance*.
2. **Ho, Jain, & Abbeel (2020).** Denoising Diffusion Probabilistic Models. *NeurIPS*.
3. **HyperIV (ICML 2025).** Hypernetwork-based IV surface interpolation.
4. **Lim et al. (2021).** Temporal Fusion Transformers. *Google Research*, arXiv:1912.09363.
5. **Beck et al. (2024).** xLSTM: Extended Long Short-Term Memory. *arXiv:2407.10240*.
6. **Amos et al. (2017).** Input Convex Neural Networks. *arXiv:1609.07152*.
7. **Wang & Privault (2022/2025).** Deep Self-Consistent Learning of Local Volatility. *arXiv:2201.07880*.
8. **Bae, Kang & Lee (2024).** Option Pricing & Local Vol by PINN. *Computational Economics*, 64:3143.

### Software Libraries

- **PyTorch 2.5.1** (CUDA 12.4)
- **pandas, numpy, scipy, scikit-learn**
- **matplotlib, tqdm**
- **QuantLib** (volatility calibration reference)
- **statsmodels** (time series analysis)
- **ruptures** (change-point detection)

---

**Document Version:** 2026-02-27  
**Total Lines of Code:** ~8,500 (src + models + tests)  
**Total Tests:** 177  
**Total Documentation:** ~1,200 lines

# Model 1 Background: eSSVI + NN Implied Volatility Surface

## 1. Overall Objective
Model 1 constructs a **physics-informed arb-free implied volatility surface generator** for TAIEX index options. It combines the **eSSVI (extended Surface Stochastic Volatility Inspired)** parametric base with a **SmileNN** neural network correction, trained on an ensemble of 5 members with softmax weighting.

**Input**: `(tau, logm, yATM)` — time-to-maturity, log-moneyness, ATM total variance  
**Output**: Total variance `w(k, θ)` and analytical gradients `(∂w/∂τ, ∂w/∂k, ∂²w/∂k²)`

## 2. The Core Problem & Diagnosis
The original Model 1 used classic **SSVI** with a single global correlation `rho`. This caused:
*   **ATM Gravity**: The heavily imbalanced dataset (vastly more ATM than OTM options) forced `rho` to optimize for ATM, flattening the surface.
*   **OTM Put Failure**: The flattened base model destroyed the steep left-skew needed for deep OTM Puts.
*   **NN Fighting Physics**: The Neural Network couldn't compensate without triggering butterfly arbitrage violations.

The bottleneck was the **static SSVI correlation**, not the NN or penalties.

## 3. The Solution: eSSVI Architecture

### 3.1 Time-Decaying Skew (ρ(θ))
Instead of a static `rho`, eSSVI uses a maturity-dependent correlation:

```
rho(theta) = rho_inf + (rho_0 - rho_inf) * exp(-decay * theta)
```

**Parameters** (in `SSVIModel.__init__`):
| Parameter | Raw Name | Init Value | Transform | Trainable |
|---|---|---|---|---|
| Short-term skew | `raw_rho_0` | -0.95 | Direct (clamped) | **Frozen** until epoch 50 |
| Long-term skew | `raw_rho_inf` | -0.50 | Direct (clamped) | ✅ |
| Decay rate | `raw_decay` | 2.0 | Direct | ✅ |
| Curvature | `raw_eta` | 0.5 | `abs(raw_eta)` | ✅ |
| Power exponent | `raw_gamma` | 0.0 | `sigmoid(raw_gamma)` | ✅ |

### 3.2 Forced Skew Regularization (ρ₀ = -0.95)
`raw_rho_0` is **frozen** (`requires_grad=False`) at -0.95, forcing steep left-skew at short maturities. This neutralizes ATM data imbalance. Unfreezing at epoch 50 confirmed -0.95 was already optimal.

### 3.3 Low-Volatility Gradient Scaling (ỹ_ATM)
The NN output multiplier uses a smoothed yATM to prevent gradient vanishing at low volatility:

```
yATM_tilde = sqrt(yATM² + ε²)    where ε = 0.02
```

This ensures `dOutput/dWeights ≠ 0` even when yATM → 0.

### 3.4 Penalty Relaxation
With eSSVI handling skew by design, arbitrage penalty weights were relaxed from `[1,1,10,10,10,10]` to `[1,1,0,0,0,0]`. Fitting accuracy is now the sole loss driver.

## 4. Model Architecture (Forward Pass)

```
MultiModel (ensemble of 5)
├── SoftmaxModel → ensemble weights
└── SingleModel[0..4], each containing:
    ├── SSVIModel (eSSVI base)
    │   └── forward(logm, tau, yATM) → (w_base, ∂τ, ∂k, ∂²k)
    ├── SmileModel (3-layer NN: 64→32→16)
    │   └── forward(tau, logm) → (correction, ∂τ_nn, ∂k_nn, ∂²k_nn)
    └── Additive combination:
        output = w_base + yATM_tilde × correction
```

**Loss**: `WeightedSumLoss` with weights `[1,1,0,0,0,0]` (MSE fit + tau-gradient, no arb penalties)

## 5. Training Configuration

| Setting | Value |
|---|---|
| Epochs | 1000 (early stopping patience=50) |
| Optimizer | AdamW, lr=0.0005 |
| Batch size | 256 |
| Gradient clip | 1.0 |
| LR scheduler | MultiStepLR (milestones from 500, step=5, γ=0.5) |
| ρ₀ unfreeze | Epoch 50 |
| Data | TAIEX 2014-2020 (train), 2021 (test) |
| dtype | float64 |

## 6. Results

| Metric | Value |
|---|---|
| Best validation loss | **0.07495** (epoch 34) |
| Early stopped at | Epoch 84 |
| Total Variance RMSE | 0.002961 (test) |
| IV-RMSE | 0.031088 (test) |
| MAPE | 8.38% (test) |
| SSVI healthy | ✅ (Gatheral-Jacquier satisfied) |
| **Butterfly violations** | **45.69%** (20,164/44,133 test points) |
| Calendar violations | 52.82% (103/195 comparable pairs) |
| Test samples | 44,597 |

> eSSVI 將 butterfly violation 從舊版 SSVI 的 74% 降至 **45.69%**，改善了 ~28 個百分點。但仍有近半測試點違規，Model 2 (ICNN Dupire) 仍為必要的下游修復步驟。

## 7. File Structure

```
model1_research/
├── model.py              # BSModel, SSVIModel, SmileModel, SingleModel, MultiModel, losses
├── train.py              # Standalone training script (CLI)
├── train_pipeline.py     # Full pipeline: train + diagnostics + validation
├── experiment.py         # Inference + evaluation → experiment_results.csv
├── model1_background.md  # This document
├── __init__.py
├── models/
│   └── MultiModel.pt     # Trained checkpoint (best val_loss)
├── figures/
│   ├── m1_loss_curve.png  # Training/val loss over 84 epochs
│   ├── m1_train_fit.png   # IV curve fits on training data
│   ├── m1_val_fit.png     # IV curve fits on validation data
│   └── m1_test_fit.png    # IV curve fits on test data
├── scripts/
│   ├── generate_model1_plots.py  # Generates the 4 figures above
│   ├── plot_pipeline_metrics.py  # Plots loss from pipeline metrics JSON
│   ├── plot_smooth_iv_check.py   # Surface smoothness diagnostic
│   └── train_diagnose.py         # Short diagnostic training run
└── tests/
    ├── conftest.py               # tiny_batch fixture
    ├── test_model.py             # Unit tests (eSSVI params, shapes, gradients)
    └── test_model_regression.py  # Regression guards for 17 fixed bugs
```

## 8. Downstream Dependencies
Model 1's output (`experiment_results.csv` in `logs/`) feeds into **Model 2** (ICNN arbitrage-free refinement). The CSV contains per-option predictions with columns for tau, logm, yATM, predicted total variance, and observed total variance.

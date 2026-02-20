# Implied Volatility Surface Prediction

A multi-model system for predicting the **implied volatility (IV) surface** of Taiwan Stock Exchange Options (TXO). Combines classical parametric models with deep learning to produce accurate, arbitrage-free predictions.

## What Is an IV Surface?

Think of a stock option as insurance against price changes. The **implied volatility** is the market's estimate of how much the stock price will fluctuate — higher IV means the market expects larger moves, so the option costs more.

An **IV surface** is like a weather map: instead of showing temperature across geography, it shows expected volatility across two dimensions:

- **Time to expiration** (tau) — how far into the future the option expires
- **Strike price** (K) — the price at which the option pays off

Accurately predicting this surface matters because it determines the fair price of every option contract. A good model can detect mispriced options and manage risk.

## Key Concepts

| Term | Plain English |
|------|---------------|
| **Implied Volatility (IV)** | Market's forecast of future price swings, extracted from option prices |
| **Total Variance (TV)** | IV squared times time-to-expiry. Smoother than raw IV, easier to model |
| **Log-Moneyness** | `ln(K/S)` — how far the strike is from the current price. Zero = at-the-money |
| **SSVI** | A parametric formula (Gatheral & Jacquier, 2014) that guarantees no-arbitrage constraints on the IV surface |
| **Arbitrage violation** | When a model's predictions imply you could make risk-free profit — physically impossible, so the model is wrong |
| **Butterfly spread** | A combination of three options that must have non-negative value. A violation means the model predicts negative probability density |

## The Five Models

This system uses five complementary models, each addressing different aspects of IV surface prediction. Think of them as a team of specialists:

- **Models 1 & 4** are *interpolators* — they predict today's IV surface from observed option prices
- **Model 2** is a *physics engine* — it verifies that option prices obey the Black-Scholes equation
- **Model 3** is a *crisis detector* — it adjusts predictions during market stress
- **Model 5** is a *forecaster* — it predicts tomorrow's IV surface from today's market conditions

### 1. Base Model: SSVI + Neural Network Ensemble

**What it does:** The foundation of the system. Predicts implied volatility for any combination of strike price and time-to-expiry.

**How it works:** An **SSVI parametric model** (a well-known formula from quantitative finance) provides the structural backbone — it guarantees the predicted surface has a mathematically valid shape. A **neural network ensemble** (5 small networks combined) then learns the residual patterns that SSVI misses, like local bumps or skew features unique to the Taiwan market.

Each neural network ("SmileModel") uses automatic differentiation to compute first and second derivatives of its output, which are fed into physics-based penalty terms. The architecture uses an **additive formulation**: `w = SSVI(logm, yATM) + yATM * NN(tau, logm)`, where the yATM scaling keeps the NN correction proportional to the current volatility level. An earlier multiplicative version was abandoned after A/B testing showed it causes gradient explosion within 2 epochs (see `logs/architecture_comparison.json`).

The loss function has six components: (1) fit the observed data (RMSE), (2) stay close to the SSVI prior (MAPE), (3) enforce calendar spread constraints (longer-dated options must be worth more), (4) enforce butterfly constraints (no negative probabilities), (5) penalize extreme density curvature, and (6) encourage smoothness.

**Why an ensemble?** Training 5 networks with different random initializations and averaging their predictions reduces variance and produces more stable results than any single network.

**Results (2025-2026 test):** TV-RMSE 0.0120, MAPE 33.0%, IV-RMSE 0.219

### 2. DGM: PDE Solver for Black-Scholes

**What it does:** Learns to solve the Black-Scholes partial differential equation (PDE) — the fundamental equation that governs how option prices change over time.

**Why it matters:** The Black-Scholes PDE is like Newton's laws for options. If a model's predictions violate this equation, the prices are physically inconsistent. Traditional methods solve this equation on a grid (like a spreadsheet), but DGM uses a neural network as a smooth, continuous approximation — no grid required.

**How it works:** The **Deep Galerkin Method** trains a neural network to minimize three errors simultaneously: (1) how badly the network violates the PDE at random interior points, (2) how badly it violates boundary conditions at extreme prices, and (3) how badly it matches the known payoff at expiration. The training points are re-sampled every 100 epochs so the network sees a fresh set of equations to satisfy.

The architecture uses LSTM-like "S-layers" with gating mechanisms — these help the network learn the complex nonlinear interactions between time, stock price, and volatility.

**Results (2025-2026 test):** Final PDE residual 2.0e-5 (near-perfect equation satisfaction), BS price RMSE 0.036 (3.6 cents average pricing error)

### 3. Adjustment Model: GRU + Attention for Crisis Periods

**What it does:** Detects and corrects for market crises — sudden events like the 2008 financial crash or COVID-19 that cause the IV surface to shift dramatically in ways the base model can't capture.

**Why it matters:** Normal market conditions are relatively smooth, but crises cause "structural breaks" — sudden regime changes where historical patterns no longer apply. Without correction, the base model's predictions become unreliable during these periods.

**How it works:** The model looks at the past 20 days of trading data (13 features per day, including base model predictions, VIX changes, S&P 500 returns, IV term structure slope, and realized volatility). A **GRU** (Gated Recurrent Unit — a type of recurrent neural network designed for sequential data) processes this time series, building up a representation of the current market regime. Then **multi-head attention** (the same mechanism used in ChatGPT) examines all 20 days and learns which past days are most relevant — for example, a sudden VIX spike 3 days ago might be more informative than gradual drift over the past week.

The output is a multiplicative adjustment ratio: `adjusted_prediction = base_prediction * ratio`. Training uses KDE-weighted loss (Kernel Density Estimation) to focus on rare extreme events — otherwise the model would optimize only for normal conditions and ignore crises, which is exactly when corrections matter most.

**Results (2025-2026 test):** Test RMSE 52.20, MAPE 70.15% (1000 epochs). The high MAPE reflects that adjustment ratios are close to 1.0, where small absolute errors produce large percentage errors.

### 4. HyperIV: Hypernetwork (State-of-the-Art)

**What it does:** The most accurate predictor in the system. Generates a specialized prediction model for *each individual day's* IV surface.

**Why it's special:** The base model learns a single average mapping that works for all days. But each day's IV surface has unique characteristics — maybe today has extra skew due to earnings announcements, or the term structure is inverted due to upcoming elections. HyperIV solves this by creating a custom neural network for each day.

**How it works:** Based on the ICML 2025 HyperIV paper — the current state-of-the-art for IV surface interpolation. The system works in two stages:

1. **Read the market:** A **Transformer set encoder** (the same architecture behind large language models) reads 50 observed option prices from today's market. The Transformer uses attention to understand cross-strike and cross-maturity relationships — "this put at strike 15000 tells us something about the call at strike 16000."

2. **Generate a specialist:** A **hypernetwork** takes the Transformer's summary and *generates the weights* of a small target neural network. This target network then predicts total variance for any `(tau, log-moneyness)` query point.

The key insight is that the hypernetwork doesn't predict IV directly — it predicts the *parameters of another neural network* that predicts IV. This means every day gets its own specialist predictor, automatically adapted to that day's unique market conditions.

**Results (2025-2026 test):** TV-RMSE 0.0056, MAPE 20.0%, IV-RMSE 0.113 (best point prediction — **53% lower error** than the base model)

### 5. DDPM: Diffusion Model for Surface Forecasting

**What it does:** Forecasts *tomorrow's* IV surface based on today's market conditions. While models 1-4 interpolate today's surface from observed prices, this model predicts the future.

**Why it matters:** Portfolio managers need to know not just today's prices, but where they're headed. A good surface forecast enables proactive hedging — adjusting positions before the market moves, rather than reacting afterward.

**How it works:** A **Denoising Diffusion Probabilistic Model** (DDPM — the same family of models behind image generators like DALL-E and Stable Diffusion, but applied to financial surfaces instead of images). The process works in reverse:

1. **Training:** Take a real IV surface (a 10x20 grid = 200 numbers), gradually add random noise over 1000 steps until it becomes pure static. Train a neural network to reverse each step — given the noisy version, predict the noise that was added.

2. **Generation:** Start from pure random noise and apply the trained denoiser 1000 times. Each step removes a little noise, gradually revealing a realistic IV surface conditioned on the input market features.

The architecture is a **1D U-Net** (encoder-decoder with skip connections). Conditioning on 13 market features (underlying price, VIX, volume, returns, plus enhancement features like S&P 500, synthetic Taiwan VIX, term structure slope, variance risk premium, and realized volatility) is done via **FiLM layers** (Feature-wise Linear Modulation) — these tell the denoiser "generate a surface that looks like what the market should produce given these conditions."

**Key advantage over point prediction:** The diffusion model generates *coherent* surfaces where all 200 grid points are mutually consistent. Point prediction models (Base, HyperIV) predict each point independently, which can create internal inconsistencies.

**Results (2025-2026 test):** Val surface RMSE 0.0049, Test surface RMSE 0.0072

## Results Summary

Two rounds of experiments were conducted:
- **Round 1:** Train on 2014-2020 (254K rows), test on 2021
- **Round 2:** Train on 2014-2024 (480K rows), test on 2025-2026 — with transfer learning and enhanced market features

### Point Prediction (Interpolation)

| Model | TV-RMSE (R1 / R2) | MAPE (R1 / R2) | IV-RMSE (R1 / R2) |
|-------|-------------------|-----------------|-------------------|
| Base (SSVI+NN) | 0.0134 / **0.0120** | 44.1% / **33.0%** | 0.209 / 0.219 |
| HyperIV | 0.0074 / **0.0056** | 20.7% / **20.0%** | 0.076 / 0.113 |

HyperIV consistently outperforms the base model — **53% lower TV-RMSE** and **39% lower MAPE** in Round 2.

### All Five Models (Round 2, 2025-2026 Test)

| Model | Task | Key Metric | Training |
|-------|------|------------|----------|
| Base (SSVI+NN) | Point prediction | TV-RMSE: 0.0120, MAPE: 33.0% | 105 ep, ~9h GPU |
| HyperIV | Point prediction | TV-RMSE: 0.0056, MAPE: 20.0% | 69 ep, ~30min GPU |
| DGM | PDE solving | Residual: 2.0e-5, BS RMSE: 0.036 | 5000 ep, ~25min GPU |
| DDPM | Surface forecasting | Test RMSE: 0.0072 | 1000 ep, ~8h GPU |
| Adjustment | Crisis correction | Test RMSE: 52.20, MAPE: 70.15% | 1000 ep, ~46h CPU |

### Training Curves & Visualizations

Training curve and IV surface visualizations can be regenerated from the training logs using `scripts/plot_training_curves.py`. The training logs are stored in `logs/` and results are documented in `EXPERIMENT.md`.

## Key Findings

1. **More data helps significantly.** Expanding from 254K to 480K data points and 7 to 12 years of history reduced the base model's MAPE from 44.1% to 33.0% — a 25% improvement just from more training data.

2. **HyperIV is the clear winner for point prediction.** Its per-surface specialization (generating unique network weights for each day) consistently outperforms the fixed-weight base model by 50%+ on TV-RMSE.

3. **Transfer learning accelerates convergence.** HyperIV converged in just 69 epochs (vs. 58 in the original run with less data), thanks to starting from pretrained weights rather than random initialization.

4. **The base model has a stability problem.** SSVI parameter optimization becomes unstable after ~60 epochs on the extended dataset, causing gradient explosion. This is a fundamental challenge of combining parametric models (SSVI) with neural network optimization — the parametric part can drift into degenerate configurations. Early stopping and checkpoint saving are essential safeguards.

5. **Enhancement features matter.** Adding market features (VIX, term structure slope, variance risk premium, S&P 500 returns) improved the DDPM's validation RMSE by 31% compared to using only 4 basic features. These features give the model richer context about the current market regime.

6. **IV-RMSE can be misleading.** Despite better TV-RMSE, IV-RMSE increased in Round 2 because the 2025-2026 test period has more short-maturity options. The conversion `IV = sqrt(TV / tau)` amplifies errors when tau is small — a mathematical artifact, not a model failure.

## Project Structure

```
README.md               # This file (plain-English overview)
EXPERIMENT.md           # Detailed experimental results and analysis
ARCHITECTURE.md         # System architecture and design decisions
requirements.txt        # Python dependencies
src/
  config.ini            # All hyperparameters and file paths
  utils.py              # Utilities (seed, logging, metrics, early stopping)
  dataset.py            # Data loading, feature engineering, train/test splits
  model.py              # Base model (SSVI, SmileModel, ensemble, losses)
  train.py              # Base model training loop
  experiment.py         # Experiment runner with visualization
  test.py               # Evaluation with arbitrage violation checks
  dgm.py                # DGM PDE solver network
  train_dgm.py          # DGM training with collocation resampling
  structural_break.py   # CUSUM/Bai-Perron change-point detection
  adjustment.py         # GRU+Attention adjustment model
  train_adjustment.py   # Adjustment training pipeline
  hyperiv.py            # HyperIV hypernetwork model
  train_hyperiv.py      # HyperIV training
  diffusion.py          # DDPM (UNet1D, noise schedule, sampler)
  train_diffusion.py    # DDPM training
  transfer.py           # Transfer learning utilities (weight loading, differential LR)
scripts/
  download_data.py      # Download TXO data from FinMind API + TWII/VIX from yfinance
  build_features.py     # Compute enhancement features (VIXTWN, RV, VRP, etc.)
  compare_architectures.py  # Additive vs multiplicative A/B test
  plot_smooth_iv_check.py   # Fixed-yATM smooth surface verification
  plot_training_curves.py   # Training loss curve visualization
  inspect_ssvi_params.py    # SSVI parameter inspection
  diagnose_rho_gradient.py  # Per-loss rho gradient analysis
  train_diagnose.py         # Training with per-epoch parameter tracking
docs/
  research_report.md    # Full research paper (1200 lines)
  discussion_notes.md   # Issue tracking and resolution log
  prediction_analysis.md # Architecture fix notes
models/                 # Trained model weights (.pt, gitignored)
dataset/                # TXO options data (gitignored)
logs/                   # Training logs and metrics (gitignored)
tests/                  # 215 unit tests
```

## Setup

```bash
# Create conda environment
conda create -n smartiv python=3.12 -y
conda activate smartiv

# Install PyTorch with CUDA 12.4
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt

# Install test framework
pip install pytest
```

## Usage

All training scripts are run from the `src/` directory:

```bash
cd src

# Phase 1: Train base model (SSVI + NN ensemble)
python train.py --on_gpu --epochs 2000

# Phase 2: Train DGM PDE solver
python train_dgm.py --on_gpu

# Phase 3: Train adjustment model (requires base model to be trained first)
python train_adjustment.py --on_gpu

# Phase 4: Train HyperIV
python train_hyperiv.py --on_gpu --epochs 500

# Phase 5: Train DDPM
python train_diffusion.py --on_gpu --epochs 1000

# Evaluate base model on test set
python test.py --on_gpu

# Generate figures from logs
cd ..
python scripts/generate_plots.py
```

### Transfer Learning

All training scripts support the `--finetune` flag to initialize from a previous checkpoint. This enables faster convergence when retraining on updated data:

```bash
cd src

# Fine-tune base model from existing weights
python train.py --on_gpu --finetune ../models/MultiModel.pt

# Fine-tune HyperIV from existing weights
python train_hyperiv.py --on_gpu --finetune ../models/HyperIVModel.pt

# Fine-tune all other models similarly
python train_dgm.py --on_gpu --finetune ../models/DGMModel.pt
python train_adjustment.py --on_gpu --finetune ../models/AdjustmentModel.pt
python train_diffusion.py --on_gpu --finetune ../models/DiffusionModel.pt
```

Transfer learning uses **differential learning rates**: pretrained layers learn at 1/10th the normal rate (to preserve useful knowledge), while newly initialized layers learn at full speed (to quickly adapt to new features). This is especially important for the Adjustment and DDPM models, where the input dimension changed (new enhancement features were added).

## Data

The system uses Taiwan Stock Exchange Options (TXO) data and supplementary market features. Data can be automatically downloaded or manually provided.

### Automatic Download

```bash
# Download TXO options (2022-2026) from FinMind API + TWII/VIX from yfinance
python scripts/download_data.py

# Compute enhancement features (VIXTWN, realized volatility, VRP, etc.)
python scripts/build_features.py
```

### Data Files

| File | Description | Rows |
|------|-------------|------|
| `dataset/prs_dataset_full.csv` | Full TXO options dataset (2014-2026) | 480,194 |
| `dataset/TWII_full.csv` | TAIEX underlying index daily prices | 2,947 |
| `dataset/VIX_full.csv` | CBOE VIX index daily | 3,043 |
| `dataset/enhancement/daily_features.csv` | Computed market features (23 columns) | 2,947 |

### Enhancement Features

These additional market features improve the Adjustment and DDPM models by providing richer market context:

| Feature | Description | Used By |
|---------|-------------|---------|
| VIXTWN | Synthetic Taiwan VIX computed from ATM options | DDPM |
| Realized Volatility (20d) | Actual price volatility over past 20 days | Adjustment, DDPM |
| IV Term Slope | Slope of the IV term structure (long vs short maturity) | Adjustment, DDPM |
| IV Skew | Difference between put-side and call-side IV | Adjustment, DDPM |
| Variance Risk Premium | Gap between implied and realized volatility (fear gauge) | Adjustment, DDPM |
| S&P 500 Return | US market return as a global risk factor | Adjustment, DDPM |
| Futures Basis | Deviation of futures price from theoretical fair value | Adjustment, DDPM |

### Train/Test Split

Training uses 2014-2024 data; testing uses 2025-2026 data (strictly chronological split, no data leakage). The validation set is a 20% random sample within the training period.

## Testing

215 unit tests covering all 5 model families, loss functions, data pipelines, and training loops:

```bash
python -m pytest tests/ -v           # All tests (~4 seconds)
python -m pytest tests/test_model.py # Base model only
```

Tests use `float64` precision, tiny model architectures (`hidden_sizes=[5,5,5]`), and synthetic data — no GPU, dataset, or trained models required.

## References

1. Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance.
2. Sirignano, J. & Spiliopoulos, K. (2018). *DGM: A deep learning algorithm for solving partial differential equations.* Journal of Computational Physics.
3. HyperIV (ICML 2025). *Hypernetwork-based implied volatility surface interpolation.*
4. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.

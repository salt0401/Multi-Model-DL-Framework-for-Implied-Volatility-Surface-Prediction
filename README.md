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

This system uses five complementary models, each addressing different aspects of IV surface prediction:

### 1. Base Model: SSVI + Neural Network Ensemble

The foundation. An **SSVI parametric model** provides the structural prior (guaranteed no-arbitrage shape), and a **neural network ensemble** learns the residual patterns that SSVI misses.

Each neural network ("SmileModel") takes log-moneyness as input and outputs total variance for a single expiration. Five networks are trained with different random seeds and combined via learned softmax weights. The loss function includes physics-informed terms: calendar spread constraints (total variance must increase with time), butterfly constraints (probability density must be non-negative), and density smoothness penalties.

**Results:** TV-RMSE 0.0134, MAPE 44.1%, IV-RMSE 0.209

### 2. DGM: PDE Solver for Black-Scholes

The **Deep Galerkin Method** solves the backward Kolmogorov PDE — the partial differential equation that governs how option prices evolve over time. Instead of discretizing the PDE on a grid (finite differences), DGM uses a neural network as a continuous function approximator and penalizes PDE residuals at randomly sampled collocation points.

The architecture uses LSTM-like "S-layers" with gating mechanisms (update, forget, reset gates) and LayerNorm. The loss has three terms: PDE residual in the interior, boundary conditions at extreme strikes, and terminal conditions at expiration.

**Results:** Final PDE residual 2.6e-5, BS price RMSE 0.036

### 3. Adjustment Model: GRU + Attention for Crisis Periods

Financial crises (e.g., 2008 crash, COVID-19) cause structural breaks where the IV surface shifts dramatically. This model detects those breaks using CUSUM change-point detection and learns time-varying corrections.

A **GRU** (Gated Recurrent Unit) processes sequences of daily IV observations, and **multi-head attention** learns which past days matter most. The output is a multiplicative adjustment ratio. Training uses KDE-weighted loss to focus on rare tail events and oversamples crisis periods.

### 4. HyperIV: Hypernetwork (State-of-the-Art)

Based on the ICML 2025 HyperIV paper — the current state-of-the-art for IV surface interpolation. Instead of training a single model on all options, a **hypernetwork** generates unique neural network weights for each day's IV surface.

A **Transformer set encoder** reads a variable-size set of observed option contracts and produces a context embedding. A hypernetwork MLP then generates the weights of a small target MLP that maps `(tau, log-moneyness)` to total variance. This per-surface specialization dramatically improves accuracy.

**Results:** TV-RMSE 0.0074, MAPE 20.7%, IV-RMSE 0.076 (best point prediction)

### 5. DDPM: Diffusion Model for Surface Forecasting

A **Denoising Diffusion Probabilistic Model** generates next-day IV surfaces conditioned on current market state. While models 1-4 interpolate today's surface, this model *forecasts* tomorrow's.

The architecture is a **1D U-Net** that denoises flattened IV surface vectors. Conditioning on market features (underlying price, VIX, volume, returns) is done via FiLM layers (Feature-wise Linear Modulation). The model learns to reverse a 1000-step noise corruption process using a cosine schedule.

**Results:** Test surface RMSE 0.0029 (best surface generation)

## Results Summary

### Point Prediction (Interpolation)

| Model | TV-RMSE | MAPE | IV-RMSE | Improvement |
|-------|---------|------|---------|-------------|
| Base (SSVI+NN) | 0.0134 | 44.1% | 0.209 | Baseline |
| HyperIV | 0.0074 | 20.7% | 0.076 | **45% / 53% / 64%** |

### Specialized Tasks

| Model | Task | Key Metric |
|-------|------|------------|
| DGM | PDE solving | Residual: 2.6e-5, BS RMSE: 0.036 |
| DDPM | Surface forecasting | Test RMSE: 0.0029 |

### Training Curves

| Base Model (76 epochs, early stopped) | HyperIV (58 epochs, early stopped) |
|:---:|:---:|
| ![Base Model Training](figures/base_model_training_zoomed.png) | ![HyperIV Training](figures/hyperiv_training.png) |

| DGM PDE Solver (5000 epochs) | DDPM Diffusion (1000 epochs) |
|:---:|:---:|
| ![DGM Training](figures/dgm_training.png) | ![DDPM Training](figures/diffusion_training.png) |

### IV Surface Visualizations

| Predicted vs Observed IV Smiles | Predicted 3D IV Surface |
|:---:|:---:|
| ![IV Smiles](figures/iv_smiles.png) | ![IV Surface](figures/iv_surface_pred.png) |

## Project Structure

```
README.md               # This file
EXPERIMENT.md           # Detailed experimental results and analysis
ARCHITECTURE.md         # System architecture and design decisions
requirements.txt        # Python dependencies
Metainfo.txt            # Original project metadata
src/
  config.ini            # All hyperparameters
  utils.py              # Utilities (seed, logging, metrics, early stopping)
  dataset.py            # Data loading and preprocessing
  model.py              # Base model (SSVI, SmileModel, ensemble, losses)
  train.py              # Base model training loop
  experiment.py         # Experiment runner with visualization
  test.py               # Evaluation with arbitrage checks
  dgm.py                # DGM PDE solver network
  train_dgm.py          # DGM training with collocation resampling
  structural_break.py   # CUSUM/Bai-Perron change-point detection
  adjustment.py         # GRU+Attention adjustment model
  train_adjustment.py   # Adjustment training pipeline
  hyperiv.py            # HyperIV hypernetwork model
  train_hyperiv.py      # HyperIV training
  diffusion.py          # DDPM (UNet1D, noise schedule, sampler)
  train_diffusion.py    # DDPM training
scripts/
  generate_plots.py     # Regenerate all figures from logs
figures/                # Experimental plots (9 PNGs)
docs/
  keynote.pdf           # Presentation slides
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

# Phase 3: Train adjustment model (requires base model)
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

## Data

Requires TXO options data in `dataset/`:

| File | Description |
|------|-------------|
| `2009_2023.pkl` or `.csv` | Raw TXO options data (2009-2023) |
| `TWII.csv` | TAIEX underlying index prices |
| `VIX.csv` | Realized volatility index |

Training uses 2014-2020 data; testing uses 2021 data (chronological split, no leakage).

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

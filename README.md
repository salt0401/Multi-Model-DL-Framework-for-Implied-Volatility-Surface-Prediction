# Smart IV Surface Analysis

Implied volatility (IV) surface prediction system for Taiwan Stock Exchange Options (TXO) using SSVI parametrization, neural network ensembles, and deep generative models.

## Models

| Phase | Model | Description | Script |
|-------|-------|-------------|--------|
| 1 | **Base (SSVI + NN Ensemble)** | SSVI prior with SmileModel neural networks, softmax-weighted ensemble. Physics-informed loss (calendar, butterfly, density constraints). | `train.py` |
| 2 | **DGM** | Deep Galerkin Method PDE solver for Black-Scholes backward Kolmogorov equation. Supports transfer learning across tau ranges. | `train_dgm.py` |
| 3 | **Adjustment (GRU + Attention)** | Time-series adjustment for structural breaks (crises). GRU encoder with multi-head attention, KDE-weighted loss, crisis oversampling. | `train_adjustment.py` |
| 4 | **HyperIV** | Hypernetwork-based IV surface interpolation (ICML 2025 SOTA). Transformer set encoder generates target MLP weights per surface. | `train_hyperiv.py` |
| 5 | **DDPM** | Conditional Denoising Diffusion Probabilistic Model for next-day IV surface forecasting. 1D U-Net with FiLM conditioning on market features. | `train_diffusion.py` |

## Project Structure

```
src/
  config.ini              # All model/training hyperparameters
  utils.py                # Utilities (seed, logging, metrics, early stopping)
  dataset.py              # Data loading, preprocessing, train/val/test splits
  model.py                # Base model (SSVI, SmileModel, MultiModel, losses)
  train.py                # Base model training
  experiment.py           # Experiment runner with visualization
  test.py                 # Evaluation with arbitrage violation checks
  dgm.py                  # DGM network and PDE loss
  train_dgm.py            # DGM training with resampling + transfer learning
  structural_break.py     # CUSUM/Bai-Perron structural break detection
  adjustment.py           # GRU+Attention adjustment model
  train_adjustment.py     # Adjustment model training pipeline
  hyperiv.py              # HyperIV model (set encoder + hypernetwork)
  train_hyperiv.py        # HyperIV training with per-surface batching
  diffusion.py            # DDPM (UNet1D, noise schedule, sampler)
  train_diffusion.py      # DDPM training for IV surface forecasting
dataset/                  # Data files (not tracked)
```

## Setup

```bash
# Create conda environment
conda create -n smartiv python=3.12 -y
conda activate smartiv

# Install PyTorch with CUDA
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install numpy pandas matplotlib tqdm scipy scikit-learn statsmodels ruptures
```

## Usage

```bash
cd src

# Train base model
python train.py --on_gpu --epochs 2000

# Evaluate on test set
python test.py --on_gpu

# Train HyperIV
python train_hyperiv.py --on_gpu --epochs 500

# Train DDPM
python train_diffusion.py --on_gpu --epochs 1000

# Train DGM PDE solver
python train_dgm.py --on_gpu

# Train adjustment model (requires trained base model)
python train_adjustment.py --on_gpu
```

## Data

Requires TXO options data in `dataset/` folder:
- `2009_2023.pkl` or `2009_2023.csv` — Raw TXO options data
- `TWII.csv` — TAIEX underlying index prices
- `VIX.csv` — VIX data (for adjustment and diffusion models)

## Configuration

All hyperparameters are in `src/config.ini`. Key sections:

- `[model_sett]` — Base model architecture (hidden sizes, ensemble count, loss weights)
- `[training]` — Epochs, batch size, learning rate schedule, date ranges
- `[hyperiv]` — HyperIV transformer and target network settings
- `[diffusion]` — DDPM grid resolution, U-Net channels, noise schedule
- `[dgm]` — DGM PDE solver domain and loss weights
- `[adjustment]` — GRU+Attention architecture, crisis event dates

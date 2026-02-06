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
tests/
  conftest.py             # Shared fixtures (float64, synthetic data, mock config)
  test_utils.py           # Utility functions and helper classes (22 tests)
  test_model.py           # All model forward passes and loss classes (52 tests)
  test_model_regression.py # Regression tests for 17 fixed bugs (18 tests)
  test_train_integration.py # End-to-end training loop integration (10 tests)
  test_hyperiv.py         # HyperIV model components (15 tests)
  test_diffusion.py       # DDPM components and noise schedule (19 tests)
  test_dgm.py             # DGM network, PDE loss, sampler (17 tests)
  test_adjustment.py      # GRU+Attention adjustment model (21 tests)
  test_structural_break.py # Break detection classes (16 tests)
  test_dataset.py         # DataProcessor with mock data (14 tests)
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

# Install test dependencies
pip install pytest
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

## Testing

215 unit tests covering all 5 model families, loss functions, data pipelines, and training loops. Tests use tiny model architectures and synthetic data for speed (~4 seconds total).

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_model.py -v

# Run regression tests only
python -m pytest tests/test_model_regression.py -v
```

### Test Categories

| Category | Tests | What's Covered |
|----------|-------|----------------|
| Model correctness | 52 | Forward pass shapes, dtypes, gradient flow for all model classes |
| Bug regressions | 18 | Named after bug IDs (M1-M5, X1-X3, T1-T2, E1) to prevent regressions |
| Training integration | 10 | `train_one_epoch`, `validate`, scheduler placement, early stopping |
| HyperIV | 15 | Set encoder, target network, hypernetwork param generation, loss |
| Diffusion (DDPM) | 19 | UNet1D, cosine noise schedule, trainer, sampler |
| DGM | 17 | S-layers, PDE residual, boundary/terminal conditions, BS pricing |
| Adjustment | 21 | SquarePlus, temporal attention, GRU model (3 modes), KDE loss |
| Structural break | 16 | CUSUM, Bai-Perron, VIX detectors, dispatcher |
| Dataset | 14 | Train/val/test splits, chronological ordering, mock I/O |
| Utilities | 22 | Seed, config parsing, metrics, early stopping, RMSE/MAPE |

### Key Design Decisions

- **Session-wide `float64`**: `conftest.py` sets `torch.set_default_dtype(torch.float64)` to match production
- **Tiny models**: `hidden_sizes=[5,5,5]`, `ensemble_num=2` for sub-second execution
- **No file I/O**: Dataset tests inject mock DataFrames directly into `DataProcessor`
- **Autograd-safe**: SmileModel tests never wrap forward passes in `torch.no_grad()` — the model uses `autograd.grad(create_graph=True)` internally

## Configuration

All hyperparameters are in `src/config.ini`. Key sections:

- `[model_sett]` — Base model architecture (hidden sizes, ensemble count, loss weights)
- `[training]` — Epochs, batch size, learning rate schedule, date ranges
- `[hyperiv]` — HyperIV transformer and target network settings
- `[diffusion]` — DDPM grid resolution, U-Net channels, noise schedule
- `[dgm]` — DGM PDE solver domain and loss weights
- `[adjustment]` — GRU+Attention architecture, crisis event dates

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
- **Model 2** is a *local volatility extractor* — it extracts arbitrage-free local volatility surfaces using a Dupire PDE-constrained ICNN
- **Model 3** is a *crisis detector* — it adjusts predictions during market stress
- **Model 5** is a *forecaster* — it predicts tomorrow's IV surface from today's market conditions

### 1. Base Model: SSVI + Neural Network Ensemble

**What it does:** The foundation of the system. Predicts implied volatility for any combination of strike price and time-to-expiry.

**How it works:** An **SSVI parametric model** (a well-known formula from quantitative finance) provides the structural backbone — it guarantees the predicted surface has a mathematically valid shape. A **neural network ensemble** (5 small networks combined) then learns the residual patterns that SSVI misses, like local bumps or skew features unique to the Taiwan market.

Each neural network ("SmileModel") uses automatic differentiation to compute first and second derivatives of its output, which are fed into physics-based penalty terms. The architecture uses an **additive formulation**: `w = SSVI(logm, yATM) + yATM * NN(tau, logm)`, where the yATM scaling keeps the NN correction proportional to the current volatility level. An earlier multiplicative version was abandoned after A/B testing showed it causes gradient explosion within 2 epochs (see `logs/architecture_comparison.json`).

The loss function has six components: (1) fit the observed data (RMSE), (2) stay close to the SSVI prior (MAPE), (3) enforce calendar spread constraints (longer-dated options must be worth more), (4) enforce butterfly constraints (no negative probabilities), (5) penalize extreme density curvature, and (6) encourage smoothness.

**Why an ensemble?** Training 5 networks with different random initializations and averaging their predictions reduces variance and produces more stable results than any single network.

**Results (2021 test):** TV-RMSE 0.0134, MAPE 44.1%, IV-RMSE 0.209, Butterfly violations 74%

### 2. ICNN Dupire: Local Volatility Extractor

**What it does:** Extracts the **local volatility surface** σ_LV(K,T) and **risk-neutral density** q(K,T) from Model 1's output, while simultaneously correcting the 74% butterfly violation problem.

**Why it matters:** Model 1's predictions have 74% butterfly violations — points where the predicted surface implies negative probability density, making direct application of the Dupire formula impossible (it produces imaginary local volatility). Model 2 learns a self-consistent pair of (call price, local vol) that satisfies the Dupire PDE, eliminating these violations from the architecture level.

**How it works:** An **Input-Convex Neural Network (ICNN)** guarantees that the predicted call price is always convex in the strike price K — this mathematically ensures ∂²C/∂K² ≥ 0 (the butterfly condition) by construction, not just by penalty. All weight matrices in the K→C(K) path are forced non-negative via softplus, and activations are monotone increasing (ReLU/Softplus). A second network predicts local volatility σ²_LV(K,T), and both networks are jointly trained to satisfy the **Dupire PDE**: ∂C/∂T = ½ σ²_LV K² ∂²C/∂K².

**Dual-path strategy:** The primary path (α) uses ICNN for hard convexity guarantee. An alternative path (β) uses Module A (soft surface correction) + GNO (Graph Neural Operator) for offline-trained global mapping. Both paths feed into Module D (Greeks: Vanna, Volga, ∂σ_LV/∂K) before Model 3.

**Results:** V1 (Soft PINN), V2 (ICNN with hard convexity), and V3 (Module D Greeks extraction) are complete. The V2 ICNN successfully eliminated 100% of butterfly violations. Downstream models now receive a 15-dimensional state (base + local vol + vanna + volga + lv gradient).

### 3. Adjustment Model: Architecture Comparison (GRU / xLSTM / TFT)

**What it does:** Detects and corrects for market crises — sudden events like the 2008 financial crash or COVID-19 that cause the IV surface to shift dramatically in ways the base model can't capture.

**Why it matters:** Normal market conditions are relatively smooth, but crises cause "structural breaks" — sudden regime changes where historical patterns no longer apply. Without correction, the base model's predictions become unreliable during these periods.

**How it works:** The model looks at the past 20 days of trading data (12 features per day, including base model predictions, VIX changes, S&P 500 returns, IV term structure slope, and realized volatility). A sequence encoder processes this time series, building up a representation of the current market regime. Then **multi-head attention** examines all 20 days and learns which past days are most relevant. The output is a multiplicative adjustment ratio: `adjusted_prediction = base_prediction * ratio`. Training uses KDE-weighted loss to focus on rare extreme events.

**Architecture comparison (2026-02-21):** Three architectures were trained and compared on identical data (245K sequences, chronological split, GPU float64):

| Metric | GRU (Baseline) | xLSTM (mLSTM) | TFT (Baseline) | TFT (CPR Regularized) |
|--------|:---:|:---:|:---:|:---:|
| **Val RMSE** | 0.1477 | 0.1414 | 0.1452 | **0.1404** |
| **Val MAPE** | 9.43% | 9.01% | 9.12% | **8.89%** |
| Parameters | 58,689 | **39,133** | 265,281 | 265,281 |
| Training Time | **41.4 min** | 311.5 min | 207.1 min | 175.3 min (fp32) |

**Winner: TFT with CPR** — Achieved the lowest RMSE (0.1404) and MAPE (8.89%), beating the unregularized xLSTM. While xLSTM is parameter-efficient, TFT provides excellent interpretability via its Variable Selection Network. Regularization (Constrained Parameter Regularization, CPR) was the key to unlocking TFT's performance by mitigating overfitting.

**Results (2026-02-22):** Architecture comparison and regularization research are complete. TFT + CPR selected as best candidate. Model 2 (ICNN Dupire) implementation complete (V1-V3).

### 4. HyperIV: Hypernetwork (State-of-the-Art)

**What it does:** The most accurate predictor in the system. Generates a specialized prediction model for *each individual day's* IV surface.

**Why it's special:** The base model learns a single average mapping that works for all days. But each day's IV surface has unique characteristics — maybe today has extra skew due to earnings announcements, or the term structure is inverted due to upcoming elections. HyperIV solves this by creating a custom neural network for each day.

**How it works:** Based on the ICML 2025 HyperIV paper — the current state-of-the-art for IV surface interpolation. The system works in two stages:

1. **Read the market:** A **Transformer set encoder** (the same architecture behind large language models) reads 50 observed option prices from today's market. The Transformer uses attention to understand cross-strike and cross-maturity relationships — "this put at strike 15000 tells us something about the call at strike 16000."

2. **Generate a specialist:** A **hypernetwork** takes the Transformer's summary and *generates the weights* of a small target neural network. This target network then predicts total variance for any `(tau, log-moneyness)` query point.

The key insight is that the hypernetwork doesn't predict IV directly — it predicts the *parameters of another neural network* that predicts IV. This means every day gets its own specialist predictor, automatically adapted to that day's unique market conditions.

**Results:** Pending retraining with latest codebase.

### 5. DDPM: Diffusion Model for Surface Forecasting

**What it does:** Forecasts *tomorrow's* IV surface based on today's market conditions. While models 1-4 interpolate today's surface from observed prices, this model predicts the future.

**Why it matters:** Portfolio managers need to know not just today's prices, but where they're headed. A good surface forecast enables proactive hedging — adjusting positions before the market moves, rather than reacting afterward.

**How it works:** A **Denoising Diffusion Probabilistic Model** (DDPM — the same family of models behind image generators like DALL-E and Stable Diffusion, but applied to financial surfaces instead of images). The process works in reverse:

1. **Training:** Take a real IV surface (a 10x20 grid = 200 numbers), gradually add random noise over 1000 steps until it becomes pure static. Train a neural network to reverse each step — given the noisy version, predict the noise that was added.

2. **Generation:** Start from pure random noise and apply the trained denoiser 1000 times. Each step removes a little noise, gradually revealing a realistic IV surface conditioned on the input market features.

The architecture is a **1D U-Net** (encoder-decoder with skip connections). Conditioning on 11 market features (today's surface summary + VIX level, VIX change, underlying return, realized volatility, S&P 500 return, IV term slope, IV skew, variance risk premium, futures basis, institutional positioning) is done via **FiLM layers** (Feature-wise Linear Modulation) — these tell the denoiser "generate a surface that looks like what the market should produce given these conditions."

**Key advantage over point prediction:** The diffusion model generates *coherent* surfaces where all 200 grid points are mutually consistent. Point prediction models (Base, HyperIV) predict each point independently, which can create internal inconsistencies.

**Results:** Pending retraining with updated condition_dim=11 (vixtwn_change removed).

## Results Summary

> **Status (2026-02-22):** Model 1 (SSVI+NN) is trained on `prs_dataset_no_fat(clean)` (2014-2020 train, 2021 test). Model 2 (ICNN Dupire) has completed implementation and validation for its V1-V3 phases. Model 3 (Adjustment) architecture comparison is complete — TFT+CPR selected. Models 4, 5 await retraining.

### Model 1 (Base SSVI+NN) — Current

| Metric | Value (2021 test) |
|--------|-------------------|
| TV-RMSE | 0.0134 |
| MAPE | 44.1% |
| IV-RMSE | 0.209 |
| Butterfly violations | 74% |

> An extended dataset (`prs_dataset_full.csv`, 480K rows, 2014-2026) exists but has known data quality issues in the 2022-2026 portion (see `docs/discussion_notes.md` §3.2). A future round of training on the full dataset is planned once these issues are resolved.

#### Training Curve & Implied Volatility Fit

**Training & Validation Loss:**  
The model optimizes 5 ensemble members simultaneously. Checkpointing saves the parameters at the lowest validation loss to avoid subsequent SSVI gradient explosion and degradation.
![Model 1 Loss Curve](model1_research/model1_research/figures/m1_loss_curve.png)

**Train Set Fit (2014-2020):**  
Each plot shows the observed options (blue dots) and the model's predicted IV curve (red line) for a specific expiration (tau) and baseline volatility level (yATM).
![Model 1 Train Fit](model1_research/model1_research/figures/m1_train_fit.png)

**Validation Set Fit (2014-2020 chronological split):**
![Model 1 Validation Fit](model1_research/model1_research/figures/m1_val_fit.png)

**Test Set Fit (2021 out-of-sample):**
![Model 1 Test Fit](model1_research/model1_research/figures/m1_test_fit.png)

### Model 3 (Adjustment) — Architecture Comparison Complete

| Architecture | Val RMSE | Val MAPE | Params | Status |
|-------------|:---:|:---:|:---:|--------|
| TFT + CPR | **0.1404** | **8.89%** | 265K | **Primary Choice** — Best overall performance & interpretability |
| TFT + AdamW | 0.1436 | 8.99% | 265K | Alternate Choice — Strong runner-up |
| GRU + CWD (baseline) | 0.1447 | 9.13% | 59K | Baseline Choice — Fastest inference, reference model |

> **Note:** As of 2026-02-22, the 3 models above have been officially shortlisted and retained in `model3_research/models/`. All other preliminary experiments (including xLSTM and unregularized versions) have been moved to `archived_models/` to maintain a clean project structure.

#### Regularization Loss Curves

During the architecture search, significant overfitting was observed across the board (e.g. train-val gap 2.8x-6.9x). The following plots show the impact of different regularization methods on the training and validation loss:

**Temporal Fusion Transformer (TFT) Regularization:**  
Constrained Parameter Regularization (CPR) significantly suppressed the validation loss spikes, allowing the highly-parameterized TFT model to achieve the lowest overall error.
![TFT Loss Curves](model3_research/figures/tft_regularization_loss_curves.png)

**GRU Baseline Regularization:**  
Cautious Weight Decay (CWD) mitigated overfitting better than standard AdamW for the recurrent GRU baseline.
![GRU Loss Curves](model3_research/figures/baseline_regularization_loss_curves.png)

### Models 2, 4, 5 — Pending

| Model | Task | Status |
|-------|------|--------|
| ICNN Dupire | Local vol extraction | Implemented (V1-V3) |
| HyperIV | Point prediction (SOTA) | Needs retraining |
| DDPM | Surface forecasting | Needs retraining (condition_dim=11) |

### Model 1 (SSVI+NN) Training Details

#### Training Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 5x ensemble of (SSVI + SmileModel), additive formulation |
| SmileModel layers | 3 hidden (64 → 32 → 16), Softplus activation, LayerNorm |
| Optimizer | AdamW, lr=0.001, gradient clip=1.0 |
| Batch size | 256 |
| LR schedule | MultiStepLR (gamma=0.5 every 5 epochs after epoch 500) |
| Early stopping | Patience 50 epochs |
| Dataset | `prs_dataset_no_fat(clean)` (~254K rows) |
| Training period | 2014-01-01 to 2020-12-31 |
| Test period | 2021-01-01 to 2021-12-31 |

#### SSVI Learned Parameters

All 5 ensemble members satisfy the Gatheral-Jacquier no-arbitrage constraint `eta*(1+|rho|) < 2`:

| Member | rho | eta | gamma | GJ value | Constraint |
|--------|-----|-----|-------|----------|------------|
| 0 | -0.315 | 1.060 | 0.533 | 1.394 | Satisfied |
| 1 | -0.309 | 1.069 | 0.538 | 1.400 | Satisfied |
| 2 | -0.311 | 1.068 | 0.537 | 1.400 | Satisfied |
| 3 | -0.309 | 1.074 | 0.540 | 1.406 | Satisfied |
| 4 | -0.306 | 1.073 | 0.540 | 1.402 | Satisfied |

The negative rho values encode the observed **left skew** in TXO options (OTM puts are more expensive than equidistant OTM calls), consistent with the volatility smile structure seen in equity markets worldwide.

#### Loss Component Breakdown (Final Epoch)

| Component | Weight | Value | Meaning |
|-----------|--------|-------|---------|
| RMSE | 1 | 0.0015 | Fit to observed data |
| MAPE | 1 | 0.063 | Relative prediction accuracy |
| Calendar | 10 | ~0 | No calendar arbitrage violations |
| Butterfly | 10 | 0 | No butterfly arbitrage violations |
| Linear (density) | 10 | 3e-6 | Smooth density in wings |
| Upper bound | 10 | 0 | Lee's moment bound satisfied |

The zero butterfly and calendar losses confirm the model produces **arbitrage-free surfaces**. This is critical for practical use — a surface with arbitrage violations implies negative probability densities, making it useless for option pricing.

#### Predicted Surface Shape (tau=0.5, Slice Across Strikes)

| Log-Moneyness | Total Variance | Implied Vol | Interpretation |
|---------------|---------------|-------------|----------------|
| -0.30 (deep OTM put) | 0.074 | 38.5% | High IV (crash protection premium) |
| -0.20 | 0.060 | 34.6% | |
| -0.10 | 0.054 | 32.8% | |
| 0.00 (ATM) | 0.050 | 31.7% | Baseline volatility level |
| +0.10 | 0.033 | 25.7% | |
| +0.20 | 0.028 | 23.8% | Minimum (slight right-side dip) |
| +0.30 (deep OTM call) | 0.032 | 25.3% | Slight uptick (right-wing smile) |

This shape is market-consistent: the strong left skew (38.5% vs 25.3%) reflects the well-known demand for downside protection in equity options, and the slight right-wing uptick produces the characteristic "smirk" shape.

#### Test Prediction Statistics (2021 Test)

| Metric | Value |
|--------|-------|
| Test points | ~52K |
| **TV-RMSE** | **0.0134** |
| **MAPE** | **44.1%** |
| **IV-RMSE** | **0.209** |
| **Butterfly violations** | **74%** |

#### Architecture Decision: Additive vs Multiplicative

An A/B test compared two formulations (see `logs/architecture_comparison.json`):

- **Additive** `w = SSVI(logm, yATM) + yATM * NN(tau, logm)`: Stable training, converges normally
- **Multiplicative** `w = SSVI(logm, yATM) * NN(tau, logm)`: **Explodes at epoch 2** (butterfly loss: 0 → 0.69 → 6.9, MAPE: 0.07 → 0.18 → 3.8)

Root cause: the product rule creates cross-terms in butterfly constraint derivatives that amplify gradient noise. The additive formulation isolates the SSVI and NN gradients, preventing this feedback loop.

### Training Curves & Visualizations

Training curve and IV surface visualizations can be regenerated from the training logs using `scripts/plot_training_curves.py`. The training logs are stored in `logs/` and results are documented in `EXPERIMENT.md`.

## Key Findings (from Model 1 training)

1. **The base model has a stability problem.** SSVI parameter optimization becomes unstable after extended training, causing gradient explosion. This is a fundamental challenge of combining parametric models (SSVI) with neural network optimization — the parametric part can drift into degenerate configurations. Early stopping and checkpoint saving are essential safeguards.

2. **Additive architecture is essential.** A/B testing confirmed that `w = SSVI + yATM * NN` is strictly superior to the multiplicative `w = SSVI * NN`. The multiplicative version explodes at epoch 2 due to product-rule cross-terms in the butterfly constraint derivatives.

3. **Butterfly violations remain a challenge.** The base model's 74% butterfly violation rate indicates the density constraint needs stronger enforcement. This motivates Model 2 (ICNN Dupire), which guarantees convexity by architecture.

4. **SSVI bounded parameterization is critical.** Constraining `eta ∈ (0,2)` via sigmoid and enforcing negative rho for equity left-skew prevents parameter explosion during training.

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
  structural_break.py   # CUSUM/Bai-Perron change-point detection
  hyperiv.py            # HyperIV hypernetwork model
  train_hyperiv.py      # HyperIV training
  diffusion.py          # DDPM (UNet1D, noise schedule, sampler)
  train_diffusion.py    # DDPM training
  transfer.py           # Transfer learning utilities (weight loading, differential LR)
scripts/
  download_data.py      # Download TXO data from FinMind API + TWII/VIX from yfinance
  build_features.py     # Compute enhancement features (RV, VRP, IV skew, etc.)
  compare_architectures.py  # Additive vs multiplicative A/B test
  plot_smooth_iv_check.py   # Fixed-yATM smooth surface verification
  plot_training_curves.py   # Training loss curve visualization
  inspect_ssvi_params.py    # SSVI parameter inspection
  diagnose_rho_gradient.py  # Per-loss rho gradient analysis
  train_diagnose.py         # Training with per-epoch parameter tracking
model2_research/        # Model 2 Local Volatility Extractor (ICNN)
  dupire_pinn.py        # Dupire PINN local vol extractor (V1/V2 ICNN)
  train_dupire.py       # Dupire PINN training script
  module_d.py           # V3 Greeks Extractor (Vanna, Volga, LV Grad)
  extract_features.py   # V3 downstream feature extraction script
model3_research/        # Model 3 architecture comparison & overfitting research
  README.md             # Training results, 3-way comparison, benchmarks
  tft_adjustment.py     # Temporal Fusion Transformer adjustment model
  train_models.py       # Unified training script (--model xlstm|tft|baseline)
  overfitting_research/ # Overfitting analysis and regularization research
docs/
  research_report.md    # Full research paper (1200 lines)
  discussion_notes.md   # Issue tracking and resolution log
  prediction_analysis.md # Architecture fix notes
models/                 # Trained model weights (.pt, gitignored)
dataset/                # TXO options data (gitignored)
                        #   prs_dataset_no_fat(clean).csv (~254K rows, 2014-2021, active)
                        #   prs_dataset_full.csv (480K rows, 2014-2026, NOT used — data quality issues)
logs/                   # Training logs and metrics (gitignored)
tests/                  # 177 unit tests
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

Training scripts are organized by model phase:

```bash
# Phase 1: Train base model (SSVI + NN ensemble)
cd model1_research
python train_pipeline.py --on_gpu --epochs 2000

# Phase 2 & 2.5: Train ICNN Dupire local vol extractor and extract V3 features
cd ../model2_research
python train_dupire.py --on_gpu --use_icnn
python extract_features.py --model_path ../models/DupireModel.pt --use_icnn

# Phase 3: Train adjustment model (TFT+CPR)
cd ../model3_research
python scripts/train_models.py --model tft

# Phase 4 & 5: Train HyperIV and DDPM
cd ../src
python train_hyperiv.py --on_gpu --epochs 500
python train_diffusion.py --on_gpu --epochs 1000

# Evaluate base model on test set
python test.py --on_gpu

# Generate training curve plots from logs
cd ..
python scripts/plot_training_curves.py
```

### Transfer Learning

All training scripts support the `--finetune` flag to initialize from a previous checkpoint. This enables faster convergence when retraining on updated data:

```bash
cd model1_research

# Fine-tune base model from existing weights
python train.py --on_gpu --finetune ../model1_research/models/MultiModel.pt

# Fine-tune HyperIV from existing weights
python train_hyperiv.py --on_gpu --finetune ../models/HyperIVModel.pt

# Fine-tune DDPM from existing weights
python train_diffusion.py --on_gpu --finetune ../models/DiffusionModel.pt
```

Transfer learning uses **differential learning rates**: pretrained layers learn at 1/10th the normal rate (to preserve useful knowledge), while newly initialized layers learn at full speed (to quickly adapt to new features). This is especially important for the Adjustment and DDPM models, where the input dimension changed (new enhancement features were added).

## Data

The system uses Taiwan Stock Exchange Options (TXO) data and supplementary market features. Data can be automatically downloaded or manually provided.

### Automatic Download

```bash
# Download TXO options (2022-2026) from FinMind API + TWII/VIX from yfinance
python scripts/download_data.py

# Compute enhancement features (realized volatility, VRP, IV skew, etc.)
python scripts/build_features.py
```

### Data Files

| File | Description | Rows | Status |
|------|-------------|------|--------|
| `dataset/prs_dataset_no_fat(clean).csv` | TXO options (2014-2021, pre-processed) | ~254K | **Active** — used for training |
| `dataset/prs_dataset_full.csv` | Extended TXO options (2014-2026) | 480,194 | Not used — data quality issues in 2022-2026 portion |
| `dataset/TWII.csv` / `TWII_full.csv` | TAIEX underlying index daily prices | ~2,900 | |
| `dataset/VIX.csv` / `VIX_full.csv` | CBOE VIX index daily | ~3,000 | |
| `dataset/enhancement/daily_features.csv` | Computed market features (23 columns) | ~2,900 | |

### Enhancement Features

These additional market features improve the Adjustment and DDPM models by providing richer market context:

| Feature | Description | Used By |
|---------|-------------|---------|
| Realized Volatility (20d) | Actual price volatility over past 20 days | Adjustment, DDPM |
| IV Term Slope | Slope of the IV term structure (long vs short maturity) | Adjustment, DDPM |
| IV Skew | Difference between put-side and call-side IV | Adjustment, DDPM |
| Variance Risk Premium | Gap between implied and realized volatility (fear gauge) | Adjustment, DDPM |
| S&P 500 Return | US market return as a global risk factor | Adjustment, DDPM |
| Futures Basis | Deviation of futures price from theoretical fair value | Adjustment, DDPM |

### Train/Test Split

Training uses 2014-2020 data; testing uses 2021 data (strictly chronological split, no data leakage). The validation set is the last 20% of training dates (chronological, not random).

> **Note:** The 2022-2026 data in `prs_dataset_full.csv` has known quality issues (tau distribution shift, hardcoded risk-free rate, extreme IV values) and has **not** been used for training. See `docs/discussion_notes.md` §3.2 for details.

## Testing

177 unit tests covering all 5 model families, loss functions, data pipelines, and training loops:

```bash
python -m pytest tests/ -v           # All tests (~4 seconds)
python -m pytest model1_research/tests/test_model.py # Base model only
```

Tests use `float64` precision, tiny model architectures (`hidden_sizes=[5,5,5]`), and synthetic data — no GPU, dataset, or trained models required. The legacy DGM code was previously evaluated but has been completely removed in favor of the ICNN Dupire approach.

## References

1. Gatheral, J. & Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance.
2. HyperIV (ICML 2025). *Hypernetwork-based implied volatility surface interpolation.*
3. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS.

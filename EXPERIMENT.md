# Experimental Results

Detailed training results for all five IV surface prediction models, trained on TXO options data (2014-2020 training, 2021 test).

## 1. Base Model (SSVI + Neural Network Ensemble)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | SSVI prior + 5 SmileModel NNs (64-32-16) |
| Ensemble method | Learned softmax weights |
| Learning rate | 0.001 |
| Batch size | 256 |
| Max epochs | 2000 |
| Early stopping patience | 50 |
| Loss weights | `[1, 1, 10, 10, 10, 10]` (data, SSVI, calendar, butterfly, density, smoothness) |
| Gradient clipping | 1.0 |

### Training

- **Epochs trained:** 76 / 2000 (early stopped)
- **Best epoch:** 26 (val loss = 2.6624)
- Initial instability: first 3 epochs had losses in the millions due to physics loss terms (calendar/butterfly constraints) calibrating against random weights
- Converged by epoch ~20, then gradually overfit
- A second instability spike at epoch 62 (train loss = 38.4) and epoch 74 (train loss = 59,351) triggered early stopping

![Base Model Training — Full](figures/base_model_training.png)

![Base Model Training — Zoomed](figures/base_model_training_zoomed.png)

### Test Metrics

| Metric | Value |
|--------|-------|
| TV-RMSE | 0.0134 |
| MAPE | 44.1% |
| IV-RMSE | 0.209 |
| Butterfly violations | 74% |

### Analysis

The high MAPE (44.1%) is driven by near-ATM options where true total variance is very small — even a small absolute error produces a large percentage error. The 74% butterfly violation rate indicates the model struggles with the curvature constraint in the wings. This is a known limitation of additive ensemble approaches: each SmileModel independently predicts a smooth curve, but the softmax-weighted combination can produce kinks.

![IV Smiles — Predicted vs Observed](figures/iv_smiles.png)

## 2. HyperIV (Hypernetwork)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | Transformer set encoder (128-dim, 4 heads, 2 layers) + Hypernetwork MLP |
| Target MLP | 64-32 hidden dims |
| Reference points | 50 per surface |
| Learning rate | 0.001 |
| Batch size | 32 (per-surface) |
| Max epochs | 500 |
| Train/Val/Test surfaces | 1372 / 344 / 244 |

### Training

- **Epochs trained:** 58 / 500 (early stopped)
- **Best epoch:** 8 (val loss = 0.000176)
- Rapid convergence: loss dropped from 2228 to 1.3e-5 in first 10 epochs
- Slight overfitting after epoch 10 (train continued improving, val plateaued)

![HyperIV Training](figures/hyperiv_training.png)

### Test Metrics

| Metric | Value |
|--------|-------|
| TV-RMSE | 0.0074 |
| MAPE | 20.7% |
| IV-RMSE | 0.076 |

### Analysis

HyperIV outperforms the base model on every metric: 45% lower TV-RMSE, 53% lower MAPE, 64% lower IV-RMSE. The key advantage is **per-surface specialization** — generating unique weights for each day means the model adapts to the specific shape of that day's IV surface rather than learning a single average mapping.

The Transformer set encoder is critical: it attends across all reference options simultaneously, capturing cross-strike and cross-maturity relationships that the base model's per-expiration SmileModels miss.

## 3. DGM (Deep Galerkin Method PDE Solver)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 3 S-layers, 64 hidden dim |
| Domain | sigma: [0.05, 1.0], t: [0.02, 2.0], S: [0.5, 1.5] |
| Loss weights | PDE: 1.0, BC: 1.0, TC: 1.0 |
| Collocation points | Interior: 5000, Boundary: 500, Terminal: 500 |
| Resample every | 100 epochs |
| Epochs | 5000 |

### Training

All four loss components decreased monotonically over 5000 epochs:

| Component | Epoch 100 | Epoch 2500 | Epoch 5000 |
|-----------|-----------|------------|------------|
| Total | 0.006215 | 0.000450 | 0.000166 |
| PDE | 0.001073 | 0.000088 | 0.000028 |
| BC | 0.001106 | 0.000025 | 0.000006 |
| TC | 0.004036 | 0.000337 | 0.000132 |

![DGM Training](figures/dgm_training.png)

### Test Metrics

| Metric | Value |
|--------|-------|
| Best PDE residual | 2.6e-5 |
| BS price RMSE | 0.036 |

### Analysis

The DGM successfully learns the Black-Scholes PDE solution with very low residual error. Terminal condition (TC) loss dominates early training because the payoff function has a kink at the strike price, which is harder for smooth neural networks to approximate. By epoch 5000, BC has dropped to 6e-6, indicating near-perfect boundary condition satisfaction.

The BS price RMSE of 0.036 means model prices differ from analytical Black-Scholes by ~3.6 cents on average — acceptable for a mesh-free PDE solver that generalizes across the full (S, t, sigma) domain.

## 4. Adjustment Model (GRU + Attention)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer GRU (64 hidden) + 4-head attention |
| Sequence length | 20 days |
| Dropout | 0.2 |
| Prediction target | Ratio (multiplicative adjustment) |
| Crisis dates | 2001-09, 2008-10, 2016-05 |
| Oversampling factor | 5x for crisis periods |
| KDE bandwidth | 0.1 |

### Analysis

The adjustment model serves as a post-processor that applies time-varying corrections during structural breaks. It was trained after the base model to capture regime-specific deviations. The KDE-weighted loss ensures the model pays attention to tail events (extreme IV values) rather than just minimizing average error.

## 5. DDPM (Diffusion Model)

### Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 1D U-Net (64-128-256 channels) |
| Surface grid | 10 tau x 20 log-moneyness = 200-dim vector |
| Diffusion steps | 1000 |
| Noise schedule | Cosine |
| Condition dim | 4 (underlying, VIX, volume, returns) |
| Learning rate | 0.0002 |
| Batch size | 16 |
| Epochs | 1000 |

### Training

Train loss decreased steadily from 0.099 to 8.9e-5 over 1000 epochs. Validation RMSE was evaluated every 100 epochs:

| Checkpoint | Val RMSE |
|------------|----------|
| Epoch 100 | 0.01267 |
| Epoch 200 | 0.00929 |
| Epoch 400 | 0.00884 |
| Epoch 500 | 0.00823 |
| Epoch 600 | 0.00713 |
| Epoch 900 | 0.00706 (best) |
| Epoch 1000 | 0.00725 |

![DDPM Training Loss](figures/diffusion_training.png)

![DDPM Validation RMSE](figures/diffusion_val_rmse.png)

### Test Metrics

| Metric | Value |
|--------|-------|
| Test surface RMSE | 0.0029 |

### Analysis

The test RMSE (0.0029) is substantially better than the best validation RMSE (0.00706). This is likely because the test set (2021) had lower IV overall than the validation period, making surfaces easier to generate. The diffusion model excels at capturing the joint distribution of the entire surface — it generates *coherent* surfaces where all grid points are consistent with each other, unlike point prediction models that predict each point independently.

## Model Comparison

### Improvement Over Baseline

| Metric | Base Model | HyperIV | Relative Improvement |
|--------|-----------|---------|---------------------|
| TV-RMSE | 0.0134 | 0.0074 | -44.8% |
| MAPE | 44.1% | 20.7% | -53.1% |
| IV-RMSE | 0.209 | 0.076 | -63.6% |

![Model Comparison](figures/model_comparison.png)

### Strengths and Weaknesses

| Model | Strengths | Weaknesses |
|-------|-----------|------------|
| Base (SSVI+NN) | Physics-informed, interpretable | High violation rate, struggles in wings |
| HyperIV | Best accuracy, per-surface adaptation | Requires many reference points per surface |
| DGM | Mesh-free PDE solving, generalizes | Doesn't directly predict IV |
| Adjustment | Handles structural breaks | Requires crisis event labels |
| DDPM | Generates coherent surfaces, forecasts | Slow sampling (1000 denoising steps) |

## Limitations

1. **Single underlying asset** — All models are trained on TXO only. Transfer to other markets would require retraining.
2. **Stale VIX proxy** — The VIX file was synthesized from 20-day realized volatility rather than a market-implied VIX for Taiwan.
3. **Base model instability** — Physics-informed loss with 6 weighted terms is sensitive to hyperparameters. Training frequently diverges without careful learning rate tuning.
4. **No model stacking** — The five models are trained independently. An ensemble or stacking approach could combine their strengths.
5. **Butterfly violations** — The base model's 74% violation rate indicates the density constraint needs stronger enforcement (e.g., Lagrangian dual or penalty scheduling).

## Future Work

- Integrate HyperIV predictions as DDPM conditioning for improved surface forecasting
- Add explicit no-arbitrage constraints to HyperIV via penalty or projection
- Explore attention-based architectures for the base model (replace per-expiration SmileModels with a single cross-expiration model)
- Extend to American-style options using the DGM PDE framework with early exercise boundary

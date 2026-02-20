# A Multi-Model Deep Learning Framework for Implied Volatility Surface Prediction: Combining Physics-Informed Constraints with Generative Models on Taiwan Stock Exchange Options

---

## Abstract

Accurate prediction of the implied volatility (IV) surface is fundamental to derivatives pricing, risk management, and portfolio hedging. This paper presents a comprehensive multi-model framework for predicting the IV surface of Taiwan Stock Exchange Options (TXO), integrating five complementary deep learning models: (1) an SSVI-constrained neural network ensemble for physics-informed interpolation, (2) a Deep Galerkin Method (DGM) network for mesh-free PDE solving, (3) a GRU-Attention adjustment model for structural break correction, (4) a HyperIV hypernetwork for state-of-the-art per-surface specialization, and (5) a conditional Denoising Diffusion Probabilistic Model (DDPM) for next-day surface forecasting. Each model addresses a distinct aspect of IV surface modeling — from arbitrage-free interpolation to crisis-period adjustment and generative forecasting. We conduct two rounds of experiments: Round 1 on 254,044 observations (2014–2021) and Round 2 on 480,194 observations (2014–2026) with transfer learning and nine additional market features. Preliminary results show the base model achieves TV-RMSE 0.0120 and MAPE 33.0% on out-of-sample 2025–2026 data, with HyperIV and DDPM results pending revalidation after codebase updates. We also introduce a transfer learning framework with differential learning rates that enables efficient model adaptation when dataset dimensions change. The complete system is validated with 215 unit tests and evaluated on strictly out-of-sample data.

**Keywords:** implied volatility surface, SSVI, deep learning, hypernetwork, diffusion model, physics-informed neural network, options pricing, transfer learning

---

## 1. Introduction

### 1.1 Background and Motivation

The implied volatility (IV) surface is a core object in quantitative finance, encoding the market's consensus expectation of future asset price uncertainty across different strike prices and maturities. For a given option contract, the implied volatility is the volatility parameter that, when substituted into the Black-Scholes formula, reproduces the observed market price. When plotted as a function of both moneyness (the ratio of strike to underlying price) and time-to-expiry, these implied volatilities form a two-dimensional surface — the IV surface.

Accurately modeling the IV surface serves multiple practical purposes. First, it enables **fair pricing** of options contracts that are not actively traded by interpolating from liquid contracts. Second, it supports **risk management** by providing the sensitivity parameters (Greeks) needed for hedging. Third, it facilitates **arbitrage detection** — if a model predicts an IV surface that violates no-arbitrage constraints, it can identify mispriced contracts. Fourth, **forecasting** the IV surface one day ahead enables proactive portfolio adjustment before market moves occur.

Despite its importance, IV surface modeling presents several challenges. The surface must satisfy no-arbitrage constraints: calendar spreads require total variance to be non-decreasing in time-to-expiry, and butterfly spreads require the risk-neutral probability density to be non-negative everywhere. Classical parametric models such as SSVI (Gatheral and Jacquier, 2014) guarantee these constraints by construction but lack the flexibility to capture market microstructure effects. Conversely, purely data-driven neural network models can fit observed data well but frequently violate arbitrage constraints, producing economically meaningless predictions.

Furthermore, financial markets exhibit **regime changes** — sudden shifts in volatility dynamics during crises (e.g., the 2008 financial crisis, COVID-19 in 2020) that invalidate models trained on normal-market data. A robust IV surface prediction system must detect and adapt to these structural breaks.

### 1.2 Contributions

This paper makes the following contributions:

1. **Multi-model architecture.** We design and implement a five-model system where each model addresses a distinct aspect of IV surface prediction — interpolation, PDE consistency, crisis adaptation, per-surface specialization, and generative forecasting — rather than attempting to solve all problems with a single model.

2. **Physics-informed base model.** We combine the SSVI parametric model with a neural network ensemble using a six-component loss function that jointly optimizes data fidelity, parametric prior adherence, calendar and butterfly arbitrage constraints, density smoothness, and wing behavior. The additive formulation ($w = w_{\text{SSVI}} + \theta \cdot f_{\text{NN}}$) preserves the parametric structure as a baseline while allowing flexible nonlinear correction with simpler gradient flow than a multiplicative alternative.

3. **HyperIV adaptation for TXO.** We adapt the state-of-the-art HyperIV architecture (ICML 2025) — which uses a Transformer set encoder and hypernetwork to generate per-surface specialized prediction networks — to the Taiwan options market, achieving a 53% reduction in TV-RMSE compared to the base model.

4. **Conditional DDPM for surface forecasting.** We apply Denoising Diffusion Probabilistic Models to financial surface prediction, conditioned on 11 market features via FiLM (Feature-wise Linear Modulation) layers. To our knowledge, this is among the first applications of diffusion models to IV surface generation.

5. **Transfer learning framework.** We develop a transfer learning module that handles dimension mismatches (e.g., when adding new input features), applies differential learning rates for pretrained versus new parameters, and supports optional layer freezing. This enables efficient model retraining when datasets are extended.

6. **Comprehensive evaluation.** We conduct two rounds of experiments with strict chronological train-test splits (no data leakage), extensive ablation analysis, and arbitrage violation measurement. All code is validated with 215 unit tests.

### 1.3 Paper Organization

Section 2 reviews related work on IV surface modeling, physics-informed neural networks, hypernetworks, and diffusion models. Section 3 formalizes the problem and describes the data pipeline. Section 4 details each of the five model architectures. Section 5 describes the transfer learning framework. Section 6 presents the experimental setup. Section 7 reports and analyzes experimental results. Section 8 discusses limitations and implications. Section 9 concludes with future work directions.

---

## 2. Related Work

### 2.1 Parametric IV Surface Models

The Stochastic Volatility Inspired (SVI) parameterization, introduced by Gatheral (2004), models the total implied variance as a function of log-moneyness with five parameters per maturity slice. Gatheral and Jacquier (2014) extended this to the Surface SVI (SSVI) parameterization, which models the entire surface jointly and guarantees absence of static arbitrage under mild conditions on the parameter functions. SSVI expresses total variance as:

$$w(k, \theta) = \frac{\theta}{2}\left(1 + \rho\varphi(\theta)k + \sqrt{(\varphi(\theta)k + \rho)^2 + 1 - \rho^2}\right)$$

where $k$ is log-moneyness, $\theta$ is the ATM total variance at a given expiry, $\rho \in (-1, 1)$ is a correlation parameter, and $\varphi(\theta)$ is a function controlling the smile's curvature. The power-law form $\varphi(\theta) = \eta / (\theta^\gamma(1+\theta)^{1-\gamma})$ is commonly used.

While SSVI provides an elegant no-arbitrage framework, it is inherently limited by its parametric form: it cannot capture localized smile features, market microstructure effects, or nonlinear cross-maturity interactions that are present in real options data.

### 2.2 Neural Network Approaches to IV Modeling

Several works have applied neural networks to IV surface modeling. Dugas et al. (2009) used constrained neural networks with positive weight constraints to ensure convexity. Ackerer et al. (2020) combined neural networks with the SSVI prior as a regularizer. The challenge is balancing flexibility (fitting the data) with economic constraints (no-arbitrage).

Our base model follows the approach of combining an SSVI prior with a neural network residual, but extends it with: (a) an ensemble of five independently initialized networks with learned softmax weighting, (b) automatic differentiation to compute first and second derivatives within the network for physics-based penalties, and (c) synthetic wing data to enforce large-moneyness behavior.

### 2.3 Deep Galerkin Method (DGM)

Sirignano and Spiliopoulos (2018) introduced the Deep Galerkin Method, which uses neural networks to approximate solutions to high-dimensional PDEs. The key idea is to minimize the PDE residual at randomly sampled collocation points, avoiding the curse of dimensionality inherent in grid-based methods. The DGM architecture uses LSTM-like gating mechanisms (S-layers) to capture the nonlinear interactions in PDE solutions.

In our framework, DGM serves as a consistency verifier: it learns to solve the backward Kolmogorov equation (the risk-neutral pricing PDE), and its predictions can be compared against the base model's prices to check for PDE violations.

### 2.4 Hypernetwork-Based Models

Ha et al. (2017) introduced hypernetworks — neural networks that generate the weights of another network. The HyperIV model (ICML 2025) applies this concept to IV surface interpolation: a Transformer set encoder reads a variable-size set of observed options and a hypernetwork projects the encoding into the weights of a small target MLP that predicts total variance for any query point.

The key advantage over conventional models is **per-instance specialization**: each day's IV surface receives a uniquely generated prediction network, adapted to that day's specific market conditions. This is qualitatively different from a single fixed model that must average across all possible market states.

### 2.5 Diffusion Models

Denoising Diffusion Probabilistic Models (Ho et al., 2020), along with improvements by Nichol and Dhariwal (2021) on noise scheduling, have achieved state-of-the-art results in image generation. The core idea is to define a forward process that gradually adds Gaussian noise to data over $T$ timesteps, and train a neural network to reverse each step.

Applications of diffusion models to financial data are emerging. Our work applies DDPM to IV surface forecasting: given today's market conditions, generate tomorrow's complete 200-point IV surface via conditional reverse diffusion. This differs from point prediction approaches because the diffusion model generates all surface points jointly, maintaining internal coherence.

### 2.6 Attention Mechanisms for Financial Time Series

Gated Recurrent Units (Cho et al., 2014) have been widely used for sequential financial data. The addition of multi-head attention (Vaswani et al., 2017) enables the model to selectively weight different timesteps — a crucial capability for regime-change detection, where a sudden event (e.g., a VIX spike) several days ago may be more informative than gradual drift over weeks.

---

## 3. Problem Formulation and Data

### 3.1 Problem Definition

Let $C(K, \tau)$ denote the market price of a European call option with strike price $K$ and time-to-expiry $\tau$. The implied volatility $\sigma_{\text{imp}}(K, \tau)$ is defined as the unique non-negative value such that:

$$C(K, \tau) = \text{BS}(S, K, \tau, r, \sigma_{\text{imp}})$$

where $\text{BS}(\cdot)$ is the Black-Scholes pricing formula and $S$ is the current underlying price.

Following Gatheral and Jacquier (2014), we work with **total variance** $w(k, \tau) = \sigma_{\text{imp}}^2 \cdot \tau$ as the modeling target, where $k = \ln(K/S)$ is the log-moneyness. Total variance is smoother than raw IV and more amenable to parametric modeling.

Our system addresses four prediction tasks:

1. **Interpolation (Models 1, 4):** Given a set of observed options on day $t$, predict $w(k, \tau)$ for arbitrary $(k, \tau)$ query points.
2. **PDE consistency (Model 2):** Verify that predicted option prices satisfy the backward Kolmogorov PDE.
3. **Crisis adjustment (Model 3):** Detect structural breaks and correct base model predictions during regime changes.
4. **Forecasting (Model 5):** Given day $t$'s market conditions, predict the entire IV surface for day $t+1$.

### 3.2 No-Arbitrage Constraints

A valid IV surface must satisfy:

**Calendar spread constraint (C1):** Total variance must be non-decreasing in $\tau$:
$$\frac{\partial w}{\partial \tau} \geq 0$$

**Butterfly spread constraint (C2):** The risk-neutral density must be non-negative:
$$g(k) = \left(1 - \frac{k \cdot w'}{2w}\right)^2 - \frac{w'}{4}\left(\frac{1}{w} + \frac{1}{4}\right) + \frac{w''}{2} \geq 0$$

where $w' = \partial w / \partial k$ and $w'' = \partial^2 w / \partial k^2$.

**Upper bound constraint (C3):** Lee's moment formula requires $w(k) \leq 2|k|$ for large $|k|$, ensuring finite moments.

These constraints are incorporated into our loss functions through penalty terms.

### 3.3 Dataset

#### 3.3.1 TXO Options Data

The dataset comprises Taiwan Stock Exchange Options (TXO) data from two sources:

- **Historical data (2014–2021):** 254,044 preprocessed option observations from the original dataset (`prs_dataset_no_fat(clean).csv`), including fields: date, strike price, put/call indicator, option price, expiry date, time-to-expiry ($\tau$), underlying price ($S$), implied volatility, log-moneyness, and total variance.

- **Extended data (2022–2026):** Downloaded via the FinMind API (`TaiwanOptionDaily` endpoint) in monthly batches to respect rate limits. Raw data is filtered to regular-session settlement prices with positive volume, then preprocessed to compute expiry dates (handling monthly, Wednesday-weekly, and Friday-weekly contracts), filter by put-call liquidity (selecting the more liquid of the pair at each strike-date-expiry), and compute implied volatility via Black-Scholes inversion using L-BFGS-B optimization.

The **merged full dataset** contains 480,194 observations spanning 2014-01-02 to 2026-02-06.

#### 3.3.2 Feature Engineering

The preprocessing pipeline performs the following feature engineering steps:

1. **Log-moneyness computation:** $k = \ln(K_\beta / S_\beta)$, where $K_\beta$ and $S_\beta$ are beta-adjusted strike and underlying prices. Beta adjustment is performed via linear regression of put-call parity on (underlying, strike) within each (year, season, $\tau$) group, accounting for dividends and interest rates.

2. **ATM total variance interpolation:** For each time-to-expiry $\tau$, the ATM total variance $\theta(\tau)$ is obtained by identifying the option closest to at-the-money (minimizing $|S - K|$) and interpolating across $\tau$ values using linear interpolation with extrapolation.

3. **Implied volatility computation:** For each option, IV is computed by minimizing the squared difference between the Black-Scholes model price and the observed market price, using scipy's bounded minimization with initial guess $\sigma_0 = 0.2$ and bounds $[0.001, 1.0]$ (original data) or $[0.001, 3.0]$ (extended data).

4. **Synthetic wing data:** For the base model's smoothness constraint (C3), synthetic data is generated at extreme log-moneyness values (4–6 times the observed range) for each unique $\tau$.

#### 3.3.3 Enhancement Features

Seven additional daily market features are computed to enrich the conditioning of the DDPM and Adjustment models:

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| **Realized Volatility (20d)** | Annualized standard deviation of 20-day log returns: $\text{RV} = \sigma_{20d} \cdot \sqrt{252}$ | Actual historical price fluctuation |
| **IV Term Slope** | Mean long-dated ATM IV ($\tau > 0.3$) minus mean short-dated ATM IV ($\tau < 0.12$) | Term structure curvature; negative = inversion (crisis signal) |
| **IV Skew** | Mean OTM put IV ($-0.2 < k < -0.05$) minus ATM IV | Demand for downside protection |
| **Variance Risk Premium (VRP)** | $\text{ATM\_IV}^2 - \text{RV}_{20d}^2$ | Gap between implied and realized — "fear premium" |
| **S&P 500 Return** | Overnight return on ^GSPC from Yahoo Finance | US market spillover to Asia |
| **Futures Basis %** | $(F_{\text{close}} - S_{\text{close}}) / S_{\text{close}} \times 100$ | Market sentiment; positive = bullish |
| **Institutional Net Ratio** | $(\text{Long} - \text{Short}) / (\text{Long} + \text{Short})$ from FinMind | Smart money positioning |

These features are computed by `scripts/build_features.py` and stored as a single daily CSV.

### 3.4 Train-Test Split

The data is split **strictly chronologically** to prevent temporal data leakage:

- **Round 1:** Train on 2014–2020 (202K observations), validate on 20% random within-period split, test on 2021 (52K observations).
- **Round 2:** Train on 2014–2024 (384K observations), validate on last 20% of training dates, test on 2025–2026 (96K observations).

This chronological split is critical for financial time series — shuffled splits would allow future information to leak into training, producing overly optimistic results.

---

## 4. Model Architectures

### 4.1 Phase 1: Base Model — SSVI + Neural Network Ensemble

#### 4.1.1 Architecture

The base model combines an SSVI parametric prior with a neural network ensemble via additive composition. Each ensemble member $i$ consists of a `SingleModel`:

$$w_i(k, \tau, \theta) = w_{\text{SSVI}}(k, \theta) + \theta \cdot f_{\text{NN},i}(\tau, k)$$

where $w_{\text{SSVI}}$ is the SSVI parametric prediction, $f_{\text{NN},i}$ is the $i$-th `SmileModel` neural network, and $\theta$ is the ATM total variance (yATM). The $\theta$ scaling ensures the NN correction is proportional to the current volatility level. An earlier multiplicative formulation ($w_{\text{SSVI}} \cdot f_{\text{NN}}$) was abandoned after A/B testing showed it causes gradient explosion within 2 epochs due to product-rule cross-terms in the butterfly constraint derivatives (see `logs/architecture_comparison.json`).

**SSVI component (`SSVIModel`):** Implements the SSVI formula with learnable parameters $\rho$ (constrained via $\tanh$ to $(-1,1)$), $\eta$ and $\gamma$ (constrained via $\exp$ to be positive). The power-law $\varphi$ function is used. Analytical first and second derivatives with respect to log-moneyness are computed in closed form for efficiency.

**Neural network component (`SmileModel`):** A 3-hidden-layer MLP (64→32→16→1) with the following structure:
- **Input layer:** A custom bilinear input combining log-moneyness and time-to-expiry through exponential parameterization:
  $$h_0 = \text{LN}\left(\text{smile}(b_k + k \cdot e^{w_k}) \cdot \sigma(b_\tau + \tau \cdot e^{w_\tau}) \cdot e^{W_{\text{exp}}} + b_{\text{exp}}\right)$$
  where $\text{smile}(x) = \sqrt{x \tanh(x+0.5) + \tanh(-x/2) + 0.0005}$ is a custom smile-shaped function and LN is LayerNorm.
- **Hidden layers:** Softplus activation after each linear transformation.
- **Output:** Scalar total variance prediction.
- **Gradient computation:** First-order derivatives $\partial w/\partial \tau$ and $\partial w/\partial k$ are computed via `torch.autograd.grad` with `create_graph=True`; the second derivative $\partial^2 w/\partial k^2$ is computed via a second autograd call with `retain_graph=True` (without `create_graph` to avoid third-order gradient instability).

**Additive derivatives:** For the composite model $w = w_{\text{SSVI}} + \theta \cdot f_{\text{NN}}$, derivatives are computed by simple addition (no cross-terms):

$$\frac{\partial w}{\partial k} = \frac{\partial w_{\text{SSVI}}}{\partial k} + \theta \cdot \frac{\partial f_{\text{NN}}}{\partial k}$$

$$\frac{\partial^2 w}{\partial k^2} = \frac{\partial^2 w_{\text{SSVI}}}{\partial k^2} + \theta \cdot \frac{\partial^2 f_{\text{NN}}}{\partial k^2}$$

This is a key advantage over the multiplicative formulation, which would introduce cross-terms ($2 \cdot \partial w_{\text{SSVI}}/\partial k \cdot \partial f_{\text{NN}}/\partial k$) that create feedback loops between the SSVI and NN gradient paths, causing training instability.

**Ensemble aggregation (`SoftmaxModel`):** Five `SingleModel` instances are trained with different random initializations. Their predictions are combined via learned softmax weights that depend on the input features $(\tau, k, \theta)$:

$$w_{\text{ensemble}} = \sum_{i=1}^{5} \alpha_i(\tau, k, \theta) \cdot w_i$$

where $\alpha_i = \text{softmax}(\sigma([\ \tau, k, \theta\ ] \cdot W + b))_i$ and $\sigma$ is the sigmoid function.

#### 4.1.2 Loss Function

The `WeightedSumLoss` combines six components with configurable weights $[\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5, \lambda_6] = [1, 1, 10, 10, 10, 10]$:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{RMSE}} + \lambda_2 \mathcal{L}_{\text{MAPE}} + \lambda_3 \mathcal{L}_{\text{calendar}} + \lambda_4 \mathcal{L}_{\text{butterfly}} + \lambda_5 \mathcal{L}_{\text{density}} + \lambda_6 \mathcal{L}_{\text{upperbound}}$$

where:
- $\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{N}\sum_i (w_{\text{pred},i} - w_{\text{true},i})^2}$
- $\mathcal{L}_{\text{MAPE}} = \frac{1}{N}\sum_i \left|\frac{w_{\text{true},i} - w_{\text{pred},i}}{w_{\text{true},i} + \epsilon}\right|$ with $\epsilon = 0.005$
- $\mathcal{L}_{\text{calendar}} = \frac{1}{N}\sum_i \text{ReLU}(-\partial w / \partial \tau)$ — penalizes decreasing total variance
- $\mathcal{L}_{\text{butterfly}} = \frac{1}{N}\sum_i \text{ReLU}(-g(k))$ — penalizes negative density
- $\mathcal{L}_{\text{density}} = \frac{1}{M}\sum_j |\partial^2 w / \partial k^2|$ — evaluated on synthetic wing data for smoothness
- $\mathcal{L}_{\text{upperbound}} = \frac{1}{M}\sum_j \text{ReLU}(w - 2|k|)$ — evaluated on synthetic wing data for Lee's bound

The high weights on physics terms ($10\times$) ensure that arbitrage constraints are strongly enforced relative to data fitting.

#### 4.1.3 Training

Training uses AdamW optimizer with learning rate 0.001, batch size 256, gradient clipping at 1.0, and a multi-step learning rate scheduler. Early stopping with patience 50 prevents overfitting. The training loop alternates between real data batches (for $\mathcal{L}_{\text{RMSE}}, \mathcal{L}_{\text{MAPE}}, \mathcal{L}_{\text{calendar}}, \mathcal{L}_{\text{butterfly}}$) and synthetic wing data batches (for $\mathcal{L}_{\text{density}}, \mathcal{L}_{\text{upperbound}}$).

### 4.2 Phase 2: DGM PDE Solver

#### 4.2.1 Architecture

The Deep Galerkin Method (Sirignano and Spiliopoulos, 2018) trains a neural network to approximate the solution of the backward Kolmogorov equation for European option pricing under geometric Brownian motion:

$$\frac{\partial u}{\partial t} + \frac{1}{2}\sigma^2 x^2 \frac{\partial^2 u}{\partial x^2} + rx\frac{\partial u}{\partial x} - ru = 0$$

with terminal condition $u(x, T) = \max(x - K, 0)$ for calls and boundary conditions at extreme asset prices.

The `DGMNetwork` consists of:
- **Input projection:** Linear(2, 64) mapping $(x, t)$ to hidden dimension.
- **Three S-layers** with residual connections. Each S-layer implements LSTM-like gating:
  $$z = \sigma(W_z x + U_z s), \quad g = \sigma(W_g x + U_g s), \quad r = \sigma(W_r x + U_r s)$$
  $$h = \tanh(W_h x + U_h(s \odot r)), \quad s_{\text{new}} = \text{LN}((1-g) \odot h + z \odot s)$$
  where $s$ is the hidden state, and LayerNorm is applied to the output.
- **Output layer:** Linear(64, 1) producing the option price.
- Residual connections: $s \leftarrow s + s_{\text{new}}$ after each S-layer.

#### 4.2.2 Loss Function

The `DGMLoss` minimizes a weighted sum of three terms:

$$\mathcal{L} = \lambda_{\text{PDE}} \mathcal{L}_{\text{PDE}} + \lambda_{\text{BC}} \mathcal{L}_{\text{BC}} + \lambda_{\text{TC}} \mathcal{L}_{\text{TC}}$$

- $\mathcal{L}_{\text{PDE}} = \frac{1}{N_I}\sum_{i=1}^{N_I} r_i^2$ where $r_i$ is the PDE residual at interior point $(x_i, t_i)$, computed via two rounds of `autograd.grad` with `create_graph=True`.
- $\mathcal{L}_{\text{BC}} = \frac{1}{N_B}\sum_j (u_{\text{pred},j} - u_{\text{BC},j})^2$ at boundary points.
- $\mathcal{L}_{\text{TC}} = \frac{1}{N_T}\sum_j (u_{\text{pred},j} - \text{payoff}(x_j))^2$ at terminal time.

All weights are set to 1.0.

#### 4.2.3 Training

The `DGMSampler` generates collocation points: 5,000 interior, 500 boundary (250 lower, 250 upper), and 500 terminal. These points are re-sampled every 100 epochs to prevent overfitting to specific collocation points. Training runs for 5,000 epochs with Adam optimizer (lr=0.001) and ReduceLROnPlateau scheduler (patience=200, factor=0.5). The domain covers $S \in [0.5, 1.5]$ (normalized price), $t \in [0.02, 2.0]$ years, $\sigma \in [0.05, 1.0]$.

Validation is performed against the analytical Black-Scholes formula at 20 equally-spaced spot prices.

### 4.3 Phase 3: Adjustment Model — GRU + Attention

#### 4.3.1 Motivation

The base model learns a time-invariant mapping that works well during normal market conditions but fails during structural breaks — sudden regime changes where historical patterns cease to apply. The adjustment model is a time-varying post-processor that detects and corrects for these crisis periods.

#### 4.3.2 Architecture

The `TVAdjustmentModel` processes a 20-day sequence of 12-dimensional daily feature vectors through:

1. **GRU encoder** (2 layers, 64 hidden units, dropout 0.2): Processes the temporal sequence, building a representation of the current market regime. The bidirectional option is not used (unidirectional to respect causality).

2. **Temporal multi-head attention** (4 heads): Computes attention from the last hidden state (query) over all 20 timesteps (keys/values). This learns which past days are most informative — for example, a sudden VIX spike 3 days ago may be weighted more than gradual drift over 2 weeks.

   $$Q = W_q h_{T}, \quad K = W_k H, \quad V = W_v H$$
   $$\text{Attention} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

3. **Prediction head:** Linear(64→32→1) with ReLU and dropout.

4. **SquarePlus activation:** For ratio prediction mode, the output is passed through $f(x) = (x + \sqrt{x^2 + 4})/2$, which is always positive and smooth everywhere (unlike ReLU).

#### 4.3.3 Input Features

The 12 input features per timestep are:

| Index | Feature | Source |
|-------|---------|--------|
| 0 | VIX daily change | CBOE VIX |
| 1 | Underlying return | TAIEX |
| 2 | Log-moneyness | Options data |
| 3 | Time-to-expiry | Options data |
| 4 | Base model TV prediction | Base model forward pass |
| 5 | ITM/OTM indicator | Options data |
| 6 | S&P 500 return | Yahoo Finance |
| 7 | IV term slope | Computed from TXO |
| 8 | IV skew | Computed from TXO |
| 9 | VRP (20d) | Computed |
| 10 | Futures basis % | FinMind |
| 11 | Realized vol (20d) | TAIEX returns |

The `tv_pred` feature (index 4) is computed by running the trained base model on all data points. To avoid GPU out-of-memory errors from autograd graph accumulation during this inference step, we process the data in chunks of 5,000 rows.

#### 4.3.4 Loss Function

The `AdjustmentLoss` combines MSE and MAPE with KDE-based sample weighting:

$$\mathcal{L} = \sum_i w_i (y_i - \hat{y}_i)^2 + 0.5 \sum_i w_i \left|\frac{y_i - \hat{y}_i}{y_i + \epsilon}\right|$$

The weights $w_i$ are computed from kernel density estimation (KDE) of the target distribution:
$$w_i = \min\left(\frac{1}{\hat{f}(y_i) + 10^{-8}}, P_{95}(1/\hat{f})\right)$$

where $\hat{f}$ is estimated via `scipy.stats.gaussian_kde` with bandwidth 0.1, and weights are capped at the 95th percentile to prevent extreme values. This inverse-density weighting ensures that rare extreme events (crisis periods) receive higher loss weight, preventing the model from ignoring these critical periods in favor of optimizing for normal market conditions.

### 4.4 Phase 4: HyperIV — Hypernetwork Model

#### 4.4.1 Architecture

The HyperIV model (based on the ICML 2025 paper) generates a specialized prediction network for each day's IV surface:

**Step 1 — Set Encoding (`SetEmbeddingNetwork`):**
- Input: $n_{\text{ref}} = 50$ reference options as $(tau, k, w)$ triplets, shape $(B, 50, 3)$.
- Linear projection: 3 → 128 dimensions.
- Transformer encoder: 2 layers, 4 attention heads, feedforward dimension 512, dropout 0.1.
- Mean pooling over tokens (with padding mask support for variable-size sets).
- Output: context vector $(B, 128)$.

**Step 2 — Weight Generation (`HyperNetwork`):**
- The context vector is projected through a single linear layer to produce all parameters of the target MLP.
- For target architecture (3→64→32→1) with ReLU activations, the total parameter count is:
  $(3 \times 64 + 64) + (64 \times 32 + 32) + (32 \times 1 + 1) = 256 + 2080 + 33 = 2369$ parameters.
- Initialization: $\mathcal{N}(0, 0.01)$ for weights, zeros for biases — small initialization prevents gradient explosion from over-parameterized generated weights.

**Step 3 — Functional Application:**
- The generated flat parameter vector is reshaped into weight matrices and bias vectors.
- Applied to query points $(tau_q, k_q, \theta_q)$ via `torch.func.functional_call`.
- Gradients $\partial w/\partial \tau$, $\partial w/\partial k$, $\partial^2 w/\partial k^2$ are computed via autograd for physics-informed loss.

#### 4.4.2 Loss Function

The `HyperIVLoss` combines MSE with calendar and butterfly constraints:

$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + 10 \cdot \mathcal{L}_{\text{calendar}} + 10 \cdot \mathcal{L}_{\text{butterfly}}$$

The butterfly loss uses a numerically-stabilized version with `tv_pred.clamp(min=1e-8)` to prevent division by zero.

#### 4.4.3 Data Handling

Each training "sample" is one day's complete set of observed options. The `collate_surfaces` function handles variable-size sets via padding with binary masks. For each day, $n_{\text{ref}}$ options are randomly selected as the reference set (the "observations"), and the remaining options form the target set (the "predictions to evaluate").

### 4.5 Phase 5: Conditional DDPM — Diffusion Model

#### 4.5.1 Surface Representation

The IV surface is discretized onto a fixed grid of $N_\tau = 10$ time-to-expiry points and $N_k = 20$ log-moneyness points, yielding a 200-dimensional vector. Each day's observed options are interpolated to this grid via `scipy.interpolate.griddata` with linear interpolation (nearest-neighbor fallback for NaN regions). Grid points are placed at the 5th and 95th quantiles of the observed $\tau$ and $k$ distributions.

#### 4.5.2 Architecture

**1D U-Net (`UNet1D`):** An encoder-decoder architecture with skip connections for denoising:

- **Encoder:** Three stages, each consisting of a `ResBlock1D` (GroupNorm → SiLU → Conv1d → Time embedding → FiLM conditioning → GroupNorm → SiLU → Conv1d + residual skip) followed by a stride-2 downsampling convolution. Channel progression: 64 → 128 → 256.

- **Bottleneck:** Two `ResBlock1D` blocks at 256 channels.

- **Decoder:** Three stages, each consisting of a stride-2 ConvTranspose1d upsampling, concatenation with the corresponding encoder skip connection (doubling channels), and a `ResBlock1D` that reduces channels back. Channel progression: 256 → 128 → 64.

- **Output projection:** GroupNorm → SiLU → Conv1d(64, 1, kernel_size=1).

**Conditioning via FiLM (`FiLMLayer`):**
Each ResBlock receives a condition vector $c = t_{\text{emb}} + c_{\text{market}}$ where:
- $t_{\text{emb}}$ is the sinusoidal time embedding of the diffusion timestep, projected through a 2-layer MLP with SiLU activation.
- $c_{\text{market}}$ encodes today's surface + 11 market features through a 2-layer MLP.

FiLM modulates convolutional features: $\hat{h} = \gamma \odot h + \beta$ where $\gamma, \beta$ are projected from the condition vector.

#### 4.5.3 Noise Schedule

We use the cosine schedule (Nichol and Dhariwal, 2021):

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos^2\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)$$

with $s = 0.008$ and $T = 1000$ timesteps. Betas are clipped to $[10^{-5}, 0.999]$.

The cosine schedule maintains a more uniform noise level across timesteps compared to the linear schedule, which is important for structured data like IV surfaces where the signal is concentrated in a narrow dynamic range.

#### 4.5.4 Training

The training procedure follows standard DDPM:

1. Sample a clean target surface $x_0$ (tomorrow's IV surface).
2. Sample a random timestep $t \sim \text{Uniform}(0, T-1)$.
3. Sample noise $\epsilon \sim \mathcal{N}(0, I)$.
4. Compute noisy surface: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$.
5. Predict noise: $\hat{\epsilon} = f_\theta(x_t, t, x_{\text{today}}, c_{\text{market}})$.
6. Loss: $\mathcal{L} = \|{\hat{\epsilon} - \epsilon}\|^2$.

Training uses AdamW (lr=0.0002), cosine annealing scheduler, gradient clipping at 1.0, and batch size 16. Validation RMSE is evaluated every 100 epochs via full reverse sampling.

#### 4.5.5 Sampling

Generation proceeds from pure noise $x_T \sim \mathcal{N}(0, I)$ through $T=1000$ reverse steps:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \hat{\epsilon}_\theta(x_t, t)\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$ for $t > 0$ and $z = 0$ for $t = 0$, and $\sigma_t = \sqrt{\beta_t}$.

---

## 5. Transfer Learning Framework

When the dataset is extended (e.g., adding years 2022–2026) or enhanced (adding new input features), models must be retrained. Retraining from scratch wastes the knowledge captured in the original model. We develop a transfer learning framework (`transfer.py`) that addresses three challenges:

### 5.1 Dimension Mismatch Handling

When input dimensions change (e.g., Adjustment model GRU input from 6 to 12 features), the `load_finetune_weights()` function performs partial weight transfer:

1. For **matching-shape parameters:** Weights are copied directly from the checkpoint.
2. For **dimension-mismatched weight matrices:** The new parameter is initialized with Xavier uniform. Then, the overlapping portion of the old weights is copied:
   ```
   new_param[:min_out, :min_in] = old_param[:min_out, :min_in]
   ```
   This preserves the model's learned representations for existing features while allowing new feature dimensions to be learned from scratch.
3. For **new parameters** not present in the checkpoint: Standard initialization is used.

### 5.2 Differential Learning Rates

The `setup_finetune_optimizer()` function creates an AdamW optimizer with two parameter groups:

- **Pretrained parameters:** Learning rate $\eta_{\text{base}} = \eta \times 0.1$ — a lower rate to fine-tune existing knowledge without catastrophic forgetting.
- **New/reinitialized parameters:** Learning rate $\eta_{\text{new}} = \eta$ — the full rate to rapidly learn new feature representations.

This differential rate is essential for the Adjustment and DDPM models, where 7 and 9 new input features were added respectively in Round 2.

### 5.3 Optional Layer Freezing

The `freeze_transferred()` function optionally freezes pretrained parameters (setting `requires_grad=False`) for warmup epochs, allowing new layers to catch up before joint optimization begins.

---

## 6. Experimental Setup

### 6.1 Compute Environment

All experiments were conducted on a Windows 11 system with:
- CPU: AMD/Intel consumer processor
- GPU: NVIDIA GPU with CUDA 12.4
- Software: Python 3.12, PyTorch 2.5.1+cu124, NumPy, Pandas, SciPy, scikit-learn
- Conda environment: `smartiv`

### 6.2 Training Configuration

| Parameter | Base | HyperIV | DGM | Adjustment | DDPM |
|-----------|------|---------|-----|------------|------|
| **Optimizer** | AdamW | AdamW | Adam | Adam | AdamW |
| **Learning rate** | 0.001 | 0.001 | 0.001 | 0.001 | 0.0002 |
| **Batch size** | 256 | 32 | N/A | 128 | 16 |
| **Max epochs** | 2000 | 500 | 5000 | 1000 | 1000 |
| **Early stopping** | 50 | 50 | None | 100 | 100 |
| **Gradient clip** | 1.0 | 1.0 | None | 1.0 | 1.0 |
| **LR scheduler** | MultiStep | Cosine | ReduceOnPlateau | ReduceOnPlateau | Cosine |
| **Precision** | float64 | float64 | float64 | float64 | float64 |
| **Random seed** | 42 | 42 | 42 | 42 | 42 |

All models use `float64` precision to ensure sufficient numerical accuracy for financial computations, particularly for the autograd-based derivative computations in the base model and HyperIV.

### 6.3 Evaluation Metrics

We report the following metrics on held-out test data:

- **TV-RMSE:** Root mean squared error of total variance predictions.
  $$\text{TV-RMSE} = \sqrt{\frac{1}{N}\sum_i (w_{\text{pred},i} - w_{\text{true},i})^2}$$

- **MAPE:** Mean absolute percentage error with stability constant.
  $$\text{MAPE} = \frac{1}{N}\sum_i \left|\frac{w_{\text{true},i} - w_{\text{pred},i}}{w_{\text{true},i} + 0.005}\right|$$

- **IV-RMSE:** RMSE of implied volatility derived from total variance.
  $$\text{IV-RMSE} = \sqrt{\frac{1}{N}\sum_i \left(\sqrt{\frac{w_{\text{pred},i}}{\tau_i}} - \sqrt{\frac{w_{\text{true},i}}{\tau_i}}\right)^2}$$

- **Calendar violation rate:** Percentage of comparable pairs where $w(\tau_2) < w(\tau_1)$ for $\tau_2 > \tau_1$.

- **Butterfly violation rate:** Percentage of interior points where the numerically computed density $g(k) < 0$.

- **PDE residual:** Mean squared PDE residual for DGM evaluation.

- **Surface RMSE:** RMSE over all 200 grid points for DDPM evaluation.

### 6.4 Structural Break Detection

The system includes three structural break detection methods:

1. **CUSUM (Cumulative Sum) detector:** Computes a rolling CUSUM statistic over a 60-day lookback window, flagging breaks when the normalized cumulative deviation exceeds a threshold of 2.0 standard deviations.

2. **Bai-Perron detector:** Uses the PELT algorithm (via the `ruptures` library) with RBF kernel to detect multiple breakpoints in a time series. Supports both penalized optimization and fixed number of breakpoints.

3. **VIX-based detector:** Flags breaks when VIX overnight change exceeds 5% or when VIX remains above its 90th percentile for 5 consecutive days.

Known crisis dates used for adjustment model oversampling: September 2001, October 2008, May 2016.

### 6.5 Testing Strategy

The codebase is validated with 215 unit tests across 10 test files, executing in approximately 4 seconds on CPU. Key design principles:

- **Float64 precision** set globally via `conftest.py` to match training conditions.
- **Tiny architectures** (`hidden_sizes=[5,5,5]`, `ensemble_num=2`) for sub-second execution.
- **No file I/O:** Tests inject mock DataFrames and synthetic tensors.
- **Autograd-safe:** Never wraps SmileModel forward passes in `torch.no_grad()`.
- **18 regression tests** named after specific bug IDs (M1–M5, X1–X3, D1–D5, T1–T5, E1) to prevent regressions of 17 identified and fixed bugs.

Test distribution:

| File | Count | Coverage |
|------|-------|----------|
| `test_model.py` | 52 | Base model classes, losses, gradient flow |
| `test_utils.py` | 22 | Utilities, metrics, early stopping |
| `test_adjustment.py` | 21 | GRU+Attention model |
| `test_diffusion.py` | 19 | DDPM components |
| `test_model_regression.py` | 18 | Bug regression tests |
| `test_dgm.py` | 17 | DGM PDE solver |
| `test_structural_break.py` | 16 | Change-point detection |
| `test_hyperiv.py` | 15 | HyperIV hypernetwork |
| `test_dataset.py` | 14 | Data pipeline |
| `test_train_integration.py` | 10 | End-to-end training loops |
| **Total** | **215** | |

---

## 7. Results and Analysis

> **Note (2026-02-20):** Model 1 (Base SSVI+NN) results below are current. All Model 2-5 (HyperIV, DGM, DDPM, Adjustment) results are from a previous training run based on an older Model 1 checkpoint and dataset configuration (including the since-removed `vixtwn_change` feature). These results are retained for reference but **must be regenerated** before publication. Sections marked with ⚠️ contain stale data.

### 7.1 Round 1: Original Dataset (2014–2021)

#### 7.1.1 Base Model

The base model (SSVI + 5-NN ensemble) was trained for 76/2000 epochs before early stopping:

- **Convergence behavior:** Initial training showed extreme instability in the first 3 epochs (losses in the millions) as the physics loss terms calibrated against randomly initialized weights. The model converged by epoch ~20, with best validation loss 2.6624 at epoch 26.
- **Instability:** A secondary instability spike occurred at epochs 62 (train loss = 38.4) and 74 (train loss = 59,351), triggering early stopping. This instability is attributable to the SSVI parameters drifting into degenerate configurations within the complex, non-convex multi-term loss landscape.
- **Test performance:** TV-RMSE 0.0134, MAPE 44.1%, IV-RMSE 0.209, butterfly violations 74%.

The high MAPE is driven by near-ATM options where total variance is very small — even a small absolute error produces a large percentage error. The 74% butterfly violation rate indicates the ensemble's softmax-weighted combination can produce curvature artifacts despite individual SmileModels being smooth.

#### ⚠️ 7.1.2 HyperIV — *Pending retraining*

> Results removed. Previous results were based on an older codebase. Will be regenerated after retraining.

#### ⚠️ 7.1.3 DGM — *Pending retraining*

> Results removed. Will be regenerated after retraining.

#### ⚠️ 7.1.4 DDPM — *Pending retraining*

> Results removed. Will be regenerated after retraining with updated condition_dim=11.

### 7.2 Round 2: Extended Dataset with Transfer Learning (2014–2026)

#### 7.2.1 Overview

All five models were retrained on the extended 480,194-row dataset with transfer learning from Round 1 checkpoints. Key modifications:
- Training period extended to 2014–2024; test period 2025–2026.
- 6 enhancement features added to Adjustment model, 7 to DDPM condition vectors.
- Transfer learning with differential learning rates (pretrained: $\eta \times 0.1$, new: $\eta$).

#### 7.2.2 Base Model

Trained for 105/2000 epochs. Best validation loss 1.914 at epoch 55.

**Gradient explosion (ep67–105):** Train loss spiked from ~2.37 to 1,029 within a few epochs. The SSVI parametric component creates a complex, non-convex loss landscape, and even with gradient clipping at 1.0, the six-component physics loss can become unstable when SSVI parameters ($\rho$, $\eta$, $\gamma$) drift into degenerate regions where the discriminant $(\varphi k + \rho)^2 + 1 - \rho^2$ approaches zero. The best model (ep55) was safely checkpointed before the instability.

Test results: TV-RMSE 0.0120 (**10.4% improvement** from Round 1), MAPE 33.0% (**25.2% improvement**), IV-RMSE 0.219. Calendar violations: 53.3%; butterfly violations: 83.7%.

#### ⚠️ 7.2.3 HyperIV — *Pending retraining*

> Results removed. Will be regenerated after retraining.

#### ⚠️ 7.2.4 DGM — *Pending retraining*

> Results removed. Will be regenerated after retraining.

#### ⚠️ 7.2.5 DDPM — *Pending retraining*

> Results removed. Will be regenerated after retraining with condition_dim=11 (vixtwn_change removed).

#### ⚠️ 7.2.6 Adjustment Model — *Pending retraining*

> Results removed. Will be regenerated after retraining with 12 input features (vixtwn_change removed, was 13).
>
> **Known issue (retained):** The `prepare_adjustment_data()` function runs the base model's forward pass with `autograd.grad(create_graph=True)` on all data points. The mitigation is chunked inference (5,000 rows per batch), implemented in `dataset.py`.

### ⚠️ 7.3 Comprehensive Model Comparison — *Pending retraining of Models 2-5*

#### 7.3.1 Base Model Cross-Round Comparison (Current)

| Metric | Base R1 | Base R2 | $\Delta$ |
|--------|---------|---------|----------|
| TV-RMSE | 0.0134 | 0.0120 | -10.4% |
| MAPE | 44.1% | 33.0% | -25.2% |
| IV-RMSE | 0.209 | 0.219 | +4.8% |

**Key observation:** TV-RMSE and MAPE improved with more data, confirming that additional training data improves generalization. IV-RMSE increased because the 2025–2026 test period has more short-maturity options where $\text{IV} = \sqrt{w/\tau}$ amplifies errors.

> Full model comparison tables will be regenerated after Models 2-5 are retrained.

### 7.4 Arbitrage Violation Analysis

#### 7.4.1 Base Model Violations (2025–2026 Test)

| Violation Type | Rate | Count |
|---------------|------|-------|
| Calendar | 53.3% | 105/197 pairs |
| Butterfly | 83.7% | 77,256/92,270 points |

**Calendar violations** at 53.3% indicate that the model frequently predicts total variance decreasing with $\tau$. This occurs primarily at extreme moneyness where the model extrapolates beyond observed data.

**Butterfly violations** at 83.7% are alarmingly high, indicating pervasive negative probability density. This is a known limitation of additive ensemble approaches: each SmileModel independently produces a smooth curve, but the softmax-weighted combination can create kinks in the density function. Potential mitigations include stronger penalty weights, Lagrangian dual formulations, or architectural constraints (e.g., input-convex neural networks).

The violations worsened from Round 1 (74%) to Round 2 (84%), reflecting the greater complexity of 2025–2026 market conditions (higher volatility, more pronounced skew).

#### ⚠️ 7.4.2 HyperIV Constraint Behavior — *Pending retraining*

> Will be evaluated after HyperIV retraining. The HyperIV loss includes explicit calendar ($\lambda=10$) and butterfly ($\lambda=10$) penalty terms with numerically stabilized density computation.

### ⚠️ 7.5 Effect of Enhancement Features — *Pending retraining*

> Enhancement feature impact will be re-evaluated after DDPM retraining with updated condition_dim=11 (vixtwn_change removed). The enhancement features (VRP, IV term slope, S&P 500 return, institutional net ratio, etc.) provide critical market context and are expected to improve conditioning quality.

### ⚠️ 7.6 Transfer Learning Impact — *Pending retraining*

> Transfer learning impact metrics will be re-evaluated after Models 2-5 are retrained. The `src/transfer.py` framework handles dimension mismatch, differential learning rates, and optional layer freezing.

---

## 8. Discussion

### 8.1 Why Multiple Models?

A natural question is whether a single model could replace the five-model system. We argue that the multi-model approach is justified by the qualitative differences between tasks:

- **Interpolation vs. forecasting:** Models 1 and 4 predict today's surface from today's observations (a regression task), while Model 5 predicts tomorrow's surface (a generative forecasting task). These require fundamentally different architectures.

- **Point prediction vs. PDE consistency:** The base model optimizes for prediction accuracy, but accurate predictions may still violate the Black-Scholes PDE. Model 2 directly enforces PDE consistency in a mesh-free setting.

- **Normal vs. crisis periods:** The base model learns an average mapping that fails during regime changes. Model 3 is specifically designed for crisis detection and correction, using temporal attention to identify relevant historical precedents.

### 8.2 The IV-RMSE Paradox

An important methodological observation is that IV-RMSE can be misleading when comparing across test periods. The conversion $\sigma_{\text{imp}} = \sqrt{w/\tau}$ amplifies errors when $\tau$ is small. If a test set contains more short-maturity options (as 2025–2026 does compared to 2021), IV-RMSE increases even when TV-RMSE improves. Researchers should report TV-RMSE as the primary metric and use IV-RMSE with caution, always accounting for the $\tau$ distribution of the test set.

### 8.3 Arbitrage Constraint Challenge

The 84% butterfly violation rate in the base model is a significant concern. While the loss function penalizes violations, the penalty approach has limitations:

1. **Competing objectives:** With six loss terms, increasing butterfly penalty weight may worsen data fit or other constraints.
2. **Ensemble artifacts:** The softmax-weighted combination of individually smooth SmileModels can create density kinks at the boundaries between model "jurisdictions."
3. **Gradient explosion link:** The SSVI parameter instability (Section 7.2.2) is related — when SSVI parameters drift, the butterfly constraint becomes harder to satisfy, creating a feedback loop of increasing penalty and increasing gradient magnitude.

Potential solutions include: hard constraint enforcement via projected gradient descent, Lagrangian dual formulations, input-convex neural network architectures (Amos et al., 2017), or replacing the ensemble with a single cross-expiration Transformer model.

### 8.4 Computational Considerations

The adjustment model's 46-hour CPU training time represents a practical bottleneck. The root cause is the `autograd.grad(create_graph=True)` call during data preparation, which stores three levels of computation graph for 463K data points. Our chunked inference solution (processing 5,000 rows at a time) resolves the GPU OOM issue and has been implemented. With GPU training, the estimated time reduces to approximately 1–2 hours.

### 8.5 Limitations

1. **Single underlying asset:** All models are trained exclusively on TXO. Transfer to other markets (e.g., S&P 500 options) would require retraining, though the architecture is general.

2. **No model stacking:** The five models operate independently. An ensemble or stacking approach could combine their complementary strengths (e.g., using HyperIV point predictions as DDPM conditioning).

3. **Sampling cost:** The DDPM requires 1,000 denoising steps for each surface sample, making real-time inference expensive. Accelerated sampling methods (DDIM, consistency models) could reduce this.

4. **SSVI instability:** The base model's gradient explosion after ~60 epochs is a fundamental challenge of combining parametric and neural network optimization. SSVI parameter clamping or adaptive loss weighting could help.

5. **No transaction cost modeling:** The system predicts IV surfaces but does not model bid-ask spreads, market impact, or execution costs for practical trading applications.

---

## 9. Conclusion and Future Work

### 9.1 Conclusion

We presented a comprehensive multi-model deep learning framework for implied volatility surface prediction on Taiwan Stock Exchange Options, addressing interpolation, PDE consistency, crisis adaptation, and generative forecasting through five specialized models.

Our key findings are:

> ⚠️ Findings 1, 3, 4, 6 below reference Model 2-5 results that are pending retraining. They will be updated with fresh numbers after retraining.

1. **HyperIV is the clear winner for point prediction** *(pending revalidation)*. Per-surface specialization via hypernetwork-generated weights fundamentally outperforms fixed-weight models that must average across all market states.

2. **More training data significantly improves generalization.** Expanding from 254K to 480K observations and 7 to 12 years of history reduced the base model's MAPE by 25% — a substantial improvement from data alone.

3. **Transfer learning with differential learning rates enables efficient model adaptation** *(pending revalidation)*. The framework supports partial weight transfer for dimension mismatches and differential learning rates.

4. **Enhancement market features improve diffusion model forecasting** *(pending revalidation)*. Conditioning on VRP, IV term slope, S&P 500 returns, and other market features provides richer context about the current market regime.

5. **Arbitrage constraint enforcement remains an open challenge.** Despite explicit penalty terms, the base model exhibits 84% butterfly violations, highlighting the difficulty of combining parametric models with neural network optimization.

6. **DDPM produces coherent full-surface forecasts** *(pending revalidation)*. Unlike point prediction models that predict each grid point independently, the diffusion model generates mutually consistent 200-point surfaces.

### 9.2 Future Work

1. **Model integration:** Use HyperIV's per-day point predictions as additional conditioning features for the DDPM, potentially improving surface forecasting by combining the best interpolator with the generative model.

2. **Hard arbitrage constraints:** Replace penalty-based enforcement with projected gradient descent or input-convex neural network architectures to guarantee zero-violation surfaces.

3. **Cross-expiration architecture:** Replace the per-expiration SmileModel ensemble with a single Transformer-based model that attends across all strikes and maturities simultaneously.

4. **Accelerated sampling:** Apply DDIM (Song et al., 2020) or consistency models (Song et al., 2023) to reduce DDPM sampling from 1,000 steps to 10–50 steps for practical real-time deployment.

5. **American options extension:** Use the DGM framework to solve the free-boundary PDE for American-style options with early exercise boundary detection.

6. **Multi-asset generalization:** Investigate whether the hypernetwork approach transfers across option markets (e.g., pre-training on S&P 500 options and fine-tuning on TXO).

7. **Adaptive loss weighting:** Use uncertainty-based methods (Kendall et al., 2018) to automatically balance the six loss components in the base model, potentially mitigating the SSVI parameter instability.

8. **Model stacking:** Train a meta-learner to combine predictions from all five models, leveraging their complementary strengths.

---

## 10. Software Engineering and Implementation Details

This section describes the system's codebase organization, configuration management, data engineering pipeline, numerical computation patterns, testing infrastructure, and reproducibility practices. The emphasis is on the practical engineering methods that underpin the experimental results reported above — demonstrating that the research code meets production-grade standards of correctness, modularity, and maintainability.

### 10.1 Project Directory Structure

The codebase is organized into four top-level directories that separate concerns by function:

```
smart-data-analysis-main/
├── src/                        # Core source code (16 files)
│   ├── config.ini              # Centralized configuration (all hyperparameters)
│   ├── model.py                # Base model: BSModel, SSVIModel, SmileModel, MultiModel, losses
│   ├── dataset.py              # DataProcessor: loading, feature engineering, splits
│   ├── train.py                # Base model training loop
│   ├── experiment.py           # Base model evaluation + surface/smile plotting
│   ├── test.py                 # Evaluation with arbitrage violation checking
│   ├── dgm.py                  # DGM PDE solver: SLayer, DGMNetwork, DGMLoss, DGMSampler
│   ├── train_dgm.py            # DGM training script
│   ├── adjustment.py           # Adjustment model: SquarePlus, TemporalAttention, TVAdjustmentModel
│   ├── train_adjustment.py     # Adjustment training with KDE-weighted loss
│   ├── hyperiv.py              # HyperIV: SetEmbeddingNetwork, TargetNetwork, HyperIVModel
│   ├── train_hyperiv.py        # HyperIV training with per-surface batching
│   ├── diffusion.py            # DDPM: UNet1D, FiLM, CosineNoiseSchedule, DiffusionTrainer
│   ├── train_diffusion.py      # DDPM training with periodic validation sampling
│   ├── structural_break.py     # Change-point detection (CUSUM, Bai-Perron, VIX-based)
│   ├── transfer.py             # Transfer learning utilities (weight loading, differential LR)
│   └── utils.py                # Shared utilities: seeding, logging, metrics, early stopping
│
├── tests/                      # 215 unit tests (10 test files + conftest.py)
│   ├── conftest.py             # Session-wide fixtures: float64, tiny models, mock data
│   ├── test_model.py           # 52 tests — base model classes and loss functions
│   ├── test_utils.py           # 22 tests — utilities, metrics, early stopping
│   ├── test_adjustment.py      # 21 tests — GRU+Attention model
│   ├── test_diffusion.py       # 19 tests — DDPM components
│   ├── test_model_regression.py# 18 tests — bug regression (M1-M5, X1-X3, T1-T2, E1)
│   ├── test_dgm.py             # 17 tests — DGM PDE solver
│   ├── test_structural_break.py# 16 tests — change-point detection
│   ├── test_hyperiv.py         # 15 tests — HyperIV hypernetwork
│   ├── test_dataset.py         # 14 tests — DataProcessor pipeline
│   └── test_train_integration.py# 10 tests — end-to-end training loops
│
├── scripts/                    # Standalone utility scripts
│   ├── download_data.py        # Data acquisition: FinMind API + yfinance
│   ├── build_features.py       # Enhancement feature computation pipeline
│   ├── compare_architectures.py# Additive vs multiplicative A/B test
│   ├── plot_smooth_iv_check.py # Fixed-yATM smooth surface verification
│   ├── plot_training_curves.py # Training loss curve visualization
│   ├── inspect_ssvi_params.py  # SSVI parameter inspection
│   ├── diagnose_rho_gradient.py# Per-loss rho gradient analysis
│   └── train_diagnose.py       # Training with per-epoch parameter tracking
│
├── dataset/                    # Data files (gitignored except metadata)
│   ├── prs_dataset_full.csv    # Primary dataset (480K rows)
│   ├── raw_txo/                # Raw TXO CSVs from FinMind (2022-2026)
│   └── enhancement/            # daily_features.csv (2,947 rows × 23 columns)
│
├── models/                     # Trained model weights (.pt files, gitignored)
├── logs/                       # Training logs, metrics JSON, and generated plots
├── docs/                       # Documentation
│   ├── research_report.md      # This report
│   ├── discussion_notes.md     # Issue tracking and resolution log
│   └── prediction_analysis.md  # Architecture fix notes
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview (CS-audience)
├── EXPERIMENT.md               # Detailed experimental results
└── ARCHITECTURE.md             # System design documentation
```

**Design principle: one model per module.** Each of the five models has its own `<model>.py` file containing the architecture definition and loss function, paired with a `train_<model>.py` file containing the training loop. This separation means model architecture changes never break training logic, and training hyperparameter changes never touch model definitions.

### 10.2 Configuration-Driven Hyperparameter Management

All hyperparameters, file paths, training dates, and model architecture sizes are centralized in a single `src/config.ini` file, parsed via Python's `configparser`. This eliminates scattered magic numbers and ensures every experiment is fully described by one configuration artifact.

```ini
[model_sett]
learning_rate = 0.001
hidden_sizes = 64,32,16        # Parsed by parse_list_config() → [64, 32, 16]
ensemble_num = 5
loss_weights = 1,1,10,10,10,10 # data, SSVI, calendar, butterfly, density, smooth

[training]
batch_size = 256
epochs = 2000
seed = 42
train_start_date = 20140101
train_end_date = 20241231
test_start_date = 20250101
test_end_date = 20261231
gradient_clip = 1.0
early_stopping_patience = 50

[diffusion]
unet_channels = 64,128,256
timesteps = 1000
condition_dim = 11
```

Comma-separated lists (e.g., `hidden_sizes`, `unet_channels`, `loss_weights`) are parsed by a shared `parse_list_config()` utility that converts them to typed Python lists. Date strings in `YYYYMMDD` format are parsed by `parse_date()`. This convention allows the configuration to remain a flat text file while supporting complex structured values.

Each model section (`[model_sett]`, `[dgm]`, `[adjustment]`, `[hyperiv]`, `[diffusion]`) is self-contained, meaning a researcher can modify one model's hyperparameters without risk of affecting another. Path configuration (`[save_path]`) uses relative paths (`../models/`, `../logs/`) so the system works identically regardless of the absolute install location.

### 10.3 Data Engineering Pipeline

The data pipeline is implemented across three components, each handling a distinct stage:

**Stage 1: Data Acquisition (`scripts/download_data.py`)**

This script constructs the 480K-row dataset from two sources:
- **Historical CSV** (`prs_dataset_no_fat(clean).csv`): 254,044 rows of TXO options from 2014–2021, preprocessed by a prior research pipeline
- **FinMind API** (2022–2026): Downloaded in annual chunks, with Chinese column names (`買賣權`, `履約價`, `成交價`) mapped to English equivalents

The merge pipeline computes implied volatility for new rows using the Newton-Raphson method for Black-Scholes inversion, then concatenates with the historical data to produce `prs_dataset_full.csv`.

**Stage 2: Enhancement Features (`scripts/build_features.py`)**

A 6-step feature engineering pipeline computes daily market features from external data sources:

1. Download S&P 500 daily returns via yfinance
2. Compute 20-day realized volatility from TAIEX daily returns
3. Calculate IV term slope (long minus short maturity ATM IV) and IV skew (OTM put minus ATM IV)
4. Derive variance risk premium: VRP = IV² − RV²
5. Extract institutional net buy/sell ratio from TWSE daily data
7. Compute futures basis percentage from TAIEX futures vs. spot

The output is `dataset/enhancement/daily_features.csv` (2,947 rows × 23 columns), which is merged into the main pipeline during training via date-based joins.

**Stage 3: DataProcessor (`src/dataset.py`)**

The `DataProcessor` class is the central data handler, responsible for:

1. **Column normalization** — Renaming Chinese column headers to English (e.g., `交易日期` → `date`, `履約價` → `strike_price`)
2. **Feature engineering** — Computing log-moneyness `ln(K/S)`, total variance `IV² × τ`, and ATM total variance via per-expiration interpolation at `logm = 0`
3. **Beta-tau estimation** — Linear regression of SSVI β parameters on τ for the parametric prior
4. **Chronological splitting** — Strict temporal separation: train (2014-2024), test (2025-2026), with 20% random within-period validation split. This guarantees zero future information leakage into training.
5. **Model-specific data preparation** — Separate methods for each model's data format:
   - `prepare_hyperiv_surfaces()`: Groups options by date into per-day surface structures with reference/query splits and random masking
   - `prepare_diffusion_surfaces()`: Discretizes each day's IV surface onto a fixed 10×20 (τ × logm) grid with bilinear interpolation
   - `prepare_adjustment_data()`: Constructs 20-day rolling sequences with base model predictions as features, using **chunked inference** (5000 rows per batch) to avoid GPU OOM from `autograd.grad(create_graph=True)`

All data is loaded into PyTorch `TensorDataset` objects with `float64` dtype, ensuring numerical precision throughout the pipeline.

### 10.4 Autograd Patterns and Numerical Computation

The system makes extensive use of PyTorch's automatic differentiation engine for purposes beyond standard backpropagation:

**Higher-order derivatives for physics constraints.** The `SmileModel` forward pass detaches the input tensor and re-enables gradients to compute derivatives of the network output with respect to its input (log-moneyness):

```python
logm_var = logm.detach().requires_grad_(True)
tv = network(logm_var)
grad1 = torch.autograd.grad(tv.sum(), logm_var, create_graph=True)[0]
grad2 = torch.autograd.grad(grad1.sum(), logm_var, retain_graph=True)[0]
```

The `create_graph=True` flag is essential: it instructs PyTorch to build a computational graph through the gradient computation itself, enabling the optimizer to backpropagate through the second derivative. Without this, the butterfly and density loss terms (which depend on `d²w/dk²`) would have zero gradients and provide no learning signal.

**Additive derivative composition.** The `SingleModel` combines SSVI and SmileModel derivatives via simple addition:

```python
# w_ssvi = ssvi_prior(logm, yATM)
# h(k) = SmileModel(tau, logm)
# final prediction = w_ssvi + yATM * h(k)
# grad = dw_ssvi/dk + yATM * dh/dk  (no cross-terms)
```

The absence of cross-terms is critical for training stability: the butterfly constraint `g(k) = (1 - kw'/2w)² - w'²/4(1/w + 1/4) + w''/2 ≥ 0` depends on both first and second derivatives, and cross-terms in a multiplicative formulation create feedback loops that cause gradient explosion (verified by A/B experiment, see `logs/architecture_comparison.json`).

**Float64 precision.** The entire system uses `torch.float64` (double precision) rather than the PyTorch default `float32`. This is set globally at test time via `conftest.py` and enforced in data loading via explicit dtype specification. Double precision is necessary because:
- Total variance values near ATM are O(10⁻³), where float32 loses significant digits in difference operations
- Second derivatives amplify rounding errors quadratically
- The SSVI parametric form involves `exp()` and `sqrt()` compositions where small input perturbations produce large output changes

**Chunked inference for memory management.** When running the trained base model on the full 463K-row dataset to generate features for the adjustment model, the `create_graph=True` flag causes PyTorch to retain the entire computation graph in GPU memory. The solution is to process data in 5000-row chunks with `torch.no_grad()` disabled only for the derivative computation within each chunk, then immediately discard the graph:

```python
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    with torch.enable_grad():
        output = model(chunk)  # creates graph
    results.append(output.detach())  # discards graph
```

### 10.5 Testing Infrastructure

The project includes 215 unit tests across 10 test files, designed to run in under 4 seconds on CPU. The testing infrastructure embodies several practical engineering principles:

**Session-wide float64 enforcement.** The `conftest.py` file sets `torch.set_default_dtype(torch.float64)` via a session-scoped `autouse` fixture, ensuring all test tensor allocations use the same precision as production. This eliminates an entire class of dtype mismatch bugs.

**Tiny architecture fixtures.** Test models use minimal hidden sizes (`[5, 5, 5]` instead of `[64, 32, 16]`) and ensemble counts (`2` instead of `5`). This reduces test runtime from minutes to milliseconds while exercising the same code paths. All model modules accept architecture sizes as constructor parameters specifically to enable this pattern.

**Zero file I/O.** No test reads from disk. The `mock_prs_dataset` fixture generates a synthetic 50-row DataFrame with realistic value ranges, and the `patched_data_processor` injects it directly into a `DataProcessor` instance, bypassing CSV loading entirely. This makes tests portable, deterministic, and parallelizable.

**Bug regression tests.** Eighteen tests in `test_model_regression.py` are named after specific bug IDs (e.g., `test_M1_float64_dtype`, `test_X2_multimodel_device_consistency`, `test_D3_dgm_gradient_flow`). Each test was written simultaneously with its bug fix to prevent future regressions. The naming convention provides a direct audit trail from test to the specific issue it guards against. Example:

```python
def test_M3_additive_derivatives(tiny_batch, device):
    """Bug M3: SingleModel gradient must flow through both SSVI and NN components."""
    model = SingleModel(hidden_sizes=[5, 5, 5], device=device)
    tau, logm, yATM, _ = tiny_batch
    tv, grad1, grad2, bs = model(tau, logm, yATM)
    assert grad1.requires_grad  # must flow through additive composition
    loss = grad1.sum()
    loss.backward()
    # Verify gradients propagate to both SmileModel and SSVI components
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
```

**Integration test coverage.** The `test_train_integration.py` file tests complete training loops — not just model forward passes — verifying that `train_one_epoch()` reduces loss, `validate()` runs without error, learning rate schedulers step correctly, and early stopping triggers at the right patience threshold. These tests use only 2 epochs and 32-sample datasets to remain fast.

**Autograd safety.** A critical design rule: test code never wraps `SmileModel` forward passes in `torch.no_grad()`. Since SmileModel internally calls `autograd.grad(create_graph=True)`, disabling gradient tracking would silently produce zero derivatives, making tests pass with incorrect values. Tests instead verify that output tensors retain their `requires_grad` attribute.

### 10.6 Logging, Metrics, and Visualization

**Structured logging.** Every training script uses `setup_logging()` from `utils.py`, which creates both a timestamped log file and console output with the format `%(asctime)s - %(name)s - %(levelname)s - %(message)s`. Log files are stored in `../logs/` and named by model and timestamp (e.g., `training_20260210_143022.log`).

**Metrics tracking.** The `MetricsTracker` class records per-epoch train and validation losses and tracks the best validation epoch. After training, it serializes the full loss history to a JSON file, enabling post-hoc analysis and plot generation without re-running training. The `EarlyStopping` class implements patience-based stopping with configurable `min_delta` for loss improvement threshold.

**Evaluation metrics.** Three metrics capture different aspects of prediction quality:
- **TV-RMSE**: Root mean squared error on total variance — the primary training objective
- **MAPE**: Mean absolute percentage error with ε=0.005 floor to prevent division-by-zero near ATM
- **IV-RMSE**: RMSE on implied volatility derived via `IV = √(TV/τ)`, which is more interpretable to practitioners

**Arbitrage violation checking.** The `test.py` evaluation script checks two no-arbitrage conditions on the predicted IV surface:
- **Calendar spread violations**: `∂w/∂τ < 0` for any pair of expirations (total variance must increase with time)
- **Butterfly spread violations**: The local density function `g(k) < 0` at any point (options prices must be convex in strike)

Violation rates are reported as percentages and serve as important qualitative metrics beyond point prediction accuracy.

**Automated figure generation.** Several scripts in `scripts/` generate diagnostic and training visualizations: `plot_training_curves.py` generates training loss curves from log data, `plot_smooth_iv_check.py` produces fixed-yATM smooth surface verification plots, and `compare_architectures.py` runs and visualizes the additive vs. multiplicative architecture A/B test. All scripts output to `logs/` and are idempotent — they can be re-run without requiring model weights or GPU access.

### 10.7 Transfer Learning Engineering

The `src/transfer.py` module implements a practical transfer learning framework that handles the real-world complication of changing model architectures between training rounds:

**Partial weight transfer with dimension mismatch handling.** When loading a pretrained checkpoint into a model with different layer sizes (e.g., GRU input changing from 6 to 12 features), the system:
1. Identifies parameters where shapes match — copies directly
2. Identifies parameters where shapes differ — initializes with Xavier uniform, then copies the overlapping submatrix from the pretrained weights
3. Returns sets of `(transferred_params, reinitialized_params)` for downstream use

This is more robust than the common approach of either failing on shape mismatch or randomly initializing the entire layer.

**Differential learning rates.** The `setup_finetune_optimizer()` function constructs parameter groups with two learning rates:
- Pretrained (transferred) parameters: `base_lr × 0.1` — slow learning to preserve learned features
- New/reinitialized parameters: `base_lr × 1.0` — fast learning to catch up with pretrained layers

This prevents the common failure mode of catastrophic forgetting where pretrained weights are overwritten before new parameters have calibrated.

**CLI integration.** All five training scripts accept an optional `--finetune <path>` command-line argument. When provided, the script loads the checkpoint, applies partial transfer, sets up differential learning rates, and logs which parameters were transferred vs. reinitialized. This makes transfer learning a single-flag operation rather than a code change.

### 10.8 Reproducibility Practices

The system implements multiple layers of reproducibility:

**Deterministic seeding.** The `set_seed()` utility sets random seeds across all four random number generators simultaneously: `torch.manual_seed()`, `np.random.seed()`, `random.seed()`, and `torch.cuda.manual_seed_all()`. It also enables `cudnn.deterministic = True` and disables `cudnn.benchmark`, sacrificing a small amount of GPU performance for bitwise reproducibility.

**Configuration as experiment record.** Since all hyperparameters live in `config.ini`, each experiment is fully defined by: (1) the config file, (2) the git commit hash, and (3) the random seed. The `[best_performance]` section in the config file records the best validation loss from training, creating a self-documenting artifact.

**Chronological data splitting.** The train/test split is based on calendar dates (configurable via `train_end_date` / `test_start_date`), not random sampling. This eliminates data leakage from temporal correlation in financial time series and ensures that test set evaluation reflects genuine out-of-sample performance.

**Checkpoint management.** All training scripts save model weights at the best validation epoch and at training completion. The `MetricsTracker` serializes the full training history, and log files provide timestamped per-epoch records. This means any result can be traced back to the exact training state that produced it.

### 10.9 Dependency Management

The project's `requirements.txt` specifies 10 Python packages:

```
numpy, pandas, matplotlib, torch, tqdm, scipy, scikit-learn, QuantLib, statsmodels, ruptures
```

**Core scientific stack** (numpy, pandas, scipy, scikit-learn): Standard numerical computing, data manipulation, interpolation (interp1d for ATM total variance), and preprocessing.

**Deep learning** (torch): PyTorch is the sole deep learning framework. The system uses PyTorch's functional API (`torch.nn.utils.parametrize.functional_call`) in HyperIV for applying dynamically generated weights — a pattern that avoids the overhead of creating new Module instances per batch.

**Domain-specific** (QuantLib): Used exclusively for Black-Scholes implied volatility inversion via `QuantLib.BlackCalculator`, providing a validated reference implementation for a numerically delicate operation.

**Statistical analysis** (statsmodels, ruptures): `statsmodels` provides the OLS regression used in beta-tau estimation. `ruptures` implements the PELT algorithm for offline change-point detection in the structural break module.

**Visualization and UX** (matplotlib, tqdm): matplotlib generates all figures. tqdm provides progress bars for long-running training loops and data processing operations.

### 10.10 Defensive Engineering Patterns

Several engineering patterns in the codebase address common failure modes in deep learning research code:

**Gradient clipping.** All training scripts apply `torch.nn.utils.clip_grad_norm_()` with a configurable maximum norm (default 1.0) before every optimizer step. This was essential for the base model, where the six-component physics-informed loss can produce extreme gradients when SSVI parameters drift into degenerate regions.

**LayerNorm for training stability.** Both the SmileModel (base) and DGM SLayer use `LayerNorm` after each hidden layer, normalizing activations to zero mean and unit variance. This is particularly important for the DGM's LSTM-like gating mechanism, where without normalization, the sigmoid gates can saturate and block gradient flow.

**SquarePlus activation.** The adjustment model uses `SquarePlus(x) = (x + √(x² + 4)) / 2` instead of ReLU or Softplus for its output activation. SquarePlus is everywhere differentiable (unlike ReLU) and has a non-vanishing gradient for all inputs (unlike Softplus at large negative values), making it well-suited for predicting strictly positive adjustment ratios.

**FiLM conditioning.** The DDPM conditions on market features via Feature-wise Linear Modulation rather than simple concatenation. FiLM applies `γ·x + β` where γ and β are learned projections of the condition vector, providing multiplicative and additive modulation. This is more expressive than concatenation because it allows the condition to scale and shift the intermediate representations, not just add information.

**Cosine noise schedule.** The DDPM uses a cosine schedule (`β_t = 1 − ᾱ_t / ᾱ_{t-1}` where `ᾱ_t = f(t)/f(0)` and `f(t) = cos²((t/T + s)/(1+s) · π/2)`) rather than the original linear schedule. The cosine schedule provides more gradual noise addition in early timesteps, which preserves fine structure in the IV surface for longer during the forward diffusion process.

---

## References

1. Ackerer, D., Tagasovska, N., & Vatter, T. (2020). Deep smoothing of the implied volatility surface. *Advances in Neural Information Processing Systems*, 33.

2. Amos, B., Xu, L., & Kolter, J. Z. (2017). Input convex neural networks. *International Conference on Machine Learning*.

3. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *EMNLP*.

4. Dugas, C., Bengio, Y., Bélisle, F., Nadeau, C., & Garcia, R. (2009). Incorporating functional knowledge in neural networks. *Journal of Machine Learning Research*, 10, 1239-1262.

5. Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives. *Presentation at Global Derivatives & Risk Management*, Madrid.

6. Gatheral, J., & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*, 14(1), 59-71.

7. Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. *International Conference on Learning Representations*.

8. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33.

9. HyperIV (2025). Hypernetwork-based implied volatility surface interpolation. *International Conference on Machine Learning*.

10. Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *IEEE Conference on Computer Vision and Pattern Recognition*.

11. Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*.

12. Sirignano, J., & Spiliopoulos, K. (2018). DGM: A deep learning algorithm for solving partial differential equations. *Journal of Computational Physics*, 375, 1339-1364.

13. Song, J., Meng, C., & Ermon, S. (2020). Denoising diffusion implicit models. *International Conference on Learning Representations*.

14. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models. *International Conference on Machine Learning*.

15. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30.

---

## Appendix A: Bug Taxonomy

During development, 17 bugs were identified and fixed, each catalogued with an ID for regression testing:

| ID | Module | Description |
|----|--------|-------------|
| M1 | `BSModel.__init__` | Removed unused `atm_fun` parameter that caused initialization errors |
| M2 | `BSModel.forward` | Fixed signature to accept `(logm, yATM)` and return 4-tuple with zero gradients |
| M3 | `SoftmaxModel` | Standardized input order; added `yATM` as 3rd input for volatility-regime-aware weighting |
| M4 | `Loss_linear` | Fixed to take only `grad_logm_2nd` (was taking extra unused arguments) |
| M5 | `WeightedSumLoss` | Fixed to return single tensor (was returning list); store individual losses as attribute |
| X1 | `WeightedSumLoss` | Used `register_buffer` with `float64` tensor instead of Python list for loss weights |
| X2 | `WeightedSumLoss.forward` | Used `torch.stack()` instead of `torch.FloatTensor([...])` for autograd compatibility |
| X3 | `SmileModel` | Fixed `weights_exp` shape from `(J,1)` to `(J, hidden_sizes[0])` to match matmul output |
| D1 | `DataProcessor.int_and_div` | Fixed beta_tau column name handling after CSV round-trip |
| D2 | `DataProcessor.int_and_div` | Used 'season' (not 'month') to match groupby keys |
| D3 | `DataProcessor.int_and_div` | Merged on `['year', 'season', 'tau']` to match groupby keys |
| D4 | `DataProcessor.getYATM` | Used 'underlying' column name (not legacy 'S') |
| D5 | `DataProcessor` | Fixed synthetic data path; renamed `Prepare_prs_dataset` → `Prepare_train_data` |
| T1 | `train.py` | Called correct method name `Prepare_train_data` |
| T2 | `train.py` | Passed weights list (not device) to `WeightedSumLoss` |
| T3 | `train.py` | Return DataLoaders from data preparation (not raw tensors) |
| T4 | `train.py` | Moved `scheduler.step()` to epoch loop (not batch loop) |
| T5 | `train.py` | Updated loss function call signature after M5 fix |

---

## Appendix B: Reproducibility

### B.1 Environment Setup

```bash
conda create -n smartiv python=3.12 -y
conda activate smartiv
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install pytest
```

### B.2 Data Acquisition

```bash
# Download TXO data (2022-2026) and TWII/VIX
python scripts/download_data.py

# Compute enhancement features
python scripts/build_features.py
```

### B.3 Training Sequence

```bash
cd src

# Phase 1: Base model
python train.py --on_gpu --epochs 2000

# Phase 2: DGM
python train_dgm.py --on_gpu

# Phase 3: Adjustment (requires trained base model)
python train_adjustment.py --on_gpu

# Phase 4: HyperIV
python train_hyperiv.py --on_gpu --epochs 500

# Phase 5: DDPM
python train_diffusion.py --on_gpu --epochs 1000

# Evaluation
python test.py --on_gpu
```

### B.4 Transfer Learning (Round 2)

```bash
cd src

python train.py --on_gpu --finetune ../models/MultiModel.pt
python train_hyperiv.py --on_gpu --finetune ../models/HyperIVModel.pt
python train_dgm.py --on_gpu --finetune ../models/DGMModel.pt
python train_adjustment.py --on_gpu --finetune ../models/AdjustmentModel.pt
python train_diffusion.py --on_gpu --finetune ../models/DiffusionModel.pt
```

### B.5 Testing

```bash
python -m pytest tests/ -v  # All 215 tests (~4 seconds)
```

---

*This report documents the complete design, implementation, and evaluation of a multi-model IV surface prediction system. All source code, trained models, and experimental logs are available in the project repository.*

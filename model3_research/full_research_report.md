# Advanced Time Series Models for IV Surface Adjustment — Full Research Report

> Research date: 2026-02-20
> Context: Replace GRU in Model 3 (TVAdjustmentModel) for TXO IV Surface Prediction system

---

## 1. Problem Specification

- **Task**: Predict adjustment ratio alpha for base model (SSVI+NN) during structural breaks
- **Input**: 20-day sliding window x 15 features per timestep
- **Features**: 
  - 6 base features: `vix_change`, `underlying_return`, `logm`, `tau`, `tv_pred`, `itm_otm`
  - 6 enhancement features: `sp500_return`, `iv_term_slope`, `iv_skew`, `vrp_20d`, `futures_basis_pct`, `rv_20d`
  - 3 Model 2 Greeks (V3): `vanna_proxy`, `volga_proxy`, `lv_gradient_K`
- **Target**: tv_ratio = tv_true / tv_pred (multiplicative correction)
- **Application**: tv_adjusted = tv_base * alpha
- **Key challenge**: Crisis periods are rare but critical (2001/09, 2008/10, 2016/05)

---

## 2. Detailed Model Evaluations

### 2.1 Temporal Fusion Transformer (TFT)

**Paper**: Lim et al. (2021), Google Research, arXiv:1912.09363
**Published**: International Journal of Forecasting, Vol 37, Issue 4

**Architecture**:
```
Input Features
    |
    v
Variable Selection Network (VSN)
    |  - Softmax gating per timestep
    |  - Learns feature importance dynamically
    v
Gated Residual Networks (GRN)
    |  - ELU activation + skip connections
    |  - Gate suppresses irrelevant paths
    v
LSTM Encoder (past) / LSTM Decoder (future)
    |
    v
Static Enrichment Layer
    |  - Enriches temporal features with static context
    v
Interpretable Multi-Head Attention
    |  - Shared V weights across heads (interpretable)
    |  - Attention weights = temporal importance
    v
Position-wise Feed-Forward
    |
    v
Quantile Output (10th, 50th, 90th percentiles)
```

**Key Innovation — Variable Selection Network**:
- Each timestep has a softmax gate over all input features
- Model learns: "in crisis, attend to VIX and returns; in calm, attend to tau and logm"
- This IS regime switching behavior, learned end-to-end

**Benchmark Results**:
- Outperformed DeepAR by 36-69%
- P50 loss 7% lower, P90 loss 9% lower than next best model
- Attention automatically identifies 2008 crisis periods with high weights

**Interpretability Features**:
1. Feature importance ranking (global and per-instance)
2. Temporal attention patterns (which past days matter)
3. Regime detection via attention pattern deviation from average

**Implementation Options**:
- `pytorch-forecasting` library (full TFT with data pipeline)
- Custom implementation (more control, our approach)

**Training Cost**: Hours on single GPU for our dataset size

---

### 2.2 xLSTM (Extended LSTM)

**Paper**: Beck et al. (2024), NXAI Lab (Sepp Hochreiter's group)
**Variants**: sLSTM (scalar) and mLSTM (matrix)

**sLSTM Architecture**:
- Exponential gating: exp(input_gate), exp(forget_gate)
- Normalizer state prevents numerical overflow
- Better gradient flow than sigmoid-based gates
- Still sequential (like GRU/LSTM)

**mLSTM Architecture**:
- Matrix memory C (not vector) — stores richer history
- Query-Key-Value retrieval (like Transformer attention)
- Covariance update rule: C_t = f_t * C_{t-1} + v_t * k_t^T
- Parallelizable (unlike sLSTM)

**Benchmark Results (xLSTMTime)**:
- Matches or exceeds Transformer-based and linear methods on LTSF benchmarks
- xLSTM-TS outperformed TCN, N-BEATS, TFT, N-HiTS, TiDE on stock price direction
- Particularly strong on datasets where temporal order matters

**Installation**:
```bash
pip install xlstm          # Official (NX-AI)
pip install torchxlstm     # Pure PyTorch alternative
```

**Integration Approach**:
- Replace `nn.GRU` with mLSTM block
- Keep existing TemporalAttention + FC + SquarePlus
- Minimal code change, maximal compatibility

---

### 2.3 Mamba / State Space Models

**Paper**: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

**Core Mechanism**:
- Selective state space: dynamically decide what to remember/forget
- Linear complexity O(n) vs Transformer's O(n^2)
- Hardware-aware parallel scan algorithm

**Why NOT suitable for our problem**:
1. **seq_len=20 is too short**: Mamba's advantage is processing seq_len > 1000 efficiently
2. **Fixed-size latent state**: compresses history into fixed vector, potentially losing crisis signals
3. **Research finding**: "Mamba excels at long-range dependencies, Transformer at short-term dynamics" — our problem is the latter
4. **S-Mamba paper**: showed advantages mainly on long sequences (96-720 steps)

**Hybrid option (SST / MoU)**:
- Multi-scale hybrid with Mamba for long-range + Transformer for short-range
- Overkill for seq_len=20; added complexity not justified

**References**:
- "Is Mamba Effective for Time Series Forecasting?" (Neurocomputing 2024)
- MambaTS (arXiv:2405.16440)
- SST: Multi-Scale Hybrid Mamba-Transformer (CIKM 2024)

---

### 2.4 Neural SDE / Neural ODE

**Libraries**: torchsde (Google Research), torchdiffeq, torchcde

**Theoretical Appeal**:
- Volatility IS a stochastic process — SDE is the natural formulation
- Neural SDE: parameterize drift f(x,t) and diffusion g(x,t) as neural networks
- Continuous-time modeling, can handle irregular timestamps
- Natural uncertainty quantification from diffusion term

**Practical Issues**:
1. **Training instability**: SDE solvers have high variance gradients
2. **Speed**: Adjoint method backprop 5-10x slower than standard
3. **Discrete jumps**: SDE is continuous — crisis "jumps" need jump-diffusion extensions
4. **Federal Reserve finding (2025)**: Nonlinear regime-switching models (THAR, STHAR) consistently outperform ML models for volatility forecasting across GFC and COVID

**Verdict**: Academically interesting but engineering cost too high for marginal benefit

---

### 2.5 Transformer Variants (PatchTST, iTransformer, Crossformer)

**PatchTST** (ICLR 2023):
- Segments time series into patches (typically 16-32 steps per patch)
- seq_len=20 → only 1-2 patches → attention has nothing to work with
- Designed for seq_len >= 96

**iTransformer** (2024):
- Inverts axis: treats each variable as a token
- 6 variables = 6 tokens → workable but designed for >20 variables
- Cross-variate dependencies are not the bottleneck in our problem

**Crossformer** (2023):
- Cross-scale attention for variable-length series
- Overkill for fixed-length seq_len=20

**TimeMixer** (ICLR 2024):
- Multiscale decomposition + MLP mixing
- Needs longer sequences for meaningful decomposition

---

### 2.6 Foundation Models (TimesFM, Chronos, Moirai)

**Not applicable because**:
- Designed for zero-shot general forecasting
- Our target (tv_ratio) is highly domain-specific
- Cannot encode the Model 1 dependency (tv_pred as input feature)
- Fine-tuning cost exceeds training a task-specific model from scratch

---

## 3. Summary Comparison

| Model | Short seq (20) | Regime detection | Interpretability | Implementation | GPU hours |
|-------|---------------|------------------|-----------------|----------------|-----------|
| GRU+Attn (current) | Good | Poor | Low | Existing | ~2h GPU |
| **TFT** | **Excellent** | **Excellent** | **Excellent** | Medium | ~3-4h GPU |
| **xLSTM (mLSTM)** | **Excellent** | Good | Low | **Minimal** | ~2h GPU |
| Mamba | Mediocre | Mediocre | Low | Medium | ~1h GPU |
| Neural SDE | Good | Poor (continuous) | Medium | Hard | ~12h GPU |
| PatchTST | Poor (too short) | N/A | Low | Easy | ~1h GPU |
| Foundation Model | N/A | N/A | N/A | N/A | N/A |

## 4. Decision

**Implement both TFT and xLSTM**, train on identical data, compare metrics.
- TFT: potential for best performance + interpretability
- xLSTM: minimal risk, guaranteed improvement over GRU

> **Update (2026-02-22)**: Both architectures successfully implemented and compared. While both outperformed the GRU baseline (xLSTM by 4.3%, TFT by 1.7%), they exhibited significant overfitting (train-val gap 2.8x – 6.9x). Parameter-level regularization experiments (AdamW, Cautious Weight Decay, Constrained Parameter Regularization) were conducted to establish a higher validation floor. Additionally, switching to `float32` training yielded a 3x speedup for TFT due to its compute-bound nature, whereas xLSTM showed no improvement as it is memory-bound.

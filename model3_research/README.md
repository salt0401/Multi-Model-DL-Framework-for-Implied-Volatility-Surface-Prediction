# Model 3 Architecture Research

## Goal
Replace GRU in TVAdjustmentModel with more advanced architectures.
Two candidates selected for implementation:
1. **Temporal Fusion Transformer (TFT)** — full replacement
2. **xLSTM (mLSTM)** — drop-in GRU replacement

## Current Model 3 Baseline
- GRU (2-layer, hidden=64) + Multi-Head Attention (4 heads) + FC + SquarePlus
- Input: 20-day sliding window x 16 features (6 base + 6 enhancement + 4 Greeks)
- Features: [vix_change, underlying_return, logm, tau, tv_pred, itm_otm] + [sp500_return, iv_term_slope, iv_skew, vrp_20d, futures_basis_pct, rv_20d] + [local_vol, vanna, volga, lv_gradient_K]
- Target: tv_ratio = tv_true / tv_pred (adjustment factor alpha)

## Candidate 1: TFT (Temporal Fusion Transformer)

### Why TFT
- Variable Selection Network: learns which features matter at each timestep
- Interpretable Multi-Head Attention: attention weights show which past days matter
- Gated Residual Networks: suppress irrelevant inputs
- Crisis-adaptive: attention pattern shifts during high-volatility periods
- Google Research paper: P50 loss 7% lower, P90 loss 9% lower than next best

### Architecture Components
1. Variable Selection Network (VSN) — per-timestep feature gating
2. Gated Residual Networks (GRN) — nonlinear skip connections
3. Static Enrichment — incorporate static covariates
4. Temporal Self-Attention — interpretable multi-head attention
5. Position-wise Feed-Forward — final prediction

### Implementation
- Library: `pytorch-forecasting` or custom implementation
- Training cost: hours on single GPU (small dataset)

### Key References
- Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (arXiv:1912.09363)
- Google Research Blog: Interpretable Deep Learning for Time Series Forecasting

---

## Candidate 2: xLSTM (Extended LSTM)

### Why xLSTM
- Minimal code change: drop-in replacement for nn.GRU
- mLSTM: matrix memory + Query-Key-Value mechanism
- sLSTM: exponential gating for better gradient flow
- Outperformed TCN, N-BEATS, TFT, N-HiTS, TiDE on stock direction prediction
- By Sepp Hochreiter (original LSTM inventor), 2024

### Architecture Variants
- **sLSTM**: scalar memory with exponential gating
- **mLSTM**: matrix memory with QKV-style retrieval (recommended)

### Implementation
- `pip install xlstm` (official) or `pip install torchxlstm`
- Replace nn.GRU with mLSTM, keep Attention + FC + SquarePlus

### Key References
- Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory"
- xLSTMTime (arXiv:2407.10240) — time series forecasting benchmark

---

## Rejected Candidates

### Mamba / State Space Models
- Core advantage is long-sequence efficiency (linear O(n))
- seq_len=20 too short to benefit; fixed-size latent state may lose info
- Better for seq_len > 100

### PatchTST / iTransformer / TimeMixer
- Designed for long sequences; PatchTST patches seq_len=20 into 2-4 tokens (meaningless)
- iTransformer needs high-dimensional features (>20 vars)
- TimeMixer multiscale decomposition needs longer sequences

### Neural SDE (torchsde)
- Theoretically elegant for volatility modeling
- Training unstable, adjoint backprop 5-10x slower
- Continuous SDE can't capture discrete jumps well
- Fed Reserve 2025 study: regime-switching models beat ML models in crisis prediction

### Foundation Models (TimesFM, Chronos, Moirai)
- Designed for zero-shot general forecasting
- tv_ratio is too domain-specific; fine-tuning cost > benefit

---

## Current Training Configuration

### Data Pipeline
- **Input**: 20-day sliding window × 16 features (6 base + 6 enhancement + 4 Greeks)
- **Target**: `tv_ratio = tv_true / tv_pred`
- **Split** (aligned with Model 1): Train < 2019-08-13, Val 2019-08-13 ~ 2020-12-31, Test 2021 (held-out)
- **GPU**: NVIDIA RTX 4060 Laptop (8.6GB)

### Models Under Evaluation

| Model | Optimizer | Dtype | Rationale |
|-------|-----------|-------|-----------|
| TFT + CPR | Constrained Parameter Regularization | float32 | Strong interpretability + NeurIPS 2024 regularizer |
| TFT + AdamW | Standard weight decay | float32 | Baseline comparison |
| GRU + CWD | Cautious Weight Decay (ICLR 2026) | float64 | Lightweight baseline |

### float32 vs float64 Hardware Observation

RTX 4060 Laptop: FP32 ~15.11 TFLOPS, FP64 ~0.236 TFLOPS (1/64 ratio).
- **TFT** is compute-bound → ~3x speedup with float32
- **xLSTM** is memory-bound → no benefit from float32
- **GRU** remains on float64 for compatibility

*Training results will be populated after the current training run completes.*

---

### TODO
- [x] Train GRU baseline for 3-way comparison
- [x] Evaluate regularization (AdamW, CWD, CPR)
- [x] Test float32 for speed
- [x] Integrate Model 2 Greeks (16-dim input)
- [x] Align data split with Model 1 (2021 test set)
- [ ] Crisis-period analysis (2020/03 COVID subset)
- [ ] Populate results from 16-dim training run

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

### 12-Way Comparison Results (3 Architectures × 4 Optimizers)

All 12 combinations trained on 16-dim input (6 base + 6 enhancement + 4 Greeks), evaluated on 2021 held-out test set.

#### TFT (Temporal Fusion Transformer) — float32, 318K params

| Optimizer | Val Loss | Val RMSE | Val MAPE | Test RMSE | Test MAPE | Best Epoch | Time (min) |
|-----------|:--------:|:--------:|:--------:|:---------:|:---------:|:----------:|:----------:|
| **CPR** ⭐ | **0.1481** | **0.1494** | **8.92%** | **0.1558** | **9.51%** | 114 | 199.0 |
| AdamW | 0.1545 | 0.1543 | 9.14% | 0.1590 | 9.75% | 51 | 63.4 |
| CWD | 0.1616 | 0.1590 | 9.53% | 0.1608 | 9.75% | 21 | 122.1 |
| Adam (no reg) | 0.1721 | 0.1643 | 10.06% | 0.1592 | 9.85% | 11 | 51.2 |

#### GRU (Baseline) — float64, 59K params

| Optimizer | Val Loss | Val RMSE | Val MAPE | Test RMSE | Test MAPE | Best Epoch | Time (min) |
|-----------|:--------:|:--------:|:--------:|:---------:|:---------:|:----------:|:----------:|
| AdamW | 0.1645 | 0.1623 | 9.51% | 0.1628 | 9.70% | 81 | 41.5 |
| CWD | 0.1669 | 0.1648 | 9.57% | 0.1658 | 9.91% | 81 | 45.0 |
| Adam (no reg) | 0.1765 | 0.1699 | 9.96% | 0.1652 | 9.87% | 29 | 29.5 |
| CPR | 0.2128 | 0.1878 | 12.02% | 0.1765 | 11.28% | 42 | 42.6 |

#### xLSTM (mLSTM) — float64, 40K params

| Optimizer | Val Loss | Val RMSE | Val MAPE | Test RMSE | Test MAPE | Best Epoch | Time (min) |
|-----------|:--------:|:--------:|:--------:|:---------:|:---------:|:----------:|:----------:|
| CWD | 0.1656 | 0.1613 | 9.69% | 0.1645 | 10.20% | 67 | 202.5 |
| Adam (no reg) | 0.1761 | 0.1683 | 10.15% | 0.1660 | 10.15% | 102 | 231.2 |
| CPR | 0.1834 | 0.1716 | 10.53% | 0.1663 | 10.25% | 87 | 227.6 |
| AdamW | 0.1679 | 0.1620 | 9.83% | 0.1679 | 10.37% | 47 | 170.3 |

#### Cross-Architecture Ranking (by Test RMSE)

| Rank | Architecture | Optimizer | Test RMSE | Test MAPE | Δ vs #1 |
|:----:|-------------|-----------|:---------:|:---------:|:-------:|
| 1 | **TFT** | **CPR** ⭐ | **0.1558** | **9.51%** | — |
| 2 | TFT | AdamW | 0.1590 | 9.75% | +0.0032 |
| 3 | TFT | Adam | 0.1592 | 9.85% | +0.0034 |
| 4 | TFT | CWD | 0.1608 | 9.75% | +0.0050 |
| 5 | GRU | AdamW | 0.1628 | 9.70% | +0.0070 |
| 6 | xLSTM | CWD | 0.1645 | 10.20% | +0.0087 |
| 7 | GRU | Adam | 0.1652 | 9.87% | +0.0094 |
| 8 | GRU | CWD | 0.1658 | 9.91% | +0.0100 |
| 9 | xLSTM | Adam | 0.1660 | 10.15% | +0.0102 |
| 10 | xLSTM | CPR | 0.1663 | 10.25% | +0.0105 |
| 11 | xLSTM | AdamW | 0.1679 | 10.37% | +0.0121 |
| 12 | GRU | CPR | 0.1765 | 11.28% | +0.0207 |

#### Key Findings

1. **TFT dominates across all optimizers**: All 4 TFT variants occupy the top 4 positions in Test RMSE. Architecture matters more than optimizer choice.
2. **CPR is architecture-dependent**: CPR is excellent for TFT (#1) but poor for GRU (#12) and mediocre for xLSTM (#10). It requires sufficient model capacity to work well.
3. **AdamW is the most robust optimizer**: Ranks consistently well across all architectures (TFT #2, GRU #5, xLSTM #11 but with good Val RMSE).
4. **xLSTM underperforms expectations**: Despite theoretical advantages, xLSTM consistently trails GRU+AdamW on Test RMSE and has noticeably worse MAPE (>10% across the board).
5. **Original selection confirmed**: TFT+CPR (#1), TFT+AdamW (#2), GRU+CWD (#8) remain the sensible picks. However, GRU+AdamW (#5) could replace GRU+CWD as the GRU baseline.

### float32 vs float64 Hardware Observation

RTX 4060 Laptop: FP32 ~15.11 TFLOPS, FP64 ~0.236 TFLOPS (1/64 ratio).

- **TFT** is compute-bound → ~3x speedup with float32
- **xLSTM** is memory-bound → no benefit from float32
- **GRU** remains on float64 for compatibility

---

### TODO

- [x] Train GRU baseline for 3-way comparison
- [x] Evaluate regularization (AdamW, CWD, CPR)
- [x] Test float32 for speed
- [x] Integrate Model 2 Greeks (16-dim input)
- [x] Align data split with Model 1 (2021 test set)
- [x] Full 12-way comparison (3 arch × 4 opt)
- [ ] Crisis-period analysis (2020/03 COVID subset)

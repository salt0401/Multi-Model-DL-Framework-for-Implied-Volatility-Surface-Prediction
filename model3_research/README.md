# Model 3 Architecture Research — 2026-02-20

## Goal
Replace GRU in TVAdjustmentModel with more advanced architectures.
Two candidates selected for implementation:
1. **Temporal Fusion Transformer (TFT)** — full replacement
2. **xLSTM (mLSTM)** — drop-in GRU replacement

## Current Model 3 Baseline
- GRU (2-layer, hidden=64) + Multi-Head Attention (4 heads) + FC + SquarePlus
- Input: 20-day sliding window x 6 features
- Features: [vix_change, underlying_return, logm, tau, tv_pred, itm_otm]
- Target: tv_ratio = tv_true / tv_pred (adjustment factor alpha)
- Last training: ~46.5 hours on CPU

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

## Training Results (2026-02-21)

### Data
- 245,228 sequences, input_dim=12 (6 base + 6 enhancement features)
- Chronological split: train=179,615 (< 2020-06-05), val=65,613
- GPU: NVIDIA RTX 4060 Laptop (8.6GB), dtype=float64

### 3-Way Model Comparison

| Metric | GRU (Baseline) | xLSTM (mLSTM) | TFT | Winner |
|--------|:---:|:---:|:---:|:---:|
| **Val Loss (best)** | 0.1639 | **0.1544** | 0.1581 | xLSTM |
| **Val RMSE** | 0.1477 | **0.1414** | 0.1452 | xLSTM |
| **Val MAPE** | 9.43% | **9.01%** | 9.12% | xLSTM |
| Parameters | 58,689 | 39,133 | 265,281 | xLSTM (fewest) |
| Best Epoch | 70 | 138 | 112 | GRU (fastest convergence) |
| Training Time | 41.4 min | 311.5 min | 207.1 min | GRU |
| Interpretability | Low | Low | **Excellent** | TFT |

#### Improvement over GRU Baseline

| Model | RMSE Improvement | MAPE Improvement |
|-------|:---:|:---:|
| xLSTM | **-4.27%** (0.1477 → 0.1414) | **-4.45%** (9.43% → 9.01%) |
| TFT | **-1.69%** (0.1477 → 0.1452) | **-3.29%** (9.43% → 9.12%) |

#### Overfitting Analysis (Train-Val Gap)

| Model | Best Train Loss | Best Val Loss | Gap Ratio |
|-------|:---:|:---:|:---:|
| GRU | ~0.055 | 0.1639 | ~3.0x |
| xLSTM | ~0.055 | 0.1544 | ~2.8x |
| TFT | ~0.023 | 0.1581 | ~6.9x |

### TFT Feature Importance (Variable Selection Network)

| Feature | Importance | Category |
|---------|:---:|----------|
| tv_pred | 21.4% | Model 1 prediction (most critical) |
| tau | 15.9% | Term structure |
| iv_term_slope | 12.7% | Enhancement: IV structure |
| rv_20d | 11.7% | Enhancement: realized vol |
| iv_skew | 10.3% | Enhancement: tail risk |
| vrp_20d | 7.0% | Enhancement: vol risk premium |
| futures_basis_pct | 6.0% | Enhancement: basis |
| itm_otm | 5.0% | Moneyness indicator |
| vix_change | 4.8% | Market volatility |
| sp500_return | 2.2% | US market proxy |
| logm | 1.9% | Log moneyness |
| underlying_return | 1.0% | TAIEX return |

### Key Findings
1. **xLSTM is the best model**: 4.3% RMSE improvement over GRU baseline, with fewest parameters (39K)
2. **TFT also beats baseline**: 1.7% RMSE improvement, but 6.8x more parameters than xLSTM
3. **GRU is fastest to train**: 41.4 min vs 207.1 (TFT) and 311.5 (xLSTM) — cuDNN optimized kernel
4. **All three models overfit**: train-val gap 2.8-6.9x. We conducted regularization experiments (AdamW, CWD, CPR) to address this — see `regularization_results.md` (and `scripts/plot_regularization_results.py`) for the full analysis.
5. **Enhancement features contribute 47.7%** of TFT importance (iv_term_slope + rv_20d + iv_skew + vrp_20d + futures_basis_pct + sp500_return)
6. **Temporal attention**: last timestep gets 20.3% weight, confirming recency matters most
7. **tv_pred is the most important feature** (21.4%) — Model 1's output is the key input

> **Note (2026-02-22):** Based on the regularization results, we have officially shortlisted 3 final candidate models. All other experimental checkpoints and logs have been moved to `archived_models/` and `archived_logs/` to keep the active directory clean.
> The 3 shortlisted models are:
> 1. `tft_cpr_fp32_AdjustmentModel.pt` (Best overall, Val=0.1521)
> 2. `tft_adamw_fp32_AdjustmentModel.pt` (Strong runner-up, Val=0.1556)
> 3. `baseline_cwd_AdjustmentModel.pt` (Best GRU baseline, Val=0.1582)

### float32 vs float64 Benchmark (2026-02-21)

RTX 4060 Laptop 理論算力: FP32 ~15.11 TFLOPS, FP64 ~0.236 TFLOPS (1/64 ratio).
Benchmark 使用 batch_size=256, seq_len=20, input_dim=12, warmup=5, batches=20.

#### Training Speed (ms/batch)

| Model | float64 | float32 | Speedup |
|-------|:---:|:---:|:---:|
| xLSTM (mLSTM) | 73.8 | 73.3 | **1.01x (無差異)** |
| TFT | 68.4 | 22.3 | **3.07x** |

#### Inference Speed (ms/batch)

| Model | float64 | float32 | Speedup |
|-------|:---:|:---:|:---:|
| xLSTM (mLSTM) | 17.5 | 17.8 | **0.98x (無差異)** |
| TFT | 25.3 | 4.5 | **5.61x** |

#### Estimated Full Training Time

| Model | float64 (actual) | float32 (estimated) | Savings |
|-------|:---:|:---:|:---:|
| xLSTM | 311.5 min | ~309 min | ~2 min (不值得切換) |
| TFT | 207.1 min | **~68 min** | **~140 min (省 2/3 時間)** |

#### Analysis

- **xLSTM 是 memory-bound**: mLSTM 的 sequential scan 每步有資料依賴，GPU 無法平行化。
  瓶頸在記憶體存取而非算力，FP32/FP64 速度完全相同。
- **TFT 是 compute-bound**: LSTM encoder + Multi-Head Attention + GRN 可大量平行化，
  受惠於 FP32 的 64x 算力優勢，實際達到 3-5x 加速。
- **精度影響**: float32 有 ~7 位十進制精度，對 tv_ratio (~1.0 ± 0.2) 的預測完全足夠。
  float64 的 ~15 位精度在此任務中無必要。
- **結論**: 僅 TFT 值得切換 float32；xLSTM 切換無效果。

---

### TODO
- [x] Train GRU baseline for 3-way comparison — Done, see results above
- [x] Add dropout/L2 regularization to reduce overfitting — Evaluated AdamW, CWD, and CPR. See `regularization_results.md`
- [x] Test with float32 for speed (RTX 4060 has 1/64 FP64 ratio) — Done, see benchmark above
- [ ] Crisis-period analysis (2020/03 COVID subset)

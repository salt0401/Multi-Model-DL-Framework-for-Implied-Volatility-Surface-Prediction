# Models 4 & 5 Completion Report

**Date:** 2026-07-06
**Scope:** Complete the untrained "later part" of the TXO IV-surface pipeline (Models 4 & 5), challenging the draft designs against 2024–2026 state of the art, sized to an RTX 4060 Laptop (8GB) / 32GB RAM machine.

---

## 1. Decisions

### Model 4 — HyperIV: KEPT (with the PIVOT upgrade)

A literature sweep (July 2026) confirmed HyperIV (Yang, Chen, Shu & Hospedales, ICML 2025, PMLR v267) remains the accuracy/efficiency frontier for sparse-quote, real-time, arbitrage-controlled IV surface construction:

- It beats SSVI, VAE, and graph-neural-operator baselines on 7 of 8 datasets in its own benchmark (e.g., SPX 1-day IV MAE 0.0075 vs GNO 0.0085, SSVI 0.0312).
- The only competitive alternative, Operator Deep Smoothing (ICLR 2025), required **77.8 GiB** training memory in HyperIV's benchmark — infeasible on 8GB — and degrades badly in VIX-like/small-market regimes resembling TXO.
- The one substantial, evidence-backed improvement is **PIVOT** (arXiv 2606.17065, June 2026): an *upgrade to* HyperIV, not a replacement. Adding a normalized Black-76 price-space auxiliary loss to the identical architecture cut price MAE 32–44% on SPX with IV MAE also improving. The price-space auxiliary (~30 lines, no custom kernels) was adopted with weight 0.1.

### Model 5 — grid DDPM: REPLACED (the "not a small gap" case)

Three independent research tracks converged against the draft (1D U-Net DDPM, 1000 steps, raw 200-dim grid):

1. **Data starvation:** the only published success of this exact design (Jin & Agarwal, arXiv 2511.07571) trained on ~7,000 daily surfaces; this project has **1,449 training pairs**. Factor-analysis literature (Cont & da Fonseca) shows ~3 factors explain >90% of surface variance — forecasting ~12 smooth factors is statistically far better conditioned than a 200-dim grid.
2. **Architecture mismatch:** the tabular-diffusion literature (TabDDPM, TabSyn ICLR 2024, CDTD ICLR 2025) converged on MLP denoisers for low-dim structured data; a U-Net's translation-equivariance buys nothing on a 200-dim fixed-semantics grid while adding memorization-prone parameters.
3. **Sampling cost:** conditional OT flow matching attains diffusion-quality samples in 10–50 Euler steps vs 1000 ancestral steps (identical-architecture comparison in arXiv 2511.19379: FM usable at 10 steps where DDPM yields pure noise below ~50; TSFlow, ICLR 2025, beats diffusion baselines on CRPS on 6/8 forecasting datasets). Measured draft cost on this machine: 2.2 s *per surface* (FP64, T=1000).

**The replacement:** grid daily surfaces (train-only quantile grid) → log(total variance) → PCA (train-fit, 12 factors) → z-scored scores → conditional OT flow matching with a FiLM residual-MLP velocity field (~0.9M params) → Euler sampling (50 steps) → inverse transform. Positivity of generated surfaces is **guaranteed by construction** (exp); anti-memorization levers per the small-data literature: dropout 0.1, weight decay 1e-3, EMA 0.999, early stopping on sampled val RMSE.

**Representation validation (measured, test period 2021):** raw random-walk tv-RMSE 0.001922; PCA truncation floor at k=12 is 0.001107 (truncation does not bind), and *projecting today's surface through the factor space alone* already improves on the raw random walk (0.001741) — the PCA denoises.

The draft DDPM files (`src/diffusion.py`, `src/train_diffusion.py`) are retained for reference but **deprecated**. Note the draft was also unusable as-written: surfaces at scale ~0.009 were diffused against N(0,1) noise with no normalization (signal ≈ 1% of noise), among other issues below.

### Rejected alternatives (so they are not re-proposed)

- **eSSVI-parameter forecasting** (hard no-arbitrage guarantee via the Hendriks–Martini/Corbetta–Martini–Mingone domain): principled, but Model 1 exists precisely because pure eSSVI cannot fit TXO surfaces — a pure-eSSVI forecaster inherits that fit ceiling as an accuracy floor, and it requires building a new per-day calibration pipeline.
- **Consistency models:** documented training instability; distillation-oriented; pointless when a 50-step MLP sampler is already sub-second.
- **GNO / Operator Deep Smoothing for Model 4:** memory-infeasible on 8GB and weaker in sparse/robustness regimes (above).
- **VolGAN and successors:** dominated on published metrics by conditional diffusion (arXiv 2511.07571).

---

## 2. Bugs found and fixed

### Baseline repairs (pre-existing, from the repo reorganization)
1. `conftest.py:13` — `sys.path` pointed to `../src` from the repo root (file was written for `tests/`); suite could not even collect.
2. `tests/test_train_integration.py:8` — stale `from train import ...` after Model 1's migration to `model1_research/`.
3. `ruptures` was in requirements.txt but not installed (3 structural-break tests failing).
Baseline after repairs: **210 passed, 0 failed**.

### Model 4 (HyperIV) — draft bugs
4. **Wrong Gatheral–Jacquier butterfly density** (`src/hyperiv.py:213`): used w′ where the formula requires **w′²**. The correct form existed in `model1_research/model.py:316`; transcription error.
5. **Same bug in the numerical checker** (`src/test.py:65`) — which means **Model 1's documented "45.69% butterfly violations" statistic was computed with a wrong density formula and needs recomputation** (flagged; not recomputed here).
6. **ReLU target MLP** — piecewise-linear, so d²w/dk² = 0 almost everywhere: the butterfly penalty was silently a no-op. Replaced with tanh + softplus (the paper's own design).
7. **Biased evaluation protocol** (`src/train_hyperiv.py:174`): reference set = first 50 options of surfaces sorted by (tau, logm) — always the short-maturity corner, mismatched with training's random sampling. Replaced with seeded random sampling, batched evaluation.
8. **No input standardization** for the set transformer; added train-stat z-scoring stored in the checkpoint.
9. **Padding diluted the loss** (padded entries pre-multiplied to 0 but still counted in the mean); replaced with proper masked reductions.
10. **FP64 on an Ada GPU** (1:64 throughput); training now float32 (configurable).

### Model 4 — training-collapse root cause (found by systematic debugging)
11. The draft (and the first fix attempt) plateaued at val RMSE 0.0187 = √E[tv²] — the model predicted **exactly zero everywhere** and was *worse than the trivial predict-yATM baseline (0.0111)*. Root cause: predicting raw total variance (median 0.005) through a softplus head is fatally ill-conditioned — pre-activations sit at log(tv) ≈ −5..−9 where softplus' gradient ≈ tv ≈ 1e-6, killing all upstream gradients (measured grad norms 1e-8). Two changes fixed it, verified by a 15-epoch probe (val RMSE 0.0105 by epoch 10):
    - Residual hypernetwork: generated params = learnable normally-initialized base + day-specific delta (pure-delta weights start at 0 where tanh activations are 0 and gradients die).
    - Scale-equivariant output `w = √(yATM² + 0.002²) · softplus(f(·))` — the target net predicts an O(1) ratio-to-ATM (Model 1's own Lesson #7 pattern), output bias init 0.54 so initial surfaces sit at the ATM level.

### Model 5 (draft DDPM) — why it could not have worked as written
12. **No normalization**: total-variance surfaces (mean 0.0089, std 0.0065 — measured) diffused against N(0,1) noise; signal-to-noise ≈ 10⁻⁴.
13. No positivity/arbitrage control on generated surfaces; no de-normalization or grid metadata persisted with checkpoints; unnormalized conditioning (raw VIX ≈ 20 mixed with returns ≈ 0.01); early stopping arithmetic dead (would require 10,000 epochs to trigger).

### Data pipeline
14. **getYATM leakage guard never engaged** (`src/dataset.py:51`): `__call__` didn't pass `train_end_date`, so the synthetic-c6 ATM curve averaged over test dates. Fixed (config-driven).
15. **Grid quantiles from the full dataset** including test dates in `Prepare_diffusion_data`; the new `Prepare_surface_panel` uses train-only quantiles.
16. **Boundary pair leak**: the pair (2020-12-31 → 2021-01-04) put the first test-day surface into training targets; `flow_surface.build_dataset` requires *tomorrow* ≤ train_end for training pairs.
17. **VIX default 0.2 vs index-scale values (~9–80)** on US holidays; replaced with merge-asof forward-fill.
18. Flagged out of scope (separate spawned task): Model 3's `sp500_return` is same-US-date aligned — realized ~15h *after* the Taiwan close whose same-day ratio it predicts — look-ahead leakage in Model 3's features (legitimate for Model 5's next-day use).

---

## 3. Final results

*(filled in after training/evaluation runs below)*

### Model 4 — HyperIV (test = 2021, 244 surfaces)

Training: 255 epochs (early stop; best epoch 203), float32 on the RTX 4060, ~45 minutes. The spike guard fired several times (measured penalty spikes previously collapsed training irrecoverably) and each time restored stable weights and halved the LR.

| Metric | Value | Reference |
|---|---|---|
| Test tv-RMSE | **0.002148** | Model 1 (global fit, same test year): 0.01977 |
| Test MAPE (tv, ε-damped) | **6.94%** | Model 1: 5.46% |
| Test IV-RMSE | **0.0395** | — |
| Calendar violations (dense grid, 48,995 pts) | **0.0000%** | — |
| Butterfly violations (corrected Gatheral g(k)) | **0.055%** | Model 1: ~46% (by the old, buggy checker) |

Caveats stated plainly: HyperIV sees 50 same-day reference quotes per surface, so its tv-RMSE is not directly comparable to Model 1's global no-reference fit — reconstructing a day's surface from a sparse quote subset is precisely its deployment job (real-time interpolation). MAPE uses the codebase's ε=0.005 damping and understates relative error on tiny total variances. Artifacts: `logs/hyperiv_results.json`, `logs/hyperiv_test_fit.png`, checkpoint `models/HyperIVModel.pt` (with train-set normalization stats and config embedded).

### Model 5 — Flow-matching surface forecaster (test = 2021, 243 pairs)

Trained in two iterations. The first (levels formulation: flow directly generates tomorrow's factor scores) was **worse than the random walk** (tv-RMSE 0.00277 vs 0.00192, DM p=0.0002) — reported and replaced. Reformulating the flow over the **z-scored daily increment** of factor scores (persistence exact by construction; the VolaDiff-style "model shocks, not levels" recommendation) fixed it:

| Method | tv-RMSE | IV-RMSE | IV-MAPE |
|---|---|---|---|
| Random walk (tomorrow = today) | 0.001922 | 0.02117 | 3.84% |
| VAR(1) on factor scores | 0.001758 | 0.02116 | 3.90% |
| **Flow matching (mean of 100 samples)** | **0.001708** | **0.02011** | **3.59%** |

- **Diebold–Mariano vs random walk: stat −4.98, p < 0.0001** — a statistically significant improvement where the literature (Goncalves–Guidolin 2006 and successors) says even parity is demanding. PCA truncation floor: 0.001107 (not binding).
- Probabilistic: CRPS 0.000785; 90%-interval empirical coverage **73.1%** — better than the levels model's 37% but still under-dispersed; widening via sample-count/temperature calibration is listed as future work. Use the intervals as indicative, not literal.
- No-arbitrage (grid resolution): generated surfaces average 2.3% calendar / 1.9% butterfly violation rates versus the **actual market surfaces' own 1.4% / 6.1%** — i.e., generated butterfly quality is 3× cleaner than the data itself; calendar is slightly above the market's (the market's calendar violations are near zero at this resolution). Positivity is exact by construction.
- Training: 1,050 epochs (early stop), ~4 minutes on CPU; sampling 100 scenarios × 243 days ≈ 20 s. The deprecated DDPM draft would have needed ~9 minutes *per evaluation pass* for the same samples at T=1000 in FP64.
- Figures: `logs/flow_eval_surfaces.png`, `logs/flow_eval_metrics.png`; full metrics `logs/flow_surface_eval.json`.

---

## 4. How to run

```bash
cd src
python train_hyperiv.py --on_gpu                 # Model 4: train + test + violations -> logs/hyperiv_results.json
python train_flow_surface.py --on_gpu            # Model 5: train -> models/FlowSurfaceModel.pt
python evaluate_surface_forecast.py              # Model 5: full evaluation -> logs/flow_surface_eval.json
python -m pytest tests/ model1_research/tests model2_research/tests -q   # full suite
```

## 5. Deferred / out of scope

- 5-model consensus/stacking — no specification exists anywhere in the docs; the "Consensus Predictions" box in COMPREHENSIVE_TECHNICAL_SUMMARY contradicts ARCHITECTURE.md ("trained independently") and remains future work.
- Model 3 retraining with corrected S&P 500 timing (spawned as a separate task).
- Recomputing Model 1's butterfly-violation statistic with the corrected density formula (`src/test.py`).
- 2022–2026 extended dataset (quarantined for data-quality reasons documented in `docs/discussion_notes.md`).
- Repair of the deprecated DDPM trainer (superseded by flow matching).

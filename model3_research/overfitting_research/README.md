# Model 3 Overfitting Research

## 問題描述

三個模型都出現顯著的 train-val gap（overfitting）：

| Model | Best Train Loss | Best Val Loss | Gap Ratio |
|-------|:---:|:---:|:---:|
| GRU | ~0.055 | 0.1639 | ~3.0x |
| xLSTM | ~0.055 | 0.1544 | ~2.8x |
| TFT | ~0.023 | 0.1581 | ~6.9x |

Early stopping 選擇 val loss 最低的 epoch，但 train loss 持續下降表示模型容量
被浪費在記憶訓練資料的 noise，而非學習更好的泛化 pattern。

## 研究方向

### 1. Parameter Drift Analysis（原創想法）
追蹤 val loss 觸底後哪些參數仍在變動，分類為：
- **Stable parameters**: 已停止變動 → 學到有用 pattern
- **Drifting parameters**: 仍在變動 → noise memorizers

根據參數群的統計分布選擇 L1（Laplace prior）或 L2（Gaussian prior）正則化。

**相關文獻：**
- Elastic Weight Consolidation (EWC) — Kirkpatrick et al., 2017, PNAS
- Synaptic Intelligence (SI) — Zenke et al., 2017, ICML
- Memory Aware Synapses (MAS) — Aljundi et al., 2018, ECCV
- Memorization Localization — Maini et al., 2023, ICML
- Spectral Bias — Rahaman et al., 2019, ICML

### 2. Cautious Weight Decay (CWD) — 見 `cwd_notes.md`
ICLR 2026 最新方法，one-line 修改，per-parameter selective weight decay。

### 3. Constrained Parameter Regularization (CPR)
Per-parameter-matrix adaptive regularization — Franke et al., 2024, NeurIPS

## 檔案結構
- `README.md` — 本文件，研究總覽
- `cwd_notes.md` — Cautious Weight Decay 詳細分析
- `literature_review.md` — 完整文獻回顧（待補）

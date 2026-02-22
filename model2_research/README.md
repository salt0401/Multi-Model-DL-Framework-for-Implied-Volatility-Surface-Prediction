# Model 2 Research — Dupire PDE-Constrained Local Volatility Extractor

## 在專案中的角色

本專案的 5-model 系統中，Model 2 扮演 **IV surface → Local Volatility surface** 的橋樑：

```
Model 1 (SSVI+NN)  ──→  Model 2 (Dupire PINN)  ──→  Model 3 (Adjustment)
   IV surface              Local vol surface           時序修正
   w(τ, logm)              σ²_LV(K, T)                (+ local vol features)
                           + Risk-neutral density
                                    │
                                    └──→  Model 5 (DDPM)
                                          次日預測
                                          (conditioned on local vol)
```

**Model 2 消費 Model 1 的輸出，產出 local volatility surface 和 risk-neutral density，
供 Model 3 和 Model 5 使用。**

### Model 1 現況（已驗證, 2026-02-22）

| 項目 | 值 |
|------|---|
| 資料集 | `prs_dataset_no_fat(clean).csv` (~254K rows, 2014-2021) |
| 訓練期間 | 2014-2020, 驗證集: 最後 20% dates (split at 2020-06-05) |
| 測試期間 | 2021 (50,310 test points) |
| 架構 | SSVI (bounded power-law) + SmileModel ×5 ensemble, additive |
| Pipeline best val loss | 0.117 (3 epochs, Stage 1) |
| Butterfly violation | **74%** (2021 test, ~37K 點 g(k) < 0) |
| SSVI mean rho | -0.310（左偏確認） |

---

## 為什麼需要這個模型

### 問題：Dupire 公式的數值不穩定性

Model 1 輸出 total variance `w(τ, logm)` 及其導數。理論上可以直接用 Dupire 公式
提取 local volatility：

```
σ²_LV(K,T) = ∂w/∂τ / [1 - (logm/w)(∂w/∂logm) + ¼(-¼ - 1/w + logm²/w²)(∂w/∂logm)² + ½(∂²w/∂logm²)]
```

但這個公式**數值上是病態的(ill-conditioned)**：
- 分母可以趨近零 → local vol 爆炸
- Model 1 的 butterfly violation rate **74%**（2021 test, 50,310 test points 中約 37K 點） → 等價於分母為負 → local vol 出現虛數
- 微小的 IV surface 誤差會被放大成 local vol 的劇烈震盪
- Model 1 使用 additive architecture (`w = SSVI + yATM·NN`)，導數不含 cross-terms，但集成的 softmax 加權仍然造成密度扭結

> **關鍵數據：** 74% butterfly violation = 74% 的測試點套 Dupire 公式會得到負的分母。
> 這就是 Model 2 存在的最主要理由。

### 解決方案：PINN + Dupire PDE 約束

用 Physics-Informed Neural Network (PINN) 把 Dupire PDE 當作 loss 約束，
讓神經網路學出一組**自洽(self-consistent)**的 call price + local vol，
而不是先算再修的傳統做法。

### 為什麼不用 Black-Scholes PDE（舊版 DGM）

舊版 Model 2 學的是固定 sigma 的 GBM backward Kolmogorov PDE，
這個問題有解析解（Black-Scholes formula），用神經網路去逼近毫無意義。
新版改為學習 **Dupire equation**——一個沒有通用解析解的 inverse problem。

---

## 技術設計

### 架構：雙網路 Self-Consistent Learning

參考 Wang & Privault (2022/2025) 和 WamOL (ICAIF 2024)：

```
(A) Price Correction Network: π_θ(k, τ)
    - 輸入: Model 1 的 call price 作為 prior
    - 架構: 3 residual blocks, 64 neurons, tanh
    - 輸出: 修正後的 call price C(K,T)

(B) Local Volatility Network: σ_LV_φ(k, τ)
    - 輸入: (k, τ)
    - 架構: 3 residual blocks, 64 neurons, tanh + softplus (確保正值)
    - 輸出: σ²_LV(K,T) > 0
```

### Loss Function（5 項）

```
L = λ_fit    · L_fit      # 符合 Model 1 的 call price
  + λ_dup    · L_dupire    # Dupire PDE 殘差: ∂π/∂τ = ½σ²_LV k² ∂²π/∂k²
  + λ_arb    · L_arb       # No-arbitrage 不等式 (calendar + butterfly + delta bounds)
  + λ_ini    · L_initial   # 邊界條件: τ=0 時 payoff = max(S-K, 0)
  + λ_smooth · L_smooth    # Local vol 平滑性 (Sobolev penalty)
```

### 輸入與輸出

**輸入（來自 Model 1）：**
- Total variance surface w(τ, logm) 及其導數
- ATM volatility yATM(date)

**輸出（供 Model 3 / Model 5）：**
- Local volatility surface σ_LV(K, T)
- Risk-neutral density q(K, T) = ∂²C/∂K²
- PDE residual map（surface 品質診斷）

### 計算需求估計（RTX 4060, float64）

| 指標         | 估計值                |
|-------------|----------------------|
| 參數量       | ~100K（兩網路各 ~50K）|
| 記憶體       | < 1 MB 權重          |
| 每 epoch     | ~2 min               |
| 日常再校準   | ~1-2 min (transfer learning) |

---

## 參考文獻

1. Wang & Privault (2022/2025). *Deep Self-Consistent Learning of Local Volatility.*
   arXiv:2201.07880 — 雙網路 + Dupire PDE 約束
2. WamOL (ICAIF 2024). *Whack-a-mole Online Learning: PINN for Intraday IV Surface.*
   arXiv:2411.02375 — 三層自適應權重 + no-arbitrage 約束
3. Bae, Kang & Lee (2024). *Option Pricing and Local Volatility Surface by PINN.*
   Computational Economics, 64:3143-3159
4. DeepSVM (2025). *Learning Stochastic Volatility Models with PI-DeepONet.*
   arXiv:2512.07162
5. ICLR 2025. *Operator Deep Smoothing for Implied Volatility.*
   arXiv:2406.11520
6. ICPINN (2025). *Improved Constrained PINNs for PDEs and Option Pricing.*
   ScienceDirect
7. Chataigner et al. *DupireNN.* GitHub: mChataign/DupireNN — 參考實作

---

## 目前進度

- [x] 確認舊版 DGM (BS PDE solver) 無實質價值
- [x] Deep research：調查 2023-2026 年 neural PDE for finance 論文
- [x] 確定新方向：Dupire PDE-Constrained Local Volatility Extraction
- [x] 完成技術設計（雙網路架構 + 5-term loss）
- [x] 驗證 Model 1 數據：確認 74% butterfly（非之前記錄的 83.7%）、2014-2020 train / 2021 test
- [x] 更新 Gemini 研究 brief (`model2_gemini_brief.md`) with verified data
- [x] 將 brief 提交 Gemini 進行 deep research
- [x] 實作 `dupire_pinn.py`（雙網路 + Dupire loss）
- [x] 實作 `train_dupire.py`（訓練腳本）
- [x] 連接 Model 1 pipeline（load MultiModel.pt → 產生 call price grid）
- [x] 訓練 + 驗證（2014-2020 train, 2021 test）
- [x] 輸出 local vol features 給 Model 3
- [x] 輸出 local vol grid 給 Model 5

> **最新狀態 (Update 2026-02)**: Model 2 (ICNN Dupire) 已順利完成 V1-V3 階段開發。其中 V2 透過 ICNN 結構成功達成 100% 消除套利違規，而 V3 (Module D) 利用 AutoGrad 進一步計算出高階希臘字母 (Vanna, Volga) 與局部波動率梯度 (Local Volatility Gradient)，以 15 維度矩陣供下游的 Model 3 擴展訓練使用。

---

## 檔案結構（規劃）

```
model2_research/
├── README.md                # 本文件
├── dupire_pinn.py           # 模型定義（雙網路 + loss）
├── train_dupire.py          # 訓練腳本
├── benchmark_dupire.py      # 驗證：與直接 Dupire 公式比較
├── logs/                    # 訓練 logs
└── models/                  # Checkpoints
```

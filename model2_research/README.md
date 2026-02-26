# Model 2 — Dupire PINN Local Volatility Extractor

## 在專案中的角色

本專案的 multi-model 系統中，Model 2 扮演 **IV surface → Local Volatility surface** 的橋樑：

```
Model 1 (eSSVI+NN)  ──→  Model 2 (ICNN Dupire PINN)  ──→  Module D (Greeks)  ──→  Model 3 (TFT)
   IV surface               Local vol surface               Vanna, Volga,          時序修正
   w(τ, logm)               σ²_LV(K, T)                    ∂σ_LV/∂K
```

Model 2 消費 Model 1 的 total variance 預測，產出 local volatility surface 供下游模型使用。

---

## 為什麼需要這個模型

### 問題：Dupire 公式的數值不穩定性

Model 1 輸出 total variance `w(τ, logm)` 及其導數。理論上可以直接用 Dupire 公式提取 local volatility：

```
σ²_LV(K,T) = ∂w/∂τ / [1 - (logm/w)(∂w/∂logm) + ¼(-¼ - 1/w + logm²/w²)(∂w/∂logm)² + ½(∂²w/∂logm²)]
```

但這個公式**數值上是病態的 (ill-conditioned)**：
- 分母可以趨近零 → local vol 爆炸
- Model 1 的 butterfly violation rate **45.69%**（2021 test）→ 分母為負 → local vol 出現虛數
- 微小的 IV surface 誤差會被放大成 local vol 的劇烈震盪

### 解決方案：ICNN + Dupire PDE 約束

使用 Input-Convex Neural Network (ICNN) 搭配 Physics-Informed Neural Network (PINN) 的方法，
從**架構層級**硬保證 ∂²C/∂K² ≥ 0（蝴蝶條件），同時透過 Dupire PDE loss 約束讓 Price Network 和 Local Vol Network 找出一組自洽解。

---

## 技術設計

### 架構：雙網路 Self-Consistent Learning

```
(A) Price Network (ICNNPriceNetwork):
    - 輸入: (K, τ)
    - 架構: 3 residual blocks, 64 neurons
    - K → output 路徑所有權重透過 softplus 強制非負 → 硬保證 ∂²C/∂K² ≥ 0
    - 輸出: 修正後的 call price C(K,T)
    - 參數量: 13,123

(B) Local Volatility Network (LocalVolNetwork):
    - 輸入: (K, τ)
    - 架構: 3 residual blocks, 64 neurons, softplus output (確保正值)
    - 輸出: σ²_LV(K,T) > 0
    - 參數量: 13,633
```

總參數量: **26,756**

### Loss Function（6 項）

```
L_total = λ_fit    · L_fit       # Model 2 價格 vs. Model 1 提供的目標 Call Price
        + λ_pde    · L_dupire    # Dupire PDE 殘差: ∂C/∂τ = ½σ²_LV K² ∂²C/∂K²
        + λ_cal    · L_calendar  # 日曆套利約束: ∂C/∂τ ≥ 0
        + λ_but    · L_butterfly # 蝴蝶套利約束: ∂²C/∂K² ≥ 0 (ICNN 下永遠 = 0)
        + λ_smooth · L_smooth    # σ_LV 平滑性 (Sobolev penalty)
        + 1.0      · L_boundary  # 邊界條件: τ→0 時 payoff = max(S-K, 0)
```

> L_fit + L_boundary 合佔 Final Loss 約 98%；PDE/Calendar/Smooth 佔 ~2%；Butterfly = 0（ICNN 架構保證）。

### 資料來源：Model 1 Pipeline 連接

Model 2 的訓練目標 `C_target` **直接來自載入的 `MultiModel.pt`**，而非靜態 CSV 或合成 BS 資料。

訓練時每次重新取樣 collocation points 時，`DupireSampler` 會：
1. 在 `(K, τ)` 域內隨機撒下 5000 個內部點
2. 將 `K → logm = ln(K/S)` 轉換後，直接 query Model 1: `MultiModel(tau, logm, yATM)`
3. 取得 `tv_pred`（total variance），透過 `_total_variance_to_call_price_tensor()` 轉成 `C_target`
4. 配合 500 個邊界點（τ→0 payoff）一起輸入 Loss Function

> `yATM` 從 `DataProcessor.prs_dataset['y_atm']` 自動計算 mean 值。

### 輸入與輸出

**輸入（來自 Model 1）：**
- 載入 `MultiModel.pt` 權重，動態查詢 total variance surface w(τ, logm)
- ATM total variance `yATM` 從 dataset 自動計算

**輸出（供 Module D / Model 3）：**
- Local volatility surface σ_LV(K, T)
- Risk-neutral density q(K, T) = ∂²C/∂K²

---

## 使用方式

```bash
cd src

# 訓練（使用 Model 1 真實數據）
python ../model2_research/train_dupire.py --on_gpu --use_icnn --use_model1

# 可選：手動指定 yATM
python ../model2_research/train_dupire.py --on_gpu --use_icnn --use_model1 --yATM 0.05

# 萃取 V3 特徵 (Module D)
python ../model2_research/extract_features.py --model_path ../models/DupireModel.pt --use_icnn
```

---

## 目前結果（2026-02-25）

### Model 1 上游現況

| 項目 | 值 |
|------|---|
| 資料集 | `prs_dataset_no_fat(clean).csv` (~254K rows, 2014-2021) |
| 架構 | eSSVI (frozen ρ₀=-0.95) + SmileModel ×5 ensemble, ε=0.02 |
| Best val loss | 0.07495 |
| Butterfly violation | **45.69%** (20,164/44,133 test points) |
| 權重檔 | `model1_research/models/MultiModel.pt` |

### Model 2 訓練結果

| 項目 | 值 |
|------|---|
| yATM (自動計算) | 0.005965 |
| Epochs | 5,000 |
| Runtime | ~5.5 min (RTX 4060, float64) |
| Final Total Loss | **0.0768** |
| Butterfly violation | **0.000** (全程 0%, ICNN 架構保證) |
| PDE Residual | 0.000103 |
| Calendar violation | 0.000026 |
| Validation local vol std | 0.0032 |
| 權重檔 | `models/DupireModel.pt` |

### Loss 收斂過程

| Epoch | Total Loss | Fit | PDE | Calendar | Butterfly |
|------:|----------:|----:|----:|---------:|----------:|
| 100 | 1.14×10¹³ | — | — | — | 0.000 |
| 1000 | 9.65×10⁹ | — | — | — | 0.000 |
| 1500 | 1.66×10⁷ | — | — | — | 0.000 |
| 1900 | 39.5 | — | — | — | 0.000 |
| 2100 | 0.085 | 0.042 | 0.000316 | 0.000273 | 0.000 |
| 5000 | **0.077** | ~0.039 | 0.000103 | 0.000026 | 0.000 |

---

## Module D (Model 2.5) — Greeks 萃取

從 Model 2 的無套利 local vol 曲面計算高階特徵，不需要額外訓練：

| 特徵 | 公式 | 預測價值 |
|------|------|---------| 
| **Local Vol** | σ_LV(K,T) | 局部波動率直接值 |
| **Vanna** | ∂²C/∂S∂σ | 捕捉偏態 + S-vol 負相關，預測單邊暴跌 |
| **Volga** | ∂²C/∂σ² | Vega 凸性，量化尾部風險預期 |
| **∂σ_LV/∂K** | local vol 對 K 的斜率 | 瞬間局部偏態 |

Model 3 input: 從 12 dim 擴展至 **16 dim** (+Local Vol, +Vanna, +Volga, +∂σ_LV/∂K)

---

## 檔案結構

```
model2_research/
├── README.md                # 本文件
├── model2_training_details.md  # 訓練細節與 Loss 解析
├── dupire_pinn.py           # 模型定義（ICNN + LocalVol + Loss + Sampler）
├── train_dupire.py          # 訓練腳本（含 Model 1 載入）
├── extract_features.py      # Module D: Greeks 萃取
├── module_d.py              # Module D: GreekExtractor 定義
└── tests/                   # Unit tests (28 項)
```

---

## 參考文獻

1. Wang & Privault (2022/2025). *Deep Self-Consistent Learning of Local Volatility.* arXiv:2201.07880
2. WamOL (ICAIF 2024). *Whack-a-mole Online Learning: PINN for Intraday IV Surface.* arXiv:2411.02375
3. Bae, Kang & Lee (2024). *Option Pricing and Local Volatility Surface by PINN.* Computational Economics, 64:3143-3159
4. Amos et al. (2017). *Input Convex Neural Networks.* arXiv:1609.07152
5. Wiedemann et al. (ICLR 2025). *Operator Deep Smoothing for Implied Volatility.* arXiv:2406.11520

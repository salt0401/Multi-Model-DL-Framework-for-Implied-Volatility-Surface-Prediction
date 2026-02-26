# Model 2 訓練細節

> 本文件記錄 Model 2 (ICNN Dupire PINN) 的 Loss Function 組成、Hyperparameters 配置，
> 以及 autograd 兼容性等工程注意事項。

---

## Loss Function 詳解

Model 2 的 Total Loss 由 **6 項**加權組成。每一項對應一個數學或物理約束：

### 各項公式

| # | Loss 項 | 公式 | λ (config) | 作用 |
|---|---------|------|-----------|------|
| 1 | **L_fit** | `mean((C_pred - C_target)²)` | 1.0 | Price Network 預測 vs. Model 1 提供的 Call Price |
| 2 | **L_pde** | `mean((∂C/∂τ - ½σ²_LV·K²·∂²C/∂K²)²)` | 10.0 | Dupire PDE 殘差，強制兩網路的 (C, σ_LV) 自洽 |
| 3 | **L_cal** | `mean(relu(-∂C/∂τ))` | 10.0 | 日曆無套利：到期越遠的選擇權不可更便宜 |
| 4 | **L_but** | `mean(relu(-∂²C/∂K²))` | 10.0 | 蝴蝶無套利：因 ICNN 架構，此項**恆等於 0** |
| 5 | **L_smooth** | `mean((∂σ²_LV/∂K)² + (∂σ²_LV/∂τ)²)` | 1.0 | Sobolev 正則化，平滑 local vol 曲面 |
| 6 | **L_bnd** | `mean((C_pred(K, τ→0) - max(S-K, 0))²)` | 1.0 | 邊界條件：到期時的 intrinsic value |

> Total Loss 中 L_fit + L_bnd 佔約 98%，PDE / Calendar / Smooth 佔 ~2%，Butterfly = 0。

### C_target 的來源

`C_target` **不是**原始市場報價，也**不是**合成 Black-Scholes 資料。它是：

```
Random (K, τ) 點  →  logm = ln(K)  →  MultiModel(τ, logm, yATM)  →  tv_pred
                                                                      ↓
                                        _total_variance_to_call_price_tensor()
                                                                      ↓
                                                                   C_target
```

也就是說，Model 2 的學習目標是 **Model 1 已經擬合好的連續 IV 曲面**，
而非離散的市場報價。Model 1 負責「市場報價 → 平滑曲面」，Model 2 負責「平滑曲面 → 無套利 Local Vol」。

---

## Hyperparameters（`config.ini [dupire]` 區段）

| 參數 | 值 | 意義 |
|------|---|------|
| hidden_dim | 64 | 兩網路的隱藏層寬度 |
| n_layers | 3 | 網路深度 |
| k_min / k_max | 0.5 / 1.5 | 正規化後 strike 範圍 |
| tau_min / tau_max | 0.02 / 2.0 | 到期日範圍 (年) |
| n_interior | 5000 | 每次取樣的內部配點數 |
| n_boundary | 500 | 邊界點數 |
| resample_every | 100 | 每 100 epochs 重新隨機取樣配點 |
| lambda_fit | 1.0 | L_fit 權重 |
| lambda_pde | 10.0 | L_pde 權重 |
| lambda_cal | 10.0 | L_cal 權重 |
| lambda_but | 10.0 | L_but 權重（因 ICNN，此項永遠 = 0） |
| lambda_smooth | 1.0 | L_smooth 權重 |
| epochs | 5000 | 訓練 epochs |
| learning_rate | 0.001 | AdamW 初始學習率 |
| gradient_clip | 1.0 | 梯度裁切上限 |

### Scheduler

使用 `ReduceLROnPlateau(patience=200, factor=0.5)`，根據 total loss 自動降低學習率。

---

## Autograd 兼容性注意事項

### Model 1 查詢不能用 `torch.no_grad()`

Model 1 的 `SmileModel.forward` 內部使用 `autograd.grad(create_graph=True)` 計算一、二階導數。
因此在 `DupireSampler._query_model1()` 中：

```python
# ✗ 錯誤：SmileModel 的 autograd.grad 無法在 no_grad() 下運行
with torch.no_grad():
    tv_pred, _, _, _ = base_model(tau, logm, yATM)

# ✓ 正確：啟用 requires_grad，查詢後 detach 切斷計算圖
tau_q = tau.clone().requires_grad_(True)
logm_q = logm.clone().requires_grad_(True)
tv_pred, _, _, _ = base_model(tau_q, logm_q, yATM)
tv_pred = tv_pred.detach()  # 切斷 Model 1 的計算圖
```

### ICNN 的凸性保證機制

`ICNNPriceNetwork` 中從 K → output 路徑的權重透過 `softplus(weight)` 強制非負，
確保 ∂²C/∂K² ≥ 0 在任何收斂狀態下都成立。這是**架構層級的硬保證**，不依賴 loss penalty。

---

## 最新訓練結果（2026-02-25）

| 項目 | 值 |
|------|---|
| CLI 指令 | `python ../model2_research/train_dupire.py --on_gpu --use_icnn --use_model1` |
| yATM (dataset mean) | 0.005965 (min=0.000028, max=0.128450) |
| 訓練時間 | ~5.5 min (RTX 4060, CUDA, float64) |
| 參數量 | 26,756 (Price: 13,123 + LocalVol: 13,633) |
| Final Total Loss | **0.0768** |
| Butterfly violation | **0.000** (全 5000 epochs) |
| Best Total Loss | 0.070 |
| Validation local vol std | 0.0032 |
| 權重存檔 | `models/DupireModel.pt` |
| 訓練 log | `logs/dupire_20260225_212117.log` |

---

## 參考文獻

1. Wang & Privault (2022/2025) — Deep Self-Consistent Learning of Local Volatility, arXiv:2201.07880
2. Amos et al. (2017) — Input Convex Neural Networks, arXiv:1609.07152
3. Bae, Kang & Lee (2024) — Option Pricing and Local Vol by PINN, Computational Economics 64:3143

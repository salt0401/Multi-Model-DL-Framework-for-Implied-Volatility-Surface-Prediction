# Model 2 訓練細節與潛在優化方向

> 本文件整合了 Model 2 的訓練策略、硬體限制緩解方法與未來可能的優化路徑。

---

## Gemini 研究結論摘要

### 核心診斷

Model 1 的 74% butterfly violation 根本原因：5 個 SmileModel 的 **Softmax 加權邊界處產生密度扭結 (kinks)**，導致 ∂²w/∂logm² 極度不穩定 → Dupire 分母崩潰。

### 推薦排序

| 排名 | 方案 | 核心技術 | 適合度 |
|:---:|------|---------|:---:|
| 🥇 **首選** | **ICNN 嚴格凸性約束** (B+F) | Input-Convex NN 硬保證 ∂²C/∂K² ≥ 0 | ⭐⭐⭐⭐⭐ |
| 🥈 次選 | GNO 圖神經算子 (E) | 離線訓練全局映射，推論瞬間完成 | ⭐⭐⭐⭐ |
| 🥉 三選 | WamOL Dupire PINN (B) | 雙網路 + WamOL 動態權重 | ⭐⭐⭐ |

### 被排除的方案

| 方案 | 排除理由 |
|------|---------|
| **Heston Calibration (C)** | 5 參數表徵力不足以捕捉 TXO 短期陡偏態；有傳統 FFT 替代 |
| **Signature Kernels (新1)** | 路徑相依 PDE 需要完全重構 Model 3 (xLSTM) 資料結構 |
| **Bayesian Neural SDE (新2)** | Euler-Maruyama 軌跡模擬在 float64 + RTX 4060 下算力不可承受 |

---

## 雙路徑比較策略

> Gemini 指出 ICNN 可以吸收 Module A，但我們**同時保留 Module A 作為獨立元件**，
> 以便在 V2 完成後進行兩條路徑的 A/B 比較。

### Path α：ICNN 直接方案（首選）

```
Model 1 → Model 2 (ICNN) → Module D (Greeks) → Model 3/5
           ↑ 架構硬保證 ∂²C/∂K² ≥ 0
           ↑ 同時完成 surface correction + local vol extraction
           ↑ Module A 功能被吸收，不需獨立步驟
```

### Path β：Module A + GNO 方案（次選組合）

```
Model 1 → Module A (Surface Correction) → GNO (次選) → Module D (Greeks) → Model 3/5
           ↑ 獨立的 no-arb 修復步驟        ↑ 離線訓練的全局映射
           ↑ 清洗 74% violation             ↑ 單次 forward pass（速度 100x）
           ↑ 輸出 clean surface              ↑ 需要 A 的 clean data 作為 training target
```

### 比較計畫

| 比較指標 | Path α (ICNN) | Path β (A + GNO) |
|---------|:---:|:---:|
| butterfly violation rate | 應為 0% (架構保證) | 取決於 Module A 修復品質 |
| σ_LV 精度 | 逐日校準 | 全局泛化 |
| 推論速度 | 每日需重新優化 | 瞬間 forward pass |
| 工程複雜度 | 較低（單一模組） | 較高（兩模組串接） |
| 訓練數據需求 | 無需配對標籤 | 需要 (IV, clean local vol) pairs |

### Module A 的雙重角色

1. **在 Path β 中**：作為獨立前處理器，接在 Model 1 和 GNO 之間
2. **作為 ICNN 的 baseline**：比較「soft correction + GNO」是否能達到「hard guarantee ICNN」的精度
3. **為 GNO 提供訓練資料**：GNO 需要大量純淨的 (IV surface, local vol surface) pairs，Module A 是唯一能提供這些的工具

> **結論：** Module A 不論最終選哪條路徑都有價值。先實作 Module A + ICNN (V2)，
> 然後用 Module A 清洗的歷史資料訓練 GNO (V3)，最後做 A/B 比較。

### Module D 保留為獨立步驟 (Model 2.5)

Gemini 建議的 3 個高價值衍生特徵（從 Model 2 的無套利 local vol 計算）：

| 特徵 | 公式 | 預測價值 |
|------|------|---------|
| **Vanna** | ∂²C/∂S∂σ | 捕捉偏態 + S-vol 負相關，預測單邊暴跌 |
| **Volga** | ∂²C/∂σ² | Vega 凸性，量化尾部風險預期 |
| **∂σ_LV/∂K** | local vol 對 K 的斜率 | 瞬間局部偏態（不受積分平滑污染） |

Model 3 input: 12 dim → **15 dim** (+Vanna, +Volga, +∂σ_LV/∂K)

---

## 最終 Pipeline 架構

```
Model 1 (SSVI+NN)     w(τ, logm), ∂w/∂τ, ∂w/∂logm, ∂²w/∂logm²
    │                  含 74% butterfly violation
    ▼
Model 2 (ICNN)         σ_LV(K,T), RN density q(K,T)
    │                  硬保證 ∂²C/∂K² ≥ 0 → 0% butterfly violation
    │                  同時完成 surface correction + local vol extraction
    ▼
Module D (Greeks)      Vanna, Volga, ∂σ_LV/∂K (from clean local vol)
    │                  不需要訓練，closed-form / autograd 計算
    ▼
Model 3 (xLSTM)        15-dim input → tv_ratio prediction
Model 5 (DDPM)         condition_dim = 11 + local vol grid
```

---

## 三階段實作計畫

### V1 — 原型驗證（軟約束 PINN）

**目標**：先用標準 MLP 跑通 Dupire pipeline，驗證 pipeline 連接

- 保留現有 `dgm.py` 的 MLP 結構
- 將 BS PDE loss 替換為 WamOL 調控的 Dupire PDE loss
- 用 2021 test 中 26% 未違反的「健康數據」驗證 σ_LV 萃取正確性
- 打通 Model 2 → Model 3 資料流

**產出**：`src/dupire_pinn.py` + `src/train_dupire.py` (V1)

---

### V2 — 核心架構確立（ICNN 植入）

**目標**：從根本消除 butterfly violation

- 將 V1 的標準 MLP 替換為 ICNN
  - K → C(K) 路徑上所有權重矩陣強制非負 (`softplus(weight)`)
  - 激勵函數限制為單調遞增 (ReLU / Softplus)
  - 參考 ARBITER 模型的 Legendre 共軛頭
- Loss 大幅簡化：蝶式約束已由架構保證，不需 soft penalty
- 驗證：全域測試網格上 Dupire 分母嚴格為正

**產出**：`src/dupire_icnn.py` + 更新 `train_dupire.py`

---

### V3 — 特徵擴張 + 算子探索

**目標**：啟動 Module D + 探索 GNO

- **Module D**：
  - 從 V2 的純淨 ICNN local vol 計算 Vanna, Volga, ∂σ_LV/∂K
  - 合併進 Model 3 (15-dim) 重新訓練
- **GNO 探索**（如果 V2 成功）：
  - 用 V2 清洗的 2014-2020 歷史資料作為 target
  - 訓練離線 GNO 模型，驗證是否能取代 V2 的逐日校準
  - 若成功 → 推論速度 100x 提升

**產出**：`scripts/compute_greeks.py` + `src/dupire_gno.py` (experimental)

---

## 各方案的保留策略

| 方案 | 狀態 | 保留位置 | 啟動條件 |
|------|------|---------|---------|
| **ICNN (B+F)** | 🟢 已完成 (V1-V3) | `src/dupire_pinn.py`, `src/module_d.py` | 無（現為主力生產模型） |
| GNO (E) | 🟡 保留研究 | `model2_research/candidates/gno/` | 未來探索 |
| WamOL PINN (B) | 🟢 已完成 (V1) | 整合至 `src/dupire_pinn.py` | 無（原型驗證完畢） |
| Heston (C) | ⚪ 存檔 | `model2_research/candidates/heston/` | 暫不開發 |
| Signature (新1) | ⚪ 存檔 | `model2_research/candidates/signature/` | 暫不開發 |
| Neural SDE (新2) | ⚪ 存檔 | `model2_research/candidates/neural_sde/` | 暫不開發 |

---

## 硬體注意事項（來自 Gemini）

1. **隔離 autograd 計算圖**：Model 2 算 Dupire 二階導數時，每個 batch 完成後必須 `detach()` + 垃圾回收，防止 VRAM 碎片化
2. **混合精度策略**：NN forward pass 可用 float32，僅在 Dupire PDE 算子和 loss gradient 時升回 float64（節省約 45% VRAM）
3. **參數預算**：ICNN 雙網路各 ~9K params (3 layers × 64 neurons)，總計 ~18K，遠在 100K 限制內

---

## 參考文獻（重要的）

1. Wang & Privault (2022/2025) — Deep Self-Consistent Learning of Local Volatility, arXiv:2201.07880
2. WamOL (ICAIF 2024) — Physics-Informed NN for Intraday IV Surface, arXiv:2411.02375
3. Wiedemann et al. (ICLR 2025) — Operator Deep Smoothing for IV, arXiv:2406.11520
4. ARBITER — Risk-Neutral Operator + Legendre conjugate head
5. Amos et al. (2017) — Input Convex Neural Networks, arXiv:1609.07152
6. Bae, Kang & Lee (2024) — Option Pricing and Local Vol by PINN, Computational Economics 64:3143

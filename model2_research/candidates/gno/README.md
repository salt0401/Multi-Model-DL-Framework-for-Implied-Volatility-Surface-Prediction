# GNO 圖神經算子 (方向 E) — Path β 核心

> 🟡 與 Module A 組成 **Path β** (比較路徑)，對照 ICNN (Path α)
> 啟動條件：Module A 清洗出足夠歷史 local vol 資料後

## 核心概念

學習 **IV surface → Local Vol surface** 的全局映射算子。
一次訓練，任何新 IV surface 只需一次 forward pass 即得 local vol。

## 與 PINN 的本質差異

| | PINN (V1/V2) | GNO (V3) |
|---|---|---|
| 訓練 | 每日逐日校準 | 離線一次性訓練 |
| 推論 | 需要迭代優化 | 單次 forward pass |
| 數據需求 | 無需配對標籤 | 需要大量 (IV, local vol) pairs |
| 泛化 | 僅限當日 | 跨日期泛化 |

## 啟動前提

1. V2 (ICNN) 成功 → 產出 2014-2020 歷史無套利 local vol
2. 用 ICNN 輸出作為 GNO 的訓練 target
3. 驗證 GNO 精度 ≈ ICNN 後取代之

## 核心參考

- Wiedemann et al. (ICLR 2025), arXiv:2406.11520
- 鄰域截斷技術：沿 τ 軸限制入度鄰域 ρ̄=0.3 → 架構自帶 calendar no-arb

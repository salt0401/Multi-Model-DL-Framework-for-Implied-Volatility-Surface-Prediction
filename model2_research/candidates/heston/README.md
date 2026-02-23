# Heston Calibration PINN (方向 C) — 存檔

> ⚪ 暫不開發。5 參數表徵力不足以捕捉 TXO 短期陡偏態。

## 排除理由

1. Heston 的 5 參數 (v₀, κ, θ, σ_v, ρ) 對 TXO 短期 smile 表現力不足
2. 傳統 FFT calibration 已經很快，PINN 優勢不夠大
3. 輸出是低維參數向量，不如完整 local vol surface 對 Model 3/5 有用

## 如果未來需要

- 可參考 Heston PDE 損失設計
- 5 維 conditioning vector 可作為 Model 3 的額外特徵（但不如 Vanna/Volga 直接）

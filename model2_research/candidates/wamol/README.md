# WamOL Dupire PINN (方向 B) — V1 原型

> 🟡 作為 V1 過渡原型使用，驗證 Dupire pipeline 後將升級至 ICNN

## 核心設計

- 雙網路：Price Network N_P(K,T) + Local Vol Network N_LV(K,T)
- 各 3 hidden layers × 64 neurons, ~9K params/network
- N_LV 輸出層用 Softplus → σ_LV > 0

## Loss (WamOL 動態權重)

```
L = λ_fit · MSE(N_P, P_target)
  + λ_PDE · MSE(∂N_P/∂T - ½K²·N_LV²·∂²N_P/∂K², 0)   # Dupire PDE
  + λ_cal · ReLU(-∂N_P/∂T)                              # Calendar
  + λ_but · ReLU(-∂²N_P/∂K²)                            # Butterfly

WamOL 三段式：
  1. 自適應 m_β：violation 區域提升 penalty
  2. 損失平衡：按梯度比例重分配 λ
  3. 時間衰減 ζ：最新數據更高權重
```

## 用途

在 V1 階段：
1. 驗證 Model 1 → Model 2 資料流
2. 確認 σ_LV 萃取在 26% 健康數據上正確
3. 為 V2 (ICNN) 建立 baseline 比較

## 參考

- Wang & Privault (2022/2025), arXiv:2201.07880
- WamOL (ICAIF 2024), arXiv:2411.02375

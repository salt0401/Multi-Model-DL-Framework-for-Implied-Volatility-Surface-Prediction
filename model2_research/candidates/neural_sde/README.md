# Bayesian Neural SDE (新興方向 2) — 存檔

> ⚪ 暫不開發。Euler-Maruyama 軌跡模擬在 float64 + RTX 4060 下算力不可承受。

## 排除理由

神經 SDE 每次迭代都需要數值求解器做連續時間軌跡模擬。
Langevin-type 後驗採樣在 float64 雙精度限制下計算成本過高。
雖然能產出帶信任區間的 local vol（對風險管理有價值），但硬體不允許。

## 參考

- Bayesian Neural SDEs for robust bounds on local vol surfaces
- Langevin-type posterior sampling

# Signature Kernels (新興方向 1) — 存檔

> ⚪ 暫不開發。需要完全重構 Model 3 (xLSTM) 資料結構。

## 排除理由

粗糙路徑理論 + Signature Kernels 能捕捉精確的路徑相依性，
但輸出的 path-dependent PDE 解與現有 xLSTM 期望的「靜態 local vol 網格」格式完全不相容。
整合需要徹底重設計 Model 3 的資料結構。

## 參考

- Rough Path Theory + Signature-based PPDE solvers
- 適合學術研究但工程整合成本過高

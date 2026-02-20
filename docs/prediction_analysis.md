### 二、問題的根本原因（從程式碼和資料中找到的）

#### 原因 4：乘法架構放大了 Prior 的錯誤（已修正 → 改為加法架構）

原始設計在 `src/model.py` 中使用乘法：
```python
output = output_Prior * output_NN   # 舊版（已廢棄）
output = output_Prior + yATM * output_NN  # 新版（加法架構）
```

舊的乘法設計 `SSVI * SmileModel` 有兩個問題：(1) 如果 SSVI 的輸出形狀就是錯的（rho 方向反了），NN 再怎麼乘也只能調整幅度，無法修正方向；(2) 乘法的 product rule 在二階導數中產生交叉項，導致 butterfly loss 與 SSVI 參數之間形成反饋迴路，訓練在第 2 epoch 就爆炸。2026-02-20 A/B 實驗確認加法架構全面優於乘法架構（詳見 `logs/architecture_comparison.json`）。

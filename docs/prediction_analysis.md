## 完整分析：為什麼預測結果跟實際差這麼多

### 一、三張圖片的問題

#### 1. Base Model Training (105 epochs, early stopped at epoch 55)

看訓練曲線，**在 epoch ~60 之後發生了嚴重的梯度爆炸**：

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 55 (best) | 2.377 | **1.914** |
| 67 | **7.047** | 1.966 |
| 69 | **34.33** | 1.994 |
| 70 | **352.8** | 2.939 |
| 74 | 4.503 | **663.6** |
| 77 | **1029.2** | 3.522 |
| 100 | 22.27 | **3779.1** |

更關鍵的問題：**best epoch 的 val loss（1.914）跟初始 epoch 1 的 val loss（1.925）幾乎一樣**。整個訓練過程中 loss 只下降了 ~0.6%。模型根本沒有學到有意義的東西。

#### 2. HyperIV Training (69 epochs, early stopped at epoch 19)

Best epoch 是 19，之後 val loss 不斷惡化。在 epoch ~55 處出現了劇烈 spike（val loss 從 ~0.00008 暴漲到 1.02）。雖然 train loss 最終恢復，但模型選用的是 epoch 19 的權重，代表後續 50 個 epoch 的訓練完全浪費。

#### 3. Predicted vs Observed IV Smiles

這張圖是最直觀的問題展示：

- **藍線（預測）的斜率方向完全反了**。台灣加權指數選擇權（TXO）應該呈現**左偏斜**（left skew）——OTM put 的 IV 高於 OTM call，這是股票指數選擇權的普遍特徵（fear premium）。但預測線向右上方翹起。
- **幅度差異巨大**：在 tau=0.0027（極短天期）中，觀測 IV 在 0.2~0.8 範圍，預測卻只有 0.05~0.1。差了 4~10 倍。
- **預測曲線幾乎是平的**：對於 tau=0.5534 和 tau=0.7452，藍線接近水平的 0.15，完全沒有捕捉到觀測資料中明顯的下降趨勢。

---

### 二、問題的根本原因（從程式碼和資料中找到的）

#### 原因 1：SSVI 參數初始化錯誤 → 偏斜方向搞反

在 `src/model.py:23`：
```python
self.raw_rho = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
```

`rho = tanh(0.5) ≈ +0.46`。在 SSVI 模型中，**正的 rho 代表右偏斜**（right skew），但股票指數選擇權需要 **負的 rho**（left skew）。初始值從一開始就指向了錯誤方向，加上訓練只改善了 0.6%，rho 根本沒機會翻轉到正確的負值。

#### 原因 2：Loss 權重嚴重偏向約束，犧牲了擬合精度

在 `src/config.ini:15`：
```
loss_weights = 1,1,10,10,10,10
```

6 個 loss 組件中，RMSE 和 MAPE 權重各為 1，但 4 個約束（calendar, butterfly, linear, upperbound）權重各為 10。**總權重 42 中有 40 是約束項（95%）**。模型幾乎把所有優化能力花在滿足數學約束上，而不是擬合實際資料。

諷刺的是，測試結果顯示約束也沒滿足好：
- Calendar 違規：105/197 = **53%**
- Butterfly 違規：77,256/92,270 = **84%**

**既沒擬合好資料，約束也沒有效執行。**

#### 原因 3：y_atm 插值品質低落（跨日期混合）

在 `src/dataset.py:247`：
```python
atm_idxs = fit_data.groupby(['tau']).apply(
    lambda x: (x['underlying'] - x['strike_price']).abs().idxmin()
)
```

這個函數對每個 unique tau，在**所有日期中**只找一個最接近 ATM 的資料點做插值。由於指數每天變動，某個 tau 的 ATM 可能來自完全不同日期的市場條件。從實驗資料可以看到，同一天（2025-01-02）所有 row 的 `y_atm` 都是 0.012，但實際 `total_var` 卻只有 0.0004~0.0016。y_atm 比真實值大了 **8~30 倍**。

#### 原因 4：乘法架構放大了 Prior 的錯誤

在 `src/model.py:128`：
```python
output = output_Prior * output_NN
```

Base Model 的設計是 `SSVI * SmileModel`。如果 SSVI 的輸出形狀就是錯的（rho 方向反了），NN 再怎麼乘也只能調整幅度，無法修正方向。這是一個架構層面的限制。

#### 原因 5：Fine-tuning 學習率太低

在 `src/train.py:161`：
```python
optimizer = setup_finetune_optimizer(model, ..., base_lr=learning_rate * 0.1, ...)
```

Fine-tuning 時，預訓練層的學習率是 `0.001 * 0.1 = 0.0001`。SSVI 的參數（rho, eta, gamma）屬於預訓練層，以這麼低的學習率，在爆炸之前的 55 個 epoch 內根本不足以做出有意義的調整。

---

### 三、總結

| 問題 | 位置 | 影響 |
|------|------|------|
| SSVI rho 初始化為正值 | `model.py:23` | IV 偏斜方向完全相反 |
| Loss 權重偏向約束 95% | `config.ini:15` | 擬合精度被嚴重犧牲 |
| y_atm 跨日期混合插值 | `dataset.py:247` | ATM 基準值偏差 8-30 倍 |
| 乘法架構 Prior × NN | `model.py:128` | 無法修正 Prior 方向錯誤 |
| Fine-tune 學習率過低 | `train.py:161` | SSVI 參數來不及修正就爆炸 |
| 梯度爆炸 | 訓練 epoch 60+ | 模型只訓練了約 55 個有效 epoch，改善僅 0.6% |

核心問題可以用一句話概括：**SSVI 從錯的方向出發、學太慢、又在學到正確方向之前就爆炸了，導致最終保存的模型幾乎等於沒訓練。**

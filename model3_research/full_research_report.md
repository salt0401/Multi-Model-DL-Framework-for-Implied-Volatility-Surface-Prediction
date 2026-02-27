# Advanced Time Series Models for IV Surface Adjustment — Full Research Report

> Research date: 2026-02-20
> Context: Replace GRU in Model 3 (TVAdjustmentModel) for TXO IV Surface Prediction system

---

## 1. Problem Specification

- **Task**: Predict adjustment ratio alpha for base model (eSSVI+NN) during structural breaks
- **Input**: 20-day sliding window x 16 features per timestep
- **Features**:
  - 6 base features: `vix_change`, `underlying_return`, `logm`, `tau`, `tv_pred`, `itm_otm`
  - 6 enhancement features: `sp500_return`, `iv_term_slope`, `iv_skew`, `vrp_20d`, `futures_basis_pct`, `rv_20d`
  - 4 Model 2 Greeks (Module D): `local_vol`, `vanna`, `volga`, `lv_gradient_K`
- **Target**: tv_ratio = tv_true / tv_pred (multiplicative correction)
- **Application**: tv_adjusted = tv_base * alpha
- **Key challenge**: Crisis periods are rare but critical (2001/09, 2008/10, 2016/05)

---

## 2. Detailed Model Evaluations

### 2.1 Temporal Fusion Transformer (TFT)

**Paper**: Lim et al. (2021), Google Research, arXiv:1912.09363
**Published**: International Journal of Forecasting, Vol 37, Issue 4

**Architecture**:

```
Input Features
    |
    v
Variable Selection Network (VSN)
    |  - Softmax gating per timestep
    |  - Learns feature importance dynamically
    v
Gated Residual Networks (GRN)
    |  - ELU activation + skip connections
    |  - Gate suppresses irrelevant paths
    v
LSTM Encoder (past) / LSTM Decoder (future)
    |
    v
Static Enrichment Layer
    |  - Enriches temporal features with static context
    v
Interpretable Multi-Head Attention
    |  - Shared V weights across heads (interpretable)
    |  - Attention weights = temporal importance
    v
Position-wise Feed-Forward
    |
    v
Quantile Output (10th, 50th, 90th percentiles)
```

**Key Innovation — Variable Selection Network**:

- Each timestep has a softmax gate over all input features
- Model learns: "in crisis, attend to VIX and returns; in calm, attend to tau and logm"
- This IS regime switching behavior, learned end-to-end

**Benchmark Results**:

- Outperformed DeepAR by 36-69%
- P50 loss 7% lower, P90 loss 9% lower than next best model
- Attention automatically identifies 2008 crisis periods with high weights

**Interpretability Features**:

1. Feature importance ranking (global and per-instance)
2. Temporal attention patterns (which past days matter)
3. Regime detection via attention pattern deviation from average

**Implementation Options**:

- `pytorch-forecasting` library (full TFT with data pipeline)
- Custom implementation (more control, our approach)

**Training Cost**: Hours on single GPU for our dataset size

---

### 2.2 xLSTM (Extended LSTM)

**Paper**: Beck et al. (2024), NXAI Lab (Sepp Hochreiter's group)
**Variants**: sLSTM (scalar) and mLSTM (matrix)

**sLSTM Architecture**:

- Exponential gating: exp(input_gate), exp(forget_gate)
- Normalizer state prevents numerical overflow
- Better gradient flow than sigmoid-based gates
- Still sequential (like GRU/LSTM)

**mLSTM Architecture**:

- Matrix memory C (not vector) — stores richer history
- Query-Key-Value retrieval (like Transformer attention)
- Covariance update rule: C_t = f_t *C_{t-1} + v_t* k_t^T
- Parallelizable (unlike sLSTM)

**Benchmark Results (xLSTMTime)**:

- Matches or exceeds Transformer-based and linear methods on LTSF benchmarks
- xLSTM-TS outperformed TCN, N-BEATS, TFT, N-HiTS, TiDE on stock price direction
- Particularly strong on datasets where temporal order matters

**Installation**:

```bash
pip install xlstm          # Official (NX-AI)
pip install torchxlstm     # Pure PyTorch alternative
```

**Integration Approach**:

- Replace `nn.GRU` with mLSTM block
- Keep existing TemporalAttention + FC + SquarePlus
- Minimal code change, maximal compatibility

---

### 2.3 Mamba / State Space Models

**Paper**: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

**Core Mechanism**:

- Selective state space: dynamically decide what to remember/forget
- Linear complexity O(n) vs Transformer's O(n^2)
- Hardware-aware parallel scan algorithm

**Why NOT suitable for our problem**:

1. **seq_len=20 is too short**: Mamba's advantage is processing seq_len > 1000 efficiently
2. **Fixed-size latent state**: compresses history into fixed vector, potentially losing crisis signals
3. **Research finding**: "Mamba excels at long-range dependencies, Transformer at short-term dynamics" — our problem is the latter
4. **S-Mamba paper**: showed advantages mainly on long sequences (96-720 steps)

**Hybrid option (SST / MoU)**:

- Multi-scale hybrid with Mamba for long-range + Transformer for short-range
- Overkill for seq_len=20; added complexity not justified

**References**:

- "Is Mamba Effective for Time Series Forecasting?" (Neurocomputing 2024)
- MambaTS (arXiv:2405.16440)
- SST: Multi-Scale Hybrid Mamba-Transformer (CIKM 2024)

---

### 2.4 Neural SDE / Neural ODE

**Libraries**: torchsde (Google Research), torchdiffeq, torchcde

**Theoretical Appeal**:

- Volatility IS a stochastic process — SDE is the natural formulation
- Neural SDE: parameterize drift f(x,t) and diffusion g(x,t) as neural networks
- Continuous-time modeling, can handle irregular timestamps
- Natural uncertainty quantification from diffusion term

**Practical Issues**:

1. **Training instability**: SDE solvers have high variance gradients
2. **Speed**: Adjoint method backprop 5-10x slower than standard
3. **Discrete jumps**: SDE is continuous — crisis "jumps" need jump-diffusion extensions
4. **Federal Reserve finding (2025)**: Nonlinear regime-switching models (THAR, STHAR) consistently outperform ML models for volatility forecasting across GFC and COVID

**Verdict**: Academically interesting but engineering cost too high for marginal benefit

---

### 2.5 Transformer Variants (PatchTST, iTransformer, Crossformer)

**PatchTST** (ICLR 2023):

- Segments time series into patches (typically 16-32 steps per patch)
- seq_len=20 → only 1-2 patches → attention has nothing to work with
- Designed for seq_len >= 96

**iTransformer** (2024):

- Inverts axis: treats each variable as a token
- 6 variables = 6 tokens → workable but designed for >20 variables
- Cross-variate dependencies are not the bottleneck in our problem

**Crossformer** (2023):

- Cross-scale attention for variable-length series
- Overkill for fixed-length seq_len=20

**TimeMixer** (ICLR 2024):

- Multiscale decomposition + MLP mixing
- Needs longer sequences for meaningful decomposition

---

### 2.6 Foundation Models (TimesFM, Chronos, Moirai)

**Not applicable because**:

- Designed for zero-shot general forecasting
- Our target (tv_ratio) is highly domain-specific
- Cannot encode the Model 1 dependency (tv_pred as input feature)
- Fine-tuning cost exceeds training a task-specific model from scratch

---

## 3. Summary Comparison

| Model | Short seq (20) | Regime detection | Interpretability | Implementation | GPU hours |
|-------|---------------|------------------|-----------------|----------------|-----------|
| GRU+Attn (current) | Good | Poor | Low | Existing | ~2h GPU |
| **TFT** | **Excellent** | **Excellent** | **Excellent** | Medium | ~3-4h GPU |
| **xLSTM (mLSTM)** | **Excellent** | Good | Low | **Minimal** | ~2h GPU |
| Mamba | Mediocre | Mediocre | Low | Medium | ~1h GPU |
| Neural SDE | Good | Poor (continuous) | Medium | Hard | ~12h GPU |
| PatchTST | Poor (too short) | N/A | Low | Easy | ~1h GPU |
| Foundation Model | N/A | N/A | N/A | N/A | N/A |

## 4. Decision

**Implement both TFT and xLSTM**, train on identical data, compare metrics.

- TFT: potential for best performance + interpretability
- xLSTM: minimal risk, guaranteed improvement over GRU

> **Update (2026-02-27)**: A complete **12-way comparison** (3 architectures × 4 optimizers) has been conducted on 16-dim input with a strictly held-out 2021 test set. Key results: TFT dominates all top 4 positions regardless of optimizer (CPR #1, AdamW #2, Adam #3, CWD #4). xLSTM underperformed expectations, trailing GRU+AdamW on all metrics. CPR is architecture-dependent — excellent for TFT (#1) but counterproductive for GRU (#12). **TFT + CPR** confirmed as the final Model 3 architecture (Test RMSE 0.1558, MAPE 9.51%). See `README.md` for the full 12-way table.

---

# Part II: 參數級自適應正則化與 Model 3 過擬合動力學研究

> 本報告基於通用正則化理論，結合本專案 Model 3（TXO 隱含波動率調整模型）的具體過擬合問題，分析各種前沿正則化方法的適用性，並給出具體的實施建議。

## 專案背景：Model 3 的過擬合現狀

Model 3 是一個時間序列調整模型，用於在市場結構性斷裂（如 COVID-19 崩盤）期間修正 Model 1（Base eSSVI+NN）的預測。完整 12-way 比較（3 架構 × 4 optimizer）結果如下（Top 5）：

| Rank | 架構 + Optimizer | 參數量 | Test RMSE | Test MAPE |
|:----:|-----------------|:------:|:---------:|:---------:|
| 1 | TFT + CPR ⭐ | 318K | **0.1558** | **9.51%** |
| 2 | TFT + AdamW | 318K | 0.1590 | 9.75% |
| 3 | TFT + Adam | 318K | 0.1592 | 9.85% |
| 4 | TFT + CWD | 318K | 0.1608 | 9.75% |
| 5 | GRU + AdamW | 59K | 0.1628 | 9.70% |

> **重要發現**：CPR 對高容量架構（TFT, 318K 參數）效果顯著（#1），但對低容量架構（GRU, 59K 參數）反而有害（排名 #12，MAPE 11.28%）。這表明 CPR 的約束參數機制需要足夠的模型容量才能正常運作。見 `README.md` 完整 12-way 表格。

**關鍵特徵**：

- 輸入：16 維特徵 × 20 步時間窗口（seq_len=20）
- 目標：tv_true / tv_pred 的比率（接近 1.0 的正數）
- 訓練集 ~80%、驗證集 ~20% 時間序列切分
- 損失函數：MSE + MAPE + KDE 加權（危機期樣本加權）

---

## 一、過擬合的動力學：損失幾何學與訓練相變

### 1.1 理論框架

神經網路的訓練表現出明顯的**相變（Phase Transitions）**。訓練過程主要由兩個阶段構成 [10]：

- **優化阶段**：訓練損失與驗證損失同步下降，模型提取數據中最顯著的泛化信號
- **正則化阶段**：驗證損失觸底，但訓練損失繼續下降 — 模型開始記憶噪聲

從參數軌跡的角度：

- 優化阶段中，參數在損失景觀中**垂直下降**，向極小值流形高效移動
- 正則化阶段中，參數開始**沿流形橫向震盪**，在不同 mini-batch 間調和矛盾的噪聲信號 [10]

### 1.2 Model 3 的具體表現

觀察三個模型的 loss curves（見 `figures/loss_curves_3way.png`）：

- **GRU & xLSTM**：驗證損失在 ~epoch 50-80 觸底，之後 train loss 繼續從 ~0.08 下降到 ~0.055，但 val loss 停滯在 ~0.15-0.16，形成典型的相變分離
- **TFT**：由於參數量最大（265K vs 39K），其 train loss 下降最快（到 ~0.023），但這也意味著其**過擬合餘量最大**，gap 高達 6.9 倍

**直接原因**：當前使用 **plain Adam（無任何 weight decay）**，模型參數的幅度在正則化阶段不受任何約束，可以自由增長以擬合噪聲。這是 overfitting 的第一大可修復因素。

### 1.3 權重幅度與噪聲擬合的數學機制

當網路試圖擬合隨機噪聲時，映射函數必須高度非線性化 [17]。要在微小的輸入區間內產生劇烈的函數值波動，多項式的係數（即權重參數）必須在幅度上顯著增長 [17]。

因此，如果某個參數在驗證損失觸底後幅度突然加速增長，該參數極大概率正在參與噪聲記憶。傳統的全局 L2 正則化試圖壓制所有參數增長，但也會傷害有價值的參數 — 這正是我們需要**參數級自適應正則化**的原因。

---

## 二、記憶的架構分層與空間定位

### 2.1 理論：記憶的層級差異

最新研究表明 [19]：

- **記憶主要發生在深層（Deeper Layers）**
- 淺層充當穩健的特徵提取器，受記憶影響極小
- 深層具備高度的函數靈活性，為了消除剩餘誤差會極度扭曲決策邊界

### 2.2 Model 3 架構中的分層映射

以 xLSTM 模型為例（39K 參數最少，overfitting 最輕）：

```
輸入 (batch, 20, 12)
    ↓
mLSTM Layer 0  ← 主要特徵提取，memorization 風險低
    ↓ LayerNorm + Dropout
mLSTM Layer 1  ← 更高的 memorization 風險
    ↓
TemporalAttention (4-head)  ← 學習時間步重要性
    ↓
FC: 64→32→ReLU→Dropout→1  ← **最高的 memorization 風險**
    ↓
SquarePlus (確保正輸出)
```

根據文獻，FC 層是最容易過擬合的部分，因為它直接接觸損失函數，有最大的自由度去擬合個別樣本的噪聲。

### 2.3 軌跡回溯："擦除"技術

實證研究證明 [19]：在驗證損失觸底後，將網路最後幾層的權重回溯到最佳 epoch 的狀態，可以恢復泛化能力。

**🟢 本專案適用性：高**

- 對我們的小模型（39K-265K），FC 層的參數量佔比顯著
- 可以定期將 FC 層權重重置為最佳 epoch 狀態，消除後續 epoch 積累的噪聲記憶
- **已實作**：`train_models.py` 中的 `--layer_reset` flag，每 20 epochs 回溯一次

---

## 三、參數級信號與噪聲的量化指標

### 3.1 梯度信噪比（GSNR）

GSNR 定義為參數梯度期望的平方與方差之比 [4]：

```
GSNR(θ_i) = E[∂L/∂θ_i]² / Var(∂L/∂θ_i)
```

- **高 GSNR**：不同 mini-batch 上梯度方向一致 → 正在學習泛化 pattern
- **低 GSNR**：梯度均值接近零但方差極大 → 被噪聲拉扯，盲目震盪

GSNR 可用於構建 per-parameter dropout mask，選擇性丟棄低 GSNR 參數 [27]。

**🔴 本專案適用性：低**

- 需要在多個 mini-batch 上累積梯度統計（均值和方差 buffer），實作複雜
- 本專案的問題不是 domain generalization（GSNR dropout 文獻的主要場景 [27]），而是 single-task overfitting
- GSNR 文獻（ICCV 2023）主要在大型視覺模型（ResNet/ViT）上驗證，對 39K 參數的 RNN/Attention 效果未經驗證
- 已選用的 CWD/CPR 方法用更簡單的機制達到類似目標：限制無益參數的權重增長

### 3.2 參數敏感度與高階曲率

Hessian 矩陣的特徵值可以區分記憶參數和泛化參數 [10]：

- 記憶參數 → 高 Hessian 特征值（損失對微擾敏感）
- 泛化參數 → 低 Hessian 特征值（位於平坦區域）

這個概念被 EWC（見第七章）用 Fisher Information Matrix 近似 — Fisher diagonal 本質上就是 Hessian diagonal 的期望值。

**🟢 本專案適用性：高（透過 EWC）**

- 我們不直接計算 Hessian（太貴），但透過 Fisher diagonal 來近似參數重要性
- **已實作**：`train_models.py` 中的 `compute_fisher_diagonal()` + `--ewc` flag

---

## 四、動態與參數級自適應正則化方法

### 4.1 自適應權重衰減（AdaDecay）

**原理**：根據參數自身的梯度幅度來按比例確定衰減強度 — 高幅度梯度（正在擬合噪聲）的參數被嚴厲懲罰，穩步下降的參數被放鬆約束 [32]。

**文獻**：Nakamura & Hong, IEEE Access 2019

**🟡 本專案適用性：中等**

- AdaDecay 的核心思想（per-parameter adaptive decay）已被更新的方法（CPR, CWD）所包含和超越
- CPR（NeurIPS 2024）用增廣拉格朗日法達到類似效果但更自動化
- CWD（ICLR 2026）用 update-parameter 方向一致性 mask 實現選擇性衰減
- 因此 **不另外實作 AdaDecay**，其精神已被 CPR 和 CWD 涵蓋

### 4.2 交叉正則化（Cross-Regularization）

**原理**：訓練數據更新模型參數 θ，驗證數據通過梯度下降更新正則化強度 λ [36]：

```
內層：θ ← θ - lr_θ × ∇_θ L_train(θ, λ)
外層：λ ← λ - lr_λ × ∇_λ L_val(θ(λ))
```

外層需要計算 θ 對 λ 的隱式梯度，涉及 Hessian-vector product（二階梯度）[36]。

**文獻**：ICML 2025, OpenReview FzvKazljRc

**🔴 本專案適用性：低**

- **二階梯度計算**：`∇_λ L_val(θ(λ))` 需要 `∂θ/∂λ`，這涉及 Hessian 矩陣。PyTorch 可實現（`torch.autograd.functional.hvp()`），但計算成本大且容易 OOM
- **論文太新**（2025 年 ICML），尚未被廣泛復現驗證
- **需要重寫訓練循環**為嵌套雙層結構，λ 是 per-parameter 向量，管理複雜
- **CPR 已做到類似效果**：CPR 用增廣拉格朗日法「自動調整」每個參數組的 regularization 強度，效果近似但實作只需替換 optimizer

---

## 五、元優化框架：L1 與 L2 正則化的動態選擇

### 5.1 雙層優化（Bi-level Optimization）

**原理**：透過外層循環，讓網路自動學習每個參數應該用 L1（稀疏化到零）還是 L2（衰減但不為零）正則化 [5]：

```
內層：更新 θ（模型參數）
外層：更新 λ（per-parameter 正則化類型和強度）
```

**🔴 本專案適用性：低**

- 同樣需要二階梯度（Hessian-vector product）
- 39K 參數的小模型，L1 vs L2 的差異不如在百萬參數模型中顯著
- 計算開銷可能比訓練本身還大

### 5.2 矩陣變元先驗（AdaReg）

**原理**：用矩陣變元正態先驗（Matrix-Variate Normal Prior）建模權重矩陣中行和列的相關性 [39]：

```
Prior: W ~ MN(0, Σ_row, Σ_col)
Covariance: Σ_row ⊗ Σ_col  (Kronecker product)
```

對一個 64×32 的 FC 層，不是用一個 λ 罰整個矩陣，而是用 64×64（行相關性）⊗ 32×32（列相關性）來表示先驗的協方差結構。

**文獻**：AdaReg, OpenReview

**🔴 本專案適用性：低**

- **模型太淺**：xLSTM 只有 ~5 個參數矩陣（2 層 mLSTM proj + Attention Wq/Wk/Wv/Wo + FC），不值得引入 Kronecker product 建模
- **Kronecker product 計算成本**：每個 epoch 都要估計和更新 Σ_row 和 Σ_col
- **文獻場景不對**：AdaReg 主要在 CNN（卷積核有強空間相關性）上驗證，對 RNN/Attention 未有可靠的實驗結果

---

## 六、貝葉斯 Spike-and-Slab 先驗

### 6.1 理論

Spike-and-Slab 是一種混合先驗分布 [43]，將每個參數分為兩類：

- **Spike（尖峰）**：集中在 0 附近（如狄拉克 δ 函數或方差極小的高斯分布）→ 參數「被關掉」
- **Slab（平板）**：寬闊的分布 → 參數自由活動

訓練時通過**變分推斷（Variational Inference）**或期望傳播，持續計算每個參數屬於 Spike/Slab 的後驗概率 [44]：

- 穩定的高 GSNR 參數 → 分配到 Slab，自由學習
- 在 val loss plateau 期間震盪的參數 → 分配到 Spike，強制歸零

### 6.2 自適應秩選擇

對大型模型，Rank-1 BNNs [47] 只在確定性權重的低秩校正項上建模後驗分布，通過自適應秩選擇（ARS）模組凍結噪聲主導的維度。

**🔴 本專案適用性：低**

- **需要變分推斷（VI）**：標準 SGD/Adam 無法訓練 Spike-and-Slab 模型。必須將整個模型改寫為 Bayesian 版本 — 每個參數變成一個分布（均值 + 方差），forward pass 需要取樣（Monte Carlo forward）。這等於重寫 `xLSTMAdjustmentModel` 和 `TFTAdjustmentModel`
- **39K 參數太小**：文獻成功案例都是 22M+ 參數的大型模型（ViT、ResNet-50）。39K 參數模型中，VI 引入的 sampling noise 可能比消除的 overfitting 還大
- **工程成本極高**：預估 2-3 天實作和調試

---

## 七、EWC 與持續學習的啟發

### 7.1 彈性權重鞏固（EWC）

**原理**：在「核心特徵已學習」後限制重要參數的移動 [51]。具體步驟：

1. 訓練到驗證損失最佳點 → 保存參數 θ*
2. 計算 **Fisher Information Matrix** 的對角線：F_i = E[(∂L/∂θ_i)²]
3. 繼續訓練，但加入 Fisher-weighted L2 penalty：

   ```
   L_total = L_task + (λ/2) × Σ_i F_i × (θ_i - θ*_i)²
   ```

高 Fisher Information 的參數（對損失敏感、學到重要特徵的）被嚴格鎖定在 θ* 附近；低 Fisher 的參數（冗餘容量）可以自由調整。

雖然 EWC 原本用於 continual learning 防止 catastrophic forgetting，但其核心機制 — **用二階資訊選擇性鎖定重要參數** — 與在單一任務過擬合阶段的參數級約束邏輯完全一致。

### 7.2 EWC vs Early Stopping

| | Early Stopping | EWC-style |
|---|---|---|
| 機制 | 所有參數同時停止更新 | 重要參數被保護，不重要的繼續學 |
| 粒度 | 全局（一刀切） | 參數級（手術刀式） |
| 理論 | 理論上可能有些參數還沒學完 | 理論上可以繼續降低有意義的 loss |

**🟢 本專案適用性：高**

- Fisher diagonal 計算只需一次完整的 forward-backward pass（~50 batches，幾分鐘）
- 對 39K 模型完全可行，無額外記憶體壓力
- **已實作**：`train_models.py` 中的 `--ewc` flag + `compute_fisher_diagonal()` + `ewc_penalty()`
- 當 early stopping 的 patience 達到一半時自動觸發 Phase 2（EWC 約束繼續訓練）

---

## 八、本專案的實施方案與結論

### 已實施的方法（4 個）

| 方案 | 方法名 | CLI Flag | 文獻 | 改動量 | 結果 |
|:---:|--------|----------|------|:------:|---|
| 1 | **AdamW** | `--optimizer adamw` | ICLR 2019 | 1 行 | 完成 (改善 2.6%) |
| 2 | **CWD**（Cautious Weight Decay）| `--optimizer cwd` | ICLR 2026, arXiv:2510.12402 | `optimizers.py` | 完成 (GRU 最佳, 改善 3.4%) |
| 3 | **CPR**（Constrained Parameter Regularization）| `--optimizer cpr` | NeurIPS 2024, Franke et al. | `optimizers.py` | 完成 (**TFT 最佳**, 突破最佳紀錄) |
| 4 | **EWC**（Fisher-weighted L2）| `--ewc` | PNAS 2017, Kirkpatrick et al. | ~80 行 | 已實作 |
|  | **Layer Resetting**（補充手段）| `--layer_reset` | arXiv:2310.07996 | ~20 行 | 已實作 |

### 排除的方法（4 個）

| 方法 | 排除原因 |
|------|----------|
| **Spike-and-Slab** | 需 Variational Inference，重寫全部模型，39K 參數太小 |
| **GSNR Dropout** | 實作複雜，文獻場景（domain generalization）與我們的 single-task overfitting 不同 |
| **Cross-regularization** | 需二階梯度（Hessian-vector product），ICML 2025 論文尚未廣泛復現 |
| **AdaReg（矩陣變元先驗）** | 需 Kronecker product，模型太淺（~5 個參數矩陣），文獻主要在 CNN 上驗證 |
| **Meta-Regularization Selection** | 需 bi-level optimization，Hessian 計算在小模型上不划算 |

### 結論

本報告的核心論點 — **將正則化從「全局靜態先驗」轉變為「參數級自適應動態約束」** — 在理論上完全正確且前沿。然而，**具體方法的選擇必須匹配模型規模和工程實際**：

- 對我們的 39K-265K 參數小型時間序列模型，最有效的方法是**低侵入性的 optimizer 層級改進**（AdamW/CWD/CPR），而非需要重寫訓練架構的複雜方法
- 最直接的改善是**從 plain Adam 切換到 AdamW**（加入 weight decay），這是目前模型完全沒有的基本正則化
- CPR 和 CWD 提供了更先進的 per-parameter adaptive decay 機制，理論上應優於全局 weight decay
- EWC 提供了「兩階段訓練 + 選擇性參數鎖定」的能力，是 Early Stopping 的精細化替代

---

## 引用文獻

1. Regularization (mathematics) - Wikipedia, <https://en.wikipedia.org/wiki/Regularization_(mathematics)>
2. Reddit: How come large weights = overfitting?, <https://www.reddit.com/r/datascience/comments/c866ka/>
3. GeeksforGeeks: Regularization in Machine Learning, <https://www.geeksforgeeks.org/machine-learning/regularization-in-machine-learning/>
4. Sun et al. "Unleashing the Power of GSNR for Zero-Shot NAS", ICCV 2023, <https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Unleashing_the_Power_of_Gradient_Signal-to-Noise_Ratio_for_Zero-Shot_NAS_ICCV_2023_paper.pdf>
5. "Meta-Regularization Selection", ResearchGate, <https://www.researchgate.net/publication/398036449>
6. "Informative Bayesian Neural Network Priors for Weak Signals", Bayesian Analysis, <https://projecteuclid.org/journals/bayesian-analysis/advance-publication/>
7. GeeksforGeeks: Training and Validation Loss in Deep Learning, <https://www.geeksforgeeks.org/deep-learning/training-and-validation-loss-in-deep-learning/>
8. CMU ML Blog: The Overfitting Iceberg, <https://blog.ml.cmu.edu/2020/08/31/4-overfitting/>
9. "On Regularization of Gradient Descent, Layer Imbalance and Flat Minima", OPT-ML 2020
10. "The Pitfalls of Memorization", arXiv:2412.07684
11. Google Developers: Overfitting — Interpreting Loss Curves, <https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves>
12. MachineLearningMastery: Weight Regularization, <https://machinelearningmastery.com/weight-regularization-to-reduce-overfitting-of-deep-learning-models/>
13. Reddit: Why large weights sign of complex network, <https://www.reddit.com/r/deeplearning/comments/ug1j99/>
14. "Decoding Generalization from Memorization in DNNs", arXiv:2501.14687
15. Medium: Early Stopping — A Simple Guide, <https://medium.com/@piyushkashyap045/>
16. "The Weights Reset Technique for DNNs Implicit Regularization", MDPI, <https://www.mdpi.com/2079-3197/11/8/148>
17. "How Does Label Noise GD Improve Generalization", NeurIPS 2025
18. "Domain Generalization Guided by GSNR of Parameters", arXiv:2310.07361
19. "Sensitivity and Generalization in Neural Networks", Google Research
20. Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019
21. Nakamura & Hong, "Adaptive Weight Decay for Deep Neural Networks", IEEE Access 2019
22. Apple MLR: "Adaptive Weight Decay", <https://machinelearning.apple.com/research/adaptive-weight>
23. "Cross-regularization: Adaptive Model Complexity through Validation Gradients", ICML 2025
24. "Learning Neural Networks with Adaptive Regularization (AdaReg)", OpenReview
25. "On Priors for Bayesian Neural Networks", eScholarship
26. "Learning Sparse DNNs with a Spike-and-Slab Prior", PMC
27. "Generalized Spike-and-Slab Priors for Bayesian Group Feature Selection", JMLR
28. "Spike-and-slab shrinkage priors for structurally sparse BNNs", arXiv:2308.09104
29. "Personalizing Low-Rank BNNs via Federated Learning", arXiv:2410.14390
30. "Posterior and Variational Inference for DNNs with Heavy-Tailed Weights", JMLR
31. Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks", PNAS 2017
32. "MemLens: Uncovering Memorization in LLMs", arXiv:2509.20909
33. "THE PITFALLS OF MEMORIZATION: WHEN MEMORIZATION HURTS GENERALIZATION", OpenReview

### 額外引用（本專案適用性分析）

- Franke et al., "Improving Deep Learning through Constrained Parameter Regularization (CPR)", NeurIPS 2024
- "Cautious Weight Decay", ICLR 2026, arXiv:2510.12402
- "Reset It and Forget It: Relearning Last-Layer Weights", arXiv:2310.07996

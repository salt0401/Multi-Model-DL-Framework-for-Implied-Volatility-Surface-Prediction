# Cautious Weight Decay (CWD) — 詳細分析

**Paper:** "Cautious Weight Decay"
**Venue:** ICLR 2026
**arXiv:** https://arxiv.org/abs/2510.12402
**OpenReview:** https://openreview.net/forum?id=Gwe6gbGng5

---

## 核心概念

### 標準 AdamW 的問題

標準 AdamW 的更新規則：
```
x_{t+1} = x_t - lr * (u_t + λ * x_t)
```
- `u_t`: optimizer 計算的更新方向（來自 gradient momentum）
- `λ * x_t`: weight decay，永遠把參數推向 0

**問題：** weight decay 永遠施加，即使 optimizer 正在把參數推離 0。
此時 decay 和 optimizer 在「打架」，decay 是反效果的。

### CWD 的解法

CWD 只在 optimizer update 和參數「方向一致」時才施加 decay：
```
x_{t+1} = x_t - lr * (u_t + λ * 𝕀(u_t ⊙ x_t ≥ 0) ⊙ x_t)
```

`𝕀(u_t ⊙ x_t ≥ 0)` 是 element-wise indicator，按每個參數維度獨立判斷。

### 直覺理解

逐參數的四種情況：

| x (參數) | u (更新) | u*x | Decay? | 解釋 |
|:---:|:---:|:---:|:---:|------|
| +5.0 | +0.1 | + | ✅ Yes | u 讓 x 變小（x-lr*u），decay 也讓 x 變小 → 方向一致 |
| +5.0 | -0.1 | - | ❌ No | u 讓 x 變大（x-lr*(-0.1)=x+），decay 卻要縮小 → 在打架 |
| -3.0 | -0.2 | + | ✅ Yes | u 讓 x 更負（|x|變大的方向...）|

更精確地說：
- **u > 0, x > 0**: update 減少 x（因為 x -= lr*u），decay 也減少 x → 一致，apply
- **u > 0, x < 0**: update 讓 x 更負（遠離0），decay 把 x 推向 0 → 一致，apply
- **u < 0, x > 0**: update 讓 x 更正（遠離0），decay 把 x 推向 0 → 衝突，skip
- **u < 0, x < 0**: update 讓 |x| 變小，decay 也讓 |x| 變小 → 一致，apply

**簡化理解：** 當 optimizer 想讓參數遠離 0（變大）時，不要 decay 來拖後腿。

---

## PyTorch 實作

### One-line 修改

```python
# 標準 AdamW（pytorch 內部邏輯簡化）
param.data.mul_(1 - lr * weight_decay)  # decay
param.data.add_(update, alpha=-lr)       # gradient step

# CWD 修改：加一行 mask
mask = (update * param.data >= 0).float()
param.data.add_(param.data * mask, alpha=-lr * weight_decay)  # selective decay
param.data.add_(update, alpha=-lr)
```

### 完整 CWD-AdamW Optimizer

```python
class CautiousAdamW(torch.optim.AdamW):
    """AdamW with Cautious Weight Decay (CWD).

    Only applies weight decay when the optimizer update direction
    aligns with the parameter sign (per-coordinate).
    """

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            wd = group['weight_decay']
            if wd == 0:
                continue

            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Update biased first and second moment estimates
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** state['step']
                bc2 = 1 - beta2 ** state['step']
                step_size = lr / bc1

                # Compute Adam update direction
                denom = (exp_avg_sq.sqrt() / (bc2 ** 0.5)).add_(eps)
                update = exp_avg / denom

                # === CWD: selective weight decay ===
                mask = (update * p.data >= 0).to(p.data.dtype)
                p.data.add_(p.data * mask, alpha=-lr * wd)

                # Apply gradient step
                p.data.add_(update, alpha=-step_size)

        return None
```

---

## 實驗結果

### ImageNet（小模型也有效果）

| Model | Params | AdamW | AdamW+CWD | 改善 |
|-------|:---:|:---:|:---:|:---:|
| ViT-S/16 | 22M | 78.84% | 79.45% | +0.61% |
| ResNet50 | 25.6M | 76.30% | 76.68% | +0.38% |
| ViT-B/16 | 86.6M | 80.15% | 80.71% | +0.56% |

### 重要發現
- **不需要新的超參數**：直接沿用原本的 weight_decay 值
- **不需要重新調參**：最佳 λ* 跟 AdamW 幾乎一樣
- **Random mask 反而變差**：證明 sign-alignment 的選擇機制是關鍵
- **所有規模都有效**：從 22M 到 billion-scale

---

## 適用於我們專案的分析

### 優勢
1. **實作極簡**：one-line 修改，不影響現有訓練流程
2. **零額外超參數**：直接用現有的 weight_decay
3. **低風險**：worst case 跟 AdamW 一樣（mask 不會讓事情變更差）
4. **理論基礎**：bilevel optimization 有嚴謹的數學推導

### 潛在限制
1. **我們目前用 Adam 而非 AdamW**：需要先加入 weight_decay
2. **模型很小（39K-265K params）**：CWD 的主要實驗是 22M+ 模型
3. **float64 + sequential scan (xLSTM)**：mask 計算增加的開銷可忽略不計
4. **不直接解決我們的核心問題**：CWD 是讓 weight decay 更有效，
   但不會告訴我們「哪些參數在記憶 noise」

### 結論
CWD 是 **low-cost, low-risk improvement**，值得嘗試但不是我們研究的主軸。
它改善了 regularization 的機制，但不提供 interpretability（不告訴我們為什麼 overfit）。

我們的原創想法（parameter drift analysis）可以跟 CWD 結合：
1. 先用 parameter drift analysis 找出 noise-memorizing parameters
2. 對這些參數用 CWD-style selective decay
3. 根據參數分布選 L1/L2 正則化類型

---

## 參考資料
- [arXiv 2510.12402](https://arxiv.org/abs/2510.12402)
- [ICLR 2026 OpenReview](https://openreview.net/forum?id=Gwe6gbGng5)
- [HTML full paper](https://arxiv.org/html/2510.12402v1)

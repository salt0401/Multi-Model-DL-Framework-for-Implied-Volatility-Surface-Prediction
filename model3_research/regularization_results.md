# Model 3 正則化實驗結果

> 基於 4 種 optimizer 的比較實驗（AdamW, CWD, CPR vs Adam baseline）
> 
> **Status**: 正在使用 16-dim input (含 Model 2 Greeks) 重新訓練。結果待更新。

## Models Under Evaluation

| Model | Optimizer | Dtype |
|-------|-----------|-------|
| TFT + CPR ⭐ | Constrained Parameter Regularization (NeurIPS 2024) | float32 |
| TFT + AdamW | Standard weight decay | float32 |
| GRU + CWD | Cautious Weight Decay (ICLR 2026) | float64 |

*Results will be populated after the current training run completes.*

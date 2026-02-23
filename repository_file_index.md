# Repository File Index

This document provides a comprehensive, file-by-file breakdown of every file currently tracked in the repository.

---

## 1. Project Root & Configuration
| File | Description | Action/Status |
|------|-------------|---------------|
| `.gitignore` | Tells Git which files/folders (like logs, cache) to ignore. | Keep |
| `ARCHITECTURE.md` | Core documentation describing the 5-Model architecture pipeline. | Keep |
| `EXPERIMENT.md` | Log of experiments, design choices, and training runs. | Keep |
| `README.md` | The main project landing page, outlining requirements and model status. | Keep |
| `requirements.txt` | Python dependency locking file (pip install -r). | Keep |

---

## 2. Root `scripts/` Directory (Analysis & Utils)
| File | Description | Action/Status |
|------|-------------|---------------|
| `build_features.py` | Generates enhancement features (e.g. S&P500 returns, realized volatility) to append to the dataset. | Keep |
| `compare_architectures.py` | A/B testing script for Additive vs Multiplicative base architectures. | Keep |
| `diagnose_rho_gradient.py` | SSVI troubleshooting script to analyze vanishing gradients on correlation parameters. | Keep |
| `download_data.py` | Pipeline for fetching raw option data from FinMind and Yahoo Finance. | Keep |
| `generate_model1_plots.py` | Generates visualization plots for Model 1 predictions. | Keep |
| `inspect_ssvi_params.py` | Tool to extract and inspect the learned SSVI parameters (rho, eta, gamma). | Keep |
| `plot_smooth_iv_check.py` | Script to verify that fixing yATM yields a smooth IV surface. | Keep |
| `plot_training_curves.py` | Simple tool to plot standard training/validation loss curves. | Keep |
| `train_diagnose.py` | Specialized training loop that tracks parameters per-epoch for debugging. | Keep |

---

## 3. `src/` Directory (Model 1 & Base Pipeline, HyperIV, DDPM)
| File | Description | Action/Status |
|------|-------------|---------------|
| `config.ini` | Core hyperparameters, paths, and training configuration. | Keep |
| `data/module_d_features.csv` | Output file for the extracted Greek features (Vanna, Volga, etc). | Keep |
| `dataset.py` | **Core**: The unified `DataProcessor` for splitting, filtering, and tensor generation. | Keep |
| `diffusion.py` | **Model 5**: DDPM (1D U-Net) architecture for IV surface forecasting. | Keep |
| `experiment.py` | **Core**: Main inference and evaluation script for Model 1 (creates fits). | Keep |
| `hyperiv.py` | **Model 4**: Hypernetwork model architecture for generating TargetMLP weights. | Keep |
| `model.py` | **Model 1**: SSVI + NN architectures and loss functions. | Keep |
| `structural_break.py` | Change-point detection logic (PELT) for volatility regimes. | Keep |
| `test.py` | Script for performing extended Model 1 evaluations (arbitrage checks). | Keep |
| `train.py` | Training script specifically for Model 1's initial prior phase. | Keep |
| `train_diffusion.py` | **Model 5**: Training script for DDPM model. | Keep |
| `train_hyperiv.py` | **Model 4**: Training script for HyperIV model. | Keep |
| `train_pipeline.py` | **Model 1**: Master two-stage training loop for SSVI + Neural Network. | Keep |
| `transfer.py` | Utilities for transfer learning (loading mismatched weights). | Keep |
| `utils.py` | Helper functions (metrics, plotting, seed setting, early stopping). | Keep |

---

## 4. `model2_research/` Directory (Model 2 Local Volatility)
| File | Description | Action/Status |
|------|-------------|---------------|
| `README.md` | Model 2 specific architecture documentation. | Keep |
| `dupire_pinn.py` | **Model 2**: ICNN based Dupire PINN Local Volatility Extractor. | Keep |
| `extract_features.py` | Command line script to run Module D feature extraction. | Keep |
| `model2_training_details.md` | Detailed log of Model 2 training dynamics and tricks used. | Keep |
| `module_d.py` | **Module D**: The `GreekExtractor` that grabs Local Vol, Vanna, Volga for Model 3. | Keep |
| `train_dupire.py` | **Model 2**: Training loop for the primary PINN local vol network. | Keep |

### `model2_research/candidates/`
*Contains experimental prior alternatives to Model 2 (Module A Soft Corrections, GNO, Heston...)*
| File | Description | Action/Status |
|------|-------------|---------------|
| `gno/README.md` | Graph Neural Operator option pricing experimental notes. | Keep |
| `heston/README.md` | Heston stochastic volatility calibration experiments. | Keep |
| `neural_sde/README.md` | Neural Stochastic Differential Equations approach experiments. | Keep |
| `signature/README.md` | Rough volatility signature methods experiments. | Keep |
| `wamol/README.md` | WAMOL alternative experiments. | Keep |
| `README.md` | General overview of alternate Model 2 approaches. | Keep |

---

## 5. `model3_research/` Directory (Model 3 Residual Adjustment)
| File | Description | Action/Status |
|------|-------------|---------------|
| `README.md` | Overview of the Model 3 architectures (TFT vs xLSTM vs GRU). | Keep |
| `full_research_report.md` | In-depth analysis report on TFT and xLSTM performance. | Keep |
| `optimizers.py` | Custom optimizers resolving overfitting (`CPR`, `CautiousAdamW`). | Keep |
| `parameter_dynamic_analysis_and_regularization.md` | Research notes on parameter dynamics vs overfitting. | Keep |
| `parameter_dynamic_analysis_and_regularization.txt` | Raw dump text for the regularization analysis. | Keep |
| `regularization_results.md` | Summary of early stopping and validation bounds after CPR. | Keep |
| `run_all_experiments.py` | Global python script to kick off multi-model runs. | Keep |
| `tft_adjustment.py` | **Model 3**: Temporal Fusion Transformer model architecture. | Keep |

### `model3_research/scripts/`
| File | Description | Action/Status |
|------|-------------|---------------|
| `benchmark_dtype.py` | Script comparing float32 vs float64 matrix operations speed. | Keep |
| `plot_loss_curves.py` | Visualizes basic train/val curves for an individual log. | Keep |
| `plot_regularization_results.py` | Overlays different regularizers (L2, Enum) on a single plot for comparisons. | Keep |
| `plot_running_logs.py` | Reads active background logs and `metrics.json` to draw real-time `live_loss_curves.png`. | Keep |
| `run_all_experiments.ps1` | PowerShell background launcher: shoots out 3 parallel Model 3 runs. | Keep |
| `run_tft_experiments.py` | Grid search script for TFT regularization. | Keep |
| `train_models.py` | Unified trainer handling GRU, xLSTM, and TFT, equipped with CPR + logging. | Keep |

### `model3_research/overfitting_research/`
| File | Description | Action/Status |
|------|-------------|---------------|
| `README.md` | Context for overfitting isolation logic. | Keep |
| `cwd_notes.md` | Notes specifically focusing on Coordinate-wise Descent effectiveness. | Keep |

### `model3_research/archived_...`
*(Directories like `archived_logs`, `archived_figures`, `archived_models`) contain hundreds of historically tracked JSON/PNG files and unselected architectures (e.g. `xlstm_adjustment.py`) from the architecture comparison phase before Greek Integration.*

---

## 6. `tests/` Directory (Unit Testing)
| File | Description | Action/Status |
|------|-------------|---------------|
| `conftest.py` | Global pytest configuration (sets default dtype to float64, fixes seeds). | Keep |
| `test_dataset.py` | Validates DataProcessor chronologic splits and tau calculations. | Keep |
| `test_diffusion.py` | Validation tests for U-Net architecture steps. | Keep |
| `test_hyperiv.py` | Tests set embedding sizes and target MLP weights dynamically generated. | Keep |
| `test_model.py` | Ensures SSVI baseline loss outputs backprop safely. | Keep |
| `test_model_regression.py` | Locks down bugs identified in Phase 1 (e.g., E1 index errors). | Keep |
| `test_structural_break.py` | Unit tests for changepoint detection boundaries. | Keep |
| `test_train_integration.py` | E2E mocked smoke test combining dataset + training loop. | Keep |
| `test_utils.py` | Checks metric bounds (MSE/MAPE stability limits). | Keep |
| `model2_research/tests/conftest.py` | Model 2 local test fixtures. | Keep |
| `model2_research/tests/test_dupire.py` | Checks local vol extraction physics bounds via PINN. | Keep |

*(Note: Data binaries like `dataset/prs_dataset_no_fat(clean).csv` and images inside `figures/` are omitted from this text description index for brevity as they are static read-only outputs.)*

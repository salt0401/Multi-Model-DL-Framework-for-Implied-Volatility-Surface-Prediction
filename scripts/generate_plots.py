#!/usr/bin/env python3
"""Generate all experimental figures from training logs.

Reads from logs/ directory (JSON metrics and .log files).
Copies existing plots from LossPlot/.
Outputs 9 PNGs to figures/.

Usage:
    python scripts/generate_plots.py
"""
import json
import re
import os
import shutil
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, 'logs')
FIG_DIR = os.path.join(ROOT, 'figures')
LOSSPLOT_DIR = os.path.join(ROOT, 'LossPlot')


def load_json(filename):
    path = os.path.join(LOG_DIR, filename)
    with open(path, 'r') as f:
        return json.load(f)


def find_latest_log(prefix):
    """Find the most recent log file matching prefix."""
    logs = sorted([f for f in os.listdir(LOG_DIR) if f.startswith(prefix) and f.endswith('.log')])
    if not logs:
        raise FileNotFoundError(f"No log file matching '{prefix}*' in {LOG_DIR}")
    return os.path.join(LOG_DIR, logs[-1])


def parse_dgm_log(path):
    """Parse DGM log: extract epoch, total, pde, bc, tc losses."""
    epochs, total, pde, bc, tc = [], [], [], [], []
    pattern = re.compile(
        r'Epoch (\d+)/\d+ - Total: ([\d.]+) - PDE: ([\d.]+) - BC: ([\d.]+) - TC: ([\d.]+)'
    )
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                total.append(float(m.group(2)))
                pde.append(float(m.group(3)))
                bc.append(float(m.group(4)))
                tc.append(float(m.group(5)))
    return epochs, total, pde, bc, tc


def parse_diffusion_log(path):
    """Parse diffusion log: extract train losses and val RMSE checkpoints."""
    train_epochs, train_losses = [], []
    val_epochs, val_rmses = [], []
    test_rmse = None
    current_epoch = None

    with open(path, 'r') as f:
        for line in f:
            m = re.search(r'Epoch (\d+)/\d+ - Train Loss: ([\d.]+)', line)
            if m:
                current_epoch = int(m.group(1))
                train_epochs.append(current_epoch)
                train_losses.append(float(m.group(2)))
                continue
            m = re.search(r'Val RMSE: ([\d.]+)', line)
            if m:
                val_epochs.append(current_epoch if current_epoch else 0)
                val_rmses.append(float(m.group(1)))
                continue
            m = re.search(r'Test RMSE: ([\d.]+)', line)
            if m:
                test_rmse = float(m.group(1))

    return train_epochs, train_losses, val_epochs, val_rmses, test_rmse


def plot_base_model_training(metrics):
    """Figure 1: Full 76 epochs, log scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(metrics['train_losses']) + 1)
    ax.semilogy(epochs, metrics['train_losses'], 'b-', alpha=0.8, label='Train Loss')
    ax.semilogy(range(1, len(metrics['val_losses']) + 1), metrics['val_losses'], 'r-', alpha=0.8, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Base Model (SSVI + NN Ensemble) — Full Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'base_model_training.png'), dpi=150)
    plt.close(fig)
    print('  [1/9] base_model_training.png')


def plot_base_model_zoomed(metrics):
    """Figure 2: Epochs 4-76, linear scale, best epoch annotated."""
    train = metrics['train_losses']
    val = metrics['val_losses']
    best_epoch = metrics['best_epoch']
    best_val = metrics['best_val_loss']

    # Zoom: skip first 3 epochs (huge initial losses)
    start = 3
    fig, ax = plt.subplots(figsize=(10, 6))
    t_epochs = range(start + 1, len(train) + 1)
    v_epochs = range(start + 1, len(val) + 1)
    ax.plot(list(t_epochs), train[start:], 'b-', alpha=0.8, label='Train Loss')
    ax.plot(list(v_epochs), val[start:], 'r-', alpha=0.8, label='Val Loss')
    ax.axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.6, label=f'Best Epoch ({best_epoch + 1})')
    ax.annotate(f'Best: {best_val:.4f}', xy=(best_epoch + 1, best_val),
                xytext=(best_epoch + 8, best_val + 0.05),
                arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Base Model — Zoomed (Epochs 4–76)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'base_model_training_zoomed.png'), dpi=150)
    plt.close(fig)
    print('  [2/9] base_model_training_zoomed.png')


def plot_hyperiv_training(metrics):
    """Figure 3: Two-panel: train (log) + val (linear)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    train = metrics['train_losses']
    val = metrics['val_losses']
    t_epochs = range(1, len(train) + 1)
    v_epochs = range(1, len(val) + 1)

    ax1.semilogy(t_epochs, train, 'b-', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss (log)')
    ax1.set_title('HyperIV — Train Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(v_epochs, val, 'r-', alpha=0.8)
    best_epoch = metrics['best_epoch']
    best_val = metrics['best_val_loss']
    ax2.axvline(best_epoch + 1, color='green', linestyle='--', alpha=0.6)
    ax2.annotate(f'Best: {best_val:.6f}\n(Epoch {best_epoch + 1})',
                 xy=(best_epoch + 1, best_val),
                 xytext=(best_epoch + 10, best_val + 0.0003),
                 arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=9)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Loss')
    ax2.set_title('HyperIV — Val Loss')
    ax2.grid(True, alpha=0.3)

    fig.suptitle('HyperIV Training (58 epochs, early stopped)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'hyperiv_training.png'), dpi=150)
    plt.close(fig)
    print('  [3/9] hyperiv_training.png')


def plot_diffusion_training(diffusion_metrics):
    """Figure 4: 1000 epochs train loss, log scale + moving average."""
    train = diffusion_metrics['train_losses']
    epochs = range(1, len(train) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, train, 'b-', alpha=0.3, label='Train Loss')

    # Moving average (window=50)
    window = 50
    if len(train) >= window:
        ma = np.convolve(train, np.ones(window) / window, mode='valid')
        ma_epochs = range(window, len(train) + 1)
        ax.semilogy(ma_epochs, ma, 'b-', linewidth=2, label=f'Moving Avg ({window}-epoch)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train Loss (log scale)')
    ax.set_title('DDPM — Training Loss (1000 Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'diffusion_training.png'), dpi=150)
    plt.close(fig)
    print('  [4/9] diffusion_training.png')


def plot_diffusion_val_rmse(val_epochs, val_rmses, test_rmse):
    """Figure 5: Val RMSE at checkpoints + test RMSE."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(val_epochs, val_rmses, 'ro-', markersize=8, label='Val RMSE')

    if test_rmse is not None:
        ax.axhline(test_rmse, color='blue', linestyle='--', alpha=0.7, label=f'Test RMSE = {test_rmse:.4f}')

    # Annotate best val
    best_idx = np.argmin(val_rmses)
    ax.annotate(f'Best: {val_rmses[best_idx]:.4f}\n(Epoch {val_epochs[best_idx]})',
                xy=(val_epochs[best_idx], val_rmses[best_idx]),
                xytext=(val_epochs[best_idx] + 50, val_rmses[best_idx] + 0.001),
                arrowprops=dict(arrowstyle='->', color='green'), color='green', fontsize=10)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('DDPM — Validation & Test RMSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'diffusion_val_rmse.png'), dpi=150)
    plt.close(fig)
    print('  [5/9] diffusion_val_rmse.png')


def plot_dgm_training(epochs, total, pde, bc, tc):
    """Figure 6: 4 lines (Total/PDE/BC/TC), log scale."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(epochs, total, 'k-', linewidth=2, label='Total')
    ax.semilogy(epochs, pde, 'b-', alpha=0.8, label='PDE Residual')
    ax.semilogy(epochs, bc, 'r-', alpha=0.8, label='Boundary Condition')
    ax.semilogy(epochs, tc, 'g-', alpha=0.8, label='Terminal Condition')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('DGM PDE Solver — Training Losses (5000 Epochs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'dgm_training.png'), dpi=150)
    plt.close(fig)
    print('  [6/9] dgm_training.png')


def plot_model_comparison():
    """Figure 7: Grouped bar chart comparing all models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Point prediction models
    models = ['Base\n(SSVI+NN)', 'HyperIV']
    tv_rmse = [0.0134, 0.0074]
    mape = [44.1, 20.7]
    iv_rmse = [0.209, 0.076]

    colors = ['#4472C4', '#ED7D31']

    axes[0].bar(models, tv_rmse, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('TV-RMSE')
    axes[0].set_title('Total Variance RMSE')
    for i, v in enumerate(tv_rmse):
        axes[0].text(i, v + 0.0003, f'{v:.4f}', ha='center', fontsize=10)

    axes[1].bar(models, mape, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('MAPE (%)')
    axes[1].set_title('Mean Absolute % Error')
    for i, v in enumerate(mape):
        axes[1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)

    axes[2].bar(models, iv_rmse, color=colors, width=0.5, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('IV-RMSE')
    axes[2].set_title('Implied Volatility RMSE')
    for i, v in enumerate(iv_rmse):
        axes[2].text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=10)

    fig.suptitle('Model Comparison: Base vs HyperIV (Point Prediction)', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'model_comparison.png'), dpi=150)
    plt.close(fig)
    print('  [7/9] model_comparison.png')


def copy_existing_plots():
    """Figures 8-9: Copy from LossPlot/ if present."""
    copied = 0
    for fname in ['iv_smiles.png', 'iv_surface_pred.png']:
        src = os.path.join(LOSSPLOT_DIR, fname)
        dst = os.path.join(FIG_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        elif os.path.exists(dst):
            copied += 1  # Already there
        else:
            print(f'  WARNING: {fname} not found in LossPlot/ or figures/')
    if copied >= 1:
        print('  [8/9] iv_smiles.png')
    if copied >= 2:
        print('  [9/9] iv_surface_pred.png')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'Generating figures in {FIG_DIR}/')
    print()

    # Load data
    base_metrics = load_json('metrics.json')
    hyperiv_metrics = load_json('hyperiv_metrics.json')
    diffusion_metrics = load_json('diffusion_metrics.json')

    dgm_log = find_latest_log('dgm_')
    dgm_epochs, dgm_total, dgm_pde, dgm_bc, dgm_tc = parse_dgm_log(dgm_log)

    diff_log = find_latest_log('diffusion_')
    _, _, val_epochs, val_rmses, test_rmse = parse_diffusion_log(diff_log)

    # Generate plots
    plot_base_model_training(base_metrics)
    plot_base_model_zoomed(base_metrics)
    plot_hyperiv_training(hyperiv_metrics)
    plot_diffusion_training(diffusion_metrics)
    plot_diffusion_val_rmse(val_epochs, val_rmses, test_rmse)
    plot_dgm_training(dgm_epochs, dgm_total, dgm_pde, dgm_bc, dgm_tc)
    plot_model_comparison()
    copy_existing_plots()

    print()
    print(f'Done. {len(os.listdir(FIG_DIR))} figures in {FIG_DIR}/')


if __name__ == '__main__':
    main()

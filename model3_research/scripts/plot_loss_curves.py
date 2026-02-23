"""Plot train/val loss curves for all three models."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False

log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')

models = {
    'GRU (Baseline)': ('baseline_metrics.json', 'baseline_results.json', '#2196F3'),
    'xLSTM (mLSTM)': ('xlstm_metrics.json', 'xlstm_results.json', '#FF5722'),
    'TFT': ('tft_metrics.json', 'tft_results.json', '#4CAF50'),
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (name, (metrics_file, results_file, color)) in enumerate(models.items()):
    ax = axes[idx]

    with open(os.path.join(log_dir, metrics_file)) as f:
        metrics = json.load(f)
    with open(os.path.join(log_dir, results_file)) as f:
        results = json.load(f)

    train = np.array(metrics['train_losses'])
    val = np.array(metrics['val_losses'])
    epochs = np.arange(1, len(train) + 1)
    best_epoch = results['best_epoch'] + 1  # 0-indexed to 1-indexed
    best_val = results['best_val_loss']

    ax.plot(epochs, train, color=color, alpha=0.8, linewidth=1.5, label='Train Loss')
    ax.plot(epochs, val, color=color, alpha=0.4, linewidth=1.5, linestyle='--', label='Val Loss')

    # Mark best epoch
    ax.axvline(x=best_epoch, color='gray', linestyle=':', alpha=0.7)
    ax.scatter([best_epoch], [best_val], color='red', s=80, zorder=5,
               marker='*', label=f'Best Val (epoch {best_epoch})')

    # Shade the gap between train and val at best epoch
    train_at_best = train[best_epoch - 1]
    gap_ratio = best_val / train_at_best

    ax.set_title(f'{name}\nBest Val Loss: {best_val:.4f} (epoch {best_epoch})', fontsize=13)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_ylim(0, max(val.max(), train[0]) * 1.05)
    ax.grid(True, alpha=0.3)

    # Add text annotation for gap at best epoch
    ax.annotate(
        f'Train: {train_at_best:.4f}\nVal: {best_val:.4f}\nGap: {gap_ratio:.1f}x',
        xy=(best_epoch, best_val),
        xytext=(best_epoch + len(epochs) * 0.08, best_val + 0.02),
        fontsize=9, color='red',
        arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
    )

fig.suptitle('Model 3 Architecture Comparison — Train vs Val Loss', fontsize=15, fontweight='bold')
plt.tight_layout()

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures', 'loss_curves_3way.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f'Saved to {save_path}')
plt.close()

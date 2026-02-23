"""Generate loss curve plots and results summary for regularization experiments.

Reads metrics JSON files from logs/ and creates:
1. GRU loss curves comparison (4 optimizers)
2. TFT loss curves comparison (4 optimizers, if available)
3. Combined comparison bar chart
4. Results summary markdown

Usage:
    python plot_regularization_results.py
"""
import json
import os
import sys
import numpy as np

# Try matplotlib; if missing, skip plots
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("WARNING: matplotlib not available, skipping plots")


def load_metrics(filepath):
    """Load train/val loss arrays from metrics JSON."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_results(filepath):
    """Load experiment results JSON."""
    with open(filepath) as f:
        return json.load(f)


def plot_loss_curves(metrics_dict, title, save_path, max_epochs=None):
    """Plot train and val loss curves for multiple optimizers.

    Args:
        metrics_dict: {optimizer_name: metrics_data}
        title: plot title
        save_path: output file path
        max_epochs: if set, truncate x-axis
    """
    if not HAS_PLT:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    colors = {
        'adam':  '#888888',
        'adamw': '#2196F3',
        'cwd':   '#4CAF50',
        'cpr':   '#FF5722',
    }
    
    for opt_name, data in metrics_dict.items():
        train = data.get('train_losses', [])
        val = data.get('val_losses', [])
        if max_epochs:
            train = train[:max_epochs]
            val = val[:max_epochs]
        epochs = range(1, len(train) + 1)
        color = colors.get(opt_name, '#999999')

        # Train loss
        axes[0].plot(epochs, train, color=color, alpha=0.8,
                     label=f'{opt_name}', linewidth=1.5)
        # Val loss
        axes[1].plot(epochs, val, color=color, alpha=0.8,
                     label=f'{opt_name}', linewidth=1.5)

    axes[0].set_title('Train Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Validation Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def plot_comparison_bar(results_dict, title, save_path):
    """Bar chart comparing val loss, RMSE, MAPE across optimizers."""
    if not HAS_PLT:
        return

    opts = list(results_dict.keys())
    val_losses = [results_dict[o]['best_val_loss'] for o in opts]
    rmses = [results_dict[o]['final_rmse'] for o in opts]
    mapes = [results_dict[o]['final_mape'] for o in opts]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#888888', '#2196F3', '#4CAF50', '#FF5722'][:len(opts)]

    # Val Loss
    bars = axes[0].bar(opts, val_losses, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Val Loss (lower = better)', fontsize=12)
    axes[0].set_ylabel('Loss')
    for bar, v in zip(bars, val_losses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # RMSE
    bars = axes[1].bar(opts, rmses, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_title('RMSE (lower = better)', fontsize=12)
    axes[1].set_ylabel('RMSE')
    for bar, v in zip(bars, rmses):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    # MAPE
    bars = axes[2].bar(opts, [m * 100 for m in mapes], color=colors,
                       edgecolor='black', linewidth=0.5)
    axes[2].set_title('MAPE % (lower = better)', fontsize=12)
    axes[2].set_ylabel('MAPE %')
    for bar, v in zip(bars, mapes):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{v*100:.2f}%', ha='center', va='bottom', fontsize=10)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {save_path}')


def generate_summary_md(all_results, fig_dir):
    """Generate markdown summary of all experiment results."""
    lines = [
        '# Model 3 正則化實驗結果',
        '',
        '> 基於 4 種 optimizer 的比較實驗（AdamW, CWD, CPR vs Adam baseline）',
        '',
    ]

    for model_name in ['GRU (baseline)', 'TFT']:
        key_prefix = 'baseline' if 'GRU' in model_name else 'tft'
        model_results = {k: v for k, v in all_results.items() if k.startswith(key_prefix)}
        if not model_results:
            continue

        lines.append(f'## {model_name}')
        lines.append('')
        lines.append('| Optimizer | Val Loss | RMSE | MAPE | Best Epoch | Time (min) |')
        lines.append('|-----------|:--------:|:----:|:----:|:----------:|:----------:|')

        # Sort by val loss
        sorted_results = sorted(model_results.items(), key=lambda x: x[1]['best_val_loss'])
        baseline_val = None
        for key, r in sorted_results:
            opt = r.get('optimizer', 'adam')
            val = r['best_val_loss']
            if opt == 'adam':
                baseline_val = val
            delta = ''
            if baseline_val and opt != 'adam':
                pct = (val - baseline_val) / baseline_val * 100
                delta = f' ({pct:+.1f}%)'
            star = ' ⭐' if key == sorted_results[0][0] and opt != 'adam' else ''
            lines.append(
                f'| {opt}{star} | {val:.6f}{delta} | '
                f'{r["final_rmse"]:.6f} | {r["final_mape"]*100:.2f}% | '
                f'{r["best_epoch"]} | {r["training_minutes"]} |'
            )
        lines.append('')

        # Add figure reference
        fig_name = f'{key_prefix}_regularization_loss_curves.png'
        fig_path = os.path.join(fig_dir, fig_name)
        if os.path.exists(fig_path):
            lines.append(f'![{model_name} Loss Curves](figures/{fig_name})')
            lines.append('')

        bar_name = f'{key_prefix}_regularization_comparison.png'
        bar_path = os.path.join(fig_dir, bar_name)
        if os.path.exists(bar_path):
            lines.append(f'![{model_name} Comparison](figures/{bar_name})')
            lines.append('')

    return '\n'.join(lines)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, 'logs')
    fig_dir = os.path.join(script_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print('=== Generating Regularization Experiment Visualizations ===\n')

    all_results = {}

    # --- GRU Results ---
    gru_metrics = {}
    gru_results = {}
    gru_files = {
        'adam':  ('baseline_metrics.json', 'baseline_results.json'),
        'adamw': ('baseline_adamw_metrics.json', 'baseline_adamw_results.json'),
        'cwd':   ('baseline_cwd_metrics.json', 'baseline_cwd_results.json'),
        'cpr':   ('baseline_cpr_metrics.json', 'baseline_cpr_results.json'),
    }

    print('GRU experiments:')
    for opt, (mf, rf) in gru_files.items():
        mp = os.path.join(log_dir, mf)
        rp = os.path.join(log_dir, rf)
        if os.path.exists(mp) and os.path.exists(rp):
            gru_metrics[opt] = load_metrics(mp)
            gru_results[opt] = load_results(rp)
            all_results[f'baseline_{opt}' if opt != 'adam' else 'baseline'] = gru_results[opt]
            print(f'  {opt}: val_loss={gru_results[opt]["best_val_loss"]:.6f}')
        else:
            print(f'  {opt}: NOT FOUND')

    if gru_metrics:
        plot_loss_curves(gru_metrics,
                         'GRU (Baseline) — Optimizer Comparison',
                         os.path.join(fig_dir, 'baseline_regularization_loss_curves.png'),
                         max_epochs=200)
        plot_comparison_bar(gru_results,
                           'GRU (Baseline) — Regularization Comparison',
                           os.path.join(fig_dir, 'baseline_regularization_comparison.png'))

    # --- TFT Results ---
    tft_metrics = {}
    tft_results = {}
    tft_files = {
        'adam':  ('tft_fp32_metrics.json', 'tft_fp32_results.json'),
        'adamw': ('tft_adamw_fp32_metrics.json', 'tft_adamw_fp32_results.json'),
        'cwd':   ('tft_cwd_fp32_metrics.json', 'tft_cwd_fp32_results.json'),
        'cpr':   ('tft_cpr_fp32_metrics.json', 'tft_cpr_fp32_results.json'),
    }

    print('\nTFT experiments:')
    for opt, (mf, rf) in tft_files.items():
        mp = os.path.join(log_dir, mf)
        rp = os.path.join(log_dir, rf)
        if os.path.exists(mp) and os.path.exists(rp):
            tft_metrics[opt] = load_metrics(mp)
            tft_results[opt] = load_results(rp)
            all_results[f'tft_{opt}' if opt != 'adam' else 'tft'] = tft_results[opt]
            print(f'  {opt}: val_loss={tft_results[opt]["best_val_loss"]:.6f}')
        else:
            print(f'  {opt}: NOT FOUND (still running?)')

    if tft_metrics:
        plot_loss_curves(tft_metrics,
                         'TFT — Optimizer Comparison (float32)',
                         os.path.join(fig_dir, 'tft_regularization_loss_curves.png'),
                         max_epochs=200)
        plot_comparison_bar(tft_results,
                           'TFT — Regularization Comparison (float32)',
                           os.path.join(fig_dir, 'tft_regularization_comparison.png'))

    # --- Summary ---
    print('\nGenerating summary markdown...')
    summary = generate_summary_md(all_results, fig_dir)
    summary_path = os.path.join(script_dir, 'regularization_results.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f'  Saved: {summary_path}')
    print('\nDone!')


if __name__ == '__main__':
    main()

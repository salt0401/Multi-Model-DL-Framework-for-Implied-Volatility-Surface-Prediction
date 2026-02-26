import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import MultiModel
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed


def plot_split(name, model, device, df, save_path):
    """Plot 5 IV curve fits, each from a single (date, exdate) option chain.
    
    Groups by (date, exdate) to guarantee exactly one option chain per subplot.
    Selects the 5 chains with the most data points.
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    
    # Group by (date, exdate) = guaranteed single option chain
    grouped = df.groupby(['date', 'exdate'])
    chain_sizes = grouped.size().sort_values(ascending=False)
    top5_keys = chain_sizes.head(5).index
    
    for ax, (date, exdate) in zip(axes, top5_keys):
        chain = grouped.get_group((date, exdate))
        
        tau_vals = chain['tau'].values
        logm_vals = chain['logm'].values
        yatm_vals = chain['y_atm'].values
        tv_true = chain['total_var'].values
        
        # Run model prediction
        tau_t = torch.tensor(tau_vals.reshape(-1, 1), dtype=torch.float64, device=device)
        logm_t = torch.tensor(logm_vals.reshape(-1, 1), dtype=torch.float64, device=device)
        yatm_t = torch.tensor(yatm_vals.reshape(-1, 1), dtype=torch.float64, device=device)
        
        tv_pred, _, _, _ = model(tau_t, logm_t, yatm_t)
        tv_pred = tv_pred.detach().cpu().numpy().flatten()
        
        # Convert total variance to implied volatility
        tau_scalar = tau_vals[0]
        iv_true = np.sqrt(np.maximum(tv_true / max(tau_scalar, 1e-8), 0))
        iv_pred = np.sqrt(np.maximum(tv_pred / max(tau_scalar, 1e-8), 0))
        
        # Sort by log-moneyness
        sort_idx = np.argsort(logm_vals)
        k_sorted = logm_vals[sort_idx]
        iv_p_sorted = iv_pred[sort_idx]
        iv_t_sorted = iv_true[sort_idx]
        
        ax.plot(k_sorted, iv_p_sorted, 'r-', label='Model 1 Predicted', linewidth=2)
        ax.plot(k_sorted, iv_t_sorted, 'b.', label='Observed Data', markersize=8, alpha=0.8)
        
        t_val = tau_scalar
        y_val = yatm_vals[0]
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        ax.set_title(f'{date_str}, tau={t_val:.4f}, yATM={y_val:.4f} (n={len(chain)})', fontsize=10)
        ax.set_xlabel('log-moneyness (k)', fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel('Implied Volatility', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([k_sorted.min() - 0.05, k_sorted.max() + 0.05])
    
    fig.suptitle(f'Model 1 IV Curve Fits ({name.capitalize()} Data)', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def main():
    set_seed(42)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    os.chdir(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    config = load_config('config.ini')
    dp = DataProcessor(config)
    dp()
    
    # Load Model 1
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'MultiModel.pt')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    epsilon = config['model_sett'].getfloat('epsilon', fallback=0.01)
    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num, epsilon=epsilon).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Generate Training Loss Plot
    metrics_file = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', 'pipeline_stage1_metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            d = json.load(f)
        plt.figure(figsize=(8, 5))
        plt.plot(d['train_losses'], label='Train Loss', color='steelblue', linewidth=2)
        plt.plot(d['val_losses'], label='Validation Loss', color='crimson', linewidth=2)
        
        best_epoch = d.get('best_epoch', 0)
        best_val = d.get('best_val_loss', 0)
        if best_epoch > 0:
            plt.axvline(x=best_epoch, color='grey', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
            plt.scatter([best_epoch], [best_val], color='black', zorder=5)
            
        plt.title('Model 1 (SSVI+NN) Training Curve', fontsize=12, fontweight='bold')
        plt.xlabel('Epoch', fontsize=10)
        plt.ylabel('Weighted Sum Loss', fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot = os.path.join(os.path.dirname(__file__), '..', 'figures', 'm1_loss_curve.png')
        os.makedirs(os.path.dirname(loss_plot), exist_ok=True)
        plt.savefig(loss_plot, dpi=150)
        plt.close()
        print(f"Saved {loss_plot}")
        
    # Prepare DataFrames directly (preserving date info)
    train_start = parse_date(config['training']['train_start_date'])
    train_end = parse_date(config['training']['train_end_date'])
    test_start = parse_date(config['training']['test_start_date'])
    test_end = parse_date(config['training']['test_end_date'])
    
    full_train = dp.prs_dataset[
        (dp.prs_dataset['date'] >= train_start) &
        (dp.prs_dataset['date'] <= train_end)
    ].sort_values('date').copy()
    
    # Chronological split (same as Prepare_train_data)
    unique_dates = np.sort(full_train['date'].unique())
    split_idx = int(len(unique_dates) * 0.8)
    val_start_date = unique_dates[split_idx]
    
    train_df = full_train[full_train['date'] < val_start_date]
    val_df = full_train[full_train['date'] >= val_start_date]
    test_df = dp.prs_dataset[
        (dp.prs_dataset['date'] >= test_start) &
        (dp.prs_dataset['date'] <= test_end)
    ]
    
    figures_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    
    print("Evaluating Train...")
    plot_split('train', model, device, train_df, os.path.join(figures_dir, 'm1_train_fit.png'))
    
    print("Evaluating Val...")
    plot_split('validation', model, device, val_df, os.path.join(figures_dir, 'm1_val_fit.png'))
    
    print("Evaluating Test...")
    plot_split('test', model, device, test_df, os.path.join(figures_dir, 'm1_test_fit.png'))
    
    print("Done!")

if __name__ == '__main__':
    main()

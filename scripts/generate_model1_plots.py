import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import MultiModel
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed

def plot_split(name, model, device, taus, logms, yatms, iv_true, iv_pred, save_path):
    fig, axes = plt.subplots(1, 5, figsize=(25, 4))
    
    # CRITICAL VISUALIZATION RULE:
    # Always group data by EXACT (tau, yATM) combinations to isolate a single 
    # option chain (one specific maturity on one specific date) per subplot. 
    # DO NOT mix data from different dates with "similar" taus, as this causes 
    # overlapping vertical scatter and destroys the single clean smile curve visualization.
    combined_keys = np.round(taus, 6).astype(str) + "_" + np.round(yatms, 6).astype(str)
    unique_keys, counts = np.unique(combined_keys, return_counts=True)
    
    # Select the 5 chains with the most data points
    top_keys = unique_keys[np.argsort(-counts)][:5]
    
    for ax, key in zip(axes, top_keys):
        mask = combined_keys == key
        
        # Get the scalar values for the title
        t_val = taus[mask][0]
        y_val = yatms[mask][0]
        
        k_slice = logms[mask]
        iv_p_slice = iv_pred[mask]
        iv_t_slice = iv_true[mask]
        
        sort_idx = np.argsort(k_slice)
        k_sorted = k_slice[sort_idx]
        iv_p_sorted = iv_p_slice[sort_idx]
        iv_t_sorted = iv_t_slice[sort_idx]
        
        # Plot predicted as a line and observed as dots
        ax.plot(k_sorted, iv_p_sorted, 'r-', label='Model 1 Predicted', linewidth=2)
        ax.plot(k_sorted, iv_t_sorted, 'b.', label='Observed Data', markersize=8, alpha=0.8)

        ax.set_title(f'tau={t_val:.4f}, yATM={y_val:.4f} (n={mask.sum()})', fontsize=11)
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

def get_predictions(model, loader, device):
    all_tau, all_logm, all_y_true, all_y_pred, all_yatm = [], [], [], [], []
    for tau, logm, y, yATM in loader:
        tau_d, logm_d, yATM_d = tau.to(device), logm.to(device), yATM.to(device)
        out, _, _, _ = model(tau_d, logm_d, yATM_d)
        
        all_tau.append(tau.numpy().flatten())
        all_logm.append(logm.numpy().flatten())
        all_y_true.append(y.numpy().flatten())
        all_y_pred.append(out.detach().cpu().numpy().flatten())
        all_yatm.append(yATM.numpy().flatten())
        
    taus = np.concatenate(all_tau)
    logms = np.concatenate(all_logm)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    yatms = np.concatenate(all_yatm)
    
    iv_true = np.sqrt(np.maximum(y_true / np.maximum(taus, 1e-8), 0))
    iv_pred = np.sqrt(np.maximum(y_pred / np.maximum(taus, 1e-8), 0))
    
    return taus, logms, yatms, iv_true, iv_pred

def main():
    set_seed(42)
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    config = load_config(os.path.join(os.path.dirname(__file__), '..', 'src', 'config.ini'))
    dp = DataProcessor(config)
    dp()
    
    # Load Model 1
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'MultiModel.pt')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Generate Training Loss Plot
    metrics_file = os.path.join(os.path.dirname(__file__), '..', 'logs', 'pipeline_stage2_metrics.json')
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
        
    print("Loading datasets to generate fit plots...")
    train_start = parse_date(config['training']['train_start_date'])
    train_end = parse_date(config['training']['train_end_date'])
    batch_size = config['training'].getint('batch_size') * 4 # Speed up inference
    
    train_loader, val_loader, _ = dp.Prepare_train_data(train_start, train_end, batch_size)
    
    test_start = parse_date(config['training']['test_start_date'])
    test_end = parse_date(config['training']['test_end_date'])
    test_loader, _ = dp.Prepare_test_data(test_start, test_end)
    
    print("Evaluating Train...")
    t1 = get_predictions(model, train_loader, device)
    plot_split('train', model, device, *t1, os.path.join(os.path.dirname(__file__), '..', 'figures', 'm1_train_fit.png'))
    
    print("Evaluating Val...")
    t2 = get_predictions(model, val_loader, device)
    plot_split('validation', model, device, *t2, os.path.join(os.path.dirname(__file__), '..', 'figures', 'm1_val_fit.png'))
    
    print("Evaluating Test...")
    t3 = get_predictions(model, test_loader, device)
    plot_split('test', model, device, *t3, os.path.join(os.path.dirname(__file__), '..', 'figures', 'm1_test_fit.png'))
    
    print("Done!")

if __name__ == '__main__':
    main()

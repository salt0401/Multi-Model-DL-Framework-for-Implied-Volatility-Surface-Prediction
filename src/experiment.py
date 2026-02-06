from model import MultiModel, SimpleLoss
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed, compute_rmse, compute_mape, compute_iv_rmse

from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def run_base_model_inference(model, test_loader, device):
    """Run model on test data. Returns (y_pred, y_true, taus, logms) as numpy arrays."""
    model.eval()
    all_preds = []
    all_true = []
    all_taus = []
    all_logms = []

    with torch.no_grad():
        for tau, logm, y, y_atm in test_loader:
            tau = tau.to(device)
            logm = logm.to(device)
            y_atm = y_atm.to(device)

            # Bug E1 fixed: unpack tuple, take first element
            output, _, _, _ = model(tau, logm, y_atm)

            all_preds.append(output.cpu().numpy())
            all_true.append(y.numpy())
            all_taus.append(tau.cpu().numpy())
            all_logms.append(logm.cpu().numpy())

    return (
        np.concatenate(all_preds).flatten(),
        np.concatenate(all_true).flatten(),
        np.concatenate(all_taus).flatten(),
        np.concatenate(all_logms).flatten(),
    )


def evaluate_predictions(tv_pred, tv_true, tau):
    """Compute and print evaluation metrics."""
    rmse = compute_rmse(tv_pred, tv_true)
    mape = compute_mape(tv_pred, tv_true)
    iv_rmse = compute_iv_rmse(tv_pred, tv_true, tau)

    print(f'  Total Variance RMSE: {rmse:.6f}')
    print(f'  Total Variance MAPE: {mape:.6f}')
    print(f'  IV-RMSE:             {iv_rmse:.6f}')
    return {'rmse': rmse, 'mape': mape, 'iv_rmse': iv_rmse}


def plot_iv_surface(tau, logm, tv_pred, save_path=None, title='Predicted IV Surface'):
    """3D surface plot of implied volatility."""
    iv_pred = np.sqrt(np.maximum(tv_pred, 0) / np.maximum(tau, 1e-8))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tau, logm, iv_pred, c=iv_pred, cmap='viridis', s=1, alpha=0.5)
    ax.set_xlabel('Time to Maturity (tau)')
    ax.set_ylabel('Log-Moneyness (k)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Surface plot saved to {save_path}')
    plt.close()


def plot_smile_by_maturity(tau, logm, tv_pred, tv_true=None, save_path=None, n_slices=5):
    """Plot IV smiles at different maturities."""
    unique_taus = np.unique(tau)
    if len(unique_taus) > n_slices:
        indices = np.linspace(0, len(unique_taus)-1, n_slices, dtype=int)
        selected_taus = unique_taus[indices]
    else:
        selected_taus = unique_taus

    fig, axes = plt.subplots(1, len(selected_taus), figsize=(4*len(selected_taus), 4), sharey=True)
    if len(selected_taus) == 1:
        axes = [axes]

    for ax, t in zip(axes, selected_taus):
        mask = np.isclose(tau, t, atol=1e-6)
        k_slice = logm[mask]
        iv_slice = np.sqrt(np.maximum(tv_pred[mask], 0) / max(t, 1e-8))
        sort_idx = np.argsort(k_slice)

        ax.plot(k_slice[sort_idx], iv_slice[sort_idx], 'b-', label='Predicted')

        if tv_true is not None:
            iv_true_slice = np.sqrt(np.maximum(tv_true[mask], 0) / max(t, 1e-8))
            ax.plot(k_slice[sort_idx], iv_true_slice[sort_idx], 'r.', label='Observed', markersize=3)

        ax.set_title(f'tau={t:.4f}')
        ax.set_xlabel('Log-Moneyness')
        if ax == axes[0]:
            ax.set_ylabel('Implied Volatility')
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Smile plot saved to {save_path}')
    plt.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--on_gpu", action='store_true')
    args = parser.parse_args()

    config = load_config('config.ini')

    start_date = parse_date(args.start_date or config['training']['test_start_date'])
    end_date = parse_date(args.end_date or config['training']['test_end_date'])

    seed = config['training'].getint('seed')
    set_seed(seed)

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    print('Preprocessing...')
    dp = DataProcessor(config)
    dp()
    test_loader, test_df = dp.Prepare_test_data(start_date, end_date)
    print('End of data preprocessing.')

    print('Model loading...')
    model_path = config['save_path']['model_path']
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print('Running inference...')
    tv_pred, tv_true, taus, logms = run_base_model_inference(model, test_loader, device)

    print('Evaluation metrics:')
    metrics = evaluate_predictions(tv_pred, tv_true, taus)

    # Add predictions to test dataframe
    test_df = test_df.copy()
    test_df['tv_pred'] = tv_pred
    test_df['iv_pred'] = np.sqrt(np.maximum(tv_pred, 0) / test_df['tau'].values)

    plot_path = config['save_path']['plot_path']
    import os
    os.makedirs(plot_path, exist_ok=True)

    plot_iv_surface(taus, logms, tv_pred,
                    save_path=f'{plot_path}/iv_surface_pred.png',
                    title='Predicted IV Surface')

    plot_smile_by_maturity(taus, logms, tv_pred, tv_true,
                           save_path=f'{plot_path}/iv_smiles.png')

    results_path = f'{plot_path}/experiment_results.csv'
    test_df.to_csv(results_path)
    print(f'Results saved to {results_path}')


if __name__ == '__main__':
    main()

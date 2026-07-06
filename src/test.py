"""Proper evaluation script for trained base model.

Loads trained model, runs on test set, computes RMSE/MAPE/IV-RMSE,
reports arbitrage constraint violations, saves results to CSV.
"""
from model1_research.model import MultiModel
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed, compute_rmse, compute_mape, compute_iv_rmse

from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
import os


def check_calendar_arbitrage(tv_pred, tau):
    """Check calendar spread arbitrage: total variance should increase with tau.

    Returns fraction of violations among comparable pairs.
    """
    sorted_idx = np.argsort(tau)
    tv_sorted = tv_pred[sorted_idx]
    tau_sorted = tau[sorted_idx]

    violations = 0
    total = 0
    for i in range(1, len(tau_sorted)):
        if tau_sorted[i] > tau_sorted[i-1]:
            total += 1
            if tv_sorted[i] < tv_sorted[i-1]:
                violations += 1

    return violations, total


def check_butterfly_arbitrage(tv_pred, logm, tau):
    """Check butterfly arbitrage via numerical second derivative of w(k).

    Groups by tau, computes d2w/dk2 numerically, counts negative density.
    """
    unique_taus = np.unique(tau)
    violations = 0
    total = 0

    for t in unique_taus:
        mask = np.isclose(tau, t, atol=1e-6)
        k = logm[mask]
        w = tv_pred[mask]

        sort_idx = np.argsort(k)
        k = k[sort_idx]
        w = w[sort_idx]

        if len(k) < 3:
            continue

        for i in range(1, len(k)-1):
            dk1 = k[i] - k[i-1]
            dk2 = k[i+1] - k[i]
            if dk1 > 0 and dk2 > 0:
                d2w = 2 * ((w[i+1] - w[i])/dk2 - (w[i] - w[i-1])/dk1) / (dk1 + dk2)
                dw = (w[i+1] - w[i-1]) / (dk1 + dk2)
                # Gatheral-Jacquier density: note the SQUARED dw term
                g = (1 - k[i]*dw/(2*w[i]))**2 - dw**2/4*(1/w[i] + 0.25) + d2w/2
                total += 1
                if g < 0:
                    violations += 1

    return violations, total


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

    # Data
    print('Loading data...')
    dp = DataProcessor(config)
    dp()
    test_loader, test_df = dp.Prepare_test_data(start_date, end_date)

    # Model
    print('Loading model...')
    model_path = config['save_path']['model_path']
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)

    if not os.path.exists(model_path):
        print(f'Model file not found at {model_path}. Train the model first.')
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Inference
    print('Running evaluation...')
    all_preds = []
    all_true = []
    all_taus = []
    all_logms = []

    # Note: SmileModel.forward uses autograd.grad(create_graph=True) internally,
    # so we cannot wrap the forward pass in torch.no_grad(). We detach outputs instead.
    for tau, logm, y, y_atm in test_loader:
        tau = tau.to(device)
        logm = logm.to(device)
        y_atm = y_atm.to(device)

        output, _, _, _ = model(tau, logm, y_atm)
        all_preds.append(output.detach().cpu().numpy())
        all_true.append(y.numpy())
        all_taus.append(tau.detach().cpu().numpy())
        all_logms.append(logm.detach().cpu().numpy())

    tv_pred = np.concatenate(all_preds).flatten()
    tv_true = np.concatenate(all_true).flatten()
    taus = np.concatenate(all_taus).flatten()
    logms = np.concatenate(all_logms).flatten()

    # Metrics
    print('\n=== Test Evaluation Results ===')
    rmse = compute_rmse(tv_pred, tv_true)
    mape = compute_mape(tv_pred, tv_true)
    iv_rmse = compute_iv_rmse(tv_pred, tv_true, taus)
    print(f'  Total Variance RMSE: {rmse:.6f}')
    print(f'  Total Variance MAPE: {mape:.4f}')
    print(f'  IV-RMSE:             {iv_rmse:.6f}')
    print(f'  N samples:           {len(tv_pred)}')

    # Arbitrage violations
    print('\n=== Arbitrage Constraint Violations ===')
    cal_viol, cal_total = check_calendar_arbitrage(tv_pred, taus)
    but_viol, but_total = check_butterfly_arbitrage(tv_pred, logms, taus)
    print(f'  Calendar: {cal_viol}/{cal_total} violations ({100*cal_viol/max(cal_total,1):.2f}%)')
    print(f'  Butterfly: {but_viol}/{but_total} violations ({100*but_viol/max(but_total,1):.2f}%)')

    # Save results
    plot_path = config['save_path']['plot_path']
    os.makedirs(plot_path, exist_ok=True)

    results = {
        'metric': ['TV_RMSE', 'TV_MAPE', 'IV_RMSE', 'Calendar_Violations', 'Butterfly_Violations', 'N_Samples'],
        'value': [rmse, mape, iv_rmse, f'{cal_viol}/{cal_total}', f'{but_viol}/{but_total}', len(tv_pred)],
    }
    results_df = pd.DataFrame(results)
    results_path = os.path.join(plot_path, 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f'\nResults saved to {results_path}')


if __name__ == '__main__':
    main()

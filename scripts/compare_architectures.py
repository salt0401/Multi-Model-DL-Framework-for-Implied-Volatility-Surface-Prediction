"""Compare additive vs multiplicative SingleModel architecture.

Trains both variants for N epochs on the same data, tracking:
- Total loss, individual loss components (RMSE, MAPE, calendar, butterfly, linear, upperbound)
- SSVI parameters (rho, eta, gamma) per epoch
- Gradient norms
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
from torch import optim
from copy import deepcopy
import json

from model import MultiModel, WeightedSumLoss, SingleModel
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed
from train import train_one_epoch, validate


# --------------- Monkey-patch SingleModel.forward for multiplicative mode ---------------
def _forward_multiplicative(self, tau, logm, yATM):
    output_Prior, grad_ttm1_prior, grad_logm1_prior, grad_logm2_prior = self.Prior(logm, yATM)
    output_NN, grad_ttm1_NN, grad_logm1_NN, grad_logm2_NN = self.NN(tau, logm)

    # Multiplicative: Prior * NN  (product rule for derivatives)
    output = output_Prior * output_NN
    # dw/dtau = dPrior/dtau * NN + Prior * dNN/dtau
    grad_ttm1 = grad_ttm1_prior * output_NN + output_Prior * grad_ttm1_NN
    # dw/dk = dPrior/dk * NN + Prior * dNN/dk
    grad_logm1 = grad_logm1_prior * output_NN + output_Prior * grad_logm1_NN
    # d2w/dk2 = d2Prior/dk2 * NN + 2 * dPrior/dk * dNN/dk + Prior * d2NN/dk2
    grad_logm2 = (grad_logm2_prior * output_NN
                  + 2 * grad_logm1_prior * grad_logm1_NN
                  + output_Prior * grad_logm2_NN)
    return output, grad_ttm1, grad_logm1, grad_logm2


_forward_additive = SingleModel.forward  # save original


def get_ssvi_params(model):
    """Extract rho, eta, gamma from all ensemble members."""
    params = []
    for sm in model.ensemble_list:
        ssvi = sm.Prior
        rho = -torch.sigmoid(ssvi.raw_rho).item()
        eta = (2 * torch.sigmoid(ssvi.raw_eta)).item()
        gamma = torch.sigmoid(ssvi.raw_gamma).item()
        params.append({'rho': rho, 'eta': eta, 'gamma': gamma})
    return params


def get_grad_norm(model):
    """Compute total gradient norm across all parameters."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def run_experiment(mode, model, train_loader, val_loader, c6_loader,
                   loss_function, device, n_epochs, gradient_clip, learning_rate):
    """Run training for n_epochs, return detailed per-epoch results."""

    # Patch forward method
    if mode == 'multiplicative':
        SingleModel.forward = _forward_multiplicative
    else:
        SingleModel.forward = _forward_additive

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    results = {
        'mode': mode,
        'epochs': [],
    }

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        n_batches = 0
        c6_iterator = iter(c6_loader)

        for tau_train, logm_train, y_train, yATM_train in train_loader:
            try:
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)
            except StopIteration:
                c6_iterator = iter(c6_loader)
                tau_syn, logm_syn, yATM_syn = next(c6_iterator)

            tau_train = tau_train.to(device)
            logm_train = logm_train.to(device)
            y_train = y_train.to(device)
            yATM_train = yATM_train.to(device)
            tau_syn = tau_syn.to(device)
            logm_syn = logm_syn.to(device)
            yATM_syn = yATM_syn.to(device)

            output_train, grad_ttm1_train, grad_logm1_train, grad_logm2_train = model(tau_train, logm_train, yATM_train)
            output_syn, _, _, grad_logm2_syn = model(tau_syn, logm_syn, yATM_syn)

            loss = loss_function(
                output_train, y_train, logm_train,
                grad_ttm1_train, grad_logm1_train, grad_logm2_train,
                output_syn, logm_syn, grad_logm2_syn
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            grad_norm = get_grad_norm(model)

            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Grab last batch's individual losses
        ind_losses = loss_function.individual_losses.cpu().tolist()

        # --- Validate ---
        val_loss = validate(model, val_loader, c6_loader, loss_function, device)

        # --- SSVI params ---
        ssvi_params = get_ssvi_params(model)

        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'individual_losses': {
                'RMSE': ind_losses[0],
                'MAPE': ind_losses[1],
                'calendar': ind_losses[2],
                'butterfly': ind_losses[3],
                'linear': ind_losses[4],
                'upperbound': ind_losses[5],
            },
            'grad_norm': grad_norm,
            'ssvi_params': ssvi_params,
        }
        results['epochs'].append(epoch_data)

        # Print progress
        rho0 = ssvi_params[0]['rho']
        eta0 = ssvi_params[0]['eta']
        gamma0 = ssvi_params[0]['gamma']
        print(f"  [{mode:14s}] Ep {epoch+1}/{n_epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"RMSE: {ind_losses[0]:.6f} | MAPE: {ind_losses[1]:.4f} | "
              f"Cal: {ind_losses[2]:.6f} | Bfly: {ind_losses[3]:.6f} | "
              f"rho={rho0:.4f} eta={eta0:.4f} gamma={gamma0:.4f} | "
              f"grad_norm={grad_norm:.2f}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--on_gpu", action='store_true')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'src'))

    config = load_config('config.ini')
    seed = config['training'].getint('seed')

    # Device
    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    print(f"Device: {device}")

    # Data (load once)
    print("Loading data...")
    torch.set_default_dtype(torch.float64)
    dp = DataProcessor(config)
    dp()

    train_start_date = parse_date(config['training']['train_start_date'])
    train_end_date = parse_date(config['training']['train_end_date'])
    batch_size = config['training'].getint('batch_size')
    train_loader, val_loader, c6_loader = dp.Prepare_train_data(train_start_date, train_end_date, batch_size)
    print(f"Data ready: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Model config
    learning_rate = config['model_sett'].getfloat('learning_rate')
    ensemble_num = config['model_sett'].getint('ensemble_num')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    loss_weights = parse_list_config(config['model_sett']['loss_weights'])
    gradient_clip = config['training'].getfloat('gradient_clip')

    all_results = {}

    for mode in ['additive', 'multiplicative']:
        print(f"\n{'='*80}")
        print(f"  Training: {mode.upper()} architecture")
        print(f"{'='*80}")

        # Fresh model + loss for each run (same seed for fair comparison)
        set_seed(seed)
        model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
        loss_function = WeightedSumLoss(weights=loss_weights).to(device)

        # Print initial SSVI params
        init_params = get_ssvi_params(model)
        print(f"  Initial SSVI params (member 0): rho={init_params[0]['rho']:.4f}, "
              f"eta={init_params[0]['eta']:.4f}, gamma={init_params[0]['gamma']:.4f}")

        results = run_experiment(
            mode, model, train_loader, val_loader, c6_loader,
            loss_function, device, args.epochs, gradient_clip, learning_rate
        )
        all_results[mode] = results

    # Restore original forward
    SingleModel.forward = _forward_additive

    # --- Summary comparison ---
    print(f"\n{'='*80}")
    print(f"  COMPARISON SUMMARY ({args.epochs} epochs)")
    print(f"{'='*80}")

    header = f"{'Metric':<20s} | {'Additive':>14s} | {'Multiplicative':>14s} | {'Winner':>14s}"
    print(header)
    print("-" * len(header))

    add_final = all_results['additive']['epochs'][-1]
    mul_final = all_results['multiplicative']['epochs'][-1]

    comparisons = [
        ('Train Loss', add_final['train_loss'], mul_final['train_loss']),
        ('Val Loss', add_final['val_loss'], mul_final['val_loss']),
        ('RMSE', add_final['individual_losses']['RMSE'], mul_final['individual_losses']['RMSE']),
        ('MAPE', add_final['individual_losses']['MAPE'], mul_final['individual_losses']['MAPE']),
        ('Calendar Loss', add_final['individual_losses']['calendar'], mul_final['individual_losses']['calendar']),
        ('Butterfly Loss', add_final['individual_losses']['butterfly'], mul_final['individual_losses']['butterfly']),
        ('Linear Loss', add_final['individual_losses']['linear'], mul_final['individual_losses']['linear']),
        ('Upperbound Loss', add_final['individual_losses']['upperbound'], mul_final['individual_losses']['upperbound']),
        ('Grad Norm', add_final['grad_norm'], mul_final['grad_norm']),
    ]

    for name, add_val, mul_val in comparisons:
        winner = 'Additive' if add_val <= mul_val else 'Multiplicative'
        print(f"{name:<20s} | {add_val:>14.6f} | {mul_val:>14.6f} | {winner:>14s}")

    # SSVI param trajectory
    print(f"\n{'='*80}")
    print(f"  SSVI PARAMETER TRAJECTORY (ensemble member 0)")
    print(f"{'='*80}")
    print(f"{'Epoch':<6s} | {'Add rho':>8s} {'Add eta':>8s} {'Add gam':>8s} | {'Mul rho':>8s} {'Mul eta':>8s} {'Mul gam':>8s}")
    print("-" * 72)
    for i in range(args.epochs):
        ap = all_results['additive']['epochs'][i]['ssvi_params'][0]
        mp = all_results['multiplicative']['epochs'][i]['ssvi_params'][0]
        print(f"{i+1:<6d} | {ap['rho']:>8.4f} {ap['eta']:>8.4f} {ap['gamma']:>8.4f} | "
              f"{mp['rho']:>8.4f} {mp['eta']:>8.4f} {mp['gamma']:>8.4f}")

    # Best val loss epoch
    add_best = min(all_results['additive']['epochs'], key=lambda x: x['val_loss'])
    mul_best = min(all_results['multiplicative']['epochs'], key=lambda x: x['val_loss'])
    print(f"\nBest val loss: Additive ep{add_best['epoch']} ({add_best['val_loss']:.6f}) vs "
          f"Multiplicative ep{mul_best['epoch']} ({mul_best['val_loss']:.6f})")

    # Save full results
    out_path = os.path.join('..', 'logs', 'architecture_comparison.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == '__main__':
    main()

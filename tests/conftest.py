"""Shared fixtures for all test modules."""
import sys
import os
import configparser

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

# Allow bare imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope='session', autouse=True)
def set_float64():
    """Set default dtype to float64 for entire session (matches production)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


@pytest.fixture
def device():
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Tiny synthetic batches
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_batch():
    """(tau, logm, yATM, y_true) tensors shape (16, 1) in realistic ranges."""
    torch.manual_seed(0)
    N = 16
    tau = torch.rand(N, 1) * 1.98 + 0.02       # [0.02, 2.0]
    logm = torch.rand(N, 1) * 1.0 - 0.5         # [-0.5, 0.5]
    yATM = torch.rand(N, 1) * 0.099 + 0.001     # [0.001, 0.1]
    y_true = torch.rand(N, 1) * 0.099 + 0.001   # [0.001, 0.1]
    return tau, logm, yATM, y_true


@pytest.fixture
def tiny_c6_batch():
    """(tau_c6, logm_c6, yATM_c6) with extreme logm values."""
    torch.manual_seed(1)
    N = 12
    tau = torch.rand(N, 1) * 1.98 + 0.02
    logm = torch.cat([
        torch.rand(6, 1) * (-2.0) - 1.0,   # negative extremes
        torch.rand(6, 1) * 2.0 + 1.0,       # positive extremes
    ])
    yATM = torch.rand(N, 1) * 0.099 + 0.001
    return tau, logm, yATM


@pytest.fixture
def tiny_multi_model():
    """MultiModel with tiny architecture for fast tests."""
    from model import MultiModel
    return MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)


# ---------------------------------------------------------------------------
# DataLoaders wrapping synthetic data
# ---------------------------------------------------------------------------

def _make_train_data(n=32):
    torch.manual_seed(42)
    tau = torch.rand(n, 1) * 1.98 + 0.02
    logm = torch.rand(n, 1) * 1.0 - 0.5
    y = torch.rand(n, 1) * 0.099 + 0.001
    yATM = torch.rand(n, 1) * 0.099 + 0.001
    return TensorDataset(tau, logm, y, yATM)


def _make_c6_data(n=12):
    torch.manual_seed(43)
    tau = torch.rand(n, 1) * 1.98 + 0.02
    logm = torch.cat([
        torch.rand(n // 2, 1) * (-2.0) - 1.0,
        torch.rand(n - n // 2, 1) * 2.0 + 1.0,
    ])
    yATM = torch.rand(n, 1) * 0.099 + 0.001
    return TensorDataset(tau, logm, yATM)


@pytest.fixture
def tiny_train_loader():
    return DataLoader(_make_train_data(32), batch_size=16, shuffle=False)


@pytest.fixture
def tiny_val_loader():
    return DataLoader(_make_train_data(16), batch_size=16, shuffle=False)


@pytest.fixture
def tiny_c6_loader():
    return DataLoader(_make_c6_data(12), batch_size=12, shuffle=False)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_config():
    """ConfigParser with all sections populated for test safety."""
    cfg = configparser.ConfigParser()
    cfg['data'] = {
        'data_folder': '../dataset/',
        'data_file': '2009_2023',
        'asset_file': 'TWII.csv',
        'prs_dataset': 'prs_dataset_no_fat(clean)',
        'vix_file': 'VIX.csv',
    }
    cfg['training'] = {
        'batch_size': '16',
        'epochs': '2',
        'seed': '42',
        'train_start_date': '20140101',
        'train_end_date': '20201231',
        'test_start_date': '20210101',
        'test_end_date': '20211231',
        'scheduler_milestones_start': '1',
        'scheduler_milestones_step': '1',
        'scheduler_gamma': '0.5',
        'gradient_clip': '1.0',
        'early_stopping_patience': '5',
    }
    cfg['model_sett'] = {
        'learning_rate': '0.01',
        'hidden_sizes': '5,5,5',
        'ensemble_num': '2',
        'best_loss': '999.0',
        'loss_weights': '1,1,10,10,10,10',
    }
    cfg['save_path'] = {
        'plot_path': '../LossPlot',
        'model_path': '../models/MultiModel.pt',
        'dgm_model_path': '../models/DGMModel.pt',
        'adjustment_model_path': '../models/AdjustmentModel.pt',
        'hyperiv_model_path': '../models/HyperIVModel.pt',
        'diffusion_model_path': '../models/DiffusionModel.pt',
        'log_dir': '../logs',
    }
    cfg['dgm'] = {
        'n_layers': '2', 'hidden_dim': '8',
        'sigma_min': '0.05', 'sigma_max': '1.0',
        'T_min': '0.02', 'T_max': '2.0',
        'S_min': '0.5', 'S_max': '1.5',
        'lambda_pde': '1.0', 'lambda_bc': '1.0', 'lambda_tc': '1.0',
        'transfer_lr': '0.0001', 'transfer_l2_lambda': '0.01',
        'resample_every': '100',
        'n_interior': '100', 'n_boundary': '20', 'n_terminal': '20',
        'epochs': '2',
    }
    cfg['adjustment'] = {
        'gru_hidden_dim': '8', 'gru_layers': '1',
        'attention_heads': '4', 'dropout': '0.1',
        'prediction_target': 'ratio',
        'event_dates': '200109,200810', 'eval_date': '202003',
        'oversampling_factor': '5', 'kde_bandwidth': '0.1',
        'epochs': '2', 'learning_rate': '0.001',
        'batch_size': '16', 'sequence_length': '5',
    }
    cfg['structural_break'] = {
        'method': 'cusum',
        'threshold': '2.0', 'lookback': '10',
        'confidence': '0.95',
        'vix_threshold': '0.05', 'sustained_days': '3',
    }
    cfg['hyperiv'] = {
        'embed_dim': '16', 'transformer_heads': '4',
        'transformer_layers': '1', 'target_hidden_dims': '8,4',
        'n_reference': '10',
        'epochs': '2', 'learning_rate': '0.001', 'batch_size': '4',
    }
    cfg['diffusion'] = {
        'n_tau_grid': '3', 'n_logm_grid': '4',
        'unet_channels': '8,16,32', 'timesteps': '10',
        'epochs': '2', 'learning_rate': '0.001', 'batch_size': '2',
        'condition_dim': '4',
    }
    cfg['hyperparams'] = {
        'random_seeds': '0,1',
        'gammas': '0.5,0.85,1',
        'init_learning_rate': '0.0003',
    }
    cfg['best_performance'] = {}
    return cfg


# ---------------------------------------------------------------------------
# Mock DataProcessor-compatible datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_prs_dataset():
    """Synthetic DataFrame mimicking preprocessed option data: 50 rows, 5 dates."""
    np.random.seed(99)
    dates = pd.date_range('2020-01-02', periods=5, freq='B')
    rows = []
    for d in dates:
        for _ in range(10):
            tau = np.random.uniform(0.02, 2.0)
            logm = np.random.uniform(-0.5, 0.5)
            iv = np.random.uniform(0.1, 0.5)
            underlying = 10000 + np.random.uniform(-500, 500)
            strike = underlying * np.exp(logm)
            total_var = iv ** 2 * tau
            rows.append({
                'date': d,
                'exdate': d + pd.Timedelta(days=int(tau * 252)),
                'tau': tau,
                'logm': logm,
                'strike_price': strike,
                'underlying': underlying,
                'put/call': 'call',
                'underlying_beta': underlying,
                'strike_price_beta': strike,
                'option_price': 100.0,
                'implied_vol': iv,
                'total_var': total_var,
                'volume': 100,
            })
    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def patched_data_processor(mock_config, mock_prs_dataset):
    """DataProcessor with prs/syn datasets injected—no file I/O."""
    from dataset import DataProcessor
    dp = DataProcessor(mock_config)

    # Inject prs_dataset
    dp.prs_dataset = mock_prs_dataset.copy()

    # Build y_atm from the injected dataset
    atm_idxs = dp.prs_dataset.groupby('tau').apply(
        lambda x: (x['underlying'] - x['strike_price']).abs().idxmin()
    )
    atm_rows = dp.prs_dataset.loc[atm_idxs]
    from scipy.interpolate import interp1d
    atm_fun = interp1d(atm_rows['tau'], atm_rows['total_var'],
                        kind='linear', fill_value='extrapolate', bounds_error=False)
    dp.prs_dataset['y_atm'] = atm_fun(dp.prs_dataset['tau'])

    # Inject syn_dataset
    taus = dp.prs_dataset['tau'].unique()
    logm_seq = [-3.0, -2.5, -2.0, 2.0, 2.5, 3.0]
    syn = pd.DataFrame(
        [(t, l) for t in taus for l in logm_seq],
        columns=['tau', 'logm'],
    )
    syn['y_atm'] = atm_fun(syn['tau'])
    dp.syn_dataset = syn

    return dp

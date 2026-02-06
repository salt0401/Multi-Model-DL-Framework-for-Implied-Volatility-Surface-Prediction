import torch
import numpy as np
import random
import logging
import json
import os
import configparser
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility across torch/numpy/random/cudnn."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir, name='training'):
    """Set up file + console logging."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(path='config.ini'):
    """Load and return a ConfigParser object."""
    config = configparser.ConfigParser()
    config.read(path)
    return config


def parse_list_config(s, dtype=float):
    """Parse a comma-separated config string into a list of dtype."""
    return [dtype(x.strip()) for x in s.split(',')]


def parse_date(s):
    """Parse date string YYYYMMDD to datetime."""
    return datetime.strptime(s.strip(), '%Y%m%d')


class MetricsTracker:
    """Track per-epoch losses and find best epoch."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def update(self, epoch, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                return True
        return False

    def save(self, path):
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class EarlyStopping:
    """Patience-based early stopping."""

    def __init__(self, patience=50, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_rmse(y_pred, y_true):
    """Compute RMSE between predictions and ground truth (numpy arrays)."""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def compute_mape(y_pred, y_true, epsilon=0.005):
    """Compute MAPE between predictions and ground truth (numpy arrays)."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)))


def compute_iv_rmse(tv_pred, tv_true, tau):
    """Compute IV-RMSE: RMSE of implied volatilities derived from total variance."""
    iv_pred = np.sqrt(np.maximum(tv_pred, 0) / tau)
    iv_true = np.sqrt(np.maximum(tv_true, 0) / tau)
    return np.sqrt(np.mean((iv_pred - iv_true) ** 2))

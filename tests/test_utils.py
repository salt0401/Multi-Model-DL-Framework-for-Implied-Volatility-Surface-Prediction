"""Tests for utils.py pure functions and helper classes."""
import json
import os

import numpy as np
import pytest
import torch

from utils import (
    set_seed,
    parse_list_config,
    parse_date,
    MetricsTracker,
    EarlyStopping,
    compute_rmse,
    compute_mape,
    compute_iv_rmse,
    load_config,
    setup_logging,
)


# ── set_seed ──────────────────────────────────────────────────────────

class TestSetSeed:
    def test_same_seed_same_randn(self):
        set_seed(0)
        a = torch.randn(5)
        set_seed(0)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_different_seeds_differ(self):
        set_seed(0)
        a = torch.randn(5)
        set_seed(1)
        b = torch.randn(5)
        assert not torch.allclose(a, b)


# ── parse_list_config ─────────────────────────────────────────────────

class TestParseListConfig:
    def test_float_list(self):
        assert parse_list_config('1.0, 2.5, 3.0') == [1.0, 2.5, 3.0]

    def test_int_list(self):
        assert parse_list_config('10,20,30', dtype=int) == [10, 20, 30]

    def test_single_value(self):
        assert parse_list_config('42', dtype=int) == [42]

    def test_extra_whitespace(self):
        assert parse_list_config('  1 ,  2 ,  3  ', dtype=int) == [1, 2, 3]


# ── parse_date ────────────────────────────────────────────────────────

class TestParseDate:
    def test_valid_date(self):
        d = parse_date('20210101')
        assert d.year == 2021 and d.month == 1 and d.day == 1

    def test_whitespace_handling(self):
        d = parse_date('  20210615  ')
        assert d.month == 6 and d.day == 15


# ── MetricsTracker ────────────────────────────────────────────────────

class TestMetricsTracker:
    def test_update_returns_true_on_improvement(self):
        mt = MetricsTracker()
        assert mt.update(0, 1.0, 0.5) is True

    def test_update_returns_false_no_improvement(self):
        mt = MetricsTracker()
        mt.update(0, 1.0, 0.5)
        assert mt.update(1, 0.9, 0.6) is False

    def test_best_epoch_tracking(self):
        mt = MetricsTracker()
        mt.update(0, 1.0, 0.5)
        mt.update(1, 0.8, 0.3)
        mt.update(2, 0.7, 0.4)
        assert mt.best_epoch == 1
        assert mt.best_val_loss == 0.3

    def test_save_to_json(self, tmp_path):
        mt = MetricsTracker()
        mt.update(0, 1.0, 0.5)
        path = str(tmp_path / 'metrics.json')
        mt.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data['best_epoch'] == 0
        assert len(data['train_losses']) == 1

    def test_update_no_val_loss(self):
        mt = MetricsTracker()
        assert mt.update(0, 1.0) is False


# ── EarlyStopping ─────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_patience_countdown(self):
        es = EarlyStopping(patience=3)
        es.step(1.0)
        assert es.step(1.5) is False  # counter=1
        assert es.step(1.5) is False  # counter=2
        assert es.step(1.5) is True   # counter=3

    def test_reset_on_improvement(self):
        es = EarlyStopping(patience=2)
        es.step(1.0)
        es.step(1.5)  # counter=1
        es.step(0.5)  # improvement -> counter=0
        assert es.counter == 0
        assert es.step(1.0) is False  # counter=1
        assert es.step(1.0) is True   # counter=2

    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        # Improvement by 0.05 < 0.1 min_delta -> no reset
        assert es.step(0.95) is False
        assert es.counter == 1


# ── compute_rmse ──────────────────────────────────────────────────────

class TestComputeRMSE:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(a, a) == pytest.approx(0.0)

    def test_known_value(self):
        pred = np.array([1.0, 1.0, 1.0])
        true = np.array([0.0, 0.0, 0.0])
        assert compute_rmse(pred, true) == pytest.approx(1.0)


# ── compute_mape ──────────────────────────────────────────────────────

class TestComputeMAPE:
    def test_zero_error(self):
        a = np.array([1.0, 2.0, 3.0])
        assert compute_mape(a, a) == pytest.approx(0.0)

    def test_epsilon_prevents_div_by_zero(self):
        pred = np.array([0.1])
        true = np.array([0.0])
        # Should not raise, epsilon=0.005 prevents division by zero
        result = compute_mape(pred, true)
        assert np.isfinite(result)


# ── compute_iv_rmse ───────────────────────────────────────────────────

class TestComputeIVRMSE:
    def test_known_values(self):
        tv = np.array([0.04])
        tau = np.array([1.0])
        # IV = sqrt(0.04/1.0) = 0.2
        assert compute_iv_rmse(tv, tv, tau) == pytest.approx(0.0)

    def test_negative_tv_clamped(self):
        tv_pred = np.array([-0.01])
        tv_true = np.array([0.04])
        tau = np.array([1.0])
        result = compute_iv_rmse(tv_pred, tv_true, tau)
        assert np.isfinite(result)


# ── load_config ───────────────────────────────────────────────────────

class TestLoadConfig:
    def test_reads_ini(self, tmp_path):
        ini = tmp_path / 'test.ini'
        ini.write_text('[section]\nkey = value\n')
        cfg = load_config(str(ini))
        assert cfg['section']['key'] == 'value'


# ── setup_logging ─────────────────────────────────────────────────────

class TestSetupLogging:
    def test_creates_dir_and_returns_logger(self, tmp_path):
        log_dir = str(tmp_path / 'logs')
        logger = setup_logging(log_dir, 'test_logger')
        assert os.path.isdir(log_dir)
        assert logger is not None
        assert logger.name == 'test_logger'

    def test_idempotent(self, tmp_path):
        log_dir = str(tmp_path / 'logs2')
        logger1 = setup_logging(log_dir, 'idem_logger')
        n_handlers = len(logger1.handlers)
        logger2 = setup_logging(log_dir, 'idem_logger')
        assert logger1 is logger2
        assert len(logger2.handlers) == n_handlers

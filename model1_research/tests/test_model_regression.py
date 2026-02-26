import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""Regression tests for the 17 fixed bugs (M1-M5, X1-X3, T1-T2, E1).

Each test is explicitly named after its bug ID to serve as a guardrail
against regressions.
"""
import inspect

import pytest
import torch

from model import (
    BSModel,
    SSVIModel,
    SmileModel,
    SingleModel,
    SoftmaxModel,
    MultiModel,
    WeightedSumLoss,
    Loss_linear,
)


class TestRegressionM1:
    """M1: BSModel.__init__ should take no args (removed unused atm_fun param)."""

    def test_bs_model_no_args(self):
        model = BSModel()
        sig = inspect.signature(BSModel.__init__)
        params = list(sig.parameters.keys())
        assert params == ['self'], f"BSModel.__init__ has unexpected params: {params}"


class TestRegressionM2:
    """M2: BSModel.forward must return 4-tuple (output, grad_ttm, grad_logm, grad_logm2)."""

    def test_four_tuple_return(self):
        model = BSModel()
        logm = torch.randn(4, 1, dtype=torch.float64)
        yATM = torch.rand(4, 1, dtype=torch.float64) + 0.01
        result = model(logm, yATM)
        assert isinstance(result, tuple)
        assert len(result) == 4


class TestRegressionM3:
    """M3: SoftmaxModel takes 3 inputs (ttm, logm, yATM), weight matrix is (3, N)."""

    def test_three_inputs_accepted(self):
        model = SoftmaxModel(ensemble_num=2)
        tau = torch.randn(4, 1, dtype=torch.float64)
        logm = torch.randn(4, 1, dtype=torch.float64)
        yATM = torch.randn(4, 1, dtype=torch.float64)
        out = model(tau, logm, yATM)
        assert out.shape == (4, 2)

    def test_weight_shape_3xN(self):
        model = SoftmaxModel(ensemble_num=5)
        assert model.in_hid_weights.shape == (3, 5)

    def test_multi_model_passes_yATM(self):
        """MultiModel.forward must pass yATM to SoftmaxModel."""
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        tau = torch.rand(4, 1, dtype=torch.float64) * 1.98 + 0.02
        logm = torch.rand(4, 1, dtype=torch.float64) - 0.5
        yATM = torch.rand(4, 1, dtype=torch.float64) * 0.1 + 0.001
        # Should not raise TypeError about missing argument
        out, _, _, _ = model(tau, logm, yATM)
        assert out.shape == (4, 1)


class TestRegressionM4:
    """M4: Loss_linear takes only grad_logm_2nd (1 arg, not 3)."""

    def test_one_arg_signature(self):
        loss_fn = Loss_linear()
        sig = inspect.signature(loss_fn.forward)
        params = [p for p in sig.parameters.keys() if p != 'self']
        assert len(params) == 1, f"Loss_linear.forward has {len(params)} params, expected 1"

    def test_accepts_single_tensor(self):
        loss_fn = Loss_linear()
        grad = torch.randn(8, 1, dtype=torch.float64)
        result = loss_fn(grad)
        assert result.dim() == 0


class TestRegressionM5:
    """M5: WeightedSumLoss returns single tensor, not tuple."""

    def test_scalar_return(self):
        loss_fn = WeightedSumLoss()
        args = [torch.randn(4, 1, dtype=torch.float64) for _ in range(9)]
        args[0] = args[0].requires_grad_(True)
        result = loss_fn(*args)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0
        assert not isinstance(result, tuple)


class TestRegressionX1:
    """X1: weights is register_buffer with float64, not torch.IntTensor."""

    def test_float64_buffer(self):
        loss_fn = WeightedSumLoss()
        assert loss_fn.weights.dtype == torch.float64

    def test_not_parameter(self):
        loss_fn = WeightedSumLoss()
        for name, _ in loss_fn.named_parameters():
            assert name != 'weights'

    def test_is_buffer(self):
        loss_fn = WeightedSumLoss()
        buffers = dict(loss_fn.named_buffers())
        assert 'weights' in buffers


class TestRegressionX2:
    """X2: torch.stack() preserves gradient, unlike torch.FloatTensor([losses])."""

    def test_gradient_preserved(self):
        loss_fn = WeightedSumLoss()
        y_pred = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        args = [y_pred] + [torch.randn(4, 1, dtype=torch.float64) for _ in range(8)]
        result = loss_fn(*args)
        assert result.requires_grad
        result.backward()
        assert y_pred.grad is not None
        assert y_pred.grad.abs().sum() > 0


class TestRegressionX3:
    """X3: weights_exp shape (J, hidden_sizes[0]) and bias_exp (1, hidden_sizes[0])."""

    def test_weights_exp_shape(self):
        model = SmileModel(hidden_sizes=[7, 5, 3])
        assert model.weights_exp.shape == (7, 7)

    def test_bias_exp_shape(self):
        model = SmileModel(hidden_sizes=[7, 5, 3])
        assert model.bias_exp.shape == (1, 7)


class TestRegressionT1:
    """T1: Prepare_train_data exists (renamed from Prepare_prs_dataset)."""

    def test_prepare_train_data_exists(self):
        from dataset import DataProcessor
        assert hasattr(DataProcessor, 'Prepare_train_data')

    def test_old_name_does_not_exist(self):
        from dataset import DataProcessor
        assert not hasattr(DataProcessor, 'Prepare_prs_dataset')


class TestRegressionT2:
    """T2: WeightedSumLoss takes weights= kwarg, rejects positional device."""

    def test_weights_kwarg(self):
        loss_fn = WeightedSumLoss(weights=[1, 2, 3, 4, 5, 6])
        expected = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float64)
        assert torch.allclose(loss_fn.weights, expected)

    def test_no_device_parameter(self):
        """Old interface had device= parameter. New interface only has weights=."""
        sig = inspect.signature(WeightedSumLoss.__init__)
        params = list(sig.parameters.keys())
        assert 'device' not in params
        assert 'weights' in params


class TestRegressionE1:
    """E1: All models return 4-tuple (output, grad_tau, grad_logm, grad_logm2)."""

    def test_bs_four_tuple_return(self):
        model = BSModel()
        logm = torch.randn(4, 1, dtype=torch.float64)
        yATM = torch.rand(4, 1, dtype=torch.float64) + 0.01
        result = model(logm, yATM)
        assert isinstance(result, tuple) and len(result) == 4

    def test_ssvi_four_tuple_return(self):
        model = SSVIModel()
        logm = torch.randn(4, 1, dtype=torch.float64)
        tau = torch.rand(4, 1, dtype=torch.float64) * 1.98 + 0.02
        yATM = torch.rand(4, 1, dtype=torch.float64) + 0.01
        result = model(logm, tau, yATM)
        assert isinstance(result, tuple) and len(result) == 4

    def test_smile_four_tuple(self):
        model = SmileModel(hidden_sizes=[5, 5, 5])
        tau = torch.rand(4, 1, dtype=torch.float64) * 1.98 + 0.02
        logm = torch.rand(4, 1, dtype=torch.float64) - 0.5
        result = model(tau, logm)
        assert isinstance(result, tuple) and len(result) == 4

    def test_multi_four_tuple(self):
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        tau = torch.rand(4, 1, dtype=torch.float64) * 1.98 + 0.02
        logm = torch.rand(4, 1, dtype=torch.float64) - 0.5
        yATM = torch.rand(4, 1, dtype=torch.float64) * 0.1 + 0.001
        result = model(tau, logm, yATM)
        assert isinstance(result, tuple) and len(result) == 4

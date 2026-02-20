"""Tests for model.py: all model forward passes and loss classes."""
import pytest
import torch
import torch.nn as nn

from model import (
    BSModel,
    SSVIModel,
    SmileModel,
    SingleModel,
    SoftmaxModel,
    MultiModel,
    RMSELoss,
    MAPELoss,
    Loss_calendar,
    Loss_butterfly,
    Loss_linear,
    Loss_upperbound,
    WeightedSumLoss,
    SimpleLoss,
)


# ── BSModel ───────────────────────────────────────────────────────────

class TestBSModel:
    def test_output_shape(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = BSModel()
        out, g1, g2, g3 = model(logm, yATM)
        assert out.shape == (16, 1)
        assert g1.shape == g2.shape == g3.shape == (16, 1)

    def test_values_yATM_plus_zeros(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = BSModel()
        out, g1, g2, g3 = model(logm, yATM)
        assert torch.allclose(out, yATM)
        assert torch.allclose(g1, torch.zeros_like(yATM))

    def test_dtype_float64(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = BSModel()
        out, _, _, _ = model(logm, yATM)
        assert out.dtype == torch.float64

    def test_no_learnable_params(self):
        model = BSModel()
        assert list(model.parameters()) == []


# ── SSVIModel ─────────────────────────────────────────────────────────

class TestSSVIModel:
    def test_output_shape(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = SSVIModel()
        out, g_tau, g_logm, g_logm2 = model(logm, yATM)
        assert out.shape == (16, 1)

    def test_dtype(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = SSVIModel()
        out, _, _, _ = model(logm, yATM)
        assert out.dtype == torch.float64

    def test_power_law_params(self):
        model = SSVIModel(phi_fun='power_law')
        param_names = [n for n, _ in model.named_parameters()]
        assert 'raw_eta' in param_names
        assert 'raw_gamma' in param_names

    def test_heston_like_params(self):
        model = SSVIModel(phi_fun='heston_like')
        param_names = [n for n, _ in model.named_parameters()]
        assert 'raw_lambda' in param_names
        assert 'raw_eta' not in param_names

    def test_rho_initial_sign_negative(self):
        """rho should initialize negative for equity index left-skew (fear premium)."""
        model = SSVIModel()
        rho = -torch.sigmoid(model.raw_rho)
        assert rho.item() < 0, f"rho should be negative for equity index, got {rho.item():.4f}"

    def test_left_skew_at_init(self):
        """At initialization, OTM put (logm<0) should have higher TV than OTM call (logm>0)."""
        model = SSVIModel()
        yATM = torch.tensor([[0.05]], dtype=torch.float64)
        logm_put = torch.tensor([[-0.3]], dtype=torch.float64)   # OTM put
        logm_call = torch.tensor([[0.3]], dtype=torch.float64)   # OTM call
        out_put, _, _, _ = model(logm_put, yATM)
        out_call, _, _, _ = model(logm_call, yATM)
        assert out_put.item() > out_call.item(), (
            f"Left skew violated: OTM put TV ({out_put.item():.6f}) "
            f"should exceed OTM call TV ({out_call.item():.6f})"
        )

    def test_rho_bounded(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = SSVIModel()
        # rho = -sigmoid(raw_rho) so should be in (-1, 0)
        rho = -torch.sigmoid(model.raw_rho)
        assert -1 < rho.item() < 0, f"rho should be in (-1, 0), got {rho.item():.4f}"

    def test_output_positive(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        # With positive yATM and small logm, SSVI output should be positive
        model = SSVIModel()
        out, _, _, _ = model(logm, yATM)
        assert (out > 0).all()

    def test_grad_logm2_positive(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = SSVIModel()
        _, _, _, grad_logm2 = model(logm, yATM)
        # Second derivative of SSVI w.r.t. logm should be positive
        assert (grad_logm2 >= 0).all()

    def test_gradient_flow(self, tiny_batch):
        _, logm, yATM, _ = tiny_batch
        model = SSVIModel()
        out, _, _, _ = model(logm, yATM)
        loss = out.sum()
        loss.backward()
        assert model.raw_rho.grad is not None


# ── SmileModel ────────────────────────────────────────────────────────

class TestSmileModel:
    def test_output_shape(self, tiny_batch):
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[5, 5, 5])
        out, g_tau, g_logm, g_logm2 = model(tau, logm)
        assert out.shape == (16, 1)
        assert g_tau.shape == g_logm.shape == g_logm2.shape == (16, 1)

    def test_dtype(self, tiny_batch):
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[5, 5, 5])
        out, _, _, _ = model(tau, logm)
        assert out.dtype == torch.float64

    def test_autograd_graph_alive(self, tiny_batch):
        """SmileModel uses autograd internally—DO NOT wrap in no_grad."""
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[5, 5, 5])
        out, g_tau, g_logm, g_logm2 = model(tau, logm)
        # All outputs should have grad_fn (graph alive)
        assert out.grad_fn is not None
        assert g_tau.grad_fn is not None

    def test_gradient_backprop(self, tiny_batch):
        """SmileModel uses autograd.grad internally, so backward still works
        on the returned output (create_graph=True keeps the graph alive)."""
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[5, 5, 5])
        # SmileModel.forward already calls autograd.grad(create_graph=True)
        # which means the output still has grad_fn and we can backward
        out, g_tau, g_logm, g_logm2 = model(tau, logm)
        # Backward through the full 4-tuple sum to exercise the whole graph
        total = out.sum() + g_tau.sum() + g_logm.sum() + g_logm2.sum()
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_different_hidden_sizes(self, tiny_batch):
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[8, 4, 4])
        out, _, _, _ = model(tau, logm)
        assert out.shape == (16, 1)

    def test_tanh_activation(self, tiny_batch):
        tau, logm, _, _ = tiny_batch
        model = SmileModel(hidden_sizes=[5, 5, 5], activation='tanh')
        out, _, _, _ = model(tau, logm)
        assert out.shape == (16, 1)

    def test_weights_exp_shape(self):
        """Regression X3: weights_exp should be (J, J) not mismatched."""
        model = SmileModel(hidden_sizes=[5, 5, 5])
        assert model.weights_exp.shape == (5, 5)
        assert model.bias_exp.shape == (1, 5)


# ── SingleModel ───────────────────────────────────────────────────────

class TestSingleModel:
    def test_output_shape(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SingleModel(hidden_sizes=[5, 5, 5])
        out, g1, g2, g3 = model(tau, logm, yATM)
        assert out.shape == (16, 1)

    def test_additive_derivatives_exist(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SingleModel(hidden_sizes=[5, 5, 5])
        out, g_tau, g_logm, g_logm2 = model(tau, logm, yATM)
        # All should be non-None tensors
        for g in (g_tau, g_logm, g_logm2):
            assert isinstance(g, torch.Tensor)
            assert g.shape == (16, 1)

    def test_bs_variant(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SingleModel(hidden_sizes=[5, 5, 5], prior='BS')
        out, _, _, _ = model(tau, logm, yATM)
        assert out.shape == (16, 1)

    def test_ssvi_variant(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SingleModel(hidden_sizes=[5, 5, 5], prior='SSVI')
        out, _, _, _ = model(tau, logm, yATM)
        assert out.shape == (16, 1)

    def test_gradient_flow(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SingleModel(hidden_sizes=[5, 5, 5])
        out, g_tau, g_logm, g_logm2 = model(tau, logm, yATM)
        total = out.sum() + g_tau.sum() + g_logm.sum() + g_logm2.sum()
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ── SoftmaxModel ─────────────────────────────────────────────────────

class TestSoftmaxModel:
    def test_output_shape(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SoftmaxModel(ensemble_num=3)
        out = model(tau, logm, yATM)
        assert out.shape == (16, 3)

    def test_weights_sum_to_one(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = SoftmaxModel(ensemble_num=3)
        out = model(tau, logm, yATM)
        sums = out.sum(dim=1)
        assert torch.allclose(sums, torch.ones(16, dtype=torch.float64), atol=1e-6)

    def test_takes_3_inputs(self, tiny_batch):
        """Bug M3: SoftmaxModel takes (ttm, logm, yATM)."""
        tau, logm, yATM, _ = tiny_batch
        model = SoftmaxModel(ensemble_num=2)
        # Should work with 3 args
        out = model(tau, logm, yATM)
        assert out.shape[1] == 2

    def test_weight_matrix_shape(self):
        """Bug M3: weight matrix is (3, N) for 3 inputs."""
        model = SoftmaxModel(ensemble_num=4)
        assert model.in_hid_weights.shape == (3, 4)


# ── MultiModel ────────────────────────────────────────────────────────

class TestMultiModel:
    def test_output_shape(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        out, g1, g2, g3 = model(tau, logm, yATM)
        assert out.shape == (16, 1)

    def test_ensemble_count(self):
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=3)
        assert len(model.ensemble_list) == 3

    def test_weighted_sum(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        out, _, _, _ = model(tau, logm, yATM)
        # Should be a weighted sum (just check it's finite)
        assert torch.isfinite(out).all()

    def test_full_gradient_flow(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        out, g_tau, g_logm, g_logm2 = model(tau, logm, yATM)
        total = out.sum() + g_tau.sum() + g_logm.sum() + g_logm2.sum()
        total.backward()
        # Check gradient reaches ensemble members
        for sm in model.ensemble_list:
            for p in sm.parameters():
                if p.requires_grad:
                    assert p.grad is not None, f"No gradient for {p.shape}"

    def test_yATM_to_softmax(self, tiny_batch):
        """Bug M3: yATM is passed to SoftmaxModel."""
        tau, logm, yATM, _ = tiny_batch
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        # Just ensure no error—SoftmaxModel expects 3 inputs
        out, _, _, _ = model(tau, logm, yATM)
        assert out.shape == (16, 1)

    def test_no_nan(self, tiny_batch):
        tau, logm, yATM, _ = tiny_batch
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        out, g1, g2, g3 = model(tau, logm, yATM)
        for t in (out, g1, g2, g3):
            assert not torch.isnan(t).any()

    def test_batch_size_one(self):
        tau = torch.rand(1, 1) * 1.98 + 0.02
        logm = torch.rand(1, 1) * 1.0 - 0.5
        yATM = torch.rand(1, 1) * 0.099 + 0.001
        model = MultiModel(hidden_sizes=[5, 5, 5], ensemble_num=2)
        out, _, _, _ = model(tau, logm, yATM)
        assert out.shape == (1, 1)


# ── RMSELoss ─────────────────────────────────────────────────────────

class TestRMSELoss:
    def test_known_value(self):
        loss_fn = RMSELoss()
        pred = torch.ones(4, 1, dtype=torch.float64)
        true = torch.zeros(4, 1, dtype=torch.float64)
        assert loss_fn(pred, true).item() == pytest.approx(1.0)

    def test_gradient_flow(self):
        loss_fn = RMSELoss()
        pred = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        true = torch.randn(4, 1, dtype=torch.float64)
        loss = loss_fn(pred, true)
        loss.backward()
        assert pred.grad is not None


# ── MAPELoss ──────────────────────────────────────────────────────────

class TestMAPELoss:
    def test_known_value(self):
        loss_fn = MAPELoss()
        pred = torch.tensor([[1.0]], dtype=torch.float64)
        true = torch.tensor([[1.0]], dtype=torch.float64)
        assert loss_fn(pred, true).item() == pytest.approx(0.0)

    def test_gradient_flow(self):
        loss_fn = MAPELoss()
        pred = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        true = torch.rand(4, 1, dtype=torch.float64) + 0.01
        loss = loss_fn(pred, true)
        loss.backward()
        assert pred.grad is not None


# ── Loss_calendar ─────────────────────────────────────────────────────

class TestLossCalendar:
    def test_zero_for_positive_grad(self):
        loss_fn = Loss_calendar()
        y_pred = torch.ones(4, 1, dtype=torch.float64)
        grad_tau = torch.ones(4, 1, dtype=torch.float64)  # all positive
        assert loss_fn(y_pred, grad_tau).item() == pytest.approx(0.0)

    def test_nonzero_for_negative_grad(self):
        loss_fn = Loss_calendar()
        y_pred = torch.ones(4, 1, dtype=torch.float64)
        grad_tau = -torch.ones(4, 1, dtype=torch.float64)
        assert loss_fn(y_pred, grad_tau).item() > 0


# ── Loss_butterfly ────────────────────────────────────────────────────

class TestLossButterfly:
    def test_scalar_output(self, tiny_batch):
        loss_fn = Loss_butterfly()
        _, logm, _, y_pred = tiny_batch
        y_pred = y_pred.abs() + 0.01  # ensure positive
        grad_logm = torch.randn_like(logm)
        grad_logm2 = torch.randn_like(logm)
        loss = loss_fn(y_pred, logm, grad_logm, grad_logm2)
        assert loss.dim() == 0

    def test_gradient_flow(self, tiny_batch):
        loss_fn = Loss_butterfly()
        _, logm, _, y_pred = tiny_batch
        y_pred = (y_pred.abs() + 0.01).requires_grad_(True)
        grad_logm = torch.randn_like(logm, requires_grad=True)
        grad_logm2 = torch.randn_like(logm, requires_grad=True)
        loss = loss_fn(y_pred, logm, grad_logm, grad_logm2)
        loss.backward()
        assert y_pred.grad is not None


# ── Loss_linear ───────────────────────────────────────────────────────

class TestLossLinear:
    def test_one_arg_signature(self):
        """Bug M4: Loss_linear only takes grad_logm_2nd."""
        loss_fn = Loss_linear()
        grad = torch.randn(4, 1, dtype=torch.float64)
        loss = loss_fn(grad)
        assert loss.dim() == 0


# ── Loss_upperbound ───────────────────────────────────────────────────

class TestLossUpperbound:
    def test_zero_when_below(self):
        loss_fn = Loss_upperbound()
        logm = torch.tensor([[0.1]], dtype=torch.float64)
        y_pred = torch.tensor([[0.0]], dtype=torch.float64)  # below 2*|logm|=0.2
        assert loss_fn(y_pred, logm).item() == pytest.approx(0.0)

    def test_nonzero_when_above(self):
        loss_fn = Loss_upperbound()
        logm = torch.tensor([[0.1]], dtype=torch.float64)
        y_pred = torch.tensor([[1.0]], dtype=torch.float64)  # above 2*|0.1|=0.2
        assert loss_fn(y_pred, logm).item() > 0


# ── WeightedSumLoss ──────────────────────────────────────────────────

class TestWeightedSumLoss:
    def _make_args(self, n=8):
        """Create 9-arg inputs for WeightedSumLoss.forward."""
        y_pred = torch.randn(n, 1, dtype=torch.float64, requires_grad=True)
        y_true = torch.randn(n, 1, dtype=torch.float64)
        logm = torch.randn(n, 1, dtype=torch.float64)
        grad_tau = torch.randn(n, 1, dtype=torch.float64)
        grad_logm = torch.randn(n, 1, dtype=torch.float64)
        grad_logm2 = torch.randn(n, 1, dtype=torch.float64)
        y_pred_c6 = torch.randn(n, 1, dtype=torch.float64)
        logm_c6 = torch.randn(n, 1, dtype=torch.float64)
        grad_logm_c6_2nd = torch.randn(n, 1, dtype=torch.float64)
        return (y_pred, y_true, logm, grad_tau, grad_logm, grad_logm2,
                y_pred_c6, logm_c6, grad_logm_c6_2nd)

    def test_returns_scalar(self):
        """Bug M5: must return single tensor, not tuple."""
        loss_fn = WeightedSumLoss()
        args = self._make_args()
        result = loss_fn(*args)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0

    def test_nine_arg_signature(self):
        loss_fn = WeightedSumLoss()
        args = self._make_args()
        # Should not raise
        loss_fn(*args)

    def test_default_weights(self):
        loss_fn = WeightedSumLoss()
        expected = torch.tensor([1, 1, 10, 10, 10, 10], dtype=torch.float64)
        assert torch.allclose(loss_fn.weights, expected)

    def test_custom_weights(self):
        loss_fn = WeightedSumLoss(weights=[2, 3, 4, 5, 6, 7])
        expected = torch.tensor([2, 3, 4, 5, 6, 7], dtype=torch.float64)
        assert torch.allclose(loss_fn.weights, expected)

    def test_buffer_not_param(self):
        """Bug X1: weights is register_buffer, not nn.Parameter."""
        loss_fn = WeightedSumLoss()
        param_names = [n for n, _ in loss_fn.named_parameters()]
        assert 'weights' not in param_names
        buffer_names = [n for n, _ in loss_fn.named_buffers()]
        assert 'weights' in buffer_names

    def test_float64(self):
        """Bug X1: weights tensor should be float64."""
        loss_fn = WeightedSumLoss()
        assert loss_fn.weights.dtype == torch.float64

    def test_individual_losses_stored(self):
        loss_fn = WeightedSumLoss()
        args = self._make_args()
        loss_fn(*args)
        assert loss_fn.individual_losses is not None
        assert loss_fn.individual_losses.shape == (6,)

    def test_gradient_flows(self):
        """Bug X2: gradient should not be detached."""
        loss_fn = WeightedSumLoss()
        args = self._make_args()
        result = loss_fn(*args)
        assert result.requires_grad

    def test_gradient_not_detached(self):
        """Bug X2: torch.stack preserves gradients unlike FloatTensor([...])."""
        loss_fn = WeightedSumLoss()
        args = self._make_args()
        result = loss_fn(*args)
        result.backward()
        # y_pred (first arg) should have gradient
        assert args[0].grad is not None


# ── SimpleLoss ────────────────────────────────────────────────────────

class TestSimpleLoss:
    def test_scalar_output(self):
        loss_fn = SimpleLoss()
        pred = torch.randn(4, 1, dtype=torch.float64)
        true = torch.randn(4, 1, dtype=torch.float64).abs() + 0.01
        loss = loss_fn(pred, true)
        assert loss.dim() == 0

    def test_gradient_flow(self):
        loss_fn = SimpleLoss()
        pred = torch.randn(4, 1, dtype=torch.float64, requires_grad=True)
        true = torch.randn(4, 1, dtype=torch.float64).abs() + 0.01
        loss = loss_fn(pred, true)
        loss.backward()
        assert pred.grad is not None

"""Tests for hyperiv.py: HyperIV model components."""
import pytest
import torch

from hyperiv import (
    SetEmbeddingNetwork,
    TargetNetwork,
    _count_target_params,
    _generate_target_params,
    HyperIVModel,
    HyperIVLoss,
)


# ── SetEmbeddingNetwork ──────────────────────────────────────────────

class TestSetEmbeddingNetwork:
    def test_output_shape(self):
        net = SetEmbeddingNetwork(input_dim=3, embed_dim=16, n_heads=4, n_layers=1)
        ref_set = torch.randn(2, 10, 3)
        out = net(ref_set)
        assert out.shape == (2, 16)

    def test_mask_handling(self):
        net = SetEmbeddingNetwork(input_dim=3, embed_dim=16, n_heads=4, n_layers=1)
        ref_set = torch.randn(2, 10, 3)
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[:, 7:] = True  # last 3 padded
        out = net(ref_set, mask=mask)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

    def test_no_mask(self):
        net = SetEmbeddingNetwork(input_dim=3, embed_dim=16, n_heads=4, n_layers=1)
        ref_set = torch.randn(2, 10, 3)
        out = net(ref_set, mask=None)
        assert out.shape == (2, 16)


# ── TargetNetwork ─────────────────────────────────────────────────────

class TestTargetNetwork:
    def test_output_shape(self):
        net = TargetNetwork(hidden_dims=(8, 4))
        x = torch.randn(5, 3)
        out = net(x)
        assert out.shape == (5, 1)

    def test_custom_hidden_dims(self):
        net = TargetNetwork(hidden_dims=(16, 8, 4))
        x = torch.randn(3, 3)
        out = net(x)
        assert out.shape == (3, 1)


# ── _count_target_params ─────────────────────────────────────────────

class TestCountTargetParams:
    def test_known_count(self):
        # (64,32): 3->64: 3*64+64=256, 64->32: 64*32+32=2080, 32->1: 32*1+1=33 = 2369
        # Actually: 3*64+64 = 256, 64*32+32 = 2080, 32*1+1 = 33, total = 2369
        assert _count_target_params((64, 32)) == 256 + 2080 + 33


# ── _generate_target_params ──────────────────────────────────────────

class TestGenerateTargetParams:
    def test_correct_keys(self):
        n_params = _count_target_params((8, 4))
        flat = torch.randn(n_params)
        params = _generate_target_params(flat, (8, 4))
        expected_keys = {'net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias',
                         'net.4.weight', 'net.4.bias'}
        assert set(params.keys()) == expected_keys

    def test_weight_shapes(self):
        n_params = _count_target_params((8, 4))
        flat = torch.randn(n_params)
        params = _generate_target_params(flat, (8, 4))
        assert params['net.0.weight'].shape == (8, 3)
        assert params['net.0.bias'].shape == (8,)
        assert params['net.2.weight'].shape == (4, 8)
        assert params['net.4.weight'].shape == (1, 4)

    def test_all_params_consumed(self):
        n_params = _count_target_params((8, 4))
        flat = torch.randn(n_params)
        params = _generate_target_params(flat, (8, 4))
        total_consumed = sum(p.numel() for p in params.values())
        assert total_consumed == n_params


# ── HyperIVModel ─────────────────────────────────────────────────────

class TestHyperIVModel:
    @pytest.fixture
    def tiny_hyperiv(self):
        return HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                            target_hidden_dims=(8, 4))

    def test_output_shapes(self, tiny_hyperiv):
        batch, n_ref, n_target = 2, 10, 5
        ref_set = torch.randn(batch, n_ref, 3)
        tau = torch.randn(batch, n_target, 1)
        logm = torch.randn(batch, n_target, 1)
        yATM = torch.randn(batch, n_target, 1)

        tv, g_tau, g_logm, g_logm2 = tiny_hyperiv(ref_set, tau, logm, yATM)
        assert tv.shape == (batch, n_target, 1)
        assert g_tau.shape == (batch, n_target, 1)
        assert g_logm.shape == (batch, n_target, 1)
        assert g_logm2.shape == (batch, n_target, 1)

    def test_gradient_flow(self, tiny_hyperiv):
        batch, n_ref, n_target = 2, 8, 3
        ref_set = torch.randn(batch, n_ref, 3)
        tau = torch.randn(batch, n_target, 1)
        logm = torch.randn(batch, n_target, 1)
        yATM = torch.randn(batch, n_target, 1)

        tv, _, _, _ = tiny_hyperiv(ref_set, tau, logm, yATM)
        loss = tv.sum()
        loss.backward()
        # Gradient should reach both encoder and hyper_proj
        assert tiny_hyperiv.set_encoder.proj.weight.grad is not None
        assert tiny_hyperiv.hyper_proj.weight.grad is not None

    def test_autograd_derivatives_nonzero(self, tiny_hyperiv):
        batch, n_ref, n_target = 2, 8, 3
        ref_set = torch.randn(batch, n_ref, 3)
        tau = torch.randn(batch, n_target, 1)
        logm = torch.randn(batch, n_target, 1)
        yATM = torch.randn(batch, n_target, 1)

        _, g_tau, g_logm, g_logm2 = tiny_hyperiv(ref_set, tau, logm, yATM)
        # At least some gradient should be nonzero
        assert g_tau.abs().sum() > 0 or g_logm.abs().sum() > 0


# ── HyperIVLoss ──────────────────────────────────────────────────────

class TestHyperIVLoss:
    def _make_args(self, batch=4, n_target=5):
        tv_pred = torch.randn(batch, n_target, 1, requires_grad=True)
        tv_true = torch.randn(batch, n_target, 1)
        logm = torch.randn(batch, n_target, 1)
        grad_tau = torch.randn(batch, n_target, 1)
        grad_logm = torch.randn(batch, n_target, 1)
        grad_logm2 = torch.randn(batch, n_target, 1)
        return tv_pred, tv_true, logm, grad_tau, grad_logm, grad_logm2

    def test_returns_4_tuple(self):
        loss_fn = HyperIVLoss()
        args = self._make_args()
        result = loss_fn(*args)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_total_is_weighted_sum(self):
        loss_fn = HyperIVLoss(w_mse=1.0, w_calendar=0.0, w_butterfly=0.0)
        args = self._make_args()
        total, mse, cal, but = loss_fn(*args)
        assert torch.allclose(total, mse, atol=1e-8)

    def test_gradient_flows(self):
        loss_fn = HyperIVLoss()
        args = self._make_args()
        total, _, _, _ = loss_fn(*args)
        total.backward()
        assert args[0].grad is not None

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

    def test_second_derivative_nonzero(self, tiny_hyperiv):
        """Smooth (tanh) target net must yield nonzero d2w/dk2 for the
        butterfly penalty to be meaningful (ReLU nets give 0 a.e.)."""
        torch.manual_seed(3)
        batch, n_ref, n_target = 2, 8, 12
        ref_set = torch.rand(batch, n_ref, 3) * 0.1
        tau = torch.rand(batch, n_target, 1)
        logm = torch.randn(batch, n_target, 1) * 0.2
        yATM = torch.rand(batch, n_target, 1) * 0.05
        _, _, _, g_logm2 = tiny_hyperiv(ref_set, tau, logm, yATM)
        assert g_logm2.abs().sum() > 0

    def test_positive_output(self, tiny_hyperiv):
        """Softplus head guarantees total variance > 0."""
        torch.manual_seed(4)
        tv, _, _, _ = tiny_hyperiv(
            torch.rand(2, 8, 3) * 0.1, torch.rand(2, 5, 1),
            torch.randn(2, 5, 1) * 0.2, torch.rand(2, 5, 1) * 0.05)
        assert (tv > 0).all()

    def test_initial_predictions_at_data_scale(self):
        """hyper_proj bias init keeps initial surfaces near tv scale (~0.01),
        not softplus(0) ~ 0.69."""
        torch.manual_seed(5)
        m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                         target_hidden_dims=(8, 4))
        tv, _, _, _ = m(torch.rand(2, 10, 3) * 0.05, torch.rand(2, 5, 1),
                        torch.randn(2, 5, 1) * 0.1, torch.rand(2, 5, 1) * 0.05)
        assert tv.median() < 0.2

    def test_learns_at_realistic_tv_scale(self):
        """Regression: raw-softplus head collapsed to predicting 0 at real tv
        scale (median ~0.005) because the saturated tail kills gradients.
        With the yATM-ratio output the model must beat the predict-zero bar
        after brief training."""
        torch.manual_seed(7)
        m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                         target_hidden_dims=(8, 4))
        loss_fn = HyperIVLoss()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)

        def make(B=8, N=30):
            tau = torch.rand(B, N, 1) * 0.5 + 0.02
            logm = torch.randn(B, N, 1) * 0.15
            yatm = torch.rand(B, N, 1) * 0.008 + 0.001   # realistic tiny scale
            tv = yatm * (1 + 2.0 * logm ** 2 / torch.sqrt(tau))
            ref = torch.cat([tau[:, :15], logm[:, :15], tv[:, :15]], dim=-1)
            return ref, tau, logm, yatm, tv

        zero_bar = None
        for i in range(100):
            ref, tau, logm, yatm, tv = make()
            if zero_bar is None:
                zero_bar = (tv ** 2).mean().item()
            tvp, gt, gl, gl2 = m(ref, tau, logm, yatm)
            total, mse, *_ = loss_fn(tvp, tv, logm, gt, gl, gl2)
            opt.zero_grad()
            total.backward()
            opt.step()
        assert mse.item() < 0.25 * zero_bar, \
            f'mse {mse.item():.2e} did not beat predict-zero bar {zero_bar:.2e}'

    def test_normalization_buffers(self):
        m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                         target_hidden_dims=(8, 4))
        m.set_normalization(torch.tensor([0.3, 0.0, 0.02]),
                            torch.tensor([0.2, 0.15, 0.01]))
        out = m(torch.rand(2, 10, 3) * 0.05, torch.rand(2, 5, 1) * 0.5,
                torch.randn(2, 5, 1) * 0.1, torch.rand(2, 5, 1) * 0.05)
        assert all(torch.isfinite(o).all() for o in out)
        # buffers persist through state_dict round-trip
        m2 = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                          target_hidden_dims=(8, 4))
        m2.load_state_dict(m.state_dict())
        assert torch.allclose(m2.feat_std, torch.tensor([0.2, 0.15, 0.01]))


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

    def test_returns_5_tuple(self):
        loss_fn = HyperIVLoss()
        args = self._make_args()
        result = loss_fn(*args)
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_total_is_weighted_sum(self):
        loss_fn = HyperIVLoss(w_mse=1.0, w_calendar=0.0, w_butterfly=0.0, w_price=0.0)
        args = self._make_args()
        total, mse, cal, but, price = loss_fn(*args)
        assert torch.allclose(total, mse, atol=1e-8)

    def test_gradient_flows(self):
        loss_fn = HyperIVLoss()
        args = self._make_args()
        total, _, _, _, _ = loss_fn(*args)
        total.backward()
        assert args[0].grad is not None

    def test_butterfly_matches_model1_reference(self):
        """g(k) must equal model1_research Loss_butterfly (w' SQUARED term)."""
        from model1_research.model import Loss_butterfly
        torch.manual_seed(0)
        w = torch.rand(20, 1) * 0.05 + 0.01
        k = torch.randn(20, 1) * 0.3
        g1 = torch.rand(20, 1) * 0.1 - 0.05
        g2 = torch.rand(20, 1) * 0.1 - 0.05
        ref = Loss_butterfly()(w, k, g1, g2)
        loss_fn = HyperIVLoss(w_price=0.0)
        _, _, _, but, _ = loss_fn(
            w.unsqueeze(0), w.unsqueeze(0), k.unsqueeze(0),
            torch.zeros(20, 1).unsqueeze(0), g1.unsqueeze(0), g2.unsqueeze(0))
        assert torch.allclose(but, ref, atol=1e-10)

    def test_price_aux_matches_scipy(self):
        """Black-76 normalized call: C = N(d1) - e^k N(d2)."""
        from scipy.stats import norm as scipy_norm
        import numpy as np
        from hyperiv import black76_call_price
        k, w = 0.05, 0.04
        d1 = (-k + w / 2) / np.sqrt(w)
        d2 = d1 - np.sqrt(w)
        expected = scipy_norm.cdf(d1) - np.exp(k) * scipy_norm.cdf(d2)
        got = black76_call_price(torch.tensor([[k]], dtype=torch.float64),
                                 torch.tensor([[w]], dtype=torch.float64))
        assert abs(got.item() - expected) < 1e-8

    def test_masked_loss_ignores_padding(self):
        """Loss with padded entries + valid_mask == loss on the unpadded slice."""
        torch.manual_seed(1)
        B, N = 2, 6
        tv_pred = torch.rand(B, N, 1) * 0.05 + 0.01
        tv_true = torch.rand(B, N, 1) * 0.05 + 0.01
        k = torch.randn(B, N, 1) * 0.2
        gt = torch.randn(B, N, 1) * 0.05
        g1 = torch.randn(B, N, 1) * 0.05
        g2 = torch.randn(B, N, 1) * 0.05
        valid = torch.zeros(B, N, dtype=torch.bool)
        valid[:, :4] = True
        loss_fn = HyperIVLoss()
        full = loss_fn(tv_pred[:, :4], tv_true[:, :4], k[:, :4],
                       gt[:, :4], g1[:, :4], g2[:, :4])
        masked = loss_fn(tv_pred, tv_true, k, gt, g1, g2, valid_mask=valid)
        assert torch.allclose(full[0], masked[0], atol=1e-10)

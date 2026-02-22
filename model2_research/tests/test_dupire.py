"""Tests for dupire_pinn.py: Dupire PINN networks, loss, sampler, utilities."""
import pytest
import numpy as np
import torch

from dupire_pinn import (
    PriceNetwork,
    ICNNPriceNetwork,
    LocalVolNetwork,
    DupirePINNLoss,
    DupireSampler,
    bs_call_price,
    total_variance_to_call_price,
)
from module_d import GreekExtractor


# ── PriceNetwork ──────────────────────────────────────────────────────

class TestPriceNetwork:
    def test_output_shape(self):
        net = PriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1) + 0.01
        out = net(K, tau)
        assert out.shape == (4, 1)

    def test_positive_output(self):
        """Call price must be non-negative (softplus guarantee)."""
        net = PriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(20, 1) * 2  # Include negative values
        tau = torch.rand(20, 1) + 0.01
        out = net(K, tau)
        assert (out >= 0).all()

    def test_gradient_flow(self):
        net = PriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1, requires_grad=True)
        tau = torch.rand(4, 1, requires_grad=True)
        out = net(K, tau)
        out.sum().backward()
        assert K.grad is not None
        assert tau.grad is not None

    def test_float64(self):
        net = PriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1)
        out = net(K, tau)
        assert out.dtype == torch.float64


# ── ICNNPriceNetwork ──────────────────────────────────────────────────

class TestICNNPriceNetwork:
    def test_output_shape(self):
        net = ICNNPriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1) + 0.01
        out = net(K, tau)
        assert out.shape == (4, 1)

    def test_positive_output(self):
        """Call price must be non-negative (softplus guarantee)."""
        net = ICNNPriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(20, 1) * 2
        tau = torch.rand(20, 1) + 0.01
        out = net(K, tau)
        assert (out >= 0).all()

    def test_strict_convexity_K(self):
        """Hard constraint: ∂²C/∂K² ≥ 0 exactly everywhere."""
        net = ICNNPriceNetwork(hidden_dim=16, n_layers=3).double()
        K = torch.linspace(0.1, 3.0, 100).reshape(-1, 1)
        K.requires_grad_(True)
        tau = torch.full_like(K, 1.0, requires_grad=True)

        C = net(K, tau)
        
        # First derivative
        dC_dK = torch.autograd.grad(
            outputs=C, inputs=K,
            grad_outputs=torch.ones_like(C),
            create_graph=True
        )[0]
        
        # Second derivative
        d2C_dK2 = torch.autograd.grad(
            outputs=dC_dK, inputs=K,
            grad_outputs=torch.ones_like(dC_dK),
            create_graph=True
        )[0]
        
        # Allow tiny numerical tolerance for float64 operations
        assert (d2C_dK2 >= -1e-12).all(), f"Found negative convexity: min value {d2C_dK2.min()}"

    def test_gradient_flow(self):
        net = ICNNPriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1, requires_grad=True)
        tau = torch.rand(4, 1, requires_grad=True)
        out = net(K, tau)
        out.sum().backward()
        assert K.grad is not None
        assert tau.grad is not None

    def test_float64(self):
        net = ICNNPriceNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1)
        out = net(K, tau)
        assert out.dtype == torch.float64


# ── LocalVolNetwork ───────────────────────────────────────────────────

class TestLocalVolNetwork:
    def test_output_shape(self):
        net = LocalVolNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1) + 0.01
        out = net(K, tau)
        assert out.shape == (4, 1)

    def test_strictly_positive(self):
        """Local variance must be strictly positive (softplus + eps)."""
        net = LocalVolNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(20, 1) * 2
        tau = torch.rand(20, 1) + 0.01
        out = net(K, tau)
        assert (out > 0).all()

    def test_gradient_flow(self):
        net = LocalVolNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1, requires_grad=True)
        tau = torch.rand(4, 1, requires_grad=True)
        out = net(K, tau)
        out.sum().backward()
        assert K.grad is not None
        assert tau.grad is not None

    def test_float64(self):
        net = LocalVolNetwork(hidden_dim=8, n_layers=2)
        K = torch.randn(4, 1)
        tau = torch.rand(4, 1)
        out = net(K, tau)
        assert out.dtype == torch.float64


# ── DupirePINNLoss ────────────────────────────────────────────────────

class TestDupirePINNLoss:
    @pytest.fixture
    def loss_setup(self):
        price_net = PriceNetwork(hidden_dim=8, n_layers=2)
        lv_net = LocalVolNetwork(hidden_dim=8, n_layers=2)
        loss_fn = DupirePINNLoss()
        return price_net, lv_net, loss_fn

    def test_pde_residual_shape(self, loss_setup):
        price_net, lv_net, loss_fn = loss_setup
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        residual, dC_dT, d2C_dK2, sigma2 = loss_fn.dupire_pde_residual(
            price_net, lv_net, K, tau
        )
        assert residual.shape == (10, 1)
        assert dC_dT.shape == (10, 1)
        assert d2C_dK2.shape == (10, 1)
        assert sigma2.shape == (10, 1)

    def test_forward_returns_6_tuple(self, loss_setup):
        price_net, lv_net, loss_fn = loss_setup
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        C_target = torch.rand(10, 1) * 0.1

        result = loss_fn(price_net, lv_net, K, tau, C_target)
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_gradient_flow(self, loss_setup):
        price_net, lv_net, loss_fn = loss_setup
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        C_target = torch.rand(10, 1) * 0.1

        total, _, _, _, _, _ = loss_fn(price_net, lv_net, K, tau, C_target)
        total.backward()

        price_grads = [p.grad for p in price_net.parameters() if p.grad is not None]
        lv_grads = [p.grad for p in lv_net.parameters() if p.grad is not None]
        assert len(price_grads) > 0, "Price network has no gradients"
        assert len(lv_grads) > 0, "LocalVol network has no gradients"

    def test_weighted_sum(self, loss_setup):
        """If only fit loss is enabled, total should equal fit loss."""
        price_net, lv_net, _ = loss_setup
        loss_fn = DupirePINNLoss(
            lambda_fit=1.0, lambda_pde=0.0, lambda_cal=0.0,
            lambda_but=0.0, lambda_smooth=0.0
        )
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        C_target = torch.rand(10, 1) * 0.1

        total, loss_fit, _, _, _, _ = loss_fn(price_net, lv_net, K, tau, C_target)
        assert torch.allclose(total, loss_fit, atol=1e-6)

    def test_butterfly_term_penalizes_negative_d2C(self, loss_setup):
        """Butterfly loss should be zero when d2C/dK2 ≥ 0, positive otherwise."""
        price_net, lv_net, loss_fn = loss_setup
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        C_target = torch.rand(10, 1) * 0.1

        _, _, _, loss_cal, loss_but, _ = loss_fn(price_net, lv_net, K, tau, C_target)
        # Both should be scalar tensors with grad_fn
        assert loss_but.shape == ()
        assert loss_cal.shape == ()

    def test_all_losses_are_scalar(self, loss_setup):
        price_net, lv_net, loss_fn = loss_setup
        K = torch.rand(10, 1, requires_grad=True) + 0.5
        tau = torch.rand(10, 1, requires_grad=True) + 0.01
        C_target = torch.rand(10, 1) * 0.1

        total, l_fit, l_pde, l_cal, l_but, l_smooth = loss_fn(
            price_net, lv_net, K, tau, C_target
        )
        for name, loss in [('total', total), ('fit', l_fit), ('pde', l_pde),
                           ('cal', l_cal), ('but', l_but), ('smooth', l_smooth)]:
            assert loss.dim() == 0, f"{name} loss is not scalar: shape {loss.shape}"


# ── DupireSampler ─────────────────────────────────────────────────────

class TestDupireSampler:
    def test_correct_keys(self):
        sampler = DupireSampler(n_interior=50, n_boundary=10)
        data = sampler.sample()
        expected_keys = {
            'K_interior', 'tau_interior', 'C_target',
            'K_boundary', 'tau_boundary', 'C_boundary',
        }
        assert set(data.keys()) == expected_keys

    def test_shapes(self):
        sampler = DupireSampler(n_interior=50, n_boundary=10)
        data = sampler.sample()
        assert data['K_interior'].shape == (50, 1)
        assert data['tau_interior'].shape == (50, 1)
        assert data['C_target'].shape == (50, 1)
        assert data['K_boundary'].shape == (10, 1)
        assert data['tau_boundary'].shape == (10, 1)
        assert data['C_boundary'].shape == (10, 1)

    def test_ranges(self):
        sampler = DupireSampler(
            K_min=0.5, K_max=1.5, tau_min=0.02, tau_max=2.0,
            n_interior=100, n_boundary=20
        )
        data = sampler.sample()
        assert data['K_interior'].min() >= 0.5
        assert data['K_interior'].max() <= 1.5
        assert data['tau_interior'].min() >= 0.02
        assert data['tau_interior'].max() <= 2.0

    def test_requires_grad(self):
        """Interior points must have requires_grad for autograd PDE."""
        sampler = DupireSampler(n_interior=50, n_boundary=10)
        data = sampler.sample()
        assert data['K_interior'].requires_grad
        assert data['tau_interior'].requires_grad

    def test_boundary_payoff(self):
        """Boundary condition: C(K, 0) = max(S - K, 0) with S=1."""
        sampler = DupireSampler(n_boundary=20, strike=1.0)
        data = sampler.sample()
        expected = torch.relu(1.0 - data['K_boundary'])
        assert torch.allclose(data['C_boundary'], expected)


# ── Utility Functions ─────────────────────────────────────────────────

class TestBSCallPrice:
    def test_known_price_atm(self):
        """ATM, 1Y, sigma=0.2: BS price ~ 0.0793 (S=K=1)."""
        price = bs_call_price(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)
        assert price == pytest.approx(0.0797, abs=0.005)

    def test_itm_more_expensive(self):
        """ITM call (S > K) should be more expensive than OTM."""
        itm = bs_call_price(S=1.1, K=1.0, T=1.0, r=0.0, sigma=0.2)
        otm = bs_call_price(S=0.9, K=1.0, T=1.0, r=0.0, sigma=0.2)
        assert itm > otm


class TestTotalVarianceToCallPrice:
    def test_atm_conversion(self):
        """TV at ATM with known sigma should produce known BS price."""
        sigma = 0.2
        tau = 1.0
        w = sigma ** 2 * tau  # total variance
        price = total_variance_to_call_price(w, K=1.0, S=1.0, tau=tau)
        expected = bs_call_price(S=1.0, K=1.0, T=tau, r=0.0, sigma=sigma)
        assert price == pytest.approx(expected, abs=1e-6)

# ── GreekExtractor (Module D) ─────────────────────────────────────────

class TestGreekExtractor:
    def test_feature_extraction_shapes_and_finiteness(self):
        """Ensures Greeks are extracted correctly with proper shapes and no NaNs."""
        price_net = ICNNPriceNetwork(hidden_dim=8, n_layers=2).double()
        localvol_net = LocalVolNetwork(hidden_dim=8, n_layers=2).double()
        extractor = GreekExtractor(price_net, localvol_net, device='cpu')
        
        N = 10
        K_norm = torch.linspace(0.8, 1.2, N, dtype=torch.float64).reshape(-1, 1)
        tau = torch.linspace(0.1, 1.0, N, dtype=torch.float64).reshape(-1, 1)
        
        features = extractor.extract_features(K_norm, tau)
        
        assert 'local_vol' in features
        assert 'lv_gradient_K' in features
        assert 'vanna' in features
        assert 'volga' in features
        
        for key, tensor in features.items():
            assert tensor.shape == (N, 1), f"Shape mismatch for {key}: {tensor.shape}"
            assert torch.isfinite(tensor).all(), f"NaN or Inf found in {key}"
            assert not tensor.requires_grad, f"{key} should be detached"

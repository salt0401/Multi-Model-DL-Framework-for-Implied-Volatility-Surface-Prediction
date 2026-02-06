"""Tests for dgm.py: DGM network, PDE loss, sampler, BS pricing."""
import pytest
import numpy as np
import torch

from dgm import (
    SLayer,
    DGMNetwork,
    DGMLoss,
    DGMSampler,
    bs_call_price,
)


# ── SLayer ────────────────────────────────────────────────────────────

class TestSLayer:
    def test_output_shape(self):
        layer = SLayer(input_dim=2, hidden_dim=8)
        x = torch.randn(4, 2)
        s_prev = torch.randn(4, 8)
        s_new = layer(x, s_prev)
        assert s_new.shape == (4, 8)

    def test_deterministic_with_zeros(self):
        layer = SLayer(input_dim=2, hidden_dim=8)
        x = torch.zeros(1, 2)
        s_prev = torch.zeros(1, 8)
        s1 = layer(x, s_prev)
        s2 = layer(x, s_prev)
        assert torch.allclose(s1, s2)

    def test_layer_norm_applied(self):
        layer = SLayer(input_dim=2, hidden_dim=8)
        x = torch.randn(10, 2)
        s_prev = torch.randn(10, 8)
        s_new = layer(x, s_prev)
        # LayerNorm output should have mean ~0, std ~1 per sample
        assert s_new.mean(dim=1).abs().max() < 0.5


# ── DGMNetwork ────────────────────────────────────────────────────────

class TestDGMNetwork:
    def test_output_shape(self):
        net = DGMNetwork(input_dim=2, hidden_dim=8, n_layers=2)
        x = torch.randn(4, 1)
        t = torch.randn(4, 1)
        out = net(x, t)
        assert out.shape == (4, 1)

    def test_residual_connections_matter(self):
        """Output differs from just input projection + output layer."""
        net = DGMNetwork(input_dim=2, hidden_dim=8, n_layers=3)
        x = torch.randn(4, 1)
        t = torch.randn(4, 1)
        out = net(x, t)
        # Just check it's not trivially zero
        assert out.abs().sum() > 0

    def test_gradient_flow(self):
        net = DGMNetwork(input_dim=2, hidden_dim=8, n_layers=2)
        x = torch.randn(4, 1, requires_grad=True)
        t = torch.randn(4, 1, requires_grad=True)
        out = net(x, t)
        out.sum().backward()
        assert x.grad is not None
        assert t.grad is not None

    def test_float64(self):
        net = DGMNetwork(input_dim=2, hidden_dim=8, n_layers=2)
        x = torch.randn(4, 1)
        t = torch.randn(4, 1)
        out = net(x, t)
        assert out.dtype == torch.float64


# ── DGMLoss ───────────────────────────────────────────────────────────

class TestDGMLoss:
    @pytest.fixture
    def loss_setup(self):
        net = DGMNetwork(input_dim=2, hidden_dim=8, n_layers=2)
        loss_fn = DGMLoss(sigma=0.2, r=0.0)
        return net, loss_fn

    def test_pde_residual_shape(self, loss_setup):
        net, loss_fn = loss_setup
        x = torch.randn(10, 1)
        t = torch.rand(10, 1) + 0.01
        residual = loss_fn.pde_residual(net, x, t)
        assert residual.shape == (10, 1)

    def test_forward_returns_4_tuple(self, loss_setup):
        net, loss_fn = loss_setup
        x_int = torch.randn(10, 1)
        t_int = torch.rand(10, 1) + 0.01
        x_bc = torch.randn(5, 1)
        t_bc = torch.rand(5, 1) + 0.01
        u_bc = torch.randn(5, 1)
        x_tc = torch.randn(5, 1)
        u_tc = torch.relu(x_tc - 1.0)

        result = loss_fn(net, x_int, t_int, x_bc, t_bc, u_bc, x_tc, u_tc)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_gradient_flow(self, loss_setup):
        net, loss_fn = loss_setup
        x_int = torch.randn(10, 1)
        t_int = torch.rand(10, 1) + 0.01
        x_bc = torch.randn(5, 1)
        t_bc = torch.rand(5, 1) + 0.01
        u_bc = torch.randn(5, 1)
        x_tc = torch.randn(5, 1)
        u_tc = torch.relu(x_tc - 1.0)

        total, _, _, _ = loss_fn(net, x_int, t_int, x_bc, t_bc, u_bc, x_tc, u_tc)
        total.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_weighted_sum(self, loss_setup):
        net, _ = loss_setup
        loss_fn = DGMLoss(sigma=0.2, r=0.0, lambda_pde=2.0, lambda_bc=0.0, lambda_tc=0.0)
        x_int = torch.randn(10, 1)
        t_int = torch.rand(10, 1) + 0.01
        x_bc = torch.randn(5, 1)
        t_bc = torch.rand(5, 1) + 0.01
        u_bc = torch.randn(5, 1)
        x_tc = torch.randn(5, 1)
        u_tc = torch.relu(x_tc - 1.0)

        total, loss_pde, _, _ = loss_fn(net, x_int, t_int, x_bc, t_bc, u_bc, x_tc, u_tc)
        assert torch.allclose(total, 2.0 * loss_pde, atol=1e-6)


# ── DGMSampler ────────────────────────────────────────────────────────

class TestDGMSampler:
    def test_correct_keys(self):
        sampler = DGMSampler(n_interior=50, n_boundary=10, n_terminal=10)
        data = sampler.sample()
        expected_keys = {'x_interior', 't_interior', 'x_terminal', 'u_terminal',
                         'x_boundary', 't_boundary', 'u_boundary'}
        assert set(data.keys()) == expected_keys

    def test_shapes(self):
        sampler = DGMSampler(n_interior=50, n_boundary=10, n_terminal=10)
        data = sampler.sample()
        assert data['x_interior'].shape == (50, 1)
        assert data['t_interior'].shape == (50, 1)
        assert data['x_terminal'].shape == (10, 1)
        assert data['x_boundary'].shape == (10, 1)  # 10//2 + 10//2

    def test_ranges(self):
        sampler = DGMSampler(S_min=0.5, S_max=1.5, T_min=0.02, T_max=2.0,
                              n_interior=100, n_boundary=20, n_terminal=20)
        data = sampler.sample()
        assert data['x_interior'].min() >= 0.5
        assert data['x_interior'].max() <= 1.5
        assert data['t_interior'].min() >= 0.02
        assert data['t_interior'].max() <= 2.0

    def test_terminal_condition(self):
        sampler = DGMSampler(strike=1.0, n_terminal=20)
        data = sampler.sample()
        expected = torch.relu(data['x_terminal'] - 1.0)
        assert torch.allclose(data['u_terminal'], expected)


# ── bs_call_price ─────────────────────────────────────────────────────

class TestBSCallPrice:
    def test_known_price(self):
        """ATM, 1Y, sigma=0.2: BS price ~ 0.0793 (S=K=1)."""
        price = bs_call_price(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)
        assert price == pytest.approx(0.0797, abs=0.005)

    def test_atm_symmetry(self):
        """For r=0, call(S, K) with S>K should be more expensive."""
        p1 = bs_call_price(S=1.1, K=1.0, T=1.0, r=0.0, sigma=0.2)
        p2 = bs_call_price(S=0.9, K=1.0, T=1.0, r=0.0, sigma=0.2)
        assert p1 > p2

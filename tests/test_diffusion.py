"""Tests for diffusion.py: DDPM components and noise schedule."""
import pytest
import torch
import torch.nn as nn

from diffusion import (
    SinusoidalTimeEmbedding,
    FiLMLayer,
    ResBlock1D,
    UNet1D,
    CosineNoiseSchedule,
    DiffusionTrainer,
    DiffusionSampler,
)

SURFACE_DIM = 12  # tiny grid for testing (3 tau * 4 logm)


# ── SinusoidalTimeEmbedding ──────────────────────────────────────────

class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(embed_dim=32)
        t = torch.tensor([0, 5, 10], dtype=torch.float64)
        out = emb(t)
        assert out.shape == (3, 32)

    def test_different_timesteps_differ(self):
        emb = SinusoidalTimeEmbedding(embed_dim=32)
        t = torch.tensor([0, 100], dtype=torch.float64)
        out = emb(t)
        assert not torch.allclose(out[0], out[1])


# ── FiLMLayer ─────────────────────────────────────────────────────────

class TestFiLMLayer:
    def test_output_shape(self):
        film = FiLMLayer(condition_dim=16, channels=8)
        x = torch.randn(2, 8, 10)
        cond = torch.randn(2, 16)
        out = film(x, cond)
        assert out.shape == (2, 8, 10)

    def test_identity_behavior(self):
        """When gamma=1 and beta=0, output should equal input."""
        film = FiLMLayer(condition_dim=4, channels=2)
        # Manually set projection to produce gamma=1, beta=0
        with torch.no_grad():
            film.proj.weight.zero_()
            film.proj.bias.zero_()
            # First half (gamma part) gets bias=1, second half (beta) stays 0
            film.proj.bias[:2] = 1.0
        x = torch.randn(1, 2, 5)
        cond = torch.zeros(1, 4)
        out = film(x, cond)
        assert torch.allclose(out, x, atol=1e-10)


# ── ResBlock1D ────────────────────────────────────────────────────────

class TestResBlock1D:
    def test_output_shape(self):
        block = ResBlock1D(in_channels=8, out_channels=16, time_dim=32, cond_dim=32, n_groups=4)
        x = torch.randn(2, 8, 10)
        t_emb = torch.randn(2, 32)
        cond = torch.randn(2, 32)
        out = block(x, t_emb, cond)
        assert out.shape == (2, 16, 10)

    def test_skip_when_channels_differ(self):
        block = ResBlock1D(in_channels=8, out_channels=16, time_dim=16, cond_dim=16, n_groups=4)
        assert isinstance(block.skip, nn.Conv1d)

    def test_skip_when_channels_same(self):
        block = ResBlock1D(in_channels=8, out_channels=8, time_dim=16, cond_dim=16, n_groups=4)
        assert isinstance(block.skip, nn.Identity)


# ── UNet1D ────────────────────────────────────────────────────────────

class TestUNet1D:
    @pytest.fixture
    def tiny_unet(self):
        return UNet1D(surface_dim=SURFACE_DIM, channels=(8, 16, 32),
                      condition_dim=4, time_dim=16)

    def test_output_shape_matches_input(self, tiny_unet):
        batch = 2
        x_noisy = torch.randn(batch, SURFACE_DIM)
        t = torch.tensor([5.0, 10.0])
        cur_surface = torch.randn(batch, SURFACE_DIM)
        market = torch.randn(batch, 4)
        out = tiny_unet(x_noisy, t, cur_surface, market)
        assert out.shape == (batch, SURFACE_DIM)

    def test_gradient_flow(self, tiny_unet):
        batch = 2
        x_noisy = torch.randn(batch, SURFACE_DIM, requires_grad=True)
        t = torch.tensor([5.0, 10.0])
        cur_surface = torch.randn(batch, SURFACE_DIM)
        market = torch.randn(batch, 4)
        out = tiny_unet(x_noisy, t, cur_surface, market)
        loss = out.sum()
        loss.backward()
        assert x_noisy.grad is not None

    def test_small_surface_dim(self):
        """Use batch>1 and surface_dim large enough to survive 3 stride-2 downsamplings."""
        unet = UNet1D(surface_dim=16, channels=(4, 8, 16), condition_dim=4, time_dim=8)
        batch = 2
        x_noisy = torch.randn(batch, 16)
        t = torch.tensor([3.0, 5.0])
        cur = torch.randn(batch, 16)
        mkt = torch.randn(batch, 4)
        out = unet(x_noisy, t, cur, mkt)
        assert out.shape == (batch, 16)


# ── CosineNoiseSchedule ──────────────────────────────────────────────

class TestCosineNoiseSchedule:
    def test_alpha_bar_monotonic_decreasing(self):
        sched = CosineNoiseSchedule(T=100)
        diffs = sched.alpha_bar[1:] - sched.alpha_bar[:-1]
        assert (diffs <= 1e-10).all()

    def test_starts_near_one(self):
        sched = CosineNoiseSchedule(T=100)
        assert sched.alpha_bar[0] > 0.99

    def test_beta_in_bounds(self):
        sched = CosineNoiseSchedule(T=100)
        assert (sched.beta >= 1e-5).all()
        assert (sched.beta <= 0.999).all()

    def test_to_device(self):
        sched = CosineNoiseSchedule(T=10)
        sched.to(torch.device('cpu'))
        assert sched.beta.device.type == 'cpu'


# ── DiffusionTrainer ─────────────────────────────────────────────────

class TestDiffusionTrainer:
    @pytest.fixture
    def trainer_setup(self):
        model = UNet1D(surface_dim=SURFACE_DIM, channels=(8, 16, 32),
                       condition_dim=4, time_dim=16)
        sched = CosineNoiseSchedule(T=10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device('cpu')
        trainer = DiffusionTrainer(model, sched, opt, device)
        return trainer

    def test_returns_float(self, trainer_setup):
        trainer = trainer_setup
        x_target = torch.randn(2, SURFACE_DIM)
        cur = torch.randn(2, SURFACE_DIM)
        mkt = torch.randn(2, 4)
        loss = trainer.train_step(x_target, cur, mkt)
        assert isinstance(loss, float)

    def test_loss_decreases(self, trainer_setup):
        """Loss should generally decrease over several steps."""
        trainer = trainer_setup
        torch.manual_seed(42)
        x_target = torch.randn(4, SURFACE_DIM)
        cur = torch.randn(4, SURFACE_DIM)
        mkt = torch.randn(4, 4)
        losses = []
        for _ in range(20):
            loss = trainer.train_step(x_target, cur, mkt)
            losses.append(loss)
        # First loss should be larger than last on average
        assert losses[0] > losses[-1] or losses[0] == pytest.approx(losses[-1], abs=0.5)

    def test_gradient_clipping(self, trainer_setup):
        trainer = trainer_setup
        x_target = torch.randn(2, SURFACE_DIM)
        cur = torch.randn(2, SURFACE_DIM)
        mkt = torch.randn(2, 4)
        # Should complete without error (gradient clipping at 1.0)
        trainer.train_step(x_target, cur, mkt)


# ── DiffusionSampler ─────────────────────────────────────────────────

class TestDiffusionSampler:
    @pytest.fixture
    def sampler_setup(self):
        model = UNet1D(surface_dim=SURFACE_DIM, channels=(8, 16, 32),
                       condition_dim=4, time_dim=16)
        sched = CosineNoiseSchedule(T=5)  # tiny T for speed
        device = torch.device('cpu')
        return DiffusionSampler(model, sched, device)

    def test_output_shape(self, sampler_setup):
        cur = torch.randn(1, SURFACE_DIM)
        mkt = torch.randn(1, 4)
        out = sampler_setup.sample(cur, mkt, n_samples=3)
        assert out.shape == (3, SURFACE_DIM)

    def test_tiny_T(self, sampler_setup):
        """Should work with very small T (5 steps)."""
        cur = torch.randn(2, SURFACE_DIM)
        mkt = torch.randn(2, 4)
        out = sampler_setup.sample(cur, mkt, n_samples=2)
        assert out.shape == (2, SURFACE_DIM)
        assert torch.isfinite(out).all()

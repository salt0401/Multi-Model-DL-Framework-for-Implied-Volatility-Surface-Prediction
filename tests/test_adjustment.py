"""Tests for adjustment.py: GRU+Attention adjustment model."""
import pytest
import numpy as np
import torch

from adjustment import (
    SquarePlus,
    TemporalAttention,
    TVAdjustmentModel,
    AdjustmentLoss,
)


# ── SquarePlus ────────────────────────────────────────────────────────

class TestSquarePlus:
    def test_always_positive(self):
        sp = SquarePlus()
        x = torch.linspace(-10, 10, 100)
        out = sp(x)
        assert (out > 0).all()

    def test_smooth_gradient(self):
        sp = SquarePlus()
        x = torch.tensor([0.0], requires_grad=True)
        out = sp(x)
        out.backward()
        assert x.grad is not None
        # Gradient at 0: (1 + 0/sqrt(4))/2 = 0.5
        assert x.grad.item() == pytest.approx(0.5, abs=0.01)

    def test_identity_like_for_large_x(self):
        sp = SquarePlus()
        x = torch.tensor([100.0])
        out = sp(x)
        # For large x, sqrt(x^2+4) ≈ x, so (x+x)/2 = x
        assert out.item() == pytest.approx(100.0, abs=0.1)


# ── TemporalAttention ────────────────────────────────────────────────

class TestTemporalAttention:
    def test_output_shape(self):
        attn = TemporalAttention(hidden_dim=8, n_heads=4)
        h = torch.randn(2, 10, 8)
        out = attn(h)
        assert out.shape == (2, 8)

    def test_mask_handling(self):
        attn = TemporalAttention(hidden_dim=8, n_heads=4)
        h = torch.randn(2, 10, 8)
        mask = torch.ones(2, 10)
        mask[:, 7:] = 0  # last 3 padded
        out = attn(h, mask=mask)
        assert out.shape == (2, 8)
        assert torch.isfinite(out).all()

    def test_multi_head_consistency(self):
        attn = TemporalAttention(hidden_dim=8, n_heads=2)
        attn.eval()  # disable dropout for deterministic output
        h = torch.randn(1, 5, 8)
        out1 = attn(h)
        out2 = attn(h)
        assert torch.allclose(out1, out2)

    def test_assertion_on_bad_dims(self):
        with pytest.raises(AssertionError):
            TemporalAttention(hidden_dim=7, n_heads=4)  # 7 not divisible by 4


# ── TVAdjustmentModel ────────────────────────────────────────────────

class TestTVAdjustmentModel:
    @pytest.fixture
    def tiny_adj(self):
        return TVAdjustmentModel(
            input_dim=6, gru_hidden_dim=8, gru_layers=1,
            attention_heads=4, dropout=0.0, prediction_target='ratio',
        )

    def test_output_shape(self, tiny_adj):
        seq = torch.randn(4, 10, 6)
        out = tiny_adj(seq)
        assert out.shape == (4, 1)

    def test_ratio_positive(self, tiny_adj):
        """In ratio mode, SquarePlus ensures positive output."""
        seq = torch.randn(4, 10, 6)
        out = tiny_adj(seq)
        assert (out > 0).all()

    def test_residual_mode(self):
        model = TVAdjustmentModel(
            input_dim=6, gru_hidden_dim=8, gru_layers=1,
            attention_heads=4, dropout=0.0, prediction_target='residual',
        )
        seq = torch.randn(4, 10, 6)
        out = model(seq)
        assert out.shape == (4, 1)
        # No SquarePlus in residual mode, so can be negative
        assert model.squareplus is None

    def test_direct_mode(self):
        model = TVAdjustmentModel(
            input_dim=6, gru_hidden_dim=8, gru_layers=1,
            attention_heads=4, dropout=0.0, prediction_target='direct',
        )
        seq = torch.randn(4, 10, 6)
        out = model(seq)
        assert out.shape == (4, 1)

    def test_adjust_ratio(self, tiny_adj):
        tv_pred = torch.tensor([[0.05]], dtype=torch.float64)
        factor = torch.tensor([[1.2]], dtype=torch.float64)
        result = tiny_adj.adjust(tv_pred, factor)
        assert result.item() == pytest.approx(0.06)

    def test_adjust_residual(self):
        model = TVAdjustmentModel(prediction_target='residual',
                                   input_dim=6, gru_hidden_dim=8,
                                   gru_layers=1, attention_heads=4)
        tv_pred = torch.tensor([[0.05]])
        factor = torch.tensor([[0.01]])
        result = model.adjust(tv_pred, factor)
        assert result.item() == pytest.approx(0.06)

    def test_adjust_direct(self):
        model = TVAdjustmentModel(prediction_target='direct',
                                   input_dim=6, gru_hidden_dim=8,
                                   gru_layers=1, attention_heads=4)
        tv_pred = torch.tensor([[0.05]])
        factor = torch.tensor([[0.07]])
        result = model.adjust(tv_pred, factor)
        assert result.item() == pytest.approx(0.07)

    def test_gradient_flow(self, tiny_adj):
        seq = torch.randn(4, 10, 6, requires_grad=True)
        out = tiny_adj(seq)
        out.sum().backward()
        assert seq.grad is not None

    def test_all_zero_mask(self, tiny_adj):
        """All-zero mask should still produce output (softmax of -inf -> uniform)."""
        seq = torch.randn(2, 5, 6)
        mask = torch.zeros(2, 5)
        out = tiny_adj(seq, mask=mask)
        # May produce NaN or finite — at least shouldn't crash
        assert out.shape == (2, 1)


# ── AdjustmentLoss ────────────────────────────────────────────────────

class TestAdjustmentLoss:
    def test_returns_scalar(self):
        loss_fn = AdjustmentLoss()
        pred = torch.randn(8, 1)
        true = torch.randn(8, 1).abs() + 0.01
        loss = loss_fn(pred, true)
        assert loss.dim() == 0

    def test_kde_weights_shape(self):
        loss_fn = AdjustmentLoss()
        targets = np.random.randn(100)
        weights = loss_fn.fit_kde_weights(targets)
        assert len(weights) == 100

    def test_kde_weights_normalized(self):
        loss_fn = AdjustmentLoss()
        targets = np.random.randn(100)
        weights = loss_fn.fit_kde_weights(targets)
        assert np.abs(weights.mean() - 1.0) < 0.1

    def test_weighted_loss(self):
        loss_fn = AdjustmentLoss()
        pred = torch.randn(8, 1)
        true = torch.randn(8, 1).abs() + 0.01
        w = torch.ones(8)
        loss = loss_fn(pred, true, sample_weights=w)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        loss_fn = AdjustmentLoss()
        pred = torch.randn(8, 1, requires_grad=True)
        true = torch.randn(8, 1).abs() + 0.01
        loss = loss_fn(pred, true)
        loss.backward()
        assert pred.grad is not None

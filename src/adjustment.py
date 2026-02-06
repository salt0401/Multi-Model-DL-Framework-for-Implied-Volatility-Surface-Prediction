"""GRU + Attention time-varying adjustment model (PDF pages 48-79).

Implements:
- SquarePlus: smooth positive activation (x + sqrt(x^2 + 4)) / 2
- TemporalAttention: multi-head attention over GRU hidden states
- TVAdjustmentModel: GRU -> Attention -> Linear -> SquarePlus
- AdjustmentLoss: MSE + MAPE with KDE-based sample weighting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SquarePlus(nn.Module):
    """Smooth positive activation: (x + sqrt(x^2 + 4)) / 2.

    Always positive, smooth everywhere (unlike ReLU).
    Useful for ensuring positive predicted ratios/volatilities.
    """

    def forward(self, x):
        return (x + torch.sqrt(x * x + 4)) / 2


class TemporalAttention(nn.Module):
    """Multi-head attention over GRU hidden states.

    Learns which timesteps in the sequence matter most for prediction.
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super(TemporalAttention, self).__init__()
        assert hidden_dim % n_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5

    def forward(self, hidden_states, mask=None):
        """Apply multi-head attention over sequence of hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) binary mask (1 = valid, 0 = padded)

        Returns:
            context: (batch, hidden_dim) attended representation
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Query from last hidden state, keys/values from all states
        query = hidden_states[:, -1:, :]  # (batch, 1, hidden_dim)

        Q = self.W_q(query).view(batch_size, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(hidden_states).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, 1, seq_len)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (batch, n_heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, self.hidden_dim)

        output = self.W_o(context)
        return output


class TVAdjustmentModel(nn.Module):
    """Time-Varying Adjustment Model: GRU -> Attention -> Linear -> SquarePlus.

    Predicts adjustment factor for base model during structural breaks.

    6 input features per timestep:
    [vix_change, underlying_return, logm, tau, tv_pred, itm_otm]

    3 prediction targets (configurable):
    - 'direct': predict tv_true directly
    - 'residual': predict tv_true - tv_pred (additive correction)
    - 'ratio': predict tv_true / tv_pred (multiplicative scaling, default per PDF)
    """

    def __init__(self, input_dim=6, gru_hidden_dim=64, gru_layers=2,
                 attention_heads=4, dropout=0.2, prediction_target='ratio'):
        super(TVAdjustmentModel, self).__init__()
        self.prediction_target = prediction_target

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = TemporalAttention(
            hidden_dim=gru_hidden_dim,
            n_heads=attention_heads,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_dim, gru_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim // 2, 1),
        )

        # For ratio target: ensure positive output
        self.squareplus = SquarePlus() if prediction_target == 'ratio' else None

    def forward(self, sequences, mask=None):
        """Forward pass.

        Args:
            sequences: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask

        Returns:
            predictions: (batch, 1)
        """
        gru_output, _ = self.gru(sequences)  # (batch, seq_len, hidden_dim)
        context = self.attention(gru_output, mask)  # (batch, hidden_dim)
        output = self.fc(context)  # (batch, 1)

        if self.squareplus is not None:
            output = self.squareplus(output)

        return output

    def adjust(self, tv_pred, adjustment_factor):
        """Apply adjustment to base model predictions.

        Args:
            tv_pred: base model total variance predictions
            adjustment_factor: output from this model

        Returns:
            adjusted total variance
        """
        if self.prediction_target == 'ratio':
            return tv_pred * adjustment_factor
        elif self.prediction_target == 'residual':
            return tv_pred + adjustment_factor
        elif self.prediction_target == 'direct':
            return adjustment_factor
        else:
            raise ValueError(f"Unknown prediction_target: {self.prediction_target}")


class AdjustmentLoss(nn.Module):
    """MSE + MAPE loss with KDE-based sample weighting for rare events.

    Rare/extreme target values (crisis periods) get higher weights
    based on inverse of their KDE density estimate.
    """

    def __init__(self, kde_bandwidth=0.1, mape_weight=0.5):
        super(AdjustmentLoss, self).__init__()
        self.kde_bandwidth = kde_bandwidth
        self.mape_weight = mape_weight
        self.sample_weights = None

    def fit_kde_weights(self, targets):
        """Compute sample weights from KDE of target distribution.

        Args:
            targets: numpy array of target values

        Returns:
            numpy array of sample weights (higher for rare values)
        """
        from scipy.stats import gaussian_kde

        targets_flat = targets.flatten()
        kde = gaussian_kde(targets_flat, bw_method=self.kde_bandwidth)
        density = kde(targets_flat)

        # Inverse density weighting, capped to avoid extreme weights
        weights = 1.0 / (density + 1e-8)
        weights = np.clip(weights, 1.0, np.percentile(weights, 95))
        weights = weights / weights.mean()  # Normalize to mean 1

        self.sample_weights = torch.tensor(weights, dtype=torch.float64)
        return weights

    def forward(self, y_pred, y_true, sample_weights=None):
        """Compute weighted MSE + MAPE loss.

        Args:
            y_pred: (batch, 1) predictions
            y_true: (batch, 1) targets
            sample_weights: (batch,) optional per-sample weights
        """
        if sample_weights is None and self.sample_weights is not None:
            sample_weights = self.sample_weights[:len(y_pred)].to(y_pred.device)

        # MSE
        mse = (y_pred - y_true) ** 2
        if sample_weights is not None:
            w = sample_weights.view(-1, 1)
            mse_loss = torch.mean(w * mse)
        else:
            mse_loss = torch.mean(mse)

        # MAPE
        epsilon = 0.005
        mape = torch.abs((y_true - y_pred) / (y_true + epsilon))
        if sample_weights is not None:
            mape_loss = torch.mean(w * mape)
        else:
            mape_loss = torch.mean(mape)

        total_loss = mse_loss + self.mape_weight * mape_loss
        return total_loss

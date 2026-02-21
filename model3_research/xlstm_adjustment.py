"""xLSTM (mLSTM) Adjustment Model — Drop-in GRU replacement.

Based on: Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory"

Key innovations over standard LSTM/GRU:
1. Exponential gating with log-space stabilization (better gradient flow)
2. Matrix memory C (d x d) instead of vector cell state (richer history)
3. Query-Key-Value retrieval mechanism (Transformer-style memory access)

Architecture: mLSTM -> TemporalAttention -> FC -> SquarePlus

Performance optimization: all 6 linear projections per layer are fused into
a single combined projection, computed once for the entire sequence before
the sequential scan. Only the state update (bmm, exp) runs in the loop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SquarePlus(nn.Module):
    """Smooth positive activation: (x + sqrt(x^2 + 4)) / 2."""

    def forward(self, x):
        return (x + torch.sqrt(x * x + 4)) / 2


class TemporalAttention(nn.Module):
    """Multi-head attention over sequential hidden states.

    Query from last hidden state, keys/values from all states.
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** 0.5

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask=None):
        B, T, _ = hidden_states.shape
        query = hidden_states[:, -1:, :]
        Q = self.W_q(query).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, self.hidden_dim)
        return self.W_o(context)


class mLSTMLayer(nn.Module):
    """Single mLSTM layer with fused projection and sequential scan.

    Optimization: all 6 projections (i, f, o, q, k, v) are fused into
    one nn.Linear call for the entire sequence. The inner loop only
    handles state updates (exp, bmm) — no Linear ops in the loop.

    Memory layout of fused projection output (dim = 4*d + 2):
        [i_logit(1), f_logit(1), o_raw(d), q(d), k(d), v(d)]
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Fused projection: 6 transforms in 1 kernel
        # Output: i(1) + f(1) + o(d) + q(d) + k(d) + v(d) = 4d + 2
        self.proj = nn.Linear(input_dim, 4 * hidden_dim + 2)

        # Initialize forget gate bias to 1.0 (encourage remembering)
        with torch.no_grad():
            self.proj.bias[1] = 1.0

        self._k_scale = hidden_dim ** 0.5

    def forward(self, x, mask=None):
        """Process entire sequence with precomputed projections.

        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask (1=valid, 0=padded)

        Returns:
            outputs: (batch, seq_len, hidden_dim)
        """
        B, T, _ = x.shape
        d = self.hidden_dim

        # === Precompute ALL projections at once (single CUDA kernel) ===
        proj = self.proj(x)  # (B, T, 4d+2)

        i_logits = proj[:, :, 0:1]              # (B, T, 1)
        f_logits = proj[:, :, 1:2]              # (B, T, 1)
        o_all = torch.sigmoid(proj[:, :, 2:2+d])  # (B, T, d)
        q_all = proj[:, :, 2+d:2+2*d]           # (B, T, d)
        k_all = proj[:, :, 2+2*d:2+3*d] / self._k_scale  # (B, T, d)
        v_all = proj[:, :, 2+3*d:]              # (B, T, d)

        # === Sequential scan (only state updates, no Linear ops) ===
        C = x.new_zeros(B, d, d)   # matrix memory
        n = x.new_zeros(B, d)      # normalizer
        m = x.new_zeros(B, 1)      # log-space stabilizer

        outputs = []
        for t in range(T):
            i_logit = i_logits[:, t]  # (B, 1)
            f_logit = f_logits[:, t]  # (B, 1)

            # Log-space stabilization
            m_new = torch.max(f_logit + m, i_logit)
            i_t = torch.exp(i_logit - m_new)
            f_t = torch.exp(f_logit + m - m_new)

            k_t = k_all[:, t]  # (B, d)
            v_t = v_all[:, t]  # (B, d)
            q_t = q_all[:, t]  # (B, d)

            # State update
            C_new = (f_t.unsqueeze(-1) * C
                     + i_t.unsqueeze(-1) * torch.bmm(v_t.unsqueeze(2), k_t.unsqueeze(1)))
            n_new = f_t * n + i_t * k_t

            # Retrieval: h = o * (C @ q / max(|n^T q|, 1))
            h_tilde = torch.bmm(C_new, q_t.unsqueeze(2)).squeeze(2)
            denom = torch.clamp(
                torch.abs((n_new * q_t).sum(dim=-1, keepdim=True)), min=1.0
            )
            h_t = o_all[:, t] * (h_tilde / denom)

            # Mask: preserve old state for padded positions
            if mask is not None:
                mt = mask[:, t].unsqueeze(-1)   # (B, 1)
                mt_3d = mt.unsqueeze(-1)        # (B, 1, 1)
                C = torch.where(mt_3d.bool(), C_new, C)
                n = torch.where(mt.bool(), n_new, n)
                m = torch.where(mt.bool(), m_new, m)
                h_t = h_t * mt
            else:
                C, n, m = C_new, n_new, m_new

            outputs.append(h_t)

        return torch.stack(outputs, dim=1)  # (B, T, d)


class mLSTM(nn.Module):
    """Multi-layer mLSTM with LayerNorm and dropout between layers."""

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            dim_in = input_dim if i == 0 else hidden_dim
            self.layers.append(mLSTMLayer(dim_in, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            if i < num_layers - 1:
                self.dropouts.append(nn.Dropout(dropout))

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask

        Returns:
            outputs: (batch, seq_len, hidden_dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, mask)
            h = self.norms[i](h)
            if i < self.num_layers - 1:
                h = self.dropouts[i](h)
        return h


class xLSTMAdjustmentModel(nn.Module):
    """mLSTM-based adjustment model: mLSTM -> Attention -> FC -> SquarePlus.

    Drop-in replacement for TVAdjustmentModel (GRU-based).
    Same forward(sequences, mask) interface, same adjust() method.
    """

    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2,
                 attention_heads=4, dropout=0.2, prediction_target='ratio'):
        super().__init__()
        self.prediction_target = prediction_target

        self.mlstm = mLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.attention = TemporalAttention(
            hidden_dim=hidden_dim,
            n_heads=attention_heads,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.squareplus = SquarePlus() if prediction_target == 'ratio' else None

    def forward(self, sequences, mask=None):
        """
        Args:
            sequences: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask
        Returns:
            predictions: (batch, 1)
        """
        hidden_states = self.mlstm(sequences, mask)   # (B, T, D)
        context = self.attention(hidden_states, mask)  # (B, D)
        output = self.fc(context)                      # (B, 1)

        if self.squareplus is not None:
            output = self.squareplus(output)
        return output

    def adjust(self, tv_pred, adjustment_factor):
        """Apply adjustment to base model predictions."""
        if self.prediction_target == 'ratio':
            return tv_pred * adjustment_factor
        elif self.prediction_target == 'residual':
            return tv_pred + adjustment_factor
        elif self.prediction_target == 'direct':
            return adjustment_factor
        raise ValueError(f"Unknown prediction_target: {self.prediction_target}")

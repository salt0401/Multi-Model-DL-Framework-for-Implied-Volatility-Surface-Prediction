"""Temporal Fusion Transformer (TFT) Adjustment Model.

Based on: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting", Google Research, arXiv:1912.09363

Simplified for our use case:
- Encoder-only (no decoder, single-step output)
- No static covariates (all features are temporal)
- Interpretable attention for temporal importance analysis

Architecture:
    Input (batch, 20, n_vars)
        |
    Variable Selection Network (VSN)
        |  - Per-variable GRN transform
        |  - Softmax selection weights -> interpretability
        v
    Selected features (batch, 20, hidden_dim)
        |
    LSTM Encoder (2 layers)
        |
    Post-LSTM GRN + LayerNorm + Skip
        |
    Interpretable Multi-Head Attention
        |  - Query from last timestep
        |  - Shared V weights across heads -> interpretability
        v
    Post-Attention GRN + LayerNorm
        |
    FC Head + SquarePlus
        |
    Output: alpha (batch, 1)

Key innovation: Variable Selection Network
    - Each timestep independently learns which features matter
    - Crisis periods: VIX change + underlying return get high weights
    - Calm periods: tau + logm dominate
    - This IS regime-switching behavior, learned end-to-end
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SquarePlus(nn.Module):
    """Smooth positive activation: (x + sqrt(x^2 + 4)) / 2."""

    def forward(self, x):
        return (x + torch.sqrt(x * x + 4)) / 2


class GatedLinearUnit(nn.Module):
    """GLU gate: sigma(W_a @ x) * (W_b @ x).

    Splits the input into two halves: one for the sigmoid gate,
    one for the linear pathway. The gate learns to suppress or
    pass through information selectively.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        x = self.fc(x)
        gate, value = x.chunk(2, dim=-1)
        return torch.sigmoid(gate) * value


class GatedResidualNetwork(nn.Module):
    """GRN: the fundamental building block of TFT.

    Processes input through:
    1. Primary linear transform + optional context injection
    2. ELU nonlinearity
    3. Secondary linear transform
    4. GLU gating (learns what to suppress)
    5. Skip connection + LayerNorm

    The skip connection + GLU combination means the network can learn
    to be a no-op when the input is already good enough.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Skip connection: project if dimensions don't match
        self.skip_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        """
        Args:
            x: (..., input_dim) input tensor (any leading dims)
            context: (..., context_dim) optional context vector
        Returns:
            output: (..., output_dim)
        """
        # Skip connection
        residual = self.skip_proj(x) if self.skip_proj else x

        # Primary path
        h = self.fc1(x)
        if self.context_proj is not None and context is not None:
            h = h + self.context_proj(context)
        h = F.elu(h)

        h = self.dropout(self.fc2(h))
        h = self.glu(h)

        return self.layernorm(residual + h)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns which input variables matter at each timestep.

    This is the key innovation of TFT for regime detection:
    - Each variable gets its own GRN transform
    - A separate GRN computes softmax weights over variables
    - Output is weighted combination of transformed variables

    The variable weights are directly interpretable:
    - High weight on vix_change -> model focuses on volatility regime
    - High weight on logm -> model focuses on moneyness structure
    - Weights shift across time -> regime-switching behavior
    """

    def __init__(self, input_dim, num_vars, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim

        # Per-variable GRN: transforms each scalar variable to hidden_dim
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout=dropout)
            for _ in range(num_vars)
        ])

        # Selection weights GRN: takes flattened input, outputs num_vars weights
        self.weight_grn = GatedResidualNetwork(
            num_vars, hidden_dim, num_vars, dropout=dropout
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, num_vars) raw feature input

        Returns:
            selected: (batch, seq_len, hidden_dim) weighted feature representation
            weights: (batch, seq_len, num_vars) variable selection weights
        """
        B, T, J = x.shape

        # Compute selection weights from raw input
        weights = self.weight_grn(x)           # (B, T, J)
        weights = F.softmax(weights, dim=-1)   # (B, T, J)

        # Transform each variable independently
        var_outputs = []
        for j in range(J):
            x_j = x[:, :, j:j+1]                  # (B, T, 1)
            transformed = self.var_grns[j](x_j)    # (B, T, hidden_dim)
            var_outputs.append(transformed)

        # Stack: (B, T, J, hidden_dim)
        var_stack = torch.stack(var_outputs, dim=2)

        # Weighted combination: (B, T, J, 1) * (B, T, J, hidden_dim) -> sum -> (B, T, hidden_dim)
        selected = (weights.unsqueeze(-1) * var_stack).sum(dim=2)

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """TFT's interpretable attention: shared V weights across heads.

    Standard MHA: each head has W_q^h, W_k^h, W_v^h (3*H projections)
    TFT MHA:      each head has W_q^h, W_k^h, but shares W_v (2*H+1 projections)

    Why shared V? When all heads extract the same "value" from each position,
    the attention weights directly tell you which timesteps matter. Different
    heads can still attend to different patterns (via separate Q, K), but
    the extracted information is consistent across heads.

    Final output: mean of per-head attended values x output projection.
    """

    def __init__(self, hidden_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.scale = self.head_dim ** 0.5

        # Per-head Q, K projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)

        # SHARED V projection (single head_dim, not full hidden_dim)
        self.W_v = nn.Linear(hidden_dim, self.head_dim)

        # Output projection
        self.W_o = nn.Linear(self.head_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask=None):
        """Query from last timestep, K/V from all timesteps.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) binary mask

        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, n_heads, seq_len) attention weights per head
        """
        B, T, _ = hidden_states.shape

        # Query from last timestep
        query = hidden_states[:, -1:, :]  # (B, 1, D)

        # Per-head Q, K
        Q = self.W_q(query).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: (B, H, 1, d_k), K: (B, H, T, d_k)

        # Shared V (same for all heads)
        V = self.W_v(hidden_states)  # (B, T, d_v) where d_v = head_dim

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, 1, T)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)           # (B, H, 1, T)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to shared V: each head produces (B, 1, d_v)
        # V is (B, T, d_v), need to apply per-head weights
        # attn_weights: (B, H, 1, T), V: (B, 1, T, d_v) -> broadcast
        V_expanded = V.unsqueeze(1)  # (B, 1, T, d_v)
        attended = torch.matmul(attn_weights, V_expanded)  # (B, H, 1, d_v)

        # Average across heads (as per TFT paper)
        context = attended.mean(dim=1).squeeze(1)  # (B, d_v)

        # Output projection
        output = self.W_o(context)  # (B, hidden_dim)

        # Return attention weights for interpretability
        attn_out = attn_weights.squeeze(2)  # (B, H, T)
        return output, attn_out


class TFTAdjustmentModel(nn.Module):
    """Temporal Fusion Transformer for IV surface adjustment.

    Full pipeline:
    1. Variable Selection Network -> learns feature importance per timestep
    2. LSTM Encoder -> captures temporal dynamics
    3. Post-LSTM GRN -> gated nonlinear processing with skip
    4. Interpretable Multi-Head Attention -> temporal importance
    5. Post-Attention GRN -> final gated processing
    6. FC Head + SquarePlus -> positive adjustment factor

    Interpretability outputs (when return_interpretability=True):
    - variable_weights: (batch, seq_len, num_vars) -- which features matter when
    - attention_weights: (batch, n_heads, seq_len) -- which timesteps matter
    """

    def __init__(self, input_dim=6, hidden_dim=64, lstm_layers=2,
                 attention_heads=4, dropout=0.1, prediction_target='ratio'):
        super().__init__()
        self.prediction_target = prediction_target
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_dim=input_dim,
            num_vars=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # 2. LSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # 3. Post-LSTM processing (GRN + skip from VSN output)
        self.post_lstm_gate = GatedLinearUnit(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)

        # 4. Interpretable Multi-Head Attention
        self.attention = InterpretableMultiHeadAttention(
            hidden_dim=hidden_dim,
            n_heads=attention_heads,
            dropout=dropout,
        )

        # 5. Post-Attention GRN
        self.post_attn_grn = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # 6. Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.squareplus = SquarePlus() if prediction_target == 'ratio' else None

    def forward(self, sequences, mask=None, return_interpretability=False):
        """
        Args:
            sequences: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask
            return_interpretability: if True, also return VSN weights and
                                     attention weights for analysis

        Returns:
            predictions: (batch, 1)
            interpretability: dict (only if return_interpretability=True)
                - 'variable_weights': (batch, seq_len, num_vars)
                - 'attention_weights': (batch, n_heads, seq_len)
        """
        # 1. Variable Selection
        selected, var_weights = self.vsn(sequences)  # (B, T, D), (B, T, J)

        # 2. LSTM Encoder
        lstm_out, _ = self.lstm(selected)  # (B, T, D)

        # 3. Post-LSTM: GLU gate + skip connection + LayerNorm
        gated = self.post_lstm_gate(lstm_out)
        enriched = self.post_lstm_norm(selected + gated)  # skip from VSN output

        # 4. Interpretable Attention (query from last timestep)
        attn_out, attn_weights = self.attention(enriched, mask)  # (B, D), (B, H, T)

        # 5. Post-Attention GRN
        final = self.post_attn_grn(attn_out)  # (B, D)

        # 6. Output
        output = self.fc(final)  # (B, 1)
        if self.squareplus is not None:
            output = self.squareplus(output)

        if return_interpretability:
            return output, {
                'variable_weights': var_weights,
                'attention_weights': attn_weights,
            }
        return output

    def adjust(self, tv_pred, adjustment_factor):
        """Apply adjustment to base model predictions (same as TVAdjustmentModel)."""
        if self.prediction_target == 'ratio':
            return tv_pred * adjustment_factor
        elif self.prediction_target == 'residual':
            return tv_pred + adjustment_factor
        elif self.prediction_target == 'direct':
            return adjustment_factor
        raise ValueError(f"Unknown prediction_target: {self.prediction_target}")

    def get_feature_importance(self, sequences, mask=None):
        """Extract per-timestep feature importance for interpretability.

        Returns the average variable selection weights across the batch,
        showing which features the model considers most important at each
        position in the sequence.

        Args:
            sequences: (batch, seq_len, input_dim)
            mask: (batch, seq_len) binary mask

        Returns:
            avg_var_weights: (seq_len, num_vars) average variable importance
            avg_attn_weights: (n_heads, seq_len) average attention weights
        """
        was_training = self.training
        self.train(False)
        with torch.no_grad():
            _, interp = self.forward(sequences, mask, return_interpretability=True)
            var_w = interp['variable_weights']     # (B, T, J)
            attn_w = interp['attention_weights']   # (B, H, T)

            if mask is not None:
                var_w = var_w * mask.unsqueeze(-1)

            avg_var = var_w.mean(dim=0)    # (T, J)
            avg_attn = attn_w.mean(dim=0)  # (H, T)

        self.train(was_training)
        return avg_var, avg_attn

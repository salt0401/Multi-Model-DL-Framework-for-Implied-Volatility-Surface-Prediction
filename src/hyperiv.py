"""HyperIV: Hypernetwork-based IV surface interpolation.

A hypernetwork generates the weights of a small target MLP from a set
embedding of observed options. The target MLP maps (tau, logm, yATM) to
total variance predictions, matching the 4-tuple interface used by the
rest of the codebase.

Reference: HyperIV (ICML 2025) — state-of-the-art for IV surface interpolation.
"""
import torch
import torch.nn as nn
import math


class SetEmbeddingNetwork(nn.Module):
    """Encode a variable-size set of reference options into a fixed-dim vector.

    Input:  (batch, n_ref, 3)  — each reference option is (tau, logm, total_var)
    Output: (batch, embed_dim) — context vector via Transformer + mean pooling
    """

    def __init__(self, input_dim=3, embed_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed_dim = embed_dim

    def forward(self, ref_set, mask=None):
        """
        Args:
            ref_set: (batch, n_ref, 3)
            mask: (batch, n_ref) bool — True where padded (to be ignored)
        Returns:
            (batch, embed_dim)
        """
        x = self.proj(ref_set)  # (batch, n_ref, embed_dim)
        x = self.transformer(x, src_key_padding_mask=mask)  # (batch, n_ref, embed_dim)
        # Mean-pool over non-padded positions
        if mask is not None:
            # Invert mask: True = valid
            valid = (~mask).unsqueeze(-1).float()  # (batch, n_ref, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        return x


class TargetNetwork(nn.Module):
    """Small MLP: (tau, logm, yATM) -> total_variance.

    Weights are NOT learned directly — they are set externally by the hypernetwork.
    This class defines the architecture (shapes) only.
    """

    def __init__(self, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        in_dim = 3  # tau, logm, yATM
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _count_target_params(hidden_dims):
    """Count total parameters in the target network."""
    sizes = [3] + list(hidden_dims) + [1]
    total = 0
    for i in range(len(sizes) - 1):
        total += sizes[i] * sizes[i + 1] + sizes[i + 1]  # weight + bias
    return total


def _generate_target_params(flat_params, hidden_dims):
    """Split flat parameter vector into list of (weight, bias) for each layer."""
    sizes = [3] + list(hidden_dims) + [1]
    params = {}
    offset = 0
    layer_idx = 0
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        w_count = fan_in * fan_out
        b_count = fan_out
        w = flat_params[offset:offset + w_count].reshape(fan_out, fan_in)
        offset += w_count
        b = flat_params[offset:offset + b_count]
        offset += b_count
        # Map to Sequential layer names: net.0, net.2, net.4, ... (skip ReLU layers)
        seq_idx = layer_idx * 2  # Linear layers at even indices (0, 2, 4, ...)
        params[f'net.{seq_idx}.weight'] = w
        params[f'net.{seq_idx}.bias'] = b
        layer_idx += 1
    return params


class HyperIVModel(nn.Module):
    """HyperIV: hypernetwork generates target MLP weights from set embedding.

    Forward returns the same 4-tuple as other models:
    (total_var, grad_tau, grad_logm, grad_logm2)
    """

    def __init__(self, embed_dim=128, n_heads=4, n_transformer_layers=2,
                 target_hidden_dims=(64, 32)):
        super().__init__()
        self.embed_dim = embed_dim
        self.target_hidden_dims = target_hidden_dims

        self.set_encoder = SetEmbeddingNetwork(
            input_dim=3, embed_dim=embed_dim,
            n_heads=n_heads, n_layers=n_transformer_layers,
        )

        # Target network (used as a template for functional_call)
        self.target_net = TargetNetwork(hidden_dims=target_hidden_dims)

        # Hypernetwork: project embedding to all target params
        n_target_params = _count_target_params(target_hidden_dims)
        self.hyper_proj = nn.Linear(embed_dim, n_target_params)

        # Initialize hyper_proj with small weights for stability
        nn.init.normal_(self.hyper_proj.weight, std=0.01)
        nn.init.zeros_(self.hyper_proj.bias)

    def forward(self, ref_set, target_tau, target_logm, target_yATM, ref_mask=None):
        """
        Args:
            ref_set:     (batch, n_ref, 3) — reference options (tau, logm, total_var)
            target_tau:  (batch, n_target, 1) — query tau values
            target_logm: (batch, n_target, 1) — query logm values
            target_yATM: (batch, n_target, 1) — query yATM values
            ref_mask:    (batch, n_ref) bool — True where padded

        Returns:
            tv_pred:    (batch, n_target, 1)
            grad_tau:   (batch, n_target, 1)
            grad_logm:  (batch, n_target, 1)
            grad_logm2: (batch, n_target, 1)
        """
        batch_size = ref_set.shape[0]
        n_target = target_tau.shape[1]

        # 1. Encode reference set
        embedding = self.set_encoder(ref_set, mask=ref_mask)  # (batch, embed_dim)

        # 2. Generate target network parameters
        flat_params = self.hyper_proj(embedding)  # (batch, n_target_params)

        # 3. Build target inputs with grad tracking
        target_tau = target_tau.detach().requires_grad_(True)
        target_logm = target_logm.detach().requires_grad_(True)
        target_input = torch.cat([target_tau, target_logm, target_yATM], dim=-1)  # (batch, n_target, 3)

        # 4. Apply generated weights per batch element
        all_preds = []
        for i in range(batch_size):
            params = _generate_target_params(flat_params[i], self.target_hidden_dims)
            pred_i = torch.func.functional_call(self.target_net, params, target_input[i])
            all_preds.append(pred_i)

        tv_pred = torch.stack(all_preds, dim=0)  # (batch, n_target, 1)

        # 5. Compute analytical gradients via autograd
        total_output = tv_pred.sum()
        grad1 = torch.autograd.grad(total_output, (target_tau, target_logm),
                                     retain_graph=True, create_graph=True)
        grad_tau = grad1[0]
        grad_logm = grad1[1]

        total_grad_logm = grad_logm.sum()
        grad_logm2 = torch.autograd.grad(total_grad_logm, target_logm,
                                          retain_graph=True, create_graph=True)[0]

        return tv_pred, grad_tau, grad_logm, grad_logm2


class HyperIVLoss(nn.Module):
    """Loss for HyperIV: MSE + physics-informed constraints.

    Same constraints as WeightedSumLoss but adapted for HyperIV interface.
    """

    def __init__(self, w_mse=1.0, w_calendar=10.0, w_butterfly=10.0):
        super().__init__()
        self.w_mse = w_mse
        self.w_calendar = w_calendar
        self.w_butterfly = w_butterfly

    def forward(self, tv_pred, tv_true, logm, grad_tau, grad_logm, grad_logm2):
        """
        All inputs: (batch, n_target, 1)
        """
        # MSE on total variance
        mse_loss = torch.mean((tv_pred - tv_true) ** 2)

        # Calendar arbitrage: dw/dtau >= 0
        calendar_loss = torch.mean(torch.relu(-grad_tau))

        # Butterfly arbitrage: density g(k) >= 0
        g_k = (1 - (logm * grad_logm) / (2 * tv_pred.clamp(min=1e-8))) ** 2 \
              - grad_logm / 4 * (1 / tv_pred.clamp(min=1e-8) + 0.25) \
              + grad_logm2 / 2
        butterfly_loss = torch.mean(torch.relu(-g_k))

        total = self.w_mse * mse_loss + self.w_calendar * calendar_loss + self.w_butterfly * butterfly_loss
        return total, mse_loss, calendar_loss, butterfly_loss

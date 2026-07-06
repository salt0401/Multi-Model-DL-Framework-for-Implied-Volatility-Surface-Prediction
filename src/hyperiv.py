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

    Activations are smooth (tanh) so second derivatives w.r.t. inputs are
    nonzero (a ReLU net is piecewise-linear: d2w/dk2 = 0 a.e., which would
    silently disable the butterfly penalty). The softplus head guarantees
    positive total variance, matching the HyperIV paper's tanh/softplus design.
    """

    def __init__(self, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        in_dim = 3  # tau, logm, yATM
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())
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

        # Hypernetwork: project embedding to a DELTA on a learnable base
        # parameter vector (residual hypernetwork). With deltas alone, the
        # generated weights start near zero, hidden activations are tanh(0)=0,
        # gradients to the generated weights vanish and training plateaus on
        # flat surfaces (which incur zero arbitrage penalty).
        n_target_params = _count_target_params(target_hidden_dims)
        self.hyper_proj = nn.Linear(embed_dim, n_target_params)
        nn.init.normal_(self.hyper_proj.weight, std=0.01)
        nn.init.zeros_(self.hyper_proj.bias)

        # Base = a normally-initialized target MLP, flattened in the exact
        # layout _generate_target_params expects ([W1,b1,W2,b2,...]).
        base_chunks = []
        for layer in self.target_net.net:
            if isinstance(layer, nn.Linear):
                base_chunks.append(layer.weight.detach().flatten())
                base_chunks.append(layer.bias.detach().flatten())
        base_flat = torch.cat(base_chunks)
        # The target net predicts the RATIO w / yATM_tilde (see forward).
        # softplus(0.54) ~= 1.0, so initial surfaces start at the ATM level.
        # Predicting raw total variance (median ~0.005) through softplus is
        # fatally ill-conditioned: pre-activations sit at log(tv) ~= -5..-9
        # where softplus' gradient ~= tv ~= 1e-6 kills all upstream gradients
        # and the model collapses to predicting zero.
        base_flat[-1] = 0.54
        self.base_params = nn.Parameter(base_flat)

        # Feature standardization stats for (tau, logm, total_var/yATM);
        # identity by default so the model works without set_normalization.
        self.register_buffer('feat_mean', torch.zeros(3))
        self.register_buffer('feat_std', torch.ones(3))

    def set_normalization(self, mean, std):
        """Set input standardization stats (3-dim: tau, logm, tv/yATM),
        computed from TRAINING data only."""
        self.feat_mean.copy_(mean.to(self.feat_mean))
        self.feat_std.copy_(std.to(self.feat_std).clamp(min=1e-8))

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

        # 1. Encode reference set (standardized with train-set stats)
        ref_norm = (ref_set - self.feat_mean) / self.feat_std
        embedding = self.set_encoder(ref_norm, mask=ref_mask)  # (batch, embed_dim)

        # 2. Generate target network parameters (base + day-specific delta)
        flat_params = self.base_params + self.hyper_proj(embedding)  # (batch, n_target_params)

        # 3. Build target inputs with grad tracking. Gradients are taken w.r.t.
        # the ORIGINAL (unnormalized) tau/logm; the affine standardization is
        # part of the autograd graph, so dw/dtau etc. come out in raw units.
        target_tau = target_tau.detach().requires_grad_(True)
        target_logm = target_logm.detach().requires_grad_(True)
        tau_n = (target_tau - self.feat_mean[0]) / self.feat_std[0]
        logm_n = (target_logm - self.feat_mean[1]) / self.feat_std[1]
        yatm_n = (target_yATM - self.feat_mean[2]) / self.feat_std[2]
        target_input = torch.cat([tau_n, logm_n, yatm_n], dim=-1)  # (batch, n_target, 3)

        # 4. Apply generated weights per batch element
        all_preds = []
        for i in range(batch_size):
            params = _generate_target_params(flat_params[i], self.target_hidden_dims)
            pred_i = torch.func.functional_call(self.target_net, params, target_input[i])
            all_preds.append(pred_i)

        tv_pred_ratio = torch.stack(all_preds, dim=0)  # (batch, n_target, 1)

        # Scale-equivariant output (Model 1's Lesson #7): the target net
        # predicts w / yATM_tilde, a well-conditioned O(1) ratio (~1 at the
        # money), instead of raw total variance whose 1e-5..1e-1 range
        # saturates the softplus head. eps floors the multiplier so gradients
        # survive near-zero yATM days.
        yatm_tilde = torch.sqrt(target_yATM ** 2 + 0.002 ** 2)
        tv_pred = yatm_tilde * tv_pred_ratio

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


def black76_call_price(logm, w):
    """Normalized Black-76 call price (forward = 1, discount = 1).

    C(k, w) = N(d1) - e^k N(d2),  d1 = (-k + w/2) / sqrt(w),  d2 = d1 - sqrt(w)
    where k is log-moneyness and w total implied variance.
    """
    w = w.clamp(min=1e-10)
    sqrt_w = torch.sqrt(w)
    d1 = (-logm + w / 2) / sqrt_w
    d2 = d1 - sqrt_w
    normal = torch.distributions.Normal(
        torch.zeros((), dtype=logm.dtype, device=logm.device),
        torch.ones((), dtype=logm.dtype, device=logm.device))
    return normal.cdf(d1) - torch.exp(logm) * normal.cdf(d2)


class HyperIVLoss(nn.Module):
    """Loss for HyperIV: MSE + physics-informed constraints + price auxiliary.

    Butterfly uses the Gatheral & Jacquier (2014) density condition (note the
    squared w' term, matching model1_research Loss_butterfly). The price
    auxiliary follows PIVOT (arXiv 2606.17065): MSE between normalized
    Black-76 prices implied by predicted vs. true total variance.
    """

    def __init__(self, w_mse=1.0, w_calendar=10.0, w_butterfly=10.0, w_price=0.1):
        super().__init__()
        self.w_mse = w_mse
        self.w_calendar = w_calendar
        self.w_butterfly = w_butterfly
        self.w_price = w_price

    @staticmethod
    def _masked_mean(x, valid_mask):
        if valid_mask is None:
            return x.mean()
        m = valid_mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum() / m.sum().clamp(min=1)

    def forward(self, tv_pred, tv_true, logm, grad_tau, grad_logm, grad_logm2,
                valid_mask=None):
        """
        All tensor inputs: (batch, n_target, 1).
        valid_mask: optional (batch, n_target) bool, True where entries are real
        (not padding). Reductions divide by the valid count only.
        """
        # MSE on total variance
        mse_loss = self._masked_mean((tv_pred - tv_true) ** 2, valid_mask)

        # Calendar arbitrage: dw/dtau >= 0
        calendar_loss = self._masked_mean(torch.relu(-grad_tau), valid_mask)

        # Butterfly arbitrage: density g(k) >= 0 (Gatheral-Jacquier 2014).
        # The 1/w factors are clamped at 1e-3 FOR THE TRAINING PENALTY ONLY:
        # at short maturities w ~ 3e-5, so unclamped 1/w amplifies penalty
        # gradients x33,000 — a single such batch was measured to throw the
        # model into the saturated-softplus collapse basin. Evaluation-time
        # violation rates use the exact formula (train_hyperiv.py).
        w_safe = tv_pred.clamp(min=1e-3)
        g_k = (1 - (logm * grad_logm) / (2 * w_safe)) ** 2 \
              - grad_logm ** 2 / 4 * (1 / w_safe + 0.25) \
              + grad_logm2 / 2
        butterfly_loss = self._masked_mean(torch.relu(-g_k), valid_mask)

        # PIVOT price-space auxiliary
        price_err = (black76_call_price(logm, tv_pred)
                     - black76_call_price(logm, tv_true)) ** 2
        price_loss = self._masked_mean(price_err, valid_mask)

        total = (self.w_mse * mse_loss + self.w_calendar * calendar_loss
                 + self.w_butterfly * butterfly_loss + self.w_price * price_loss)
        return total, mse_loss, calendar_loss, butterfly_loss, price_loss

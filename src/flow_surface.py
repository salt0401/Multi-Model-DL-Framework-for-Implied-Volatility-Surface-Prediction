"""Model 5 (replacement): conditional flow matching over PCA factors of the
log total-variance surface.

Replaces the draft grid-space DDPM (src/diffusion.py, deprecated). Rationale
(see docs/model45_completion_report.md): at ~1,450 training pairs a 200-dim
grid generative model is data-starved; ~3 factors explain >90% of IV-surface
variance (Cont & da Fonseca), the tabular-diffusion literature (TabDDPM,
TabSyn, CDTD) converged on MLP denoisers for low-dim data, and conditional
OT flow matching attains diffusion-quality samples in 10-50 Euler steps
instead of 1000 ancestral steps.

Pipeline: surface (D,) --log--> PCA (k factors, train-fit) --z-score--> z1;
a FiLM residual MLP v(z_t, t, cond) is trained with the conditional OT
flow-matching objective; sampling integrates dz/dt = v from z0 ~ N(0, I),
t: 0 -> 1, then inverts (de-z-score, PCA-reconstruct, exp) — positivity of
total variance is guaranteed by construction.
"""
import math

import numpy as np
import torch
import torch.nn as nn


# ── Preprocessing ─────────────────────────────────────────────────────

class FactorPreprocessor:
    """log -> PCA -> per-score z-normalization, fitted on TRAIN surfaces only."""

    def __init__(self, n_components=None, ev_target=0.99, max_components=12):
        self.n_components = n_components
        self.ev_target = ev_target
        self.max_components = max_components
        self.n_components_ = None
        self.log_mean_ = None       # (D,)
        self.components_ = None     # (k, D)
        self.score_mean_ = None     # (k,)
        self.score_std_ = None      # (k,)
        self.explained_variance_ratio_ = None

    def fit(self, S):
        """S: (N, D) strictly positive total-variance surfaces."""
        L = np.log(np.asarray(S, dtype=np.float64))
        self.log_mean_ = L.mean(axis=0)
        Lc = L - self.log_mean_
        U, sv, Vt = np.linalg.svd(Lc, full_matrices=False)
        ev_ratio = sv ** 2 / np.sum(sv ** 2)
        if self.n_components is not None:
            k = int(self.n_components)
        else:
            k = int(np.searchsorted(np.cumsum(ev_ratio), self.ev_target) + 1)
        k = max(1, min(k, self.max_components, len(sv)))
        self.n_components_ = k
        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = ev_ratio[:k]
        scores = Lc @ self.components_.T
        self.score_mean_ = scores.mean(axis=0)
        self.score_std_ = np.clip(scores.std(axis=0), 1e-12, None)
        return self

    def transform(self, S):
        L = np.log(np.asarray(S, dtype=np.float64))
        scores = (L - self.log_mean_) @ self.components_.T
        return (scores - self.score_mean_) / self.score_std_

    def inverse(self, Z):
        scores = np.asarray(Z, dtype=np.float64) * self.score_std_ + self.score_mean_
        L = scores @ self.components_ + self.log_mean_
        return np.exp(L)

    def to_dict(self):
        return {
            'n_components_': self.n_components_,
            'log_mean_': self.log_mean_.tolist(),
            'components_': self.components_.tolist(),
            'score_mean_': self.score_mean_.tolist(),
            'score_std_': self.score_std_.tolist(),
            'explained_variance_ratio_': self.explained_variance_ratio_.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        pp = cls()
        pp.n_components_ = d['n_components_']
        pp.log_mean_ = np.asarray(d['log_mean_'])
        pp.components_ = np.asarray(d['components_'])
        pp.score_mean_ = np.asarray(d['score_mean_'])
        pp.score_std_ = np.asarray(d['score_std_'])
        pp.explained_variance_ratio_ = np.asarray(d['explained_variance_ratio_'])
        return pp


class CondScaler:
    """Z-score scaler for condition vectors, fitted on TRAIN rows only."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, C):
        C = np.asarray(C, dtype=np.float64)
        self.mean_ = C.mean(axis=0)
        self.std_ = np.clip(C.std(axis=0), 1e-12, None)
        return self

    def transform(self, C):
        return (np.asarray(C, dtype=np.float64) - self.mean_) / self.std_

    def inverse(self, Z):
        return np.asarray(Z, dtype=np.float64) * self.std_ + self.mean_

    def to_dict(self):
        return {'mean_': self.mean_.tolist(), 'std_': self.std_.tolist()}

    @classmethod
    def from_dict(cls, d):
        cs = cls()
        cs.mean_ = np.asarray(d['mean_'])
        cs.std_ = np.asarray(d['std_'])
        return cs


# ── Velocity network ──────────────────────────────────────────────────

class _TimeEmbedding(nn.Module):
    """Sinusoidal embedding of flow time t in [0, 1]."""

    def __init__(self, embed_dim=64, max_freq=1000.0):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer('freqs', freqs)

    def forward(self, t):
        args = t.unsqueeze(-1) * self.freqs.to(t.dtype)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class _FiLMBlock(nn.Module):
    """Pre-norm residual MLP block with FiLM conditioning."""

    def __init__(self, hidden, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.film = nn.Linear(cond_dim, hidden * 2)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, h, cond):
        x = self.norm(h)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        x = x * (1 + gamma) + beta
        x = self.fc2(self.drop(self.act(self.fc1(x))))
        return h + x


class VelocityMLP(nn.Module):
    """v_theta(z_t, t, cond) for conditional flow matching on k-dim scores."""

    def __init__(self, dim, cond_dim, hidden=256, n_blocks=3, dropout=0.1,
                 time_dim=64):
        super().__init__()
        self.time_embed = _TimeEmbedding(time_dim)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.input_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            [_FiLMBlock(hidden, hidden, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    def forward(self, z_t, t, cond):
        """z_t: (B, dim), t: (B,), cond: (B, cond_dim) -> (B, dim)"""
        c = self.cond_proj(torch.cat([cond, self.time_embed(t)], dim=-1))
        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, c)
        return self.out_proj(self.out_norm(h))


# ── Conditional OT flow matching ──────────────────────────────────────

def fm_loss(model, z1, cond, generator=None):
    """Conditional OT flow-matching loss (Lipman et al. 2023 / rectified flow).

    t ~ U(0,1), z0 ~ N(0,I), z_t = (1-t) z0 + t z1, target velocity = z1 - z0.
    """
    B = z1.shape[0]
    t = torch.rand(B, device=z1.device, dtype=z1.dtype, generator=generator)
    z0 = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype,
                     generator=generator)
    t_col = t.unsqueeze(-1)
    z_t = (1 - t_col) * z0 + t_col * z1
    v_target = z1 - z0
    v_pred = model(z_t, t, cond)
    return torch.mean((v_pred - v_target) ** 2)


@torch.no_grad()
def sample_flow(model, cond, n_steps=50, n_samples=1, generator=None):
    """Euler integration of dz/dt = v(z, t, cond) from z0 ~ N(0, I).

    Args:
        cond: (B, cond_dim)
    Returns:
        (n_samples, B, dim)
    """
    was_training = model.training
    model.eval()
    dim = model.out_proj.out_features
    B = cond.shape[0]
    device = cond.device
    dtype = cond.dtype

    z = torch.randn(n_samples * B, dim, device=device, dtype=dtype,
                    generator=generator)
    cond_rep = cond.repeat(n_samples, 1)  # sample-major: (n_samples*B, c)

    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((n_samples * B,), i * dt, device=device, dtype=dtype)
        z = z + dt * model(z, t, cond_rep)

    if was_training:
        model.train()
    return z.reshape(n_samples, B, dim)


# ── EMA ───────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: p.detach().clone()
                       for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for name, p in model.named_parameters():
            p.copy_(self.shadow[name])

    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.shadow = state['shadow']


def make_ema_model(model, ckpt):
    """Load a checkpoint into `model`, preferring the EMA shadow weights."""
    model.load_state_dict(ckpt['state_dict'])
    shadow = ckpt.get('ema_shadow', {}).get('shadow')
    if shadow:
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(shadow[name].to(device=p.device, dtype=p.dtype))
    return model


# ── Dataset assembly ──────────────────────────────────────────────────

def build_dataset(panel, train_end_date, test_start_date, val_frac=0.15):
    """Pair consecutive days and split chronologically without leakage.

    A pair (today -> tomorrow) is TRAIN only if tomorrow <= train_end_date
    (the draft's mask leaked the first test-day surface into training
    targets), TEST if today >= test_start_date; anything in between is
    dropped. The last val_frac of train pairs (chronological) become val.

    Returns dict of splits; each split is a dict with S_today, S_tomorrow
    (raw surfaces), C (raw conditions for today), dates (today timestamps).
    """
    dates = panel['dates']
    S = panel['surfaces']
    C = panel['conditions']

    pairs = []
    for i in range(len(dates) - 1):
        pairs.append((i, i + 1))

    def _split(name, keep):
        idx_today = [i for i, j in pairs if keep(dates[i], dates[j])]
        return {
            'S_today': S[idx_today],
            'S_tomorrow': S[[i + 1 for i in idx_today]],
            'C': C[idx_today],
            'dates': [dates[i] for i in idx_today],
        }

    train_all = _split('train', lambda d0, d1: d1 <= train_end_date)
    test = _split('test', lambda d0, d1: d0 >= test_start_date)

    n_train_pairs = len(train_all['dates'])
    n_val = max(1, int(n_train_pairs * val_frac))
    split_at = n_train_pairs - n_val
    train = {k: v[:split_at] for k, v in train_all.items()}
    val = {k: v[split_at:] for k, v in train_all.items()}

    return {'train': train, 'val': val, 'test': test}

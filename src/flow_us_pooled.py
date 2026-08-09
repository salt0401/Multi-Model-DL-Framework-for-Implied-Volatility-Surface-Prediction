"""Pooled cross-sectional flow-matching forecaster for the US panel.

Replaces the seven independent per-ticker models with ONE model over all
eight symbols (Mag 7 + SPY), for reasons that are measured rather than
assumed: the first principal component explains ~77% of cross-sectional
variation in single-name IV level and skew and ~87% of firm-level 30-day
implied variance, so the seven series carry mostly one signal plus noise,
and ~700 training pairs per ticker is thin for a private model each.

Design points that matter:

- PER-TICKER LOG-MEAN BEFORE THE SVD. Pooling raw log total variance would
  spend PC1 on the level gap between TSLA (~55 vol) and MSFT (~22) — the
  factors would encode ticker identity instead of dynamics. Each ticker is
  centred on its own train-period mean surface first, so the shared basis
  describes shared *movement*.
- TINY TICKER IDENTITY. An 8-dim embedding with dropout, nothing else, so
  training starts at full pooling and per-ticker deviation must be earned
  against ~800 own observations.
- EVENT CLOCK. Factors are forecast on DE-EVENTED surfaces and the known
  future event variance is re-added analytically at reconstruction: the
  earnings sawtooth is deterministic and public, so letting the flow learn
  it would burn capacity on something we can look up.
- SEED ENSEMBLE. Velocity fields from N seeds are averaged at sampling time.
  At ~6k samples of dimension ~6 an epoch is milliseconds, which makes this
  the cheapest reliable accuracy lever available.
"""
import numpy as np
import torch

from flow_surface import VelocityMLP, fm_loss, sample_flow, EMA


class PooledFactorPreprocessor:
    """Shared PCA basis over per-ticker-centred log total variance."""

    def __init__(self, n_components=6, ev_target=0.99, max_components=10):
        self.n_components = n_components
        self.ev_target = ev_target
        self.max_components = max_components
        self.log_mean_ = {}          # per ticker
        self.components_ = None      # (k, D) shared
        self.score_mean_ = {}        # per ticker
        self.score_std_ = {}
        self.n_components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, surfaces_by_ticker):
        """surfaces_by_ticker: {ticker: (N_t, D) strictly positive surfaces}."""
        centred = []
        for t, S in surfaces_by_ticker.items():
            L = np.log(np.asarray(S, dtype=np.float64))
            self.log_mean_[t] = L.mean(axis=0)
            centred.append(L - self.log_mean_[t])
        X = np.concatenate(centred, axis=0)
        _, sv, Vt = np.linalg.svd(X, full_matrices=False)
        ev = sv ** 2 / np.sum(sv ** 2)
        k = (int(self.n_components) if self.n_components
             else int(np.searchsorted(np.cumsum(ev), self.ev_target) + 1))
        k = max(1, min(k, self.max_components, len(sv)))
        self.n_components_ = k
        self.components_ = Vt[:k]
        self.explained_variance_ratio_ = ev[:k]

        for t, S in surfaces_by_ticker.items():
            Z = (np.log(np.asarray(S, dtype=np.float64))
                 - self.log_mean_[t]) @ self.components_.T
            self.score_mean_[t] = Z.mean(axis=0)
            self.score_std_[t] = np.clip(Z.std(axis=0), 1e-12, None)
        return self

    def transform(self, ticker, S):
        Z = (np.log(np.asarray(S, dtype=np.float64))
             - self.log_mean_[ticker]) @ self.components_.T
        return (Z - self.score_mean_[ticker]) / self.score_std_[ticker]

    def inverse(self, ticker, Z):
        Zs = (np.asarray(Z, dtype=np.float64) * self.score_std_[ticker]
              + self.score_mean_[ticker])
        return np.exp(Zs @ self.components_ + self.log_mean_[ticker])

    def reconstruction_mse(self, ticker, S):
        return float(np.mean((self.inverse(ticker, self.transform(ticker, S))
                              - np.asarray(S)) ** 2))

    def to_dict(self):
        return {
            'n_components_': self.n_components_,
            'components_': self.components_.tolist(),
            'log_mean_': {t: v.tolist() for t, v in self.log_mean_.items()},
            'score_mean_': {t: v.tolist() for t, v in self.score_mean_.items()},
            'score_std_': {t: v.tolist() for t, v in self.score_std_.items()},
            'explained_variance_ratio_': self.explained_variance_ratio_.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        pp = cls()
        pp.n_components_ = d['n_components_']
        pp.components_ = np.asarray(d['components_'])
        pp.log_mean_ = {t: np.asarray(v) for t, v in d['log_mean_'].items()}
        pp.score_mean_ = {t: np.asarray(v) for t, v in d['score_mean_'].items()}
        pp.score_std_ = {t: np.asarray(v) for t, v in d['score_std_'].items()}
        pp.explained_variance_ratio_ = np.asarray(d['explained_variance_ratio_'])
        return pp


class PooledVelocity(torch.nn.Module):
    """VelocityMLP conditioned additionally on a small ticker embedding."""

    def __init__(self, dim, cond_dim, n_tickers, embed_dim=8, hidden=256,
                 n_blocks=3, dropout=0.1, embed_dropout=0.12):
        super().__init__()
        self.embed = torch.nn.Embedding(n_tickers, embed_dim)
        torch.nn.init.normal_(self.embed.weight, std=0.01)
        self.embed_dropout = embed_dropout
        self.core = VelocityMLP(dim=dim, cond_dim=cond_dim + embed_dim,
                                hidden=hidden, n_blocks=n_blocks,
                                dropout=dropout)

    def forward(self, z_t, t, cond, ticker_idx):
        e = self.embed(ticker_idx)
        if self.training and self.embed_dropout > 0:
            keep = (torch.rand(e.shape[0], 1, device=e.device)
                    > self.embed_dropout).to(e.dtype)
            e = e * keep                      # start from full pooling
        return self.core(z_t, t, torch.cat([cond, e], dim=-1))


class _Bound(torch.nn.Module):
    """Bind a ticker index so the ensemble can reuse flow_surface samplers."""

    def __init__(self, model, ticker_idx):
        super().__init__()
        self.model = model
        self.register_buffer('idx', ticker_idx, persistent=False)
        self.out_proj = model.core.out_proj

    def forward(self, z_t, t, cond):
        # sample_flow tiles the conditioning n_samples times (sample-major),
        # so the bound ticker index has to be tiled to match.
        idx = self.idx
        if idx.shape[0] != cond.shape[0]:
            idx = idx.repeat(cond.shape[0] // idx.shape[0])
        return self.model(z_t, t, cond, idx)


class VelocityEnsemble(torch.nn.Module):
    """Average the velocity fields of several independently seeded models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.out_proj = models[0].out_proj

    def forward(self, z_t, t, cond):
        return torch.stack([m(z_t, t, cond) for m in self.models]).mean(0)


def bind(model, ticker_idx):
    return _Bound(model, ticker_idx)


def pooled_fm_loss(model, z1, cond, ticker_idx, generator=None):
    """Conditional OT flow matching with the ticker index threaded through."""
    B = z1.shape[0]
    t = torch.rand(B, device=z1.device, dtype=z1.dtype, generator=generator)
    z0 = torch.randn(z1.shape, device=z1.device, dtype=z1.dtype,
                     generator=generator)
    tc = t.unsqueeze(-1)
    z_t = (1 - tc) * z0 + tc * z1
    return torch.mean((model(z_t, t, cond, ticker_idx) - (z1 - z0)) ** 2)


@torch.no_grad()
def sample_pooled(models, cond, ticker_idx, n_steps=50, n_samples=1,
                  generator=None):
    """Sample from an ensemble of pooled models for one ticker."""
    bound = [bind(m, ticker_idx) for m in models]
    ens = VelocityEnsemble(bound) if len(bound) > 1 else bound[0]
    return sample_flow(ens, cond, n_steps=n_steps, n_samples=n_samples,
                       generator=generator)

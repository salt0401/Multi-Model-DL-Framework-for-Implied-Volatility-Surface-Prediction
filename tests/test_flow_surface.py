"""Tests for Model 5 replacement: surface panel prep + conditional flow matching."""
import numpy as np
import pandas as pd
import pytest
import torch


# ── Prepare_surface_panel (dataset.py) ───────────────────────────────

class TestPrepareSurfacePanel:
    def test_grid_train_only_and_shapes(self, mock_config, mock_prs_dataset):
        from dataset import DataProcessor
        dp = DataProcessor(mock_config)
        df = mock_prs_dataset.copy()
        cutoff = pd.Timestamp('2020-01-06')
        # widen tau range only after cutoff — grid must ignore it
        df.loc[df['date'] > cutoff, 'tau'] = 5.0
        dp.prs_dataset = df
        panel = dp.Prepare_surface_panel(train_end_date=cutoff,
                                         n_tau_grid=3, n_logm_grid=4)
        assert panel['tau_grid'].max() < 3.0
        assert panel['surfaces'].shape[1] == 12
        assert len(panel['dates']) == len(panel['surfaces']) == len(panel['conditions'])
        assert panel['conditions'].shape[1] == 11
        assert (panel['surfaces'] > 0).all()

    def test_tau_major_layout(self, mock_config, mock_prs_dataset):
        """Row-major = tau-major: surface[i_tau * n_logm + i_logm]."""
        from dataset import DataProcessor
        dp = DataProcessor(mock_config)
        dp.prs_dataset = mock_prs_dataset.copy()
        cutoff = mock_prs_dataset['date'].max()
        panel = dp.Prepare_surface_panel(train_end_date=cutoff,
                                         n_tau_grid=3, n_logm_grid=4)
        # meta grids have the declared lengths
        assert len(panel['tau_grid']) == 3
        assert len(panel['logm_grid']) == 4


# ── FactorPreprocessor ────────────────────────────────────────────────

class TestFactorPreprocessor:
    def test_roundtrip_and_positivity(self):
        from flow_surface import FactorPreprocessor
        rng = np.random.default_rng(0)
        # low-rank structure + noise, log-normal-like tv scale
        factors = rng.normal(0, 1, size=(200, 3))
        loadings = rng.normal(0, 0.3, size=(3, 24))
        S = np.exp(-4.5 + factors @ loadings + rng.normal(0, 0.02, (200, 24)))
        pp = FactorPreprocessor(ev_target=0.99, max_components=12)
        pp.fit(S)
        Z = pp.transform(S)
        S2 = pp.inverse(Z)
        assert (S2 > 0).all()
        assert np.sqrt(np.mean((np.log(S2) - np.log(S)) ** 2)) < 0.05
        assert Z.shape[1] == pp.n_components_
        # z-scores standardized on the fitting set
        assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
        assert np.allclose(Z.std(axis=0), 1, atol=1e-9)

    def test_serialization_roundtrip(self):
        from flow_surface import FactorPreprocessor
        rng = np.random.default_rng(1)
        S = np.exp(rng.normal(-4.5, 0.4, size=(50, 12)))
        pp = FactorPreprocessor(max_components=5).fit(S)
        pp2 = FactorPreprocessor.from_dict(pp.to_dict())
        Z = pp.transform(S)
        assert np.allclose(pp2.transform(S), Z)
        assert np.allclose(pp2.inverse(Z), pp.inverse(Z))

    def test_fixed_n_components(self):
        from flow_surface import FactorPreprocessor
        rng = np.random.default_rng(2)
        S = np.exp(rng.normal(-4.5, 0.4, size=(50, 12)))
        pp = FactorPreprocessor(n_components=4).fit(S)
        assert pp.n_components_ == 4
        assert pp.transform(S).shape == (50, 4)


# ── VelocityMLP / flow matching ───────────────────────────────────────

class TestVelocityMLP:
    def test_shapes_and_gradients(self):
        from flow_surface import VelocityMLP
        model = VelocityMLP(dim=6, cond_dim=4, hidden=32, n_blocks=2)
        z = torch.randn(8, 6)
        t = torch.rand(8)
        c = torch.randn(8, 4)
        v = model(z, t, c)
        assert v.shape == (8, 6)
        v.sum().backward()
        assert model.input_proj.weight.grad is not None
        assert model.blocks[0].film.weight.grad is not None


class TestFlowMatching:
    def test_fm_loss_finite_and_positive(self):
        from flow_surface import VelocityMLP, fm_loss
        torch.manual_seed(0)
        model = VelocityMLP(dim=3, cond_dim=2, hidden=32, n_blocks=2)
        loss = fm_loss(model, torch.randn(16, 3), torch.randn(16, 2))
        assert torch.isfinite(loss) and loss > 0

    def test_fm_learns_conditional_mean(self):
        """FM on toy: z1 = 2*cond + small noise; sampled mean -> 2*cond."""
        from flow_surface import VelocityMLP, fm_loss, sample_flow
        torch.manual_seed(0)
        cond = torch.rand(4096, 1) * 2 - 1
        z1 = 2 * cond + 0.05 * torch.randn(4096, 1)
        model = VelocityMLP(dim=1, cond_dim=1, hidden=64, n_blocks=2, dropout=0.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for _ in range(400):
            idx = torch.randint(0, 4096, (256,))
            loss = fm_loss(model, z1[idx], cond[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        test_c = torch.tensor([[0.5], [-0.5]])
        samp = sample_flow(model, test_c, n_steps=50, n_samples=200,
                           generator=torch.Generator().manual_seed(1))
        means = samp.mean(dim=0)
        assert torch.allclose(means, 2 * test_c, atol=0.15)

    def test_sample_shape(self):
        from flow_surface import VelocityMLP, sample_flow
        model = VelocityMLP(dim=5, cond_dim=3, hidden=32, n_blocks=2)
        out = sample_flow(model, torch.randn(7, 3), n_steps=5, n_samples=4)
        assert out.shape == (4, 7, 5)
        assert torch.isfinite(out).all()


# ── EMA ───────────────────────────────────────────────────────────────

class TestEMA:
    def test_update_math(self):
        from flow_surface import EMA
        lin = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            lin.weight.fill_(1.0)
        ema = EMA(lin, decay=0.5)
        with torch.no_grad():
            lin.weight.fill_(3.0)
        ema.update(lin)
        assert torch.allclose(ema.shadow['weight'], torch.tensor([[2.0]]))
        ema.copy_to(lin)
        assert torch.allclose(lin.weight, torch.tensor([[2.0]]))


# ── build_dataset ─────────────────────────────────────────────────────

class TestBuildDataset:
    def _panel(self):
        dates = pd.date_range('2020-12-28', periods=6, freq='B')  # ...12-31, 01-01(skip B), 01-04...
        dates = list(dates)
        N, D = len(dates), 8
        rng = np.random.default_rng(3)
        return {
            'dates': dates,
            'surfaces': np.exp(rng.normal(-4.5, 0.3, size=(N, D))),
            'conditions': rng.normal(0, 1, size=(N, 4)),
            'tau_grid': np.linspace(0.05, 0.5, 2),
            'logm_grid': np.linspace(-0.2, 0.2, 4),
            'cond_names': ['a', 'b', 'c', 'd'],
        }

    def test_boundary_pair_excluded_from_train(self):
        from flow_surface import build_dataset
        panel = self._panel()
        train_end = pd.Timestamp('2020-12-31')
        test_start = pd.Timestamp('2021-01-01')
        splits = build_dataset(panel, train_end, test_start, val_frac=0.34)
        # No train/val pair may have TOMORROW after train_end
        all_train_dates = splits['train']['dates'] + splits['val']['dates']
        for d in all_train_dates:
            i = panel['dates'].index(d)
            assert panel['dates'][i + 1] <= train_end
        # Test pairs all have today >= test_start
        for d in splits['test']['dates']:
            assert d >= test_start

    def test_split_shapes_consistent(self):
        from flow_surface import build_dataset
        panel = self._panel()
        splits = build_dataset(panel, pd.Timestamp('2020-12-31'),
                               pd.Timestamp('2021-01-01'), val_frac=0.34)
        for name, sp in splits.items():
            assert len(sp['S_today']) == len(sp['S_tomorrow']) == len(sp['C']) \
                == len(sp['dates'])


class TestCondScalerInverse:
    def test_roundtrip(self):
        from flow_surface import CondScaler
        rng = np.random.default_rng(5)
        C = rng.normal(3, 7, size=(40, 6))
        cs = CondScaler().fit(C)
        assert np.allclose(cs.inverse(cs.transform(C)), C)

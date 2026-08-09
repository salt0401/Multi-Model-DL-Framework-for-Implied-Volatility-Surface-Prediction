"""Tests for the US event layer (us_events.py) and Global eSSVI (svi_us.py)."""
import numpy as np
import pandas as pd
import pytest


# ── Earnings calendar / event columns ─────────────────────────────────

def _cal(tmp_path):
    from us_events import EarningsCalendar
    pd.DataFrame({
        'ticker': ['AAPL'] * 3,
        'earnings_date': pd.to_datetime(['2024-02-01', '2024-05-02', '2024-08-01']),
        'source': ['yfinance'] * 3, 'confidence': [1.0] * 3,
    }).to_csv(tmp_path / 'earnings_dates_v2.csv', index=False)
    return EarningsCalendar.from_data_dir(str(tmp_path))


class TestEarningsCalendar:
    def test_count_in_is_half_open(self, tmp_path):
        cal = _cal(tmp_path)
        # (start, end] — an expiry landing exactly on the event counts it
        n = cal.count_in('AAPL', np.array(['2024-01-10'], dtype='datetime64[ns]'),
                         np.array(['2024-02-01'], dtype='datetime64[ns]'))
        assert n[0] == 1
        # an expiry the day before does not
        n = cal.count_in('AAPL', np.array(['2024-01-10'], dtype='datetime64[ns]'),
                         np.array(['2024-01-31'], dtype='datetime64[ns]'))
        assert n[0] == 0

    def test_count_spanning_two_events(self, tmp_path):
        cal = _cal(tmp_path)
        n = cal.count_in('AAPL', np.array(['2024-01-10'], dtype='datetime64[ns]'),
                         np.array(['2024-06-01'], dtype='datetime64[ns]'))
        assert n[0] == 2

    def test_days_to_next_and_since_last(self, tmp_path):
        cal = _cal(tmp_path)
        w = np.array(['2024-01-22'], dtype='datetime64[ns]')
        assert cal.days_to_next('AAPL', w)[0] == 10
        w2 = np.array(['2024-02-11'], dtype='datetime64[ns]')
        assert cal.days_since_last('AAPL', w2)[0] == 10

    def test_unknown_ticker_is_safe(self, tmp_path):
        cal = _cal(tmp_path)
        w = np.array(['2024-01-22'], dtype='datetime64[ns]')
        assert cal.days_to_next('ZZZZ', w)[0] == 120


class TestDecomposition:
    def _synthetic(self, sigma_d=0.30, jump=0.06):
        """Build quotes that satisfy w = sigma_d^2 * tau + n * J^2 exactly."""
        rows = []
        d = pd.Timestamp('2024-01-15')
        for tau_days, n in [(14, 0), (30, 1), (45, 1)]:
            tau = tau_days / 365.25
            w = sigma_d ** 2 * tau + n * jump ** 2
            for k in np.linspace(-0.1, 0.1, 7):
                rows.append({'ticker': 'AAPL', 'date': d,
                             'expiration': d + pd.Timedelta(days=tau_days),
                             'tau': tau, 'logm': k, 'total_var': w,
                             'n_earnings': n})
        return pd.DataFrame(rows)

    def test_recovers_known_components(self):
        from us_events import decompose_variance
        df = self._synthetic(sigma_d=0.30, jump=0.06)
        dec = decompose_variance(df)
        assert len(dec) == 1
        assert dec['diffusive_vol'].iloc[0] == pytest.approx(0.30, abs=0.01)
        assert dec['implied_move'].iloc[0] == pytest.approx(0.06, abs=0.005)
        assert dec['r2'].iloc[0] > 0.999

    def test_unidentified_when_all_expiries_share_event_count(self):
        from us_events import decompose_variance
        df = self._synthetic()
        df['n_earnings'] = 1                     # collinear design
        dec = decompose_variance(df)
        assert not bool(dec['identified'].iloc[0])

    def test_event_time_is_increasing_in_tau(self):
        from us_events import event_time
        tau = np.array([0.03, 0.08, 0.12])
        n = np.array([0, 1, 1])
        te = event_time(tau, n, diffusive_var=0.09, jump_var=0.0036)
        assert np.all(np.diff(te) > 0)
        # an event adds strictly positive extra clock time
        assert event_time([0.08], [1], 0.09, 0.0036)[0] > 0.08


# ── Global eSSVI ──────────────────────────────────────────────────────

class TestGlobalESSVI:
    # Realistic, arbitrage-FREE truth. psi must be small relative to theta:
    # phi = psi/theta, and phi ~ 100 (which psi=1.1 at theta=0.01 implies)
    # produces a surface whose own slices cross, so no arbitrage-free model
    # could reproduce it. test_truth_is_arbitrage_free guards this.
    TRUTH = [(0.010, -0.35, 0.10, 0.04),
             (0.020, -0.30, 0.14, 0.08),
             (0.030, -0.25, 0.17, 0.12)]

    def _slices(self, n_slices=3, noise=0.0, seed=0):
        from svi_us import essvi_w
        rng = np.random.default_rng(seed)
        k = np.linspace(-0.35, 0.35, 25)
        out = []
        for (theta, rho, psi, tau) in self.TRUTH[:n_slices]:
            w = essvi_w(k, theta, rho, psi)
            if noise:
                w = w * (1 + rng.normal(0, noise, len(k)))
            out.append({'k': k, 'w': w, 'tau': tau,
                        'weight': np.ones_like(k)})
        return out

    def test_truth_is_arbitrage_free(self):
        """Guard the fixture: a crossing truth would make recovery impossible."""
        from svi_us import arbitrage_report
        t = {'theta': np.array([p[0] for p in self.TRUTH]),
             'rho': np.array([p[1] for p in self.TRUTH]),
             'psi': np.array([p[2] for p in self.TRUTH]),
             'tau': np.array([p[3] for p in self.TRUTH])}
        rep = arbitrage_report(t)
        assert rep['calendar_violation_rate'] == 0.0
        assert rep['butterfly_ok']

    def test_butterfly_holds_by_construction_for_random_params(self):
        """Both Gatheral-Jacquier conditions AND the density itself.

        Regression guard: an earlier parameterization enforced only
        psi(1+|rho|) < 4, which is insufficient at small theta — the density
        went to -2.7 on real SPY slices while this gate still passed.
        """
        from svi_us import unpack, _density
        rng = np.random.default_rng(1)
        k = np.linspace(-0.6, 0.6, 121)
        for _ in range(300):
            p = rng.normal(0, 6, 9)          # deliberately extreme
            theta, rho, psi = unpack(p, 3)
            one_p = 1 + np.abs(rho)
            assert np.all(psi * one_p < 4.0 + 1e-12)          # condition (i)
            assert np.all(psi ** 2 * one_p / theta <= 4.0 + 1e-9)  # condition (ii)
            assert np.all(np.diff(theta) > 0)   # strict ATM calendar
            assert np.all(theta > 0)
            for i in range(3):
                assert np.min(_density(k, theta[i], rho[i], psi[i])) > 0

    def test_density_positive_on_fitted_surfaces(self):
        from svi_us import fit_snapshot, arbitrage_report
        for seed in range(5):
            fit = fit_snapshot(self._slices(noise=0.05, seed=seed))
            rep = arbitrage_report(fit)
            assert rep['min_density'] > 0
            assert rep['butterfly_ok']

    def test_recovers_known_surface(self):
        from svi_us import fit_snapshot, iv_rmse
        slices = self._slices()
        fit = fit_snapshot(slices)
        assert iv_rmse(fit, slices) < 0.01       # < 1 vol point on clean data

    def test_no_calendar_crossing_after_projection(self):
        from svi_us import fit_snapshot, arbitrage_report
        fit = fit_snapshot(self._slices(noise=0.05, seed=3))
        rep = arbitrage_report(fit)
        assert rep['calendar_violation_rate'] == 0.0
        assert rep['butterfly_ok']
        assert rep['theta_increasing']

    def test_enforce_calendar_fixes_a_crossing(self):
        from svi_us import enforce_calendar, arbitrage_report
        bad = {'theta': np.array([0.05, 0.0501]),
               'rho': np.array([-0.2, -0.9]),
               'psi': np.array([0.5, 3.0]),
               'tau': np.array([0.04, 0.08])}
        assert arbitrage_report(bad)['calendar_violation_rate'] > 0
        assert arbitrage_report(enforce_calendar(bad))['calendar_violation_rate'] == 0.0

    def test_cv_is_finite_and_worse_than_in_sample(self):
        from svi_us import fit_snapshot, iv_rmse, cv_iv_rmse
        slices = self._slices(noise=0.05, seed=7)
        cv = cv_iv_rmse(slices)
        ins = iv_rmse(fit_snapshot(slices), slices)
        assert np.isfinite(cv) and cv >= ins * 0.5


# ── M4 event head ─────────────────────────────────────────────────────

class TestHyperIVEventHead:
    def _model(self):
        import torch
        from hyperiv import HyperIVModel
        torch.manual_seed(0)
        m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1,
                         target_hidden_dims=(8, 4))
        m.enable_event_head(n_tickers=3,
                            sigma_j_init=torch.tensor([0.05, 0.08, 0.09]))
        # eval mode: the set encoder uses dropout, so two forward passes in
        # train mode differ and none of these comparisons would be meaningful.
        m.eval()
        return m

    def _inputs(self, B=2, N=5):
        import torch
        return (torch.rand(B, 8, 3) * 0.05, torch.rand(B, N, 1) * 0.2,
                torch.randn(B, N, 1) * 0.15, torch.rand(B, N, 1) * 0.05)

    def test_zero_events_leaves_surface_unchanged(self):
        """Index symbols (n_events = 0) must be numerically untouched."""
        import torch
        m = self._model()
        ref, tau, logm, yatm = self._inputs()
        idx = torch.zeros(2, dtype=torch.long)
        base, *_ = m(ref, tau, logm, yatm)
        with_zero, *_ = m(ref, tau, logm, yatm,
                          n_events=torch.zeros_like(tau), ticker_idx=idx)
        assert torch.allclose(base, with_zero, atol=1e-12)

    def test_event_raises_total_variance(self):
        import torch
        m = self._model()
        ref, tau, logm, yatm = self._inputs()
        idx = torch.zeros(2, dtype=torch.long)
        base, *_ = m(ref, tau, logm, yatm,
                     n_events=torch.zeros_like(tau), ticker_idx=idx)
        one, *_ = m(ref, tau, logm, yatm,
                    n_events=torch.ones_like(tau), ticker_idx=idx)
        assert (one > base).all()

    def test_event_term_scales_with_count_and_sigma(self):
        import torch
        m = self._model()
        ref, tau, logm, yatm = self._inputs()
        idx0 = torch.zeros(2, dtype=torch.long)
        b, *_ = m(ref, tau, logm, yatm, n_events=torch.zeros_like(tau),
                  ticker_idx=idx0)
        one, *_ = m(ref, tau, logm, yatm, n_events=torch.ones_like(tau),
                    ticker_idx=idx0)
        two, *_ = m(ref, tau, logm, yatm,
                    n_events=2 * torch.ones_like(tau), ticker_idx=idx0)
        # additive and linear in the event count
        assert torch.allclose(two - b, 2 * (one - b), atol=1e-10)
        # a higher-sigma ticker gets a larger bump
        idx2 = torch.full((2,), 2, dtype=torch.long)
        big, *_ = m(ref, tau, logm, yatm, n_events=torch.ones_like(tau),
                    ticker_idx=idx2)
        assert (big - b).mean() > (one - b).mean()

    def test_event_head_is_trainable(self):
        import torch
        m = self._model()
        ref, tau, logm, yatm = self._inputs()
        idx = torch.zeros(2, dtype=torch.long)
        out, *_ = m(ref, tau, logm, yatm, n_events=torch.ones_like(tau),
                    ticker_idx=idx)
        out.sum().backward()
        assert m.log_sigma_j.grad is not None
        assert torch.isfinite(m.log_sigma_j.grad).all()

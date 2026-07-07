"""Tests for the Mag 7 US options data layer (src/us_dataset.py)."""
import numpy as np
import pandas as pd
import pytest

from us_dataset import UsOptionsProcessor, build_us_pairs, DAYS_PER_YEAR


def _write_synthetic(dirpath, tickers=('AAPL',), dates=None, spot=100.0,
                     fb_alias=False):
    """Write minimal chains_*.csv + spots.csv + vix_us.csv into dirpath."""
    if dates is None:
        dates = pd.date_range('2023-01-02', periods=8, freq='2B')
    rows = []
    rng = np.random.default_rng(0)
    for t in tickers:
        for d in dates:
            for exp_days in (14, 30, 46):
                exp = d + pd.Timedelta(days=exp_days)
                for k in np.linspace(spot * 0.8, spot * 1.2, 9):
                    for cp in ('Call', 'Put'):
                        iv = 0.25 + 0.2 * (np.log(k / spot)) ** 2 + rng.uniform(0, 0.01)
                        rows.append({
                            'date': d.date(), 'act_symbol': t,
                            'expiration': exp.date(), 'strike': round(k, 2),
                            'call_put': cp, 'bid': 1.0, 'ask': 1.2,
                            'vol': round(iv, 4), 'delta': 0.5, 'gamma': 0.01,
                            'theta': -0.02, 'vega': 0.1, 'rho': 0.01,
                            'ticker': t,
                        })
    df = pd.DataFrame(rows)
    for t in tickers:
        name = 'FB' if (fb_alias and t == 'META') else t
        sub = df[df['act_symbol'] == t].copy()
        sub['act_symbol'] = name
        sub.drop(columns='ticker').to_csv(dirpath / f'chains_{name}.csv',
                                          index=False)

    spot_dates = pd.bdate_range(dates[0] - pd.Timedelta(days=5),
                                dates[-1] + pd.Timedelta(days=5))
    spots = pd.concat([
        pd.DataFrame({'date': spot_dates, 'close_unadj': spot,
                      'dividend': 0.0, 'ticker': t}) for t in tickers])
    spots.to_csv(dirpath / 'spots.csv', index=False)

    pd.DataFrame({'date': spot_dates,
                  'Close': 18.0 + np.arange(len(spot_dates)) * 0.1}) \
        .to_csv(dirpath / 'vix_us.csv', index=False)
    return dates


class TestBuild:
    def test_basic_build_columns_and_filters(self, tmp_path):
        _write_synthetic(tmp_path)
        proc = UsOptionsProcessor(str(tmp_path))
        df = proc.build(tickers=['AAPL'])
        assert set(['mid', 'spread', 'tau', 'logm', 'total_var',
                    'y_atm', 'forward', 'spot_syn']).issubset(df.columns)
        # parity forward: synthetic C_mid == P_mid, so F == median strike
        # and spot_syn is positive and at spot scale
        assert (df['spot_syn'] > 50).all() and (df['spot_syn'] < 200).all()
        assert (df['bid'] > 0).all()
        assert (df['logm'].abs() <= 0.6).all()
        assert (df['tau'] >= 3 / DAYS_PER_YEAR - 1e-12).all()
        assert (df['y_atm'] > 0).all()
        # w = vol^2 * tau
        np.testing.assert_allclose(df['total_var'],
                                   df['vol'] ** 2 * df['tau'], rtol=1e-12)

    def test_saturday_maps_to_friday_close(self, tmp_path):
        sat = pd.DatetimeIndex([pd.Timestamp('2023-01-14')])  # Saturday
        _write_synthetic(tmp_path, dates=sat)
        proc = UsOptionsProcessor(str(tmp_path))
        df = proc.build(tickers=['AAPL'])
        assert len(df) > 0          # merge_asof backward found Friday's close
        assert (df['close_yf'] == 100.0).all()

    def test_fb_relabeled_as_meta(self, tmp_path):
        _write_synthetic(tmp_path, tickers=('META',), fb_alias=True)
        proc = UsOptionsProcessor(str(tmp_path))
        df = proc.build(tickers=['META'])
        assert len(df) > 0
        assert (df['ticker'] == 'META').all()

    def test_dedup_on_resume_duplicates(self, tmp_path):
        _write_synthetic(tmp_path)
        # Simulate resume artifact: duplicate the whole file content
        p = tmp_path / 'chains_AAPL.csv'
        df = pd.read_csv(p)
        pd.concat([df, df]).to_csv(p, index=False)
        proc = UsOptionsProcessor(str(tmp_path))
        chains = proc.load_chains(['AAPL'])
        assert not chains.duplicated(
            subset=['ticker', 'date', 'expiration', 'strike', 'call_put']).any()


class TestPanelAndPairs:
    def _panel(self, tmp_path):
        dates = _write_synthetic(tmp_path)
        proc = UsOptionsProcessor(str(tmp_path))
        df = proc.build(tickers=['AAPL'])
        panel = proc.prepare_surface_panel(df, 'AAPL',
                                           train_end_date=dates[5])
        return panel, dates

    def test_panel_shapes_and_positivity(self, tmp_path):
        panel, dates = self._panel(tmp_path)
        n_tau, n_logm = len(panel['tau_grid']), len(panel['logm_grid'])
        assert n_tau == 3 and n_logm == 15
        assert panel['surfaces'].shape == (len(panel['dates']), n_tau * n_logm)
        assert (panel['surfaces'] > 0).all()
        assert panel['conditions'].shape == (len(panel['dates']), 7)

    def test_pairs_have_gap_days_condition(self, tmp_path):
        panel, dates = self._panel(tmp_path)
        splits = build_us_pairs(panel, train_end_date=dates[5],
                                test_start_date=dates[6])
        for sp in splits.values():
            if len(sp['dates']) == 0:
                continue
            assert sp['C'].shape[1] == 8          # 7 base + gap_days
            gaps = sp['C'][:, -1]
            assert (gaps >= 1).all() and (gaps <= 7).all()

    def test_hyperiv_surfaces_pooled_format(self, tmp_path):
        dates = _write_synthetic(tmp_path, tickers=('AAPL', 'MSFT'))
        proc = UsOptionsProcessor(str(tmp_path))
        df = proc.build(tickers=['AAPL', 'MSFT'])
        surfaces = proc.prepare_hyperiv_surfaces(df)
        assert len(surfaces) == 2 * len(dates)
        d, t, tens = surfaces[0]
        assert t in ('AAPL', 'MSFT')
        assert len(tens) == 4
        assert tens[0].shape[1] == 1

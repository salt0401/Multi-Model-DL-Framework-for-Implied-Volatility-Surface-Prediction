"""Tests for structural_break.py: break detection classes."""
import configparser

import numpy as np
import pandas as pd
import pytest

from structural_break import (
    CUSUMDetector,
    BaiPerronDetector,
    VIXBreakDetector,
    detect_structural_break,
)


# ── CUSUMDetector ─────────────────────────────────────────────────────

class TestCUSUMDetector:
    def test_no_break_in_stationary(self):
        np.random.seed(42)
        series = pd.Series(np.random.randn(200), index=pd.date_range('2020-01-01', periods=200))
        detector = CUSUMDetector(threshold=5.0, lookback=30)
        result = detector.detect(series)
        assert not result['is_break'].any()

    def test_break_on_level_shift(self):
        np.random.seed(42)
        values = np.concatenate([np.random.randn(100), np.random.randn(100) + 10])
        series = pd.Series(values, index=pd.date_range('2020-01-01', periods=200))
        detector = CUSUMDetector(threshold=2.0, lookback=30)
        result = detector.detect(series)
        assert result['is_break'].any()

    def test_threshold_sensitivity(self):
        np.random.seed(42)
        values = np.concatenate([np.random.randn(100), np.random.randn(100) + 3])
        series = pd.Series(values, index=pd.date_range('2020-01-01', periods=200))
        # Low threshold -> more breaks
        det_low = CUSUMDetector(threshold=1.0, lookback=30)
        det_high = CUSUMDetector(threshold=10.0, lookback=30)
        breaks_low = det_low.detect(series)['is_break'].sum()
        breaks_high = det_high.detect(series)['is_break'].sum()
        assert breaks_low >= breaks_high

    def test_result_columns(self):
        series = pd.Series(np.random.randn(100), index=pd.date_range('2020-01-01', periods=100))
        detector = CUSUMDetector(threshold=2.0, lookback=20)
        result = detector.detect(series)
        assert set(result.columns) == {'date', 'cusum_stat', 'is_break'}

    def test_online_short_history(self):
        detector = CUSUMDetector(threshold=2.0, lookback=30)
        history = np.random.randn(10)  # less than lookback
        is_break, stat = detector.detect_online(5.0, history)
        assert is_break is False
        assert stat == 0.0

    def test_online_spike(self):
        np.random.seed(42)
        detector = CUSUMDetector(threshold=2.0, lookback=30)
        history = np.random.randn(50)  # stationary history
        is_break, stat = detector.detect_online(100.0, history)
        assert bool(is_break) is True


# ── BaiPerronDetector ─────────────────────────────────────────────────

class TestBaiPerronDetector:
    def test_single_break(self):
        np.random.seed(42)
        values = np.concatenate([np.random.randn(50), np.random.randn(50) + 10])
        detector = BaiPerronDetector(n_bkps=1)
        bkps = detector.detect(pd.Series(values))
        assert len(bkps) == 1
        assert 30 <= bkps[0] <= 70

    def test_no_trailing_index(self):
        np.random.seed(42)
        values = np.random.randn(100)
        detector = BaiPerronDetector(pen=1.0)
        bkps = detector.detect(pd.Series(values))
        # Last element (== len(series)) should be removed
        if bkps:
            assert bkps[-1] < 100

    def test_pen_parameter(self):
        np.random.seed(42)
        values = np.concatenate([np.random.randn(50), np.random.randn(50) + 5])
        # High penalty = fewer breaks
        det_high = BaiPerronDetector(pen=100.0)
        bkps = det_high.detect(pd.Series(values))
        assert len(bkps) <= 2


# ── VIXBreakDetector ──────────────────────────────────────────────────

class TestVIXBreakDetector:
    def _make_vix_df(self, n=50, base_close=20.0, spike_idx=None, spike_change=None):
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        close = np.full(n, base_close)
        vix_change = np.zeros(n)
        if spike_idx is not None and spike_change is not None:
            vix_change[spike_idx] = spike_change
        return pd.DataFrame({'date': dates, 'Close': close, 'vix_change': vix_change})

    def test_overnight_spike(self):
        df = self._make_vix_df(spike_idx=10, spike_change=0.15)
        detector = VIXBreakDetector(vix_threshold=0.05)
        result = detector.detect(df)
        assert bool(result.loc[10, 'is_break']) is True

    def test_sustained_elevation(self):
        dates = pd.date_range('2020-01-01', periods=20, freq='B')
        close = np.concatenate([np.full(10, 20.0), np.full(10, 50.0)])
        vix_change = np.zeros(20)
        df = pd.DataFrame({'date': dates, 'Close': close, 'vix_change': vix_change})
        detector = VIXBreakDetector(vix_threshold=0.5, sustained_days=3, sustained_level=40.0)
        result = detector.detect(df)
        # Should find break after sustained elevation
        assert result['is_break'].any()

    def test_calm_market(self):
        df = self._make_vix_df(n=50)
        detector = VIXBreakDetector(vix_threshold=0.05, sustained_days=5, sustained_level=50.0)
        result = detector.detect(df)
        assert not result['is_break'].any()

    def test_result_columns(self):
        df = self._make_vix_df()
        detector = VIXBreakDetector()
        result = detector.detect(df)
        assert set(result.columns) == {'date', 'is_break', 'reason'}


# ── detect_structural_break (dispatcher) ─────────────────────────────

class TestDetectStructuralBreak:
    def _make_config(self, method='cusum'):
        cfg = configparser.ConfigParser()
        cfg['structural_break'] = {
            'method': method,
            'threshold': '2.0',
            'lookback': '20',
            'vix_threshold': '0.05',
            'sustained_days': '3',
        }
        return cfg

    def test_cusum_dispatch(self):
        cfg = self._make_config('cusum')
        series = pd.Series(np.random.randn(100), index=pd.date_range('2020-01-01', periods=100))
        result = detect_structural_break(cfg, series)
        assert isinstance(result, pd.DataFrame)

    def test_vix_requires_data(self):
        cfg = self._make_config('vix')
        series = pd.Series(np.random.randn(100))
        with pytest.raises(ValueError, match="VIX data required"):
            detect_structural_break(cfg, series, vix_df=None)

    def test_unknown_raises(self):
        cfg = self._make_config('unknown_method')
        series = pd.Series(np.random.randn(100))
        with pytest.raises(ValueError, match="Unknown structural break method"):
            detect_structural_break(cfg, series)

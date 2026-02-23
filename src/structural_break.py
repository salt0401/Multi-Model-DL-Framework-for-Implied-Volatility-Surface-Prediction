"""Structural break detection for IV surface regime changes (PDF pages 70-75).

Implements:
- CUSUMDetector: Cumulative sum test for mean shifts; rolling online detection
- BaiPerronDetector: Multiple break detection via ruptures library
- VIXBreakDetector: VIX overnight threshold + sustained elevation check
- detect_structural_break(): Unified interface selecting method by config
"""
import numpy as np
import pandas as pd


class CUSUMDetector:
    """Cumulative sum (CUSUM) test for mean shifts in time series.

    Implements rolling online detection with configurable threshold.
    """

    def __init__(self, threshold=2.0, lookback=60):
        self.threshold = threshold
        self.lookback = lookback

    def detect(self, series):
        """Detect structural breaks in a pandas Series.

        Args:
            series: pd.Series with datetime index and numeric values

        Returns:
            pd.DataFrame with columns ['date', 'cusum_stat', 'is_break']
        """
        values = series.values.astype(float)
        n = len(values)
        results = []

        for i in range(self.lookback, n):
            window = values[i - self.lookback:i]
            mu = np.mean(window)
            sigma = np.std(window, ddof=1)
            if sigma < 1e-10:
                sigma = 1e-10

            # CUSUM statistic: cumulative deviation from mean, normalized
            cusum = np.cumsum(window - mu) / (sigma * np.sqrt(self.lookback))
            stat = np.max(np.abs(cusum))
            is_break = stat > self.threshold

            results.append({
                'date': series.index[i] if hasattr(series, 'index') else i,
                'cusum_stat': stat,
                'is_break': is_break,
            })

        return pd.DataFrame(results)

    def detect_online(self, new_value, history):
        """Single-step online detection.

        Args:
            new_value: new observation
            history: numpy array of recent values (at least lookback length)

        Returns:
            (is_break, cusum_stat)
        """
        if len(history) < self.lookback:
            return False, 0.0

        window = history[-self.lookback:]
        mu = np.mean(window)
        sigma = np.std(window, ddof=1)
        if sigma < 1e-10:
            return False, 0.0

        deviation = (new_value - mu) / sigma
        return abs(deviation) > self.threshold, abs(deviation)


class BaiPerronDetector:
    """Multiple structural break detection using ruptures library."""

    def __init__(self, n_bkps=None, pen=None, min_size=20):
        self.n_bkps = n_bkps
        self.pen = pen
        self.min_size = min_size

    def detect(self, series):
        """Detect multiple breakpoints.

        Args:
            series: pd.Series or numpy array

        Returns:
            list of breakpoint indices
        """
        try:
            import ruptures as rpt
        except ImportError:
            raise ImportError("ruptures package required for BaiPerronDetector. Install with: pip install ruptures")

        values = np.array(series, dtype=float).reshape(-1, 1)

        model = rpt.Pelt(model="rbf", min_size=self.min_size).fit(values)
        if self.pen is not None:
            breakpoints = model.predict(pen=self.pen)
        elif self.n_bkps is not None:
            # Use Binseg for known number of breakpoints
            model = rpt.Binseg(model="rbf", min_size=self.min_size).fit(values)
            breakpoints = model.predict(n_bkps=self.n_bkps)
        else:
            breakpoints = model.predict(pen=1.0)

        # Remove the last element (always = len(series))
        if breakpoints and breakpoints[-1] == len(values):
            breakpoints = breakpoints[:-1]

        return breakpoints


class VIXBreakDetector:
    """Simple VIX-based structural break detection.

    Detects breaks when:
    1. VIX overnight change exceeds threshold, OR
    2. VIX remains elevated above a level for sustained_days
    """

    def __init__(self, vix_threshold=0.05, sustained_days=5, sustained_level=None):
        self.vix_threshold = vix_threshold
        self.sustained_days = sustained_days
        self.sustained_level = sustained_level

    def detect(self, vix_df):
        """Detect breaks from VIX data.

        Args:
            vix_df: DataFrame with 'date', 'Close', 'vix_change' columns

        Returns:
            pd.DataFrame with columns ['date', 'is_break', 'reason']
        """
        df = vix_df.copy().sort_values('date').reset_index(drop=True)
        results = []

        if self.sustained_level is None:
            self.sustained_level = df['Close'].quantile(0.9)

        for i in range(len(df)):
            is_break = False
            reason = ''

            # Check overnight spike
            if 'vix_change' in df.columns and pd.notna(df.loc[i, 'vix_change']):
                if abs(df.loc[i, 'vix_change']) > self.vix_threshold:
                    is_break = True
                    reason = f'VIX overnight change: {df.loc[i, "vix_change"]:.4f}'

            # Check sustained elevation
            if i >= self.sustained_days:
                recent = df.loc[i - self.sustained_days + 1:i, 'Close']
                if (recent > self.sustained_level).all():
                    is_break = True
                    reason = reason or f'VIX sustained above {self.sustained_level:.1f}'

            results.append({
                'date': df.loc[i, 'date'],
                'is_break': is_break,
                'reason': reason,
            })

        return pd.DataFrame(results)


def detect_structural_break(config, data, vix_df=None):
    """Unified interface for structural break detection.

    Args:
        config: ConfigParser object with [structural_break] section
        data: pd.Series of the time series to test (e.g., IV changes)
        vix_df: optional VIX DataFrame for VIX-based detection

    Returns:
        Detection results (format depends on method)
    """
    sb_cfg = config['structural_break']
    method = sb_cfg.get('method', 'cusum')

    if method == 'cusum':
        detector = CUSUMDetector(
            threshold=sb_cfg.getfloat('threshold'),
            lookback=sb_cfg.getint('lookback'),
        )
        return detector.detect(data)

    elif method == 'bai_perron':
        detector = BaiPerronDetector()
        return detector.detect(data)

    elif method == 'vix':
        if vix_df is None:
            raise ValueError("VIX data required for method='vix'")
        detector = VIXBreakDetector(
            vix_threshold=sb_cfg.getfloat('vix_threshold'),
            sustained_days=sb_cfg.getint('sustained_days'),
        )
        return detector.detect(vix_df)

    else:
        raise ValueError(f"Unknown structural break method: {method}")

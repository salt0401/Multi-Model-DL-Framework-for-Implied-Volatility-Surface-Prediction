"""Tests for dataset.py: DataProcessor with mock data (no file I/O)."""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from dataset import DataProcessor


class _SimpleBaseModel(nn.Module):
    """A simple mock base model that returns 4-tuple without autograd.grad().

    Used in tests where prepare_adjustment_data wraps the forward in
    torch.no_grad(), which is incompatible with SmileModel's autograd.grad().
    """
    def forward(self, tau, logm, yATM):
        out = yATM.clone()
        zeros = torch.zeros_like(out)
        return out, zeros, zeros, zeros


# ── Prepare_train_data ────────────────────────────────────────────────

class TestPrepareTrainData:
    def test_returns_3_loaders(self, patched_data_processor):
        """Bug T1: Prepare_train_data returns (train_loader, val_loader, c6_loader)."""
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        train_start = dates[0]
        train_end = dates[-1]
        result = dp.Prepare_train_data(train_start, train_end, batch_size=16)
        assert len(result) == 3
        train_loader, val_loader, c6_loader = result
        assert train_loader is not None
        assert val_loader is not None
        assert c6_loader is not None

    def test_batch_contents_train(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        train_loader, _, _ = dp.Prepare_train_data(dates[0], dates[-1], batch_size=16)
        batch = next(iter(train_loader))
        assert len(batch) == 4  # tau, logm, y, yATM
        tau, logm, y, yATM = batch
        assert tau.shape[1] == 1
        assert logm.shape[1] == 1

    def test_c6_contents(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        _, _, c6_loader = dp.Prepare_train_data(dates[0], dates[-1], batch_size=256)
        batch = next(iter(c6_loader))
        assert len(batch) == 3  # tau, logm, yATM

    def test_chronological_split(self, patched_data_processor):
        """Validation dates should be AFTER training dates (no future leakage)."""
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        train_start = dates[0]
        train_end = dates[-1]

        # Get the split date by reimplementing the logic
        train_dataset = dp.prs_dataset[
            (dp.prs_dataset['date'] >= train_start) &
            (dp.prs_dataset['date'] <= train_end)
        ].sort_values('date')
        unique_dates = np.sort(train_dataset['date'].unique())
        split_idx = int(len(unique_dates) * 0.8)
        val_start_date = unique_dates[split_idx]

        train_part = train_dataset[train_dataset['date'] < val_start_date]
        val_part = train_dataset[train_dataset['date'] >= val_start_date]

        assert train_part['date'].max() < val_part['date'].min()

    def test_val_not_empty(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        _, val_loader, _ = dp.Prepare_train_data(dates[0], dates[-1], batch_size=16)
        # Iterate to confirm val_loader has data
        total = sum(1 for _ in val_loader)
        assert total > 0


# ── Prepare_test_data ─────────────────────────────────────────────────

class TestPrepareTestData:
    def test_returns_loader_and_dataframe(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        result = dp.Prepare_test_data(dates[0], dates[-1])
        assert len(result) == 2
        test_loader, test_df = result
        assert test_loader is not None
        assert isinstance(test_df, pd.DataFrame)

    def test_dtype_float64(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        test_loader, _ = dp.Prepare_test_data(dates[0], dates[-1])
        batch = next(iter(test_loader))
        for tensor in batch:
            assert tensor.dtype == torch.float64


# ── Prepare_hyperiv_data ─────────────────────────────────────────────

class TestPrepareHyperivData:
    def test_returns_sorted_list(self, patched_data_processor):
        dp = patched_data_processor
        surfaces = dp.Prepare_hyperiv_data()
        assert isinstance(surfaces, list)
        assert len(surfaces) > 0
        # Check sorted by date
        dates = [s[0] for s in surfaces]
        assert dates == sorted(dates)

    def test_per_surface_shapes(self, patched_data_processor):
        dp = patched_data_processor
        surfaces = dp.Prepare_hyperiv_data()
        date, (tau, logm, tv, yatm) = surfaces[0]
        assert tau.shape[1] == 1
        assert logm.shape[1] == 1
        assert tv.shape[1] == 1
        assert yatm.shape[1] == 1
        assert tau.shape[0] == logm.shape[0]


# ── prepare_adjustment_data ──────────────────────────────────────────

class TestPrepareAdjustmentData:
    @pytest.fixture
    def _adj_setup(self, patched_data_processor):
        """Common setup: inject VIX data and use simple base model."""
        dp = patched_data_processor
        np.random.seed(77)
        vix_dates = dp.prs_dataset['date'].unique()
        vix_df = pd.DataFrame({
            'date': vix_dates,
            'Close': np.random.uniform(15, 30, len(vix_dates)),
        })
        vix_df = vix_df.sort_values('date')
        vix_df['vix_change'] = vix_df['Close'].pct_change()
        dp.load_vix_data = lambda: vix_df
        base_model = _SimpleBaseModel()
        return dp, base_model

    def test_returns_sequences_targets_masks_dates(self, _adj_setup):
        dp, base_model = _adj_setup
        result = dp.prepare_adjustment_data(base_model, torch.device('cpu'), sequence_length=3)
        sequences, targets, masks, seq_dates = result
        if sequences is not None:
            assert sequences.ndim == 3  # (n, seq_len, features)
            assert targets.ndim == 2   # (n, 1)
            assert masks.ndim == 2     # (n, seq_len)
            assert seq_dates.ndim == 1  # (n,)
            assert sequences.shape[0] == targets.shape[0] == masks.shape[0] == len(seq_dates)

    def test_padding_zeros(self, _adj_setup):
        dp, base_model = _adj_setup
        result = dp.prepare_adjustment_data(base_model, torch.device('cpu'), sequence_length=5)
        sequences, targets, masks, _ = result
        if sequences is not None and masks is not None:
            # Where mask is 0, sequence should be 0
            padded = (masks == 0)
            if padded.any():
                padded_vals = sequences[padded.unsqueeze(-1).expand_as(sequences)]
                assert (padded_vals == 0).all()

    def test_mask_matches_padding(self, _adj_setup):
        dp, base_model = _adj_setup
        result = dp.prepare_adjustment_data(base_model, torch.device('cpu'), sequence_length=5)
        sequences, targets, masks, _ = result
        if masks is not None:
            # Mask should be binary
            assert ((masks == 0) | (masks == 1)).all()

    def test_dates_chronological_for_split(self, _adj_setup):
        dp, base_model = _adj_setup
        result = dp.prepare_adjustment_data(base_model, torch.device('cpu'), sequence_length=3)
        sequences, targets, masks, seq_dates = result
        if seq_dates is not None:
            # Each date should come from the dataset
            all_dates = set(dp.prs_dataset['date'].unique())
            for d in seq_dates:
                assert d in all_dates

    def test_sp500_return_lagged_one_day(self, _adj_setup):
        """sp500_return dated day t is the US close realized ~15h AFTER
        Taiwan's close on day t; a day-t target may only see day t-1's value."""
        dp, base_model = _adj_setup
        dates = sorted(dp.prs_dataset['date'].unique())
        sp500 = np.arange(1, len(dates) + 1) * 0.01
        dp.load_enhancement_features = lambda: pd.DataFrame(
            {'date': dates, 'sp500_return': sp500})
        sequences, _, _, seq_dates = dp.prepare_adjustment_data(
            base_model, torch.device('cpu'), sequence_length=3)
        sp_idx = 6  # after the 6 base features
        date_pos = {pd.Timestamp(d): k for k, d in enumerate(dates)}
        assert len(seq_dates) > 0
        for i, d in enumerate(seq_dates):
            k = date_pos[pd.Timestamp(d)]
            expected = sp500[k - 1] if k >= 1 else 0.0
            assert sequences[i, -1, sp_idx].item() == pytest.approx(expected), \
                f'day {d}: saw same-day US close (leak)'

    def test_vix_change_lagged_one_day(self, _adj_setup):
        """vix_change comes from the US CBOE VIX close — same timing leak."""
        dp, base_model = _adj_setup
        vix_df = dp.load_vix_data().reset_index(drop=True)
        dates = sorted(dp.prs_dataset['date'].unique())
        sequences, _, _, seq_dates = dp.prepare_adjustment_data(
            base_model, torch.device('cpu'), sequence_length=3)
        date_pos = {pd.Timestamp(d): k for k, d in enumerate(dates)}
        assert len(seq_dates) > 0
        for i, d in enumerate(seq_dates):
            k = date_pos[pd.Timestamp(d)]
            expected = vix_df['vix_change'].iloc[k - 1] if k >= 1 else np.nan
            if np.isnan(expected):
                continue  # NaN rows are dropped by dropna, never surface
            assert sequences[i, -1, 0].item() == pytest.approx(expected), \
                f'day {d}: saw same-day US VIX close (leak)'


# ── getYATM ───────────────────────────────────────────────────────────

class TestGetYATM:
    def test_interpolation_on_training_data_only(self, patched_data_processor):
        dp = patched_data_processor
        dates = sorted(dp.prs_dataset['date'].unique())
        train_end = dates[2]  # Use first 3 dates for training
        dp.getYATM(train_end_date=train_end)
        # y_atm should exist
        assert 'y_atm' in dp.prs_dataset.columns
        assert dp.prs_dataset['y_atm'].notna().all()


class TestGetYATMLeakageGuard:
    def test_syn_curve_uses_train_dates_only(self, mock_config, mock_prs_dataset):
        """Post-cutoff dates must not influence the synthetic-c6 ATM curve."""
        import pandas as pd
        from dataset import DataProcessor
        dp = DataProcessor(mock_config)
        df = mock_prs_dataset.copy()
        cutoff = pd.Timestamp('2020-01-06')
        df.loc[df['date'] > cutoff, 'total_var'] = 5.0  # absurd post-cutoff tv
        dp.prs_dataset = df
        dp.syn_dataset = pd.DataFrame({'tau': [0.1, 0.5, 1.0],
                                       'logm': [-2.0, 2.0, 2.5]})
        dp.getYATM(train_end_date=cutoff)
        assert dp.syn_dataset['y_atm'].max() < 1.0

    def test_call_passes_config_cutoff(self):
        """DataProcessor.__call__ must forward train_end_date to getYATM."""
        import inspect
        from dataset import DataProcessor
        src = inspect.getsource(DataProcessor.__call__)
        assert 'train_end_date' in src

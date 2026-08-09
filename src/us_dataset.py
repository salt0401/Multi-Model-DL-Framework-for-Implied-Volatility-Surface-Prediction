"""Data layer for the Mag 7 US options branch.

Consumes the DoltHub chain snapshots downloaded by
scripts/download_us_options.py plus split-UNADJUSTED spots (as-traded strikes
require as-traded closes; adjusted closes would corrupt log-moneyness by the
split factor on pre-split dates).

Source quirks handled here:
- META traded as FB pre-2022-06: chains_FB.csv rows are relabeled META; on
  overlap dates META rows win.
- Snapshot dates include Saturdays (2019 era): spot joined via backward
  merge-asof (Sat -> Friday close).
- Snapshots are ~Mon/Wed/Fri: the forecast target is the NEXT SNAPSHOT, and
  the calendar gap to it (known in advance) is a condition feature.
- Provider-computed IVs are consumed directly (no Black-Scholes inversion);
  w = vol^2 * tau with tau = calendar days / 365.25.
"""
import json
import os

import numpy as np
import pandas as pd

MAG7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

TAU_GRID_DAYS = (15.0, 30.0, 45.0)
DAYS_PER_YEAR = 365.25


class UsOptionsProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    # ── Loading ───────────────────────────────────────────────────────

    def load_chains(self, tickers=None):
        tickers = tickers or MAG7
        frames = []
        for t in tickers:
            for fname, label in [(f'chains_{t}.csv', t)] + (
                    [('chains_FB.csv', 'META')] if t == 'META' else []):
                path = os.path.join(self.data_dir, fname)
                if not os.path.exists(path):
                    continue
                df = pd.read_csv(path)
                df['ticker'] = label
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f'no chain CSVs found in {self.data_dir}')
        df = pd.concat(frames, ignore_index=True)

        for col in ['strike', 'bid', 'ask', 'vol', 'delta', 'gamma', 'theta',
                    'vega', 'rho']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])

        # Resume-safety dedup + FB/META overlap resolution (META wins)
        df['_src_meta'] = (df['act_symbol'] == 'META').astype(int)
        df = (df.sort_values('_src_meta')
                .drop_duplicates(subset=['ticker', 'date', 'expiration',
                                         'strike', 'call_put'], keep='last')
                .drop(columns='_src_meta'))
        return df

    def load_spots(self):
        spots = pd.read_csv(os.path.join(self.data_dir, 'spots.csv'),
                            parse_dates=['date'])
        spots['close_unadj'] = pd.to_numeric(spots['close_unadj'],
                                             errors='coerce')
        return spots.dropna(subset=['close_unadj']).sort_values('date')

    def load_vix(self):
        path = os.path.join(self.data_dir, 'vix_us.csv')
        if not os.path.exists(path):
            return None
        vix = pd.read_csv(path, parse_dates=['date']).sort_values('date')
        vix['vix_change'] = vix['Close'].pct_change()
        return vix

    # ── Core table ────────────────────────────────────────────────────

    def build(self, tickers=None, max_abs_logm=0.6, min_tau_days=3.0):
        """Chains + parity forwards + filters + w + y_atm.

        Moneyness is measured against the PUT-CALL-PARITY-IMPLIED FORWARD
        per (ticker, date, expiration): F = median over strikes of
        (K + C_mid - P_mid). This is computed from the chains themselves and
        is therefore immune to stock-split adjustment: yfinance closes are
        split-adjusted even with auto_adjust=False, and joining them against
        as-traded strikes shifts logm by ln(split factor) pre-split — which
        silently deleted all pre-split history through the |logm| filter
        (measured: NVDA lost everything before its 2024 10:1 split).
        The shortest-expiration forward also serves as the as-traded spot
        proxy ('spot_syn') for strategy hedging; the (adjusted) yfinance
        close is kept ONLY for return features, where adjustment is correct.
        """
        chains = self.load_chains(tickers)
        spots = self.load_spots()

        merged = []
        for t, grp in chains.groupby('ticker'):
            sp = spots[spots['ticker'] == t][['date', 'close_unadj']]
            grp = grp.sort_values('date')
            if sp.empty:
                # No yfinance series for this symbol (e.g. SPY was not in the
                # spot fetch). The spot is only used for return features —
                # moneyness comes from the parity forward — so carry NaN
                # rather than silently dropping the whole symbol.
                grp = grp.assign(close_unadj=np.nan)
                merged.append(grp)
                continue
            merged.append(pd.merge_asof(grp, sp.sort_values('date'),
                                        on='date', direction='backward'))
        df = pd.concat(merged, ignore_index=True)
        df = df.rename(columns={'close_unadj': 'close_yf'})
        df = df.dropna(subset=['vol', 'bid', 'ask'])

        df['mid'] = (df['bid'] + df['ask']) / 2
        df['spread'] = df['ask'] - df['bid']
        df['tau'] = (df['expiration'] - df['date']).dt.days / DAYS_PER_YEAR

        # Parity forwards (before any filtering, both legs required)
        piv = df.pivot_table(index=['ticker', 'date', 'expiration', 'strike'],
                             columns='call_put', values='mid').reset_index()
        piv = piv.dropna(subset=['Call', 'Put'])
        piv['F'] = piv['strike'] + piv['Call'] - piv['Put']
        fwd = (piv.groupby(['ticker', 'date', 'expiration'])['F']
                  .median().rename('forward').reset_index())
        df = df.merge(fwd, on=['ticker', 'date', 'expiration'], how='left')
        df = df.dropna(subset=['forward'])
        df = df[df['forward'] > 0]

        # As-traded spot proxy: shortest-expiration forward per (ticker, date)
        near = (fwd.sort_values('expiration')
                   .groupby(['ticker', 'date']).first()['forward']
                   .rename('spot_syn').reset_index())
        df = df.merge(near, on=['ticker', 'date'], how='left')

        df['logm'] = np.log(df['strike'] / df['forward'])
        df['total_var'] = df['vol'] ** 2 * df['tau']

        df = df[(df['bid'] > 0)
                & (df['vol'] > 0.01) & (df['vol'] < 5.0)
                & (df['logm'].abs() <= max_abs_logm)
                & (df['tau'] >= min_tau_days / DAYS_PER_YEAR)]
        df = self._add_y_atm(df)
        return df.reset_index(drop=True)

    @staticmethod
    def _add_y_atm(df):
        """Per-(ticker, date) ATM total-variance term structure, interpolated
        at each row's tau (same construction as the TXO getYATM)."""
        from scipy.interpolate import interp1d
        chunks = []
        for (t, d), grp in df.groupby(['ticker', 'date']):
            atm = (grp.loc[grp.groupby('expiration')['logm']
                           .transform(lambda s: s.abs().rank(method='first')) == 1]
                   [['tau', 'total_var']].sort_values('tau'))
            if len(atm) >= 2:
                fn = interp1d(atm['tau'], atm['total_var'], kind='linear',
                              fill_value='extrapolate', bounds_error=False)
                vals = fn(grp['tau'].values)
            else:
                vals = np.full(len(grp), atm['total_var'].iloc[0])
            chunks.append(pd.Series(np.clip(vals, 1e-8, None), index=grp.index))
        df = df.copy()
        df['y_atm'] = pd.concat(chunks)
        return df

    # ── Model-facing views ────────────────────────────────────────────

    def prepare_hyperiv_surfaces(self, df):
        """Pooled per-(date, ticker) surfaces for the cross-asset HyperIV.

        Returns list of (date, ticker, (tau, logm, total_var, y_atm)) with
        float64 column tensors, sorted by date then ticker.
        """
        import torch
        surfaces = []
        cols = ['tau', 'logm', 'total_var', 'y_atm']
        if 'n_earnings' in df.columns:
            cols.append('n_earnings')        # drives the M4 event head
        for (d, t), grp in df.sort_values(['date', 'ticker', 'tau', 'logm']) \
                             .groupby(['date', 'ticker'], sort=True):
            if len(grp) < 20:
                continue
            tens = tuple(torch.from_numpy(grp[[c]].to_numpy(dtype='float64'))
                         for c in cols)
            surfaces.append((d, t, tens))
        return surfaces

    def prepare_surface_panel(self, df, ticker, train_end_date,
                              tau_grid_days=TAU_GRID_DAYS, n_logm=15):
        """Per-snapshot gridded surfaces + conditions for one ticker.

        Mirrors DataProcessor.Prepare_surface_panel: tau-major layout,
        logm grid from TRAIN-period quantiles, linear griddata + nearest fill.
        Conditions: [spot return since prev snapshot, realized vol of last 5
        snapshot returns, VIX level, VIX change since prev snapshot,
        ATM tv level (30d), term slope (45d-15d ATM), skew (w(-0.2)-w(+0.2)
        at 30d)] — gap_days is appended at pairing time (build_us_pairs).
        """
        from scipy.interpolate import griddata

        sub = df[df['ticker'] == ticker].sort_values(['date', 'tau', 'logm'])
        tau_grid = np.asarray(tau_grid_days, dtype=float) / DAYS_PER_YEAR

        train_sub = sub[sub['date'] <= train_end_date]
        logm_grid = np.linspace(train_sub['logm'].quantile(0.05),
                                train_sub['logm'].quantile(0.95), n_logm)
        tt, ll = np.meshgrid(tau_grid, logm_grid, indexing='ij')
        grid_points = np.column_stack([tt.ravel(), ll.ravel()])

        daily = {}
        spot_by_date = {}
        for d, grp in sub.groupby('date'):
            if len(grp) < 10 or grp['tau'].nunique() < 2:
                continue
            try:
                vals = griddata(grp[['tau', 'logm']].values,
                                grp['total_var'].values, grid_points,
                                method='linear')
                if np.any(np.isnan(vals)):
                    near = griddata(grp[['tau', 'logm']].values,
                                    grp['total_var'].values, grid_points,
                                    method='nearest')
                    vals = np.where(np.isnan(vals), near, vals)
                daily[d] = np.clip(vals.astype('float64'), 1e-8, None)
                # Split-ADJUSTED close is the right series for return features.
                # Symbols with no yfinance spot (SPY) fall back to the parity
                # forward, which is safe for them because they have not split;
                # using it for a name that HAS split would inject a spurious
                # -75%-style return on the split date.
                s = grp['close_yf'].iloc[0]
                spot_by_date[d] = (s if np.isfinite(s)
                                   else grp['spot_syn'].iloc[0])
            except Exception:
                continue

        dates = sorted(daily.keys())
        S = np.stack([daily[d] for d in dates])
        n_tau, n_lg = len(tau_grid), len(logm_grid)
        W = S.reshape(-1, n_tau, n_lg)

        spot = np.array([spot_by_date[d] for d in dates])
        ret = np.zeros(len(dates))
        ret[1:] = np.diff(spot) / spot[:-1]
        rv5 = pd.Series(ret).rolling(5).std().fillna(0.02).values

        vix = self.load_vix()
        if vix is not None:
            vd = pd.merge_asof(pd.DataFrame({'date': dates}),
                               vix[['date', 'Close']], on='date',
                               direction='backward')
            vix_level = vd['Close'].ffill().fillna(20.0).values
            vix_chg = np.zeros(len(dates))
            vix_chg[1:] = np.diff(vix_level) / np.clip(vix_level[:-1], 1e-9, None)
        else:
            vix_level = np.full(len(dates), 20.0)
            vix_chg = np.zeros(len(dates))

        i30 = int(np.argmin(np.abs(np.asarray(tau_grid_days) - 30)))
        i15, i45 = 0, len(tau_grid_days) - 1
        atm_col = int(np.argmin(np.abs(logm_grid)))
        lo_col = int(np.argmin(np.abs(logm_grid + 0.2)))
        hi_col = int(np.argmin(np.abs(logm_grid - 0.2)))
        atm30 = W[:, i30, atm_col]
        term_slope = W[:, i45, atm_col] - W[:, i15, atm_col]
        skew = W[:, i30, lo_col] - W[:, i30, hi_col]

        conditions = np.column_stack([ret, rv5, vix_level, vix_chg,
                                      atm30, term_slope, skew])
        cond_names = ['ret', 'rv5', 'vix_level', 'vix_change',
                      'atm30_tv', 'term_slope', 'skew']

        return {
            'ticker': ticker,
            'dates': dates,
            'surfaces': S,
            'conditions': conditions,
            'tau_grid': tau_grid,
            'logm_grid': logm_grid,
            'cond_names': cond_names,
        }


def build_us_pairs(panel, train_end_date, test_start_date, val_frac=0.15):
    """flow_surface.build_dataset + gap_days (known ahead: the snapshot
    calendar is deterministic at decision time) appended to conditions."""
    from flow_surface import build_dataset
    splits = build_dataset(panel, train_end_date, test_start_date,
                           val_frac=val_frac)
    date_index = {d: i for i, d in enumerate(panel['dates'])}
    for sp in splits.values():
        gaps = []
        for d in sp['dates']:
            i = date_index[d]
            gaps.append((panel['dates'][i + 1] - panel['dates'][i]).days)
        sp['C'] = np.column_stack([sp['C'], np.asarray(gaps, dtype=float)])
    return splits

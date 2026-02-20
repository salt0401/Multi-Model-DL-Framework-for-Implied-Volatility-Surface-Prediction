import numpy as np
import pandas as pd
import pickle as pkl
import os
from ast import literal_eval

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
from scipy.interpolate import interp1d

from torch import from_numpy
from torch.utils.data import TensorDataset, DataLoader


class DataProcessor():
    def __init__(self, config):
        """Config-driven initialization.

        Args:
            config: configparser.ConfigParser object with [data] section.
        """
        folder = config['data']['data_folder']
        file_base = config['data']['data_file']
        asset = config['data']['asset_file']
        prs_dataset = config['data']['prs_dataset']

        self.folder = folder
        file = folder + file_base
        self.preprocessed = f'{folder}{prs_dataset}.csv'
        self.syn_c6 = f'{folder}syn_data_c6.csv'

        self.pkl = f'{file}.pkl'
        self.raw = f'{file}.csv'
        self.columnHandled = f'{file}_columnHandled.csv'
        self.filtered = f'{file}_filtered.csv'

        self.intDivConcated = f'{file}_intDivConcatnated.csv'
        self.betaTau = f'{file}_betaTau.csv'

        self.impliVol = f'{file}_impliVol.csv'

        self.asset = folder + asset

        self.config = config

    def __call__(self):
        self.prs_dataset = self.preprocess()
        self.syn_dataset = self.synthesize()
        self.getYATM()

    def preprocess(self):
        print('Preprocessing dataset...')
        if os.path.exists(self.preprocessed):
            prs_dataset = pd.read_csv(self.preprocessed, parse_dates=['date', 'exdate'], index_col=0)
            # Handle legacy column name: CSV may have 'S' instead of 'underlying'
            if 'S' in prs_dataset.columns and 'underlying' not in prs_dataset.columns:
                prs_dataset = prs_dataset.rename(columns={'S': 'underlying'})
            return prs_dataset

        print('\tPreprocessed dataset not exist, preprocessing...')
        df = self.pkl_to_csv()
        df = self.columnHandling(df)
        df = self.filter(df)
        df = self.int_and_div(df)
        prs_dataset = self.impli_vol(df)

        return prs_dataset

    def pkl_to_csv(self):
        print('\tConverting pickle file to csv...')
        if os.path.exists(self.raw):
            print('\t\tconverted csv file already exist.')
            df = pd.read_csv(self.raw, dtype='object')
            return df

        print('\t\tNo csv file, converting pickle...')
        with open(self.pkl, "rb") as f:
            obj = pkl.load(f)

            df = pd.DataFrame(obj)
            print(df.head())
            df.to_csv(self.raw)
        return df

    def columnHandling(self, df):
        print('\tHandling columns...')
        if os.path.exists(self.columnHandled):
            print('\t\tcolumn-filtered dataset already exist')
            filtered_dataset = pd.read_csv(self.columnHandled, dtype='object', parse_dates=['date', 'exdate'])
            return filtered_dataset

        print('\t\tNo column-filtered dataset exist')
        print('\t\tfiltering columns...')
        df = df[['交易日期','履約價','買賣權','成交量','結算價','到期日期','time to maturity']]
        df.columns = ['date', 'strike_price', 'put/call', 'volume', 'option_price', 'exdate', 'tau']
        df['date'] = pd.to_datetime(df['date'])
        df['exdate'] = pd.to_datetime(df['exdate'])

        df['put/call'] = df['put/call'].replace({'買權': 'call', '賣權': 'put'})
        df['tau'] = df['tau'].astype(float)/252
        df = df[df['option_price'] != '-']
        df = df[df['tau'] > 0]

        print('\t\tconcating underlying asset...')
        asset = pd.read_csv(self.asset, index_col=None, parse_dates=['date'])
        df = pd.merge(df, asset[['date', 'Adj Close']], on='date', how='left')
        df = df.rename(columns={'Adj Close': 'underlying'})
        df = df[df['underlying'].notnull()]

        print(df.head())
        df.to_csv(self.columnHandled)
        return df

    def filter(self, df):
        print('\tSpliting data into put and call...')
        if os.path.exists(self.filtered):
            print('\t\tput/call splited data already exist')
            filtered_dataset = pd.read_csv(self.filtered, dtype='object', parse_dates=['date', 'exdate'], index_col=0)
            return filtered_dataset

        print('\t\tNo put/call-filtered dataset exist')
        call = df.iloc[::2].reset_index(drop=True)
        put = df.iloc[1::2].reset_index(drop=True)
        print(call.head())
        print(put.head())

        put_call_filter = call['volume'].astype(int) > put['volume'].astype(int)
        filtered_dataset = call.where(put_call_filter, put).drop('Unnamed: 0', axis=1)
        print(filtered_dataset.head())
        filtered_dataset['parity'] = call['option_price'].astype(float) - put['option_price'].astype(float)
        print(filtered_dataset.head())
        filtered_dataset.to_csv(self.filtered)

        return filtered_dataset

    def fit_beta_tau(self, daily_timely_data):
        X = daily_timely_data[['underlying', 'strike_price']]
        y = daily_timely_data['parity']
        model = LinearRegression(fit_intercept=False, n_jobs=-1)
        model.fit(X, y)
        return model.coef_

    def int_and_div(self, df):
        print('\tConcating dividend and interest rates...')
        if os.path.exists(self.intDivConcated):
            print('\t\tint and div already concatnated')
            intDiv = pd.read_csv(self.intDivConcated, parse_dates=['date', 'exdate'], index_col=None)
            return intDiv

        df['strike_price'] = df['strike_price'].astype(float)
        df['underlying'] = df['underlying'].astype(float)
        df['tau'] = df['tau'].astype(float)
        df['year'] = df['date'].dt.year
        df['season'] = (df['date'].dt.month - 1) // 3 + 1
        near_atm = df[(df['strike_price'] >= (1-0.075)*df['underlying']) & (df['strike_price'] <= (1+0.075)*df['underlying'])]

        if os.path.exists(self.betaTau):
            print('\t\tFitted beta at diffenent taus already exist')
            beta_tau = pd.read_csv(self.betaTau, dtype='object', parse_dates=['date'], index_col=None)
            beta_tau['betas'] = beta_tau['betas'].apply(literal_eval)
            beta_tau['tau'] = beta_tau['tau'].astype(float)
        else:
            print("\t\tBeta at different taus hasn't fitted, fitting...")
            beta_tau = near_atm.groupby(['year', 'season', 'tau']).apply(self.fit_beta_tau)
            beta_tau.to_csv(self.betaTau)
            # Bug D1 fixed: dict not set
            beta_tau = beta_tau.rename(columns={'0': 'betas'})

        beta_tau[['beta_S', 'beta_K']] = beta_tau['betas'].apply(pd.Series)

        # Bug D3 fixed: merge on ['year', 'season', 'tau'] to match groupby keys; add year/season to df
        intDiv = pd.merge(df, beta_tau, on=['year', 'season', 'tau'], how='left')
        intDiv['underlying_beta'] = intDiv['underlying'] * intDiv['beta_S']
        intDiv['strike_price_beta'] = intDiv['strike_price'] * intDiv['beta_K']
        intDiv['logm'] = np.log(intDiv['strike_price_beta']/intDiv['underlying_beta'])
        # Bug D2 fixed: 'season' not 'month'
        intDiv.drop(columns=['year', 'season', 'parity', 'betas', 'beta_S', 'beta_K'], inplace=True)

        intDiv.to_csv(self.intDivConcated)

        return intDiv

    def impli_vol(self, df):
        print('\tConcatnating total variance when atm...')
        if os.path.exists(self.preprocessed):
            print('\t\tTotal variance already concatnated')
            yATM = pd.read_csv(self.preprocessed, parse_dates=['date', 'exdate'], index_col=0)
            return yATM

        print('\t\ttotal variance not concatnated, calculating implied volatility...')
        def implied_vol(row):
            def BS(S_beta, K_beta, logm, tau, implied_vol, indic='call'):
                d1 = -logm/np.sqrt(tau)/implied_vol + np.sqrt(tau)*implied_vol/2
                d2 = -logm/np.sqrt(tau)/implied_vol - np.sqrt(tau)*implied_vol/2
                if indic == 'call':
                    return S_beta*norm.cdf(d1)-K_beta*norm.cdf(d2)
                elif indic == 'put':
                    return K_beta*norm.cdf(-d2)-S_beta*norm.cdf(-d1)

            def error(sigma):
                return (BS(row['underlying_beta'], row['strike_price_beta'], row['logm'], row['tau'], sigma, row['put/call']) - row['option_price'])**2

            return minimize(error, x0=[0.2], bounds=[(0.001, 1)])['x'][0]

        df['implied_vol'] = df.apply(implied_vol, axis=1)
        df.to_csv(self.impliVol)
        df['total_var'] = np.square(df['implied_vol'])*df['tau']

        return df

    def synthesize(self):
        print('Making Synthetic data for constraint c6...')
        if os.path.exists(self.syn_c6):
            print('\tSynthetic data already exist')
            syn_c6 = pd.read_csv(self.syn_c6, index_col=0)
            return syn_c6

        print('\tNo synthetic data, generating...')
        taus = self.prs_dataset['tau'].unique()
        min_logm = self.prs_dataset['logm'].min()
        max_logm = self.prs_dataset['logm'].max()

        logm_seq = [min_logm * 6, min_logm * 5, min_logm * 4, max_logm * 4, max_logm * 5, max_logm * 6]

        syn_c6 = pd.DataFrame([(tau, logm) for tau in taus for logm in logm_seq], columns=['tau', 'logm'])
        print(syn_c6.head())
        # Bug D5-related fix: use self.syn_c6 path instead of hardcoded
        syn_c6.to_csv(self.syn_c6)

        return syn_c6

    def getYATM(self, train_end_date=None):
        """Compute per-date ATM total variance interpolation.

        For each date, finds the ATM option per tau (strike closest to
        underlying) and builds a per-date interpolation curve tau -> total_var.
        Each row gets y_atm from its own date's ATM term structure, preserving
        volatility regime information (high y_atm during crashes, low in calm).

        For synthetic data (no date column), uses the mean ATM curve across
        training dates.  If train_end_date is provided, the synthetic curve
        averages only over training dates to avoid data leakage.
        """
        print('Computing per-date ATM total variance')

        # Step 1: Find ATM row for each (date, tau)
        atm_idxs = self.prs_dataset.groupby(['date', 'tau']).apply(
            lambda x: (x['underlying'] - x['strike_price']).abs().idxmin()
        )
        atm_rows = self.prs_dataset.loc[atm_idxs][['date', 'tau', 'total_var']]
        print(f'  {len(atm_rows)} ATM points across {atm_rows["date"].nunique()} dates')

        # Step 2: Build per-date interpolation functions
        interp_dict = {}
        for date, grp in atm_rows.groupby('date'):
            grp_sorted = grp.sort_values('tau')
            taus = grp_sorted['tau'].values
            tvs = grp_sorted['total_var'].values
            if len(taus) >= 2:
                interp_dict[date] = interp1d(
                    taus, tvs, kind='linear',
                    fill_value='extrapolate', bounds_error=False
                )
            else:
                val = tvs[0]
                interp_dict[date] = lambda tau, v=val: np.full_like(
                    np.asarray(tau, dtype=np.float64), v
                )

        # Step 3: Assign y_atm per row using its own date's ATM curve
        chunks = []
        for date, group in self.prs_dataset.groupby('date'):
            fn = interp_dict[date]
            vals = fn(group['tau'].values)
            chunks.append(pd.Series(vals, index=group.index))
        self.prs_dataset['y_atm'] = pd.concat(chunks)
        self.prs_dataset['y_atm'] = self.prs_dataset['y_atm'].clip(lower=1e-8)

        # Step 4: Mean ATM curve for synthetic data
        if train_end_date is not None:
            train_atm = atm_rows[atm_rows['date'] <= train_end_date]
        else:
            train_atm = atm_rows
        mean_atm = train_atm.groupby('tau')['total_var'].mean().reset_index()
        mean_atm = mean_atm.sort_values('tau')
        syn_fn = interp1d(
            mean_atm['tau'].values, mean_atm['total_var'].values,
            kind='linear', fill_value='extrapolate', bounds_error=False
        )
        self.syn_dataset['y_atm'] = syn_fn(self.syn_dataset['tau'].values)
        self.syn_dataset['y_atm'] = self.syn_dataset['y_atm'].clip(lower=1e-8)

        print(f'  y_atm range: [{self.prs_dataset["y_atm"].min():.6f}, '
              f'{self.prs_dataset["y_atm"].max():.6f}]')

    def Prepare_train_data(self, train_start_date, train_end_date, batch_size=256):
        """Return (train_loader, val_loader, c6_loader) using proper TensorDataset/DataLoader.

        Bug T1/D5 fixed: renamed from Prepare_prs_dataset.
        Bug T3 fixed: returns DataLoaders instead of raw tensors.
        Chronological split: last 20% of dates used for validation to prevent
        temporal data leakage (financial time series must not shuffle future into past).
        """
        print('Preparing train data')
        train_dataset = self.prs_dataset[
            (self.prs_dataset['date'] >= train_start_date) &
            (self.prs_dataset['date'] <= train_end_date)
        ].sort_values('date')

        # Chronological split: use last 20% of dates as validation
        unique_dates = train_dataset['date'].unique()
        unique_dates = np.sort(unique_dates)
        split_idx = int(len(unique_dates) * 0.8)
        val_start_date = unique_dates[split_idx]

        train_part = train_dataset[train_dataset['date'] < val_start_date]
        val_part = train_dataset[train_dataset['date'] >= val_start_date]

        tau_train = from_numpy(train_part[['tau']].to_numpy(dtype='float64'))
        logm_train = from_numpy(train_part[['logm']].to_numpy(dtype='float64'))
        y_train = from_numpy(train_part[['total_var']].to_numpy(dtype='float64'))
        yATM_train = from_numpy(train_part[['y_atm']].to_numpy(dtype='float64'))

        tau_val = from_numpy(val_part[['tau']].to_numpy(dtype='float64'))
        logm_val = from_numpy(val_part[['logm']].to_numpy(dtype='float64'))
        y_val = from_numpy(val_part[['total_var']].to_numpy(dtype='float64'))
        yATM_val = from_numpy(val_part[['y_atm']].to_numpy(dtype='float64'))

        train_ds = TensorDataset(tau_train, logm_train, y_train, yATM_train)
        val_ds = TensorDataset(tau_val, logm_val, y_val, yATM_val)

        train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

        # Synthetic c6 data
        tau_c6 = from_numpy(self.syn_dataset[['tau']].to_numpy(dtype='float64'))
        logm_c6 = from_numpy(self.syn_dataset[['logm']].to_numpy(dtype='float64'))
        yATM_c6 = from_numpy(self.syn_dataset[['y_atm']].to_numpy(dtype='float64'))

        c6_ds = TensorDataset(tau_c6, logm_c6, yATM_c6)
        c6_loader = DataLoader(dataset=c6_ds, batch_size=batch_size, shuffle=True)

        return train_loader, val_loader, c6_loader

    def Prepare_test_data(self, test_start_date, test_end_date):
        """Return (test_loader, test_df)."""
        print('Preparing test data')
        test_dataset = self.prs_dataset[
            (self.prs_dataset['date'] >= test_start_date) &
            (self.prs_dataset['date'] <= test_end_date)
        ]
        tau = from_numpy(test_dataset[['tau']].to_numpy(dtype='float64'))
        logm = from_numpy(test_dataset[['logm']].to_numpy(dtype='float64'))
        y = from_numpy(test_dataset[['total_var']].to_numpy(dtype='float64'))
        y_atm = from_numpy(test_dataset[['y_atm']].to_numpy(dtype='float64'))

        test_ds = TensorDataset(tau, logm, y, y_atm)
        test_loader = DataLoader(dataset=test_ds, batch_size=256, shuffle=False)

        return test_loader, test_dataset

    def load_vix_data(self):
        """Load VIX data for overnight changes (used by adjustment model)."""
        vix_file = self.config['data'].get('vix_file', 'VIX.csv')
        vix_path = os.path.join(self.folder, vix_file)
        if not os.path.exists(vix_path):
            print(f'VIX file not found at {vix_path}')
            return None
        vix = pd.read_csv(vix_path, parse_dates=['date'])
        vix = vix.sort_values('date')
        vix['vix_change'] = vix['Close'].pct_change()
        return vix

    def load_enhancement_features(self):
        """Load daily enhancement features if available.

        Returns DataFrame with date-indexed features from
        dataset/enhancement/daily_features.csv, or None if not found.

        Features include: rv_5d..rv_60d, sp500_return, us10y_yield, usdtwd,
        atm_iv, iv_term_slope, iv_skew,
        inst_net_ratio, put_call_oi_ratio, futures_basis_pct, vrp_20d, etc.
        """
        enhance_path = os.path.join(self.folder, 'enhancement', 'daily_features.csv')
        if not os.path.exists(enhance_path):
            return None
        df = pd.read_csv(enhance_path, parse_dates=['date'])
        df = df.sort_values('date')
        print(f'  Loaded enhancement features: {len(df)} rows, {len(df.columns)-1} features')
        return df

    def prepare_adjustment_data(self, base_model, device, sequence_length=20):
        """Prepare time-series sequences for the adjustment model.

        Returns padded sequences with features:
        [vix_change, underlying_return, logm, tau, tv_pred, itm_otm]
        """
        import torch

        vix = self.load_vix_data()
        if vix is None:
            return None, None, None

        df = self.prs_dataset.copy()
        df = df.sort_values(['date', 'tau', 'logm'])

        # Compute underlying returns
        daily_underlying = df.groupby('date')['underlying'].first().reset_index()
        daily_underlying = daily_underlying.sort_values('date')
        daily_underlying['underlying_return'] = daily_underlying['underlying'].pct_change()

        df = pd.merge(df, daily_underlying[['date', 'underlying_return']], on='date', how='left')
        df = pd.merge(df, vix[['date', 'vix_change']], on='date', how='left')

        # Merge enhancement features if available
        enhance = self.load_enhancement_features()
        if enhance is not None:
            enhance_cols = ['sp500_return', 'iv_term_slope',
                            'iv_skew', 'vrp_20d', 'futures_basis_pct', 'rv_20d']
            avail_cols = [c for c in enhance_cols if c in enhance.columns]
            if avail_cols:
                df = pd.merge(df, enhance[['date'] + avail_cols], on='date', how='left')
                for c in avail_cols:
                    df[c] = df[c].fillna(0.0)

        # Compute model predictions in chunks to avoid GPU OOM.
        # SmileModel.forward uses autograd.grad(create_graph=True) for second-order
        # derivatives, which stores 3 layers of computation graphs simultaneously.
        # Processing all rows at once exhausts GPU memory; chunking fixes this.
        base_model.eval()
        chunk_size = 5000
        tau_np = df[['tau']].to_numpy(dtype='float64')
        logm_np = df[['logm']].to_numpy(dtype='float64')
        yatm_np = df[['y_atm']].to_numpy(dtype='float64')
        n = len(df)
        tv_pred_list = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            tau_t = from_numpy(tau_np[start:end]).to(device)
            logm_t = from_numpy(logm_np[start:end]).to(device)
            yatm_t = from_numpy(yatm_np[start:end]).to(device)
            pred, _, _, _ = base_model(tau_t, logm_t, yatm_t)
            tv_pred_list.append(pred.detach().cpu().numpy().flatten())
        df['tv_pred'] = np.concatenate(tv_pred_list)

        df['itm_otm'] = (df['logm'] > 0).astype(float)
        df['tv_ratio'] = df['total_var'] / (df['tv_pred'] + 1e-8)

        # Build sequences per option (grouped by strike + expiry)
        # Base features + enhancement features if available
        features_cols = ['vix_change', 'underlying_return', 'logm', 'tau', 'tv_pred', 'itm_otm']
        if enhance is not None:
            avail_cols = [c for c in ['sp500_return', 'iv_term_slope',
                                       'iv_skew', 'vrp_20d', 'futures_basis_pct', 'rv_20d']
                          if c in df.columns]
            features_cols = features_cols + avail_cols
        target_col = 'tv_ratio'

        df = df.dropna(subset=features_cols + [target_col])

        sequences = []
        targets = []
        masks = []
        seq_dates = []

        for _, group in df.groupby(['strike_price', 'exdate']):
            group = group.sort_values('date')
            feats = group[features_cols].values
            tgts = group[target_col].values
            dates = group['date'].values

            for i in range(len(group)):
                start = max(0, i - sequence_length + 1)
                seq = feats[start:i+1]
                pad_len = sequence_length - len(seq)
                mask = np.concatenate([np.zeros(pad_len), np.ones(len(seq))])
                if pad_len > 0:
                    seq = np.vstack([np.zeros((pad_len, len(features_cols))), seq])
                sequences.append(seq)
                targets.append(tgts[i])
                masks.append(mask)
                seq_dates.append(dates[i])

        sequences = np.array(sequences, dtype='float64')
        targets = np.array(targets, dtype='float64').reshape(-1, 1)
        masks = np.array(masks, dtype='float64')
        seq_dates = np.array(seq_dates)

        return (
            torch.from_numpy(sequences),
            torch.from_numpy(targets),
            torch.from_numpy(masks),
            seq_dates,
        )

    def Prepare_hyperiv_data(self):
        """Group options by date for HyperIV training.

        Returns:
            List of (date, (tau, logm, total_var, y_atm)) tuples,
            sorted by date. Each element is a per-day tensor set.
        """
        print('Preparing HyperIV data...')
        import torch

        df = self.prs_dataset.copy()
        df = df.sort_values(['date', 'tau', 'logm'])
        df = df.dropna(subset=['tau', 'logm', 'total_var', 'y_atm'])

        surfaces = []
        for date, group in df.groupby('date'):
            tau = torch.from_numpy(group[['tau']].to_numpy(dtype='float64'))
            logm = torch.from_numpy(group[['logm']].to_numpy(dtype='float64'))
            tv = torch.from_numpy(group[['total_var']].to_numpy(dtype='float64'))
            yatm = torch.from_numpy(group[['y_atm']].to_numpy(dtype='float64'))
            surfaces.append((date, (tau, logm, tv, yatm)))

        surfaces.sort(key=lambda x: x[0])
        print(f'  {len(surfaces)} daily surfaces prepared')
        return surfaces

    def Prepare_diffusion_data(self, train_end_date, test_start_date,
                                n_tau_grid=10, n_logm_grid=20, batch_size=16):
        """Prepare gridded IV surface pairs for diffusion model.

        Interpolates each day's options to a fixed (tau, logm) grid.
        Returns consecutive-day pairs: (surface_today, surface_tomorrow, condition_features).

        Condition features: [VIX, VIX_change, underlying_return, realized_vol_5d]
        """
        import torch
        from scipy.interpolate import griddata

        print('Preparing diffusion data...')

        df = self.prs_dataset.copy()
        df = df.sort_values(['date', 'tau', 'logm'])
        df = df.dropna(subset=['tau', 'logm', 'total_var'])

        # Build fixed grid
        tau_range = np.linspace(df['tau'].quantile(0.05), df['tau'].quantile(0.95), n_tau_grid)
        logm_range = np.linspace(df['logm'].quantile(0.05), df['logm'].quantile(0.95), n_logm_grid)
        tau_grid, logm_grid = np.meshgrid(tau_range, logm_range)
        grid_points = np.column_stack([tau_grid.ravel(), logm_grid.ravel()])
        surface_dim = n_tau_grid * n_logm_grid

        # Compute daily gridded surfaces
        daily_surfaces = {}
        for date, group in df.groupby('date'):
            points = group[['tau', 'logm']].values
            values = group['total_var'].values
            if len(points) < 5:
                continue
            try:
                grid_values = griddata(points, values, grid_points, method='linear')
                # Fill NaNs with nearest neighbor
                if np.any(np.isnan(grid_values)):
                    nearest = griddata(points, values, grid_points, method='nearest')
                    grid_values = np.where(np.isnan(grid_values), nearest, grid_values)
                daily_surfaces[date] = grid_values.astype('float64')
            except Exception:
                continue

        # Load VIX and underlying for condition features
        vix = self.load_vix_data()
        daily_underlying = df.groupby('date')['underlying'].first().reset_index()
        daily_underlying = daily_underlying.sort_values('date')
        daily_underlying['underlying_return'] = daily_underlying['underlying'].pct_change()
        daily_underlying['realized_vol_5d'] = daily_underlying['underlying_return'].rolling(5).std()

        # Build date-indexed features
        feature_dict = {}
        if vix is not None:
            vix_dict = dict(zip(vix['date'], zip(vix['Close'], vix['vix_change'])))
        else:
            vix_dict = {}
        und_dict = dict(zip(daily_underlying['date'],
                             zip(daily_underlying['underlying_return'],
                                 daily_underlying['realized_vol_5d'])))

        # Load enhancement features if available
        enhance = self.load_enhancement_features()
        if enhance is not None:
            enhance_dict = {}
            enhance_cols = [c for c in enhance.columns if c != 'date']
            for _, row in enhance.iterrows():
                enhance_dict[row['date']] = {c: row[c] for c in enhance_cols}
        else:
            enhance_dict = {}

        sorted_dates = sorted(daily_surfaces.keys())

        # Build consecutive pairs
        surfaces_today = []
        surfaces_tomorrow = []
        cond_features = []

        for i in range(len(sorted_dates) - 1):
            d_today = sorted_dates[i]
            d_tomorrow = sorted_dates[i + 1]

            s_today = daily_surfaces[d_today]
            s_tomorrow = daily_surfaces[d_tomorrow]

            # Condition features for today
            vix_val, vix_change = vix_dict.get(d_today, (0.2, 0.0))
            und_ret, rvol = und_dict.get(d_today, (0.0, 0.01))
            if np.isnan(vix_val) or vix_val is None:
                vix_val = 0.2
            if np.isnan(vix_change) or vix_change is None:
                vix_change = 0.0
            if np.isnan(und_ret) or und_ret is None:
                und_ret = 0.0
            if np.isnan(rvol) or rvol is None:
                rvol = 0.01

            # Base 4 features (always present for backward compatibility)
            feats = [float(vix_val), float(vix_change), float(und_ret), float(rvol)]

            # Append enhancement features if available
            if enhance_dict:
                enh = enhance_dict.get(d_today, {})
                for col in ['sp500_return', 'iv_term_slope', 'iv_skew',
                            'vrp_20d', 'futures_basis_pct', 'rv_20d',
                            'inst_net_ratio']:
                    val = enh.get(col, 0.0)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0.0
                    feats.append(float(val))

            surfaces_today.append(s_today)
            surfaces_tomorrow.append(s_tomorrow)
            cond_features.append(feats)

        surfaces_today = np.array(surfaces_today, dtype='float64')
        surfaces_tomorrow = np.array(surfaces_tomorrow, dtype='float64')
        cond_features = np.array(cond_features, dtype='float64')

        print(f'  {len(surfaces_today)} consecutive-day pairs on {surface_dim}-dim grid')

        # Split by date
        dates_arr = sorted_dates[:-1]  # dates for "today"
        train_mask = np.array([d <= train_end_date for d in dates_arr])
        test_mask = np.array([d >= test_start_date for d in dates_arr])
        val_mask = ~train_mask & ~test_mask

        def make_loader(mask, shuffle):
            if not np.any(mask):
                return None
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(surfaces_today[mask]),
                torch.from_numpy(surfaces_tomorrow[mask]),
                torch.from_numpy(cond_features[mask]),
            )
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        # If no explicit val dates, split train chronologically
        if not np.any(val_mask):
            train_indices = np.where(train_mask)[0]
            split = int(len(train_indices) * 0.85)
            actual_train = np.zeros(len(dates_arr), dtype=bool)
            actual_val = np.zeros(len(dates_arr), dtype=bool)
            actual_train[train_indices[:split]] = True
            actual_val[train_indices[split:]] = True
            train_loader = make_loader(actual_train, shuffle=True)
            val_loader = make_loader(actual_val, shuffle=False)
        else:
            train_loader = make_loader(train_mask, shuffle=True)
            val_loader = make_loader(val_mask, shuffle=False)

        test_loader = make_loader(test_mask, shuffle=False)

        return train_loader, val_loader, test_loader

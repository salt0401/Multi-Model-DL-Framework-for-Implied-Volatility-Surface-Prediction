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

    def getYATM(self):
        print('concating total variance when atm')
        # Bug D4 fixed: use 'underlying' not 'S'
        atm_idxs = self.prs_dataset.groupby(['tau']).apply(lambda x: (x['underlying'] - x['strike_price']).abs().idxmin())
        atm_rows = self.prs_dataset.loc[atm_idxs]

        atm_fun = interp1d(atm_rows['tau'], atm_rows['total_var'], kind='linear', fill_value="extrapolate", bounds_error=False)
        self.prs_dataset['y_atm'] = atm_fun(self.prs_dataset['tau'])
        self.syn_dataset['y_atm'] = atm_fun(self.syn_dataset['tau'])

    def Prepare_train_data(self, train_start_date, train_end_date, batch_size=256):
        """Return (train_loader, val_loader, c6_loader) using proper TensorDataset/DataLoader.

        Bug T1/D5 fixed: renamed from Prepare_prs_dataset.
        Bug T3 fixed: returns DataLoaders instead of raw tensors.
        """
        print('Preparing train data')
        train_dataset = self.prs_dataset[
            (self.prs_dataset['date'] >= train_start_date) &
            (self.prs_dataset['date'] <= train_end_date)
        ]
        tau = from_numpy(train_dataset[['tau']].to_numpy(dtype='float64'))
        logm = from_numpy(train_dataset[['logm']].to_numpy(dtype='float64'))
        y = from_numpy(train_dataset[['total_var']].to_numpy(dtype='float64'))
        y_atm = from_numpy(train_dataset[['y_atm']].to_numpy(dtype='float64'))

        tau_train, tau_val, logm_train, logm_val, y_train, y_val, yATM_train, yATM_val = \
            train_test_split(tau, logm, y, y_atm, test_size=0.2, shuffle=True)

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

        # Compute model predictions
        base_model.eval()
        with torch.no_grad():
            tau_t = from_numpy(df[['tau']].to_numpy(dtype='float64')).to(device)
            logm_t = from_numpy(df[['logm']].to_numpy(dtype='float64')).to(device)
            yatm_t = from_numpy(df[['y_atm']].to_numpy(dtype='float64')).to(device)
            tv_pred, _, _, _ = base_model(tau_t, logm_t, yatm_t)
            df['tv_pred'] = tv_pred.cpu().numpy().flatten()

        df['itm_otm'] = (df['logm'] > 0).astype(float)
        df['tv_ratio'] = df['total_var'] / (df['tv_pred'] + 1e-8)

        # Build sequences per option (grouped by strike + expiry)
        features_cols = ['vix_change', 'underlying_return', 'logm', 'tau', 'tv_pred', 'itm_otm']
        target_col = 'tv_ratio'

        df = df.dropna(subset=features_cols + [target_col])

        sequences = []
        targets = []
        masks = []

        for _, group in df.groupby(['strike_price', 'exdate']):
            group = group.sort_values('date')
            feats = group[features_cols].values
            tgts = group[target_col].values

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

        sequences = np.array(sequences, dtype='float64')
        targets = np.array(targets, dtype='float64').reshape(-1, 1)
        masks = np.array(masks, dtype='float64')

        return (
            torch.from_numpy(sequences),
            torch.from_numpy(targets),
            torch.from_numpy(masks),
        )

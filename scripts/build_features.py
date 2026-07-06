"""
Build enhancement features for IV surface prediction.

Produces a single daily features CSV with columns that can be merged into
the DDPM and Adjustment model pipelines.

Features:
    External (downloaded):
        - sp500_return: S&P 500 close-to-close return, dated at the US close.
          The US close on date t is realized ~15h AFTER Taiwan's close on the
          same calendar date, so same-day-target consumers must lag it one
          Taiwan trading day (see dataset.py prepare_adjustment_data);
          next-day-target consumers (Model 5 conditioning) use it as-is.
        - us10y_yield: US 10-Year Treasury yield (risk-free rate proxy)
        - usdtwd: USD/TWD exchange rate
        - futures_basis: TAIEX futures close - spot close (basis points)
        - futures_volume: front-month TAIEX futures volume

    Computed from TWII:
        - rv_5d, rv_10d, rv_20d, rv_60d: realized volatility at multiple horizons
        - vrp_20d: variance risk premium (ATM IV^2 - RV_20d^2)

    Computed from preprocessed dataset:
        - iv_term_slope: ATM IV term structure slope (long - short)
        - iv_skew: IV skew (OTM put IV - ATM IV)
        - atm_iv: at-the-money implied volatility

    Aggregated from enhancement data:
        - foreign_net_buy: foreign institutional net options volume
        - dealer_net_buy: dealer net options volume
        - put_call_oi_ratio: put/call open interest ratio

Usage:
    python scripts/build_features.py

Output:
    dataset/enhancement/daily_features.csv
"""

import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
ENHANCE_DIR = os.path.join(DATASET_DIR, 'enhancement')

OUTPUT_PATH = os.path.join(ENHANCE_DIR, 'daily_features.csv')

os.makedirs(ENHANCE_DIR, exist_ok=True)


# ===========================================================================
# 1. External data downloads
# ===========================================================================
def download_external_data(start='2014-01-01'):
    """Download S&P 500, US 10Y yield, USD/TWD, and TAIEX futures."""
    import yfinance as yf

    end = datetime.today().strftime('%Y-%m-%d')
    frames = {}

    # S&P 500
    print('  Downloading S&P 500...')
    sp = yf.download('^GSPC', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(sp.columns, pd.MultiIndex):
        sp.columns = sp.columns.get_level_values(0)
    if not sp.empty:
        sp = sp[['Close']].reset_index()
        sp.columns = ['date', 'sp500_close']
        sp['date'] = pd.to_datetime(sp['date']).dt.tz_localize(None)
        # Dated at the US close: same-calendar-date Taiwan targets must lag this.
        sp['sp500_return'] = sp['sp500_close'].pct_change()
        frames['sp500'] = sp[['date', 'sp500_close', 'sp500_return']]
        print(f'    {len(sp)} rows')

    # US 10Y Treasury yield
    print('  Downloading US 10Y yield...')
    tnx = yf.download('^TNX', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(tnx.columns, pd.MultiIndex):
        tnx.columns = tnx.columns.get_level_values(0)
    if not tnx.empty:
        tnx = tnx[['Close']].reset_index()
        tnx.columns = ['date', 'us10y_yield']
        tnx['date'] = pd.to_datetime(tnx['date']).dt.tz_localize(None)
        tnx['us10y_yield'] = tnx['us10y_yield'] / 100.0  # convert % to decimal
        frames['us10y'] = tnx
        print(f'    {len(tnx)} rows')

    # USD/TWD
    print('  Downloading USD/TWD...')
    fx = yf.download('TWD=X', start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(fx.columns, pd.MultiIndex):
        fx.columns = fx.columns.get_level_values(0)
    if not fx.empty:
        fx = fx[['Close']].reset_index()
        fx.columns = ['date', 'usdtwd']
        fx['date'] = pd.to_datetime(fx['date']).dt.tz_localize(None)
        frames['usdtwd'] = fx
        print(f'    {len(fx)} rows')

    # TAIEX futures
    print('  Downloading TAIEX futures...')
    try:
        from FinMind.data import DataLoader
        api = DataLoader()
        futures_dfs = []
        for year in range(2014, datetime.today().year + 1):
            for half in [('01-01', '06-30'), ('07-01', '12-31')]:
                s = f'{year}-{half[0]}'
                e = f'{year}-{half[1]}'
                if datetime.strptime(e, '%Y-%m-%d') > datetime.today():
                    e = datetime.today().strftime('%Y-%m-%d')
                if datetime.strptime(s, '%Y-%m-%d') > datetime.today():
                    break
                try:
                    df = api.taiwan_futures_daily(futures_id='TX', start_date=s, end_date=e)
                    if len(df) > 0:
                        futures_dfs.append(df)
                except Exception:
                    pass
                time.sleep(2)
        if futures_dfs:
            fut = pd.concat(futures_dfs, ignore_index=True)
            fut['date'] = pd.to_datetime(fut['date'])
            # Get front-month regular session data
            fut = fut[fut['trading_session'] == 'position']
            # For each date, take the nearest contract (smallest contract_date)
            fut = fut.sort_values(['date', 'contract_date'])
            fut_front = fut.groupby('date').first().reset_index()
            fut_front = fut_front[['date', 'close', 'settlement_price', 'volume', 'open_interest']]
            fut_front.columns = ['date', 'futures_close', 'futures_settlement',
                                 'futures_volume', 'futures_oi']
            frames['futures'] = fut_front
            print(f'    {len(fut_front)} rows')
    except Exception as e:
        print(f'    Futures download failed: {e}')

    return frames


# ===========================================================================
# 2. Compute realized volatility from TWII
# ===========================================================================
def compute_realized_vol(twii_path):
    """Compute multi-horizon realized volatility from TWII returns."""
    print('  Computing realized volatility...')
    twii = pd.read_csv(twii_path, parse_dates=['date'])
    twii = twii.sort_values('date')
    twii['return'] = np.log(twii['Adj Close'] / twii['Adj Close'].shift(1))

    for window in [5, 10, 20, 60]:
        twii[f'rv_{window}d'] = twii['return'].rolling(window).std() * np.sqrt(252)

    result = twii[['date', 'rv_5d', 'rv_10d', 'rv_20d', 'rv_60d']].copy()
    print(f'    {len(result)} rows, rv_20d mean={result["rv_20d"].mean():.4f}')
    return result


# ===========================================================================
# 3. Compute IV surface features from preprocessed data
# ===========================================================================
def compute_iv_surface_features(preprocessed_path):
    """Compute daily ATM IV, term structure slope, and skew from preprocessed data."""
    print('  Computing IV surface features...')
    df = pd.read_csv(preprocessed_path, index_col=0, parse_dates=['date'])

    # Handle legacy column name
    if 'S' in df.columns and 'underlying' not in df.columns:
        df = df.rename(columns={'S': 'underlying'})
    elif 'underlying' not in df.columns and 'S' not in df.columns:
        print('    WARNING: no underlying column found')
        return pd.DataFrame()

    und_col = 'underlying' if 'underlying' in df.columns else 'S'

    daily_features = []

    for date, day_data in df.groupby('date'):
        # ATM: contracts closest to moneyness = 1
        day_data = day_data.copy()
        day_data['abs_logm'] = day_data['logm'].abs()

        # ATM IV: average IV of contracts near the money
        atm = day_data[day_data['abs_logm'] < 0.02]
        if len(atm) < 1:
            atm = day_data.nsmallest(3, 'abs_logm')

        atm_iv = atm['impl_volatility'].mean()

        # Term structure slope: long-dated ATM IV - short-dated ATM IV
        short_term = day_data[(day_data['abs_logm'] < 0.05) & (day_data['tau'] < 0.12)]
        long_term = day_data[(day_data['abs_logm'] < 0.05) & (day_data['tau'] > 0.3)]

        if len(short_term) > 0 and len(long_term) > 0:
            iv_term_slope = long_term['impl_volatility'].mean() - short_term['impl_volatility'].mean()
        else:
            iv_term_slope = np.nan

        # IV skew: OTM put IV - ATM IV (logm < -0.05 are OTM puts)
        otm_puts = day_data[(day_data['logm'] < -0.05) & (day_data['logm'] > -0.2)]
        if len(otm_puts) > 0:
            iv_skew = otm_puts['impl_volatility'].mean() - atm_iv
        else:
            iv_skew = np.nan

        daily_features.append({
            'date': date,
            'atm_iv': atm_iv,
            'iv_term_slope': iv_term_slope,
            'iv_skew': iv_skew,
        })

    result = pd.DataFrame(daily_features)
    result['date'] = pd.to_datetime(result['date'])
    print(f'    {len(result)} rows, atm_iv mean={result["atm_iv"].mean():.4f}')
    return result


# ===========================================================================
# 4. Aggregate institutional data
# ===========================================================================
def aggregate_institutional(inst_path):
    """Aggregate institutional CSV into daily net buy/sell flows."""
    print('  Aggregating institutional data...')
    if not os.path.exists(inst_path):
        print('    Institutional data not found')
        return pd.DataFrame()

    df = pd.read_csv(inst_path, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])

    # The institutional_investors column has Chinese text for category names
    # We want: foreign investors net and dealers net
    # Aggregate net volume = long_deal_volume - short_deal_volume by date and category
    df['net_volume'] = df['long_deal_volume'] - df['short_deal_volume']

    # Group by date and aggregate across all categories
    daily = df.groupby('date').agg(
        total_net_volume=('net_volume', 'sum'),
        total_long_volume=('long_deal_volume', 'sum'),
        total_short_volume=('short_deal_volume', 'sum'),
    ).reset_index()

    # Compute net buy ratio
    daily['institutional_net_ratio'] = daily['total_net_volume'] / (
        daily['total_long_volume'] + daily['total_short_volume'] + 1e-8
    )

    result = daily[['date', 'total_net_volume', 'institutional_net_ratio']]
    result.columns = ['date', 'inst_net_volume', 'inst_net_ratio']
    print(f'    {len(result)} rows')
    return result


# ===========================================================================
# 5. Aggregate open interest data
# ===========================================================================
def aggregate_open_interest(oi_path):
    """Load and compute put/call OI ratio."""
    print('  Loading open interest data...')
    if not os.path.exists(oi_path):
        print('    Open interest data not found')
        return pd.DataFrame()

    df = pd.read_csv(oi_path, parse_dates=['date'])
    # Columns: date, total_oi_call, total_oi_put, total_volume_call, total_volume_put
    if 'total_oi_put' in df.columns and 'total_oi_call' in df.columns:
        df['put_call_oi_ratio'] = df['total_oi_put'] / (df['total_oi_call'] + 1e-8)
        df['put_call_volume_ratio'] = df['total_volume_put'] / (df['total_volume_call'] + 1e-8)
        result = df[['date', 'put_call_oi_ratio', 'put_call_volume_ratio']]
    else:
        result = df[['date']].copy()

    print(f'    {len(result)} rows')
    return result


# ===========================================================================
# 6. Compute futures basis
# ===========================================================================
def compute_futures_basis(futures_df, twii_path):
    """Compute futures-spot basis (futures close - TWII close)."""
    print('  Computing futures basis...')
    if futures_df is None or futures_df.empty:
        print('    No futures data')
        return pd.DataFrame()

    twii = pd.read_csv(twii_path, parse_dates=['date'])
    merged = pd.merge(futures_df, twii[['date', 'Adj Close']], on='date', how='inner')
    merged['futures_basis'] = merged['futures_close'] - merged['Adj Close']
    merged['futures_basis_pct'] = merged['futures_basis'] / merged['Adj Close'] * 100

    result = merged[['date', 'futures_basis', 'futures_basis_pct',
                     'futures_volume', 'futures_oi']]
    print(f'    {len(result)} rows, basis mean={result["futures_basis"].mean():.1f}')
    return result


# ===========================================================================
# Main: merge all features
# ===========================================================================
def main():
    print('=' * 60)
    print('Building Enhancement Features')
    print('=' * 60)

    twii_path = os.path.join(DATASET_DIR, 'TWII_full.csv')
    preprocessed_path = os.path.join(DATASET_DIR, 'prs_dataset_full.csv')
    inst_path = os.path.join(ENHANCE_DIR, 'institutional.csv')
    oi_path = os.path.join(ENHANCE_DIR, 'open_interest.csv')

    if not os.path.exists(twii_path):
        twii_path = os.path.join(DATASET_DIR, 'TWII.csv')
    if not os.path.exists(preprocessed_path):
        preprocessed_path = os.path.join(DATASET_DIR, 'prs_dataset_no_fat(clean).csv')

    # Step 1: Download external data
    print('\n[1/7] Downloading external data...')
    ext = download_external_data()

    # Step 2: Compute realized volatility
    print('\n[2/6] Computing realized volatility...')
    rv = compute_realized_vol(twii_path)

    # Step 3: Compute IV surface features
    print('\n[3/6] Computing IV surface features...')
    iv_features = compute_iv_surface_features(preprocessed_path)

    # Step 4: Aggregate institutional data
    print('\n[4/6] Aggregating institutional data...')
    inst = aggregate_institutional(inst_path)

    # Step 5: Aggregate open interest
    print('\n[5/6] Aggregating open interest...')
    oi = aggregate_open_interest(oi_path)

    # Step 6: Merge everything
    print('\n[6/6] Merging all features...')

    # Start with realized vol (has the full date range from TWII)
    daily = rv.copy()

    # Merge each feature set
    for name, df in [
        ('sp500', ext.get('sp500')),
        ('us10y', ext.get('us10y')),
        ('usdtwd', ext.get('usdtwd')),
        ('iv_features', iv_features),
        ('inst', inst),
        ('oi', oi),
    ]:
        if df is not None and not df.empty:
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
            daily = pd.merge(daily, df, on='date', how='left')
            print(f'  Merged {name}: {len(df)} rows')

    # Merge futures data and compute basis
    futures_df = ext.get('futures')
    if futures_df is not None and not futures_df.empty:
        basis = compute_futures_basis(futures_df, twii_path)
        if not basis.empty:
            daily = pd.merge(daily, basis, on='date', how='left')
            print(f'  Merged futures_basis: {len(basis)} rows')

    # Compute variance risk premium (ATM IV^2 - RV_20d^2)
    if 'atm_iv' in daily.columns and 'rv_20d' in daily.columns:
        daily['vrp_20d'] = daily['atm_iv'] ** 2 - daily['rv_20d'] ** 2
        print('  Computed variance risk premium (vrp_20d)')

    # Sort and save
    daily = daily.sort_values('date').reset_index(drop=True)
    daily.to_csv(OUTPUT_PATH, index=False)

    print(f'\n{"=" * 60}')
    print(f'Output: {OUTPUT_PATH}')
    print(f'Shape: {daily.shape}')
    print(f'Date range: {daily["date"].min()} to {daily["date"].max()}')
    print(f'\nColumns ({len(daily.columns)}):')
    for col in daily.columns:
        non_null = daily[col].notna().sum()
        pct = non_null / len(daily) * 100
        print(f'  {col:30s}  {non_null:>5d} non-null ({pct:.0f}%)')

    print(f'\n{"=" * 60}')
    print('Enhancement features ready!')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()

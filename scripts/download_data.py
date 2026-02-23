"""
Download and preprocess TXO option data (2014-present) with enhancement features.

Data Sources:
    1. TXO Options    -> FinMind API (TaiwanOptionDaily)
    2. TAIEX Index    -> Yahoo Finance (^TWII)
    3. CBOE VIX       -> Yahoo Finance (^VIX)
    4. Institutional  -> FinMind API (TaiwanOptionInstitutionalInvestors)
    5. Put/Call Ratio  -> FinMind API (TaiwanOptionDailyPutCallRatio) / TAIFEX

Usage:
    python scripts/download_data.py                    # download all + merge
    python scripts/download_data.py --skip-existing    # only download 2022-present
    python scripts/download_data.py --download-only    # download raw files, no merge
    python scripts/download_data.py --enhance-only     # only download enhancement data

Output:
    dataset/prs_dataset_full.csv           - merged preprocessed dataset (2014-present)
    dataset/TWII_full.csv                  - TAIEX underlying (2014-present)
    dataset/VIX_full.csv                   - CBOE VIX (2014-present)
    dataset/raw_txo/TXO_{year}.csv         - raw yearly TXO data from FinMind
    dataset/enhancement/institutional.csv  - institutional buy/sell data
    dataset/enhancement/put_call_ratio.csv - daily put/call ratio
    dataset/enhancement/open_interest.csv  - daily aggregate open interest
"""

import os
import sys
import time
import argparse
import warnings
from datetime import datetime, timedelta
from calendar import monthrange

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, 'dataset')
RAW_TXO_DIR = os.path.join(DATASET_DIR, 'raw_txo')
ENHANCE_DIR = os.path.join(DATASET_DIR, 'enhancement')

EXISTING_PREPROCESSED = os.path.join(DATASET_DIR, 'prs_dataset_no_fat(clean).csv')
OUTPUT_PREPROCESSED = os.path.join(DATASET_DIR, 'prs_dataset_full.csv')
OUTPUT_TWII = os.path.join(DATASET_DIR, 'TWII_full.csv')
OUTPUT_VIX = os.path.join(DATASET_DIR, 'VIX_full.csv')

os.makedirs(RAW_TXO_DIR, exist_ok=True)
os.makedirs(ENHANCE_DIR, exist_ok=True)


# ===========================================================================
# 1. Download TWII and VIX via yfinance
# ===========================================================================
def download_twii_vix(start='2014-01-01', end=None):
    """Download TAIEX index and CBOE VIX from Yahoo Finance."""
    import yfinance as yf

    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    print(f'[1/5] Downloading TWII ({start} to {end})...')
    twii = yf.download('^TWII', start=start, end=end, progress=False, auto_adjust=False)
    if twii.empty:
        print('  WARNING: TWII download returned empty. Check network/yfinance.')
        return None, None

    # Flatten multi-level columns if present (yfinance >= 0.2.31)
    if isinstance(twii.columns, pd.MultiIndex):
        twii.columns = twii.columns.get_level_values(0)

    # Handle both old ('Adj Close') and new ('Close') yfinance column names
    close_col = 'Adj Close' if 'Adj Close' in twii.columns else 'Close'
    twii_out = twii[[close_col]].reset_index()
    twii_out.columns = ['date', 'Adj Close']
    twii_out['date'] = pd.to_datetime(twii_out['date']).dt.tz_localize(None)
    twii_out.to_csv(OUTPUT_TWII, index=False)
    print(f'  TWII saved: {len(twii_out)} rows -> {OUTPUT_TWII}')

    print(f'[2/5] Downloading VIX ({start} to {end})...')
    vix = yf.download('^VIX', start=start, end=end, progress=False, auto_adjust=False)
    if vix.empty:
        print('  WARNING: VIX download returned empty.')
        return twii_out, None

    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    vix_out = vix[['Close']].reset_index()
    vix_out.columns = ['date', 'Close']
    vix_out['date'] = pd.to_datetime(vix_out['date']).dt.tz_localize(None)
    vix_out.to_csv(OUTPUT_VIX, index=False)
    print(f'  VIX saved: {len(vix_out)} rows -> {OUTPUT_VIX}')

    return twii_out, vix_out


# ===========================================================================
# 2. Download TXO Options via FinMind
# ===========================================================================
def contract_date_to_expiry(contract_date_str):
    """Convert FinMind contract_date to expiry date.

    Formats:
        '202401'    -> Monthly: 3rd Wednesday of Jan 2024
        '202401W2'  -> Wednesday weekly: 2nd Wednesday of Jan 2024
        '202507F2'  -> Friday weekly: 2nd Friday of Jul 2025
    """
    contract_date_str = str(contract_date_str).strip()

    def find_weekdays(year, month, target_weekday):
        """Find all occurrences of a weekday (0=Mon, 2=Wed, 4=Fri) in a month."""
        days = []
        for day in range(1, monthrange(year, month)[1] + 1):
            d = datetime(year, month, day)
            if d.weekday() == target_weekday:
                days.append(d)
        return days

    if 'W' in contract_date_str:
        # Wednesday weekly: e.g., '202401W2'
        parts = contract_date_str.split('W')
        ym = parts[0]
        week_num = int(parts[1])
        year = int(ym[:4])
        month = int(ym[4:])
        wednesdays = find_weekdays(year, month, 2)
        idx = min(week_num - 1, len(wednesdays) - 1)
        return wednesdays[idx]

    elif 'F' in contract_date_str:
        # Friday weekly: e.g., '202507F2'
        parts = contract_date_str.split('F')
        ym = parts[0]
        week_num = int(parts[1])
        year = int(ym[:4])
        month = int(ym[4:])
        fridays = find_weekdays(year, month, 4)
        idx = min(week_num - 1, len(fridays) - 1)
        return fridays[idx]

    else:
        # Monthly: 3rd Wednesday of the month
        year = int(contract_date_str[:4])
        month = int(contract_date_str[4:])
        wednesdays = find_weekdays(year, month, 2)
        return wednesdays[2] if len(wednesdays) >= 3 else wednesdays[-1]


def download_txo_finmind(start_year=2022, end_year=None):
    """Download TXO daily data from FinMind, one month at a time to stay within rate limits."""
    from FinMind.data import DataLoader

    if end_year is None:
        end_year = datetime.today().year

    api = DataLoader()

    print(f'[3/5] Downloading TXO options ({start_year}-{end_year}) from FinMind...')
    print('  (This may take a while due to API rate limits)')

    for year in range(start_year, end_year + 1):
        outpath = os.path.join(RAW_TXO_DIR, f'TXO_{year}.csv')
        if os.path.exists(outpath):
            existing = pd.read_csv(outpath)
            print(f'  {year}: already downloaded ({len(existing)} rows), skipping')
            continue

        year_dfs = []
        for month in range(1, 13):
            start_date = f'{year}-{month:02d}-01'
            end_day = monthrange(year, month)[1]
            end_date = f'{year}-{month:02d}-{end_day:02d}'

            # Don't request future dates
            if datetime(year, month, end_day) > datetime.today():
                end_date = datetime.today().strftime('%Y-%m-%d')
                if datetime(year, month, 1) > datetime.today():
                    break

            try:
                df = api.taiwan_option_daily(
                    option_id='TXO',
                    start_date=start_date,
                    end_date=end_date,
                )
                if len(df) > 0:
                    year_dfs.append(df)
                    print(f'  {year}-{month:02d}: {len(df)} rows', end='')
                else:
                    print(f'  {year}-{month:02d}: no data', end='')
            except Exception as e:
                print(f'  {year}-{month:02d}: ERROR {e}', end='')

            # Rate limit: ~300 requests/hour for free tier
            time.sleep(2.5)
            print('', flush=True)

        if year_dfs:
            year_df = pd.concat(year_dfs, ignore_index=True)
            year_df.to_csv(outpath, index=False)
            print(f'  -> {year} total: {len(year_df)} rows saved to {outpath}')
        else:
            print(f'  -> {year}: no data collected')


def load_raw_txo():
    """Load all downloaded raw TXO CSVs and return combined DataFrame."""
    dfs = []
    for f in sorted(os.listdir(RAW_TXO_DIR)):
        if f.startswith('TXO_') and f.endswith('.csv'):
            dfs.append(pd.read_csv(os.path.join(RAW_TXO_DIR, f)))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# 3. Preprocess new TXO data to match existing format
# ===========================================================================
def preprocess_new_txo(raw_txo, twii):
    """Convert raw FinMind TXO data to the same format as existing preprocessed data.

    Output columns: date, exdate, option_price, strike_price, invm, tau, S, r, d,
                    impl_volatility, volume, logm, total_var
    """
    print('  Preprocessing new TXO data...')
    df = raw_txo.copy()

    # Filter: regular session only, non-zero settlement price
    df = df[df['trading_session'] == 'position'].copy()
    df = df[df['settlement_price'] > 0].copy()
    df = df[df['volume'] > 0].copy()

    # Compute expiry dates
    print('    Computing expiry dates...')
    expiry_cache = {}
    def get_expiry(cd):
        if cd not in expiry_cache:
            expiry_cache[cd] = contract_date_to_expiry(cd)
        return expiry_cache[cd]

    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = df['contract_date'].apply(get_expiry)
    df['exdate'] = pd.to_datetime(df['exdate'])

    # tau = (exdate - date) / 252 trading days
    df['tau'] = (df['exdate'] - df['date']).dt.days / 365.0
    df = df[df['tau'] > 0].copy()

    # Rename columns to match existing format
    df = df.rename(columns={
        'settlement_price': 'option_price',
        'call_put': 'put/call',
    })
    df['strike_price'] = df['strike_price'].astype(float)

    # Merge underlying price
    print('    Merging underlying prices...')
    twii = twii.copy()
    twii['date'] = pd.to_datetime(twii['date'])
    df = pd.merge(df, twii[['date', 'Adj Close']], on='date', how='left')
    df = df.rename(columns={'Adj Close': 'S'})
    df = df.dropna(subset=['S'])
    df['S'] = df['S'].astype(float)

    # Filter by liquidity: select the more liquid of call/put at each (date, strike, exdate)
    print('    Filtering by put/call liquidity...')
    df = df.sort_values(['date', 'exdate', 'strike_price', 'put/call'])

    filtered_rows = []
    for (date, exdate, strike), group in df.groupby(['date', 'exdate', 'strike_price']):
        calls = group[group['put/call'] == 'call']
        puts = group[group['put/call'] == 'put']
        if len(calls) > 0 and len(puts) > 0:
            call_vol = calls['volume'].values[0]
            put_vol = puts['volume'].values[0]
            filtered_rows.append(calls.iloc[0] if call_vol >= put_vol else puts.iloc[0])
        elif len(calls) > 0:
            filtered_rows.append(calls.iloc[0])
        elif len(puts) > 0:
            filtered_rows.append(puts.iloc[0])

    df = pd.DataFrame(filtered_rows)

    # Compute inverse moneyness and log-moneyness
    # Use simple risk-free rate estimate (Taiwan 1Y rate ~ 1.5-2.0%)
    # For consistency with existing data, use a simple put-call parity proxy
    df['r'] = 0.015  # approximate Taiwan short-term rate
    df['d'] = 0       # dividend yield (same as existing data)

    df['invm'] = df['strike_price'] / df['S']

    # Compute beta-adjusted log-moneyness (simplified version matching existing pipeline)
    # For new data, use direct log-moneyness since we don't have parity data for all pairs
    df['logm'] = np.log(df['strike_price'] / df['S'])

    # Compute implied volatility via Black-Scholes
    print('    Computing implied volatility (this may take a few minutes)...')
    df = compute_implied_vol(df)

    # Compute total variance
    df['total_var'] = df['impl_volatility'] ** 2 * df['tau']

    # Select and order columns to match existing format
    cols = ['date', 'exdate', 'option_price', 'strike_price', 'invm', 'tau',
            'S', 'r', 'd', 'impl_volatility', 'volume', 'logm', 'total_var']

    # Rename S to match existing column name
    result = df[cols].copy()

    # Filter out extreme/invalid values
    result = result[result['impl_volatility'] > 0.001]
    result = result[result['impl_volatility'] < 3.0]
    result = result[result['total_var'] > 0]

    print(f'    New data preprocessed: {len(result)} rows')
    return result


def compute_implied_vol(df):
    """Compute Black-Scholes implied volatility for each option row."""
    def bs_price(S, K, tau, sigma, r, indicator):
        if sigma <= 0 or tau <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        if indicator == 'call':
            return S * np.exp(-0 * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
        else:
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-0 * tau) * norm.cdf(-d1)

    def find_iv(row):
        S = float(row['S'])
        K = float(row['strike_price'])
        tau = float(row['tau'])
        price = float(row['option_price'])
        r = float(row['r'])
        indicator = row['put/call']

        def objective(sigma):
            return (bs_price(S, K, tau, sigma[0], r, indicator) - price) ** 2

        try:
            result = minimize(objective, x0=[0.2], bounds=[(0.001, 3.0)], method='L-BFGS-B')
            if result.success and result.fun < (price * 0.1) ** 2:
                return result.x[0]
            else:
                return np.nan
        except Exception:
            return np.nan

    # Process in chunks for progress reporting
    n = len(df)
    chunk_size = max(1, n // 20)
    ivs = []
    for i in range(0, n, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_ivs = chunk.apply(find_iv, axis=1)
        ivs.append(chunk_ivs)
        pct = min(100, (i + chunk_size) / n * 100)
        print(f'      IV computation: {pct:.0f}%', end='\r')

    print()
    df = df.copy()
    df['impl_volatility'] = pd.concat(ivs)
    df = df.dropna(subset=['impl_volatility'])
    return df


# ===========================================================================
# 4. Download enhancement data
# ===========================================================================
def download_enhancement_data(start_year=2014, end_year=None):
    """Download additional features: institutional flows, put/call ratio, open interest."""
    from FinMind.data import DataLoader

    if end_year is None:
        end_year = datetime.today().year

    api = DataLoader()

    # --- Institutional investors ---
    inst_path = os.path.join(ENHANCE_DIR, 'institutional.csv')
    if not os.path.exists(inst_path):
        print('[4/5] Downloading institutional investor data...')
        inst_dfs = []
        for year in range(start_year, end_year + 1):
            for half in [('01-01', '06-30'), ('07-01', '12-31')]:
                start_date = f'{year}-{half[0]}'
                end_date = f'{year}-{half[1]}'
                if datetime.strptime(end_date, '%Y-%m-%d') > datetime.today():
                    end_date = datetime.today().strftime('%Y-%m-%d')
                if datetime.strptime(start_date, '%Y-%m-%d') > datetime.today():
                    break
                try:
                    df = api.taiwan_option_institutional_investors(
                        option_id='TXO',
                        start_date=start_date,
                        end_date=end_date,
                    )
                    if len(df) > 0:
                        inst_dfs.append(df)
                        print(f'  {start_date} to {end_date}: {len(df)} rows')
                except Exception as e:
                    print(f'  {start_date} to {end_date}: ERROR {e}')
                time.sleep(3)

        if inst_dfs:
            inst_df = pd.concat(inst_dfs, ignore_index=True)
            inst_df.to_csv(inst_path, index=False, encoding='utf-8-sig')
            print(f'  Institutional data saved: {len(inst_df)} rows -> {inst_path}')
    else:
        print('[4/5] Institutional data already exists, skipping')

    # --- Open interest (aggregate from raw TXO data) ---
    oi_path = os.path.join(ENHANCE_DIR, 'open_interest.csv')
    if not os.path.exists(oi_path):
        print('  Computing aggregate open interest from raw TXO data...')
        raw = load_raw_txo()
        if not raw.empty:
            raw['date'] = pd.to_datetime(raw['date'])
            oi = raw[raw['trading_session'] == 'position'].groupby(['date', 'call_put']).agg(
                total_oi=('open_interest', 'sum'),
                total_volume=('volume', 'sum'),
            ).reset_index()
            oi_pivot = oi.pivot_table(index='date', columns='call_put',
                                       values=['total_oi', 'total_volume'],
                                       fill_value=0)
            oi_pivot.columns = ['_'.join(col).strip() for col in oi_pivot.columns]
            oi_pivot = oi_pivot.reset_index()
            oi_pivot.to_csv(oi_path, index=False)
            print(f'  Open interest saved: {len(oi_pivot)} rows -> {oi_path}')
    else:
        print('  Open interest data already exists, skipping')

    # --- Put/Call ratio from FinMind ---
    pcr_path = os.path.join(ENHANCE_DIR, 'put_call_ratio.csv')
    if not os.path.exists(pcr_path):
        print('[5/5] Downloading put/call ratio...')
        pcr_dfs = []
        for year in range(start_year, end_year + 1):
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'
            if datetime.strptime(end_date, '%Y-%m-%d') > datetime.today():
                end_date = datetime.today().strftime('%Y-%m-%d')
            if datetime.strptime(start_date, '%Y-%m-%d') > datetime.today():
                break
            try:
                # Use the FinMind API endpoint for put/call ratio
                import requests
                url = "https://api.finmindtrade.com/api/v4/data"
                params = {
                    "dataset": "TaiwanOptionDailyPutCallRatio",
                    "data_id": "TXO",
                    "start_date": start_date,
                    "end_date": end_date,
                }
                resp = requests.get(url, params=params, timeout=30)
                data = resp.json()
                if data.get('status') == 200 and data.get('data'):
                    df = pd.DataFrame(data['data'])
                    pcr_dfs.append(df)
                    print(f'  {year}: {len(df)} rows')
                else:
                    print(f'  {year}: no data or error ({data.get("msg", "")})')
            except Exception as e:
                print(f'  {year}: ERROR {e}')
            time.sleep(3)

        if pcr_dfs:
            pcr_df = pd.concat(pcr_dfs, ignore_index=True)
            pcr_df.to_csv(pcr_path, index=False)
            print(f'  Put/call ratio saved: {len(pcr_df)} rows -> {pcr_path}')
    else:
        print('[5/5] Put/call ratio already exists, skipping')


# ===========================================================================
# 5. Merge old + new data
# ===========================================================================
def merge_datasets():
    """Merge existing 2014-2021 preprocessed data with new 2022+ data."""
    print('\nMerging datasets...')

    # Load existing preprocessed data
    if not os.path.exists(EXISTING_PREPROCESSED):
        print(f'  WARNING: Existing data not found at {EXISTING_PREPROCESSED}')
        return None

    existing = pd.read_csv(EXISTING_PREPROCESSED, index_col=0)
    existing['date'] = pd.to_datetime(existing['date'])
    existing['exdate'] = pd.to_datetime(existing['exdate'])

    # Handle legacy column name
    if 'S' in existing.columns and 'underlying' not in existing.columns:
        pass  # keep as 'S' for consistency
    elif 'underlying' in existing.columns and 'S' not in existing.columns:
        existing = existing.rename(columns={'underlying': 'S'})

    print(f'  Existing data: {len(existing)} rows ({existing["date"].min().date()} to {existing["date"].max().date()})')

    # Load and preprocess new TXO data
    raw_txo = load_raw_txo()
    if raw_txo.empty:
        print('  No new TXO data found. Run with download first.')
        # Still output existing data with updated TWII/VIX
        existing.to_csv(OUTPUT_PREPROCESSED)
        print(f'  Output (existing only): {len(existing)} rows -> {OUTPUT_PREPROCESSED}')
        return existing

    # Load TWII for new data
    if os.path.exists(OUTPUT_TWII):
        twii = pd.read_csv(OUTPUT_TWII, parse_dates=['date'])
    else:
        print('  WARNING: TWII data not found. Cannot preprocess new TXO data.')
        existing.to_csv(OUTPUT_PREPROCESSED)
        return existing

    raw_txo['date'] = pd.to_datetime(raw_txo['date'])
    existing_max_date = existing['date'].max()
    new_txo = raw_txo[raw_txo['date'] > existing_max_date]
    print(f'  New raw TXO data: {len(new_txo)} rows ({new_txo["date"].min().date()} to {new_txo["date"].max().date()})')

    if new_txo.empty:
        print('  No new data after existing date range.')
        existing.to_csv(OUTPUT_PREPROCESSED)
        return existing

    new_preprocessed = preprocess_new_txo(new_txo, twii)

    # Combine
    combined = pd.concat([existing, new_preprocessed], ignore_index=True)
    combined = combined.sort_values(['date', 'tau', 'logm']).reset_index(drop=True)

    # Remove duplicates if any
    combined = combined.drop_duplicates(subset=['date', 'exdate', 'strike_price', 'tau'], keep='first')

    combined.to_csv(OUTPUT_PREPROCESSED)
    print(f'\n  MERGED OUTPUT: {len(combined)} rows ({combined["date"].min().date()} to {combined["date"].max().date()})')
    print(f'  Saved to: {OUTPUT_PREPROCESSED}')

    # Print yearly breakdown
    combined['year'] = combined['date'].dt.year
    print('\n  Rows per year:')
    for year, count in combined.groupby('year').size().items():
        marker = ' (NEW)' if year > existing_max_date.year else ''
        print(f'    {year}: {count:>8,}{marker}')

    return combined


# ===========================================================================
# 6. Update VIX file for the pipeline
# ===========================================================================
def update_pipeline_vix():
    """Copy the full VIX to the standard pipeline location."""
    src = OUTPUT_VIX
    dst = os.path.join(DATASET_DIR, 'VIX.csv')
    backup = os.path.join(DATASET_DIR, 'VIX_backup_2014_2021.csv')

    if os.path.exists(src):
        if os.path.exists(dst) and not os.path.exists(backup):
            # Backup original
            import shutil
            shutil.copy2(dst, backup)
            print(f'  Original VIX backed up to {backup}')

        import shutil
        shutil.copy2(src, dst)
        print(f'  VIX.csv updated with full date range')


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description='Download and merge TXO option data')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Only download data after existing date range')
    parser.add_argument('--download-only', action='store_true',
                        help='Download raw files only, skip merge/preprocessing')
    parser.add_argument('--enhance-only', action='store_true',
                        help='Only download enhancement data')
    parser.add_argument('--start-year', type=int, default=2022,
                        help='Start year for TXO download (default: 2022, existing data covers 2014-2021)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year for download (default: current year)')
    parser.add_argument('--no-enhance', action='store_true',
                        help='Skip enhancement data download')
    args = parser.parse_args()

    print('=' * 60)
    print('TXO Data Download & Preprocessing Pipeline')
    print('=' * 60)
    print(f'Project root: {PROJECT_ROOT}')
    print(f'Dataset dir:  {DATASET_DIR}')
    print()

    if args.enhance_only:
        download_enhancement_data(start_year=2014, end_year=args.end_year)
        print('\nDone!')
        return

    # Step 1-2: Download TWII and VIX
    twii, vix = download_twii_vix()

    # Step 3: Download TXO options from FinMind
    download_txo_finmind(start_year=args.start_year, end_year=args.end_year)

    if args.download_only:
        print('\nDownload complete (--download-only). Skipping merge.')
        return

    # Step 4-5: Enhancement data
    if not args.no_enhance:
        download_enhancement_data(start_year=2014, end_year=args.end_year)
    else:
        print('[4/5] Skipping enhancement data (--no-enhance)')
        print('[5/5] Skipping put/call ratio (--no-enhance)')

    # Step 6: Merge and preprocess
    merge_datasets()

    # Step 7: Update pipeline VIX
    update_pipeline_vix()

    print('\n' + '=' * 60)
    print('Pipeline complete!')
    print()
    print('Output files:')
    for path in [OUTPUT_PREPROCESSED, OUTPUT_TWII, OUTPUT_VIX]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  {os.path.basename(path):40s} {size_mb:.1f} MB')
    for path in [os.path.join(ENHANCE_DIR, f) for f in
                 ['institutional.csv', 'open_interest.csv', 'put_call_ratio.csv']]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f'  enhancement/{os.path.basename(path):28s} {size_mb:.1f} MB')

    print()
    print('Next steps:')
    print('  1. Review the merged data in dataset/prs_dataset_full.csv')
    print('  2. Update config.ini to point to the new dataset:')
    print('     prs_dataset = prs_dataset_full')
    print('  3. Update training date ranges in config.ini')
    print('  4. Re-train models with the extended dataset')
    print('=' * 60)


if __name__ == '__main__':
    main()

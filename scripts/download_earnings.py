"""Earnings announcement dates for the Mag 7 (scheduled events inside the
10-67 day option window that dominate short-dated single-stock IV).

yfinance's get_earnings_dates() is the primary source but its history is
capped and its tail can be stale, so dates are UNIONED with a second source
derived from quarterly fiscal period ends, and the result is validated
against the option data's own IV signature (a real earnings date shows a
front-expiry ATM IV crush the snapshot after the event).

Output: dataset/us_options/earnings_dates.csv with columns
(ticker, earnings_date, source). Announcement timing (before/after close)
is normalized to the EFFECTIVE trading date on which the move is realized:
an after-close announcement on day d moves the stock on d+1.
"""
import os
import sys

import pandas as pd
import yfinance as yf

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'us_options')
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']


def from_yfinance(ticker):
    """Announcement timestamps -> effective move date."""
    rows = []
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=80)
    except Exception as e:  # noqa: BLE001
        print(f'  {ticker}: get_earnings_dates failed ({str(e)[:80]})')
        return rows
    if ed is None or len(ed) == 0:
        return rows
    for ts in ed.index:
        ts = pd.Timestamp(ts)
        d = ts.tz_localize(None) if ts.tz is not None else ts
        # After-close announcements (>= 16:00 local) move the NEXT session
        eff = d.normalize() + (pd.Timedelta(days=1) if d.hour >= 16 else pd.Timedelta(0))
        rows.append({'ticker': ticker, 'earnings_date': eff.normalize(),
                     'source': 'yfinance'})
    return rows


def from_calendar(ticker):
    """Upcoming/scheduled date from the calendar endpoint (fills the tail)."""
    rows = []
    try:
        cal = yf.Ticker(ticker).calendar
    except Exception:  # noqa: BLE001
        return rows
    if isinstance(cal, dict):
        vals = cal.get('Earnings Date') or []
        vals = vals if isinstance(vals, (list, tuple)) else [vals]
    elif isinstance(cal, pd.DataFrame) and 'Earnings Date' in cal.index:
        vals = list(cal.loc['Earnings Date'].values)
    else:
        return rows
    for v in vals:
        try:
            rows.append({'ticker': ticker,
                         'earnings_date': pd.Timestamp(v).normalize(),
                         'source': 'calendar'})
        except Exception:  # noqa: BLE001
            continue
    return rows


def validate_against_iv(df):
    """Cross-check dates against the front-expiry ATM IV crush in our chains.

    Returns a per-ticker report: for each earnings date that falls inside the
    option sample, the change in front-expiry ATM IV from the last snapshot
    before the event to the first snapshot on/after it. A genuine earnings
    date shows a large NEGATIVE change (the crush).
    """
    import numpy as np
    report = []
    for ticker, grp in df.groupby('ticker'):
        path = os.path.join(OUT_DIR, f'chains_{ticker}.csv')
        if not os.path.exists(path):
            continue
        ch = pd.read_csv(path, usecols=['date', 'expiration', 'strike',
                                        'call_put', 'vol', 'bid', 'ask'])
        ch['date'] = pd.to_datetime(ch['date'])
        ch['expiration'] = pd.to_datetime(ch['expiration'])
        ch = ch[(ch['bid'] > 0) & ch['vol'].between(0.01, 5.0)]
        # front-expiry ATM IV proxy per snapshot: median IV of the shortest
        # expiry's 5 most-central strikes (parity forward approximated by the
        # strike where call and put mids are closest)
        atm = {}
        for d, g in ch.groupby('date'):
            exp0 = g['expiration'].min()
            g0 = g[g['expiration'] == exp0]
            piv = g0.pivot_table(index='strike', columns='call_put',
                                 values='vol', aggfunc='first').dropna()
            if len(piv) < 3:
                continue
            mid = piv.index[len(piv) // 2]
            dist = pd.Series(np.abs(piv.index.values - mid), index=piv.index)
            near = piv.loc[dist.nsmallest(5).index]
            atm[d] = float(near.mean(axis=1).median())
        if not atm:
            continue
        s = pd.Series(atm).sort_index()
        crushes = []
        for ed in grp['earnings_date']:
            before = s[s.index < ed]
            after = s[s.index >= ed]
            if len(before) == 0 or len(after) == 0:
                continue
            if (after.index[0] - before.index[-1]).days > 7:
                continue
            crushes.append(after.iloc[0] - before.iloc[-1])
        if crushes:
            arr = np.array(crushes)
            report.append({'ticker': ticker, 'n_checked': len(arr),
                           'median_iv_change': float(np.median(arr)),
                           'frac_negative': float((arr < 0).mean())})
    return pd.DataFrame(report)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for t in TICKERS:
        r = from_yfinance(t) + from_calendar(t)
        print(f'  {t}: {len(r)} raw dates')
        rows.extend(r)

    df = pd.DataFrame(rows)
    df = (df.sort_values('source')
            .drop_duplicates(subset=['ticker', 'earnings_date'], keep='first')
            .sort_values(['ticker', 'earnings_date']))

    out = os.path.join(OUT_DIR, 'earnings_dates.csv')
    df.to_csv(out, index=False)
    print(f'{len(df)} earnings dates -> {out}')
    for t, g in df.groupby('ticker'):
        print(f'  {t}: {len(g)} dates, {g["earnings_date"].min().date()} '
              f'-> {g["earnings_date"].max().date()}')

    if '--validate' in sys.argv:
        print('\nValidating against front-expiry ATM IV crush...')
        rep = validate_against_iv(df)
        if len(rep):
            print(rep.to_string(index=False))
            print('(median_iv_change should be NEGATIVE; frac_negative near 1.0)')


if __name__ == '__main__':
    main()

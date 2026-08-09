"""Repair the earnings calendar by extrapolating the quarterly schedule and
refining each date against the local IV-crush signature.

WHY THIS IS NEEDED: yfinance returns only 2 of 4 announcements for 2025 and 1
for 2026 across every Mag 7 ticker. That gap sits inside the test window
(test starts 2025-01-01), so ~19% of the option sample would be silently
mislabelled "no earnings before expiry" — corrupting every event feature
exactly where it is evaluated.

WHY THIS IS LEGITIMATE: earnings dates are publicly scheduled weeks in
advance, so a backtest that knows them is realistic — this reconstructs a
calendar that was knowable at the time. The reconstruction is driven by the
CADENCE (each firm announces on a stable ~91-day cycle at a stable point in
the quarter), which is causal; the option data is used only to nudge each
predicted date onto the exact day within a +/-10 day window.

A global "find the biggest IV crashes" detector was tried first and rejected:
it scored 0.48-0.96 recall and produced false positives in high-volatility
regimes (8 detections for MSFT in 2020 against a true 4), because a
market-wide vol collapse looks exactly like a crush. The cadence prior
removes that failure mode.

ACCURACY IS MEASURED, NOT ASSUMED: the script back-tests the procedure by
hiding the 2024 announcements, predicting them from <=2023 data, and
reporting the day error.

Output: dataset/us_options/earnings_dates_v2.csv
  (ticker, earnings_date, source in {yfinance, reconstructed}, confidence)
"""
import os

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'us_options')
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
REFINE_WINDOW = 10          # days either side of the cadence prediction
CYCLE = 91                  # quarterly cadence


def front_atm_iv(ticker):
    """Front-expiry ATM implied vol per snapshot (the crush observable).

    META traded as FB before 2022-06, so its early history lives in a
    separate file; reading only chains_META.csv silently truncates META's
    sample to 2022+ and makes the cadence walk skip 2019-2021 entirely.
    """
    files = [f'chains_{ticker}.csv']
    if ticker == 'META':
        files.append('chains_FB.csv')
    frames = []
    for f in files:
        p = os.path.join(DATA, f)
        if os.path.exists(p):
            frames.append(pd.read_csv(p, usecols=['date', 'expiration',
                                                  'strike', 'call_put',
                                                  'vol', 'bid', 'ask']))
    ch = pd.concat(frames, ignore_index=True)
    ch['date'] = pd.to_datetime(ch['date'])
    ch['expiration'] = pd.to_datetime(ch['expiration'])
    ch = ch[(ch['bid'] > 0) & ch['vol'].between(0.01, 5.0)]
    out = {}
    for d, g in ch.groupby('date'):
        g0 = g[g['expiration'] == g['expiration'].min()]
        piv = g0.pivot_table(index='strike', columns='call_put', values='vol',
                             aggfunc='first').dropna()
        if len(piv) < 3:
            continue
        mid = piv.index[len(piv) // 2]
        dist = pd.Series(np.abs(piv.index.values - mid), index=piv.index)
        out[d] = float(piv.loc[dist.nsmallest(3).index].mean(axis=1).median())
    return pd.Series(out).sort_index()


def refine(pred, iv, window=REFINE_WINDOW):
    """Snap a predicted date onto the largest local relative IV crush.

    Returns (refined_date, relative_crush). The crush is observed on the first
    post-announcement snapshot, so the announcement's effective move date is
    that snapshot itself (the day the market gaps).
    """
    rel = (iv.diff() / iv.shift(1)).dropna()
    lo, hi = pred - pd.Timedelta(days=window), pred + pd.Timedelta(days=window)
    local = rel[(rel.index >= lo) & (rel.index <= hi)]
    if len(local) == 0:
        return pred, np.nan
    return local.idxmin(), float(local.min())


def extrapolate(known, until, iv, start=None):
    """Walk the quarterly cadence across the whole span, refining each step
    against the local crush and RE-ANCHORING whenever a known announcement is
    nearby.

    Walking the full span (rather than only past the last known date) fills
    interior holes too — yfinance's coverage is patchy mid-history for META
    and TSLA, not just at the tail. Re-anchoring on known dates stops the
    91-day cycle drifting over a multi-year walk.
    """
    known = sorted(known)
    kn = pd.DatetimeIndex(known)
    out = []
    cursor = start if start is not None else known[0]
    seen = {known[0]}
    while cursor < until:
        pred = cursor + pd.Timedelta(days=CYCLE)
        if pred > until:
            break
        got, crush = refine(pred, iv)
        cand = got if (np.isfinite(crush) and crush < -0.04) else pred

        # Re-anchor on a known date if one sits near the candidate
        near = kn[np.abs((kn - cand).days) <= REFINE_WINDOW] if len(kn) else []
        if len(near):
            chosen, src = near[0], 'yfinance'
        else:
            chosen, src = cand, 'reconstructed'

        if chosen not in seen:
            out.append({'date': chosen, 'crush': crush, 'source': src})
            seen.add(chosen)
        cursor = chosen
    return out


def backtest(known, iv):
    """Hide 2024 announcements, predict them from <=2023, report day error."""
    hist = [k for k in known if k < pd.Timestamp('2024-01-01')]
    truth = [k for k in known if pd.Timestamp('2024-01-01') <= k
             < pd.Timestamp('2025-01-01')]
    if len(hist) < 4 or not truth:
        return None
    preds = extrapolate(hist, pd.Timestamp('2024-12-31'), iv)
    errs = []
    for t in truth:
        if not preds:
            continue
        d = min(preds, key=lambda p: abs((p['date'] - t).days))
        errs.append(abs((d['date'] - t).days))
    return {'n': len(errs), 'median_err_days': float(np.median(errs)),
            'max_err_days': int(np.max(errs)),
            'within_3d': float(np.mean(np.array(errs) <= 3))} if errs else None


def main():
    truth = pd.read_csv(os.path.join(DATA, 'earnings_dates.csv'),
                        parse_dates=['earnings_date'])
    rows, diag = [], []

    for t in TICKERS:
        iv = front_atm_iv(t)
        if len(iv) == 0:
            continue
        lo, hi = iv.index[0], iv.index[-1]
        known = sorted(truth[(truth.ticker == t)
                             & truth.earnings_date.between(
                                 lo - pd.Timedelta(days=120), hi)]
                       ['earnings_date'].unique())
        known = [pd.Timestamp(k) for k in known]
        if len(known) < 4:
            continue

        bt = backtest(known, iv)

        # yfinance is reliable where it has coverage; the cadence walk fills
        # interior holes and the 2025-2026 tail, re-anchoring on known dates.
        reliable = [k for k in known if k < pd.Timestamp('2025-01-01')]
        for k in reliable:
            if lo - pd.Timedelta(days=CYCLE) <= k <= hi:
                rows.append({'ticker': t, 'earnings_date': k,
                             'source': 'yfinance', 'confidence': 1.0})
        # Step the cadence anchor back to the start of the option sample so
        # the walk also covers years where yfinance has no coverage at all
        # (META has nothing before 2022, TSLA nothing before 2021).
        anchor = known[0]
        while anchor > lo:
            anchor -= pd.Timedelta(days=CYCLE)
        chain = extrapolate(reliable, hi, iv, start=anchor)
        n_recon = 0
        for r in chain:
            if r['source'] == 'yfinance':
                continue
            n_recon += 1
            rows.append({'ticker': t, 'earnings_date': r['date'],
                         'source': 'reconstructed',
                         'confidence': 0.9 if (np.isfinite(r['crush'])
                                               and r['crush'] < -0.04) else 0.5})
        d = {'ticker': t, 'known_pre2025': len(reliable),
             'reconstructed': n_recon}
        if bt:
            d.update(bt)
        diag.append(d)

    out = (pd.DataFrame(rows)
             .sort_values(['ticker', 'earnings_date'])
             .drop_duplicates(subset=['ticker', 'earnings_date']))
    path = os.path.join(DATA, 'earnings_dates_v2.csv')
    out.to_csv(path, index=False)

    print('BACKTEST OF THE RECONSTRUCTION (2024 hidden, predicted from <=2023):')
    print(pd.DataFrame(diag).to_string(index=False))
    print(f'\n{len(out)} dates -> {path}')
    ed = out[out.earnings_date >= '2019-01-01']
    print('\nCOVERAGE BY YEAR (target ~4/ticker/year):')
    print(ed.assign(y=ed.earnings_date.dt.year)
            .pivot_table(index='y', columns='ticker', values='earnings_date',
                         aggfunc='count').fillna(0).astype(int).to_string())


if __name__ == '__main__':
    main()

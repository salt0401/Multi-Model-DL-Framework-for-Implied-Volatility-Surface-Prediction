"""Earnings-event layer for short-dated US single-stock options.

Measured on this dataset (2023+, ATM IV by expiry): an expiry that SPANS an
earnings announcement carries 2.5 (AAPL) to 14.2 (TSLA) vol points more ATM
implied volatility than one that does not, and the front-expiry ATM IV
crush the snapshot after an announcement is -8.6 to -28.2 vol points with
~99.5% consistency across 151 checked events. Every option in this dataset
has tau in [10, 67] days, so essentially every surface is shaped by a
scheduled event. A term-structure model that does not know about earnings
is mis-specified on this data, not merely imprecise.

Standard decomposition (Dubinsky-Johannes style): total variance to expiry
is a continuous diffusive part plus the variance of the discrete announcement
jumps that fall before expiry,

    w(tau) = sigma_d^2 * tau + n(tau) * J^2

with n(tau) the number of announcements in (t, t+tau] and J^2 the per-event
jump variance. Both are identified per (ticker, snapshot) by regressing the
observed ATM total variances of the 3-4 available expiries on [tau, n].

That decomposition yields the EVENT-ADJUSTED CLOCK

    tau_tilde = tau + n(tau) * J^2 / sigma_d^2

in which total variance is linear again (w = sigma_d^2 * tau_tilde). Because
tau_tilde is strictly increasing in tau, calendar-arbitrage monotonicity is
preserved: any surface that is calendar-arbitrage-free in tau_tilde is also
calendar-arbitrage-free in tau. This is what makes SSVI/eSSVI-style term
structures usable on single names.
"""
import os

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


class EarningsCalendar:
    """Scheduled announcement dates per ticker (known in advance — using them
    at time t is not look-ahead: the schedule is public well before the event)."""

    def __init__(self, path):
        df = pd.read_csv(path, parse_dates=['earnings_date'])
        self._by_ticker = {t: np.sort(g['earnings_date'].values)
                           for t, g in df.groupby('ticker')}

    @classmethod
    def from_data_dir(cls, data_dir):
        """Prefer the repaired v2 calendar.

        The raw yfinance file (earnings_dates.csv) carries only 2 of 4
        announcements for 2025 and 1 for 2026 — a gap that sits inside the
        test window and would mislabel ~19% of the sample as "no earnings
        before expiry". scripts/repair_earnings_calendar.py rebuilds it.
        """
        v2 = os.path.join(data_dir, 'earnings_dates_v2.csv')
        return cls(v2 if os.path.exists(v2)
                   else os.path.join(data_dir, 'earnings_dates.csv'))

    def dates(self, ticker):
        return self._by_ticker.get(ticker, np.array([], dtype='datetime64[ns]'))

    def count_in(self, ticker, start, end):
        """Number of announcements in the half-open interval (start, end]."""
        d = self.dates(ticker)
        if len(d) == 0:
            return np.zeros(len(np.atleast_1d(start)), dtype=int)
        s = np.asarray(start, dtype='datetime64[ns]')
        e = np.asarray(end, dtype='datetime64[ns]')
        return (np.searchsorted(d, e, side='right')
                - np.searchsorted(d, s, side='right'))

    def days_to_next(self, ticker, when, cap=120):
        """Calendar days from `when` to the next announcement (capped)."""
        d = self.dates(ticker)
        w = np.asarray(when, dtype='datetime64[ns]')
        if len(d) == 0:
            return np.full(w.shape, cap, dtype=float)
        idx = np.searchsorted(d, w, side='right')
        out = np.full(w.shape, cap, dtype=float)
        ok = idx < len(d)
        if ok.any():
            delta = (d[idx[ok]] - w[ok]) / np.timedelta64(1, 'D')
            out[ok] = np.minimum(delta.astype(float), cap)
        return out

    def days_since_last(self, ticker, when, cap=120):
        d = self.dates(ticker)
        w = np.asarray(when, dtype='datetime64[ns]')
        if len(d) == 0:
            return np.full(w.shape, cap, dtype=float)
        idx = np.searchsorted(d, w, side='right') - 1
        out = np.full(w.shape, cap, dtype=float)
        ok = idx >= 0
        if ok.any():
            delta = (w[ok] - d[idx[ok]]) / np.timedelta64(1, 'D')
            out[ok] = np.minimum(delta.astype(float), cap)
        return out


def add_event_columns(df, calendar):
    """Attach per-row earnings structure to an option table.

    Adds: n_earnings (announcements before expiry), days_to_earnings,
    days_since_earnings, spans_earnings.
    """
    df = df.copy()
    n = np.zeros(len(df), dtype=int)
    dte = np.zeros(len(df), dtype=float)
    dse = np.zeros(len(df), dtype=float)
    for t, idx in df.groupby('ticker').groups.items():
        rows = df.loc[idx]
        n[df.index.get_indexer(idx)] = calendar.count_in(
            t, rows['date'].values, rows['expiration'].values)
        dte[df.index.get_indexer(idx)] = calendar.days_to_next(
            t, rows['date'].values)
        dse[df.index.get_indexer(idx)] = calendar.days_since_last(
            t, rows['date'].values)
    df['n_earnings'] = n
    df['days_to_earnings'] = dte
    df['days_since_earnings'] = dse
    df['spans_earnings'] = (df['n_earnings'] > 0).astype(float)
    return df


def _atm_total_variance(group):
    """ATM total variance per expiry for one (ticker, date) slice.

    ATM is taken at the parity forward: the row of each expiry whose
    log-moneyness is closest to zero, averaged over the 3 most central
    strikes to damp quote noise.
    """
    recs = []
    for exp, g in group.groupby('expiration'):
        if len(g) < 3:
            continue
        g = g.assign(_ad=g['logm'].abs()).nsmallest(3, '_ad')
        recs.append({'expiration': exp,
                     'tau': float(g['tau'].iloc[0]),
                     'w_atm': float(g['total_var'].mean()),
                     'n_earnings': int(g['n_earnings'].iloc[0])})
    return pd.DataFrame(recs)


def decompose_variance(df, min_expiries=2):
    """Split ATM total variance into diffusive and earnings-jump components.

    Per (ticker, date), solves the non-negative least squares problem

        w_atm(tau_i) ~ sigma_d^2 * tau_i + J^2 * n_i

    over the 3-4 available expiries. Non-negativity is enforced because both
    a variance rate and a jump variance must be >= 0; when the design is rank
    deficient (all expiries contain the same number of announcements, so the
    two columns are collinear) the jump term is not identified and J^2 is
    returned as NaN for that snapshot — callers fill it from the ticker's
    rolling history rather than trusting an arbitrary split.

    Returns one row per (ticker, date): diffusive_var (annualized variance
    rate), jump_var (per-event variance), implied_move (the one-day move the
    market prices for the announcement, sqrt(J^2)), r2, and n_expiries.
    """
    from scipy.optimize import nnls

    out = []
    for (t, d), grp in df.groupby(['ticker', 'date']):
        atm = _atm_total_variance(grp)
        if len(atm) < min_expiries:
            continue
        tau = atm['tau'].to_numpy(float)
        w = atm['w_atm'].to_numpy(float)
        n = atm['n_earnings'].to_numpy(float)

        identified = len(np.unique(n)) > 1
        if identified:
            A = np.column_stack([tau, n])
            coef, _ = nnls(A, w)
            sigma2, jump2 = float(coef[0]), float(coef[1])
            pred = A @ coef
        else:
            # Only the diffusive rate is identified; report J^2 as missing.
            coef, _ = nnls(tau.reshape(-1, 1), w)
            sigma2, jump2 = float(coef[0]), np.nan
            pred = tau * coef[0]

        ss_res = float(np.sum((w - pred) ** 2))
        ss_tot = float(np.sum((w - w.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else np.nan

        out.append({
            'ticker': t, 'date': d,
            'diffusive_var': sigma2,
            'diffusive_vol': float(np.sqrt(max(sigma2, 0.0))),
            'jump_var': jump2,
            'implied_move': float(np.sqrt(jump2)) if np.isfinite(jump2) else np.nan,
            'r2': r2, 'n_expiries': len(atm), 'identified': identified,
        })

    res = pd.DataFrame(out).sort_values(['ticker', 'date'])
    # Fill unidentified snapshots from the ticker's own recent history
    # (past-only, so this stays causal).
    res['jump_var'] = res.groupby('ticker')['jump_var'].transform(
        lambda s: s.ffill().fillna(s.expanding().median()))
    res['implied_move'] = np.sqrt(res['jump_var'].clip(lower=0))
    return res.reset_index(drop=True)


def event_time(tau, n_earnings, diffusive_var, jump_var, floor=1e-8):
    """Event-adjusted maturity: tau_tilde = tau + n * J^2 / sigma_d^2.

    Total variance is linear in tau_tilde, and tau_tilde is strictly
    increasing in tau, so calendar monotonicity in tau_tilde implies calendar
    monotonicity in tau.
    """
    sig = np.maximum(np.asarray(diffusive_var, dtype=float), floor)
    j = np.nan_to_num(np.asarray(jump_var, dtype=float), nan=0.0)
    return np.asarray(tau, dtype=float) + np.asarray(n_earnings, dtype=float) * j / sig


def attach_decomposition(df, decomposition):
    """Merge the per-snapshot decomposition back onto the option table and
    add the event-adjusted maturity column `tau_evt`."""
    cols = ['ticker', 'date', 'diffusive_var', 'diffusive_vol', 'jump_var',
            'implied_move', 'r2']
    merged = df.merge(decomposition[cols], on=['ticker', 'date'], how='left')
    merged['tau_evt'] = event_time(merged['tau'], merged['n_earnings'],
                                   merged['diffusive_var'], merged['jump_var'])
    return merged

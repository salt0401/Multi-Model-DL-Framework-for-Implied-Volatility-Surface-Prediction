"""Vol-timing strategy prototype on Mag 7: flow-forecast-driven ATM straddles.

Signal (per ticker, per snapshot t in the TEST period): the flow model's mean
forecast of the 30d ATM IV at the next snapshot minus today's 30d ATM IV.
If |signal| > theta (theta = 1 std of the signal on TRAIN pairs), enter a
delta-hedged ATM straddle in the expiration nearest 30d: long vol if the
forecast is up, short if down. Exit at the next snapshot.

Execution realism and its limits — stated bluntly:
- Fills at MID; costs = HALF the quoted bid-ask spread per leg per side
  (entry + exit), from the real quotes in the data.
- Delta hedge approximated once at entry with provider deltas
  (PnL -= entry_straddle_delta * spot move); no intra-period re-hedge.
- No early-exercise/assignment modeling, no borrow/financing costs, no
  slippage beyond half-spread, mid may be stale on wide markets.
- Trades are skipped when the exact contracts are missing at exit
  (skips are counted and reported).
"""
from us_dataset import UsOptionsProcessor, build_us_pairs, MAG7, DAYS_PER_YEAR
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          sample_flow, make_ema_model)
from utils import load_config, parse_date, set_seed, setup_logging

from argparse import ArgumentParser

import json
import numpy as np
import pandas as pd
import torch
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _atm_iv30_from_grid(S, tau_grid, logm_grid):
    """(N, D) surfaces -> 30d ATM implied vol via the grid cell nearest
    (tau=30/365.25, logm=0)."""
    n_tau, n_logm = len(tau_grid), len(logm_grid)
    W = S.reshape(-1, n_tau, n_logm)
    i30 = int(np.argmin(np.abs(tau_grid - 30.0 / DAYS_PER_YEAR)))
    j0 = int(np.argmin(np.abs(logm_grid)))
    w30 = W[:, i30, j0]
    return np.sqrt(np.clip(w30, 1e-12, None) / tau_grid[i30])


def _forecast_iv30(model, pp, cs, ics, splits_part, tau_grid, logm_grid,
                   device, dtype, n_samples, n_steps, seed):
    Z_today = pp.transform(splits_part['S_today'])
    cond = np.concatenate([Z_today, cs.transform(splits_part['C'])], axis=1)
    g = torch.Generator(device=device.type).manual_seed(seed)
    raw = sample_flow(model, torch.as_tensor(cond, dtype=dtype, device=device),
                      n_steps=n_steps, n_samples=n_samples,
                      generator=g).cpu().numpy()
    inc_mean = ics.inverse(raw.mean(axis=0))
    S_pred = pp.inverse(Z_today + inc_mean)
    return _atm_iv30_from_grid(S_pred, tau_grid, logm_grid)


def _straddle_quotes(day_chain, spot):
    """Nearest-30d ATM straddle legs for one (ticker, date) chain slice.

    Returns dict with expiration, strike, call/put mids, spreads, deltas —
    or None if either leg is missing.
    """
    if day_chain.empty:
        return None
    tau_days = (day_chain['expiration'] - day_chain['date']).dt.days
    exp = day_chain.loc[(tau_days - 30).abs().idxmin(), 'expiration']
    ch = day_chain[day_chain['expiration'] == exp]
    strike = ch.loc[(ch['strike'] - spot).abs().idxmin(), 'strike']
    legs = {}
    for cp in ('Call', 'Put'):
        leg = ch[(ch['strike'] == strike) & (ch['call_put'] == cp)]
        if leg.empty:
            return None
        r = leg.iloc[0]
        legs[cp] = {'mid': (r['bid'] + r['ask']) / 2,
                    'spread': r['ask'] - r['bid'], 'delta': r['delta']}
    return {'expiration': exp, 'strike': strike, **legs}


def run_ticker(ticker, table, proc, config, device, dtype, logger,
               n_samples=64):
    cfg = config['flow_us']
    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])

    ckpt = torch.load(f'../models/FlowSurface_{ticker}.pt',
                      map_location=device, weights_only=True)
    pp = FactorPreprocessor.from_dict(ckpt['preprocessor'])
    cs = CondScaler.from_dict(ckpt['cond_scaler'])
    ics = CondScaler.from_dict(ckpt['increment_scaler'])
    tau_grid = np.asarray(ckpt['tau_grid'])
    logm_grid = np.asarray(ckpt['logm_grid'])
    model = VelocityMLP(**ckpt['model_kwargs']).to(device)
    model = make_ema_model(model, ckpt)
    model.eval()
    n_steps = ckpt.get('n_sample_steps', 50)

    panel = proc.prepare_surface_panel(table, ticker, train_end,
                                       n_logm=cfg.getint('n_logm'))
    splits = build_us_pairs(panel, train_end, test_start)

    # Threshold from TRAIN-period signals (no test leakage)
    iv30_train_fc = _forecast_iv30(model, pp, cs, ics, splits['train'],
                                   tau_grid, logm_grid, device, dtype,
                                   n_samples=32, n_steps=n_steps, seed=11)
    iv30_train_now = _atm_iv30_from_grid(splits['train']['S_today'],
                                         tau_grid, logm_grid)
    theta = float(np.std(iv30_train_fc - iv30_train_now))

    # Test signals
    iv30_fc = _forecast_iv30(model, pp, cs, ics, splits['test'], tau_grid,
                             logm_grid, device, dtype, n_samples=n_samples,
                             n_steps=n_steps, seed=13)
    iv30_now = _atm_iv30_from_grid(splits['test']['S_today'],
                                   tau_grid, logm_grid)
    signals = iv30_fc - iv30_now

    sub = table[table['ticker'] == ticker]
    date_index = {d: i for i, d in enumerate(panel['dates'])}
    trades = []
    skipped = 0
    for i, d in enumerate(splits['test']['dates']):
        sig = signals[i]
        if abs(sig) <= theta:
            continue
        d_next = panel['dates'][date_index[d] + 1]
        chain_t = sub[sub['date'] == d]
        chain_n = sub[sub['date'] == d_next]
        if chain_t.empty or chain_n.empty:
            skipped += 1
            continue
        spot_t = chain_t['spot_syn'].iloc[0]
        spot_n = chain_n['spot_syn'].iloc[0]
        entry = _straddle_quotes(chain_t, spot_t)
        if entry is None:
            skipped += 1
            continue
        exit_ch = chain_n[(chain_n['expiration'] == entry['expiration'])
                          & (chain_n['strike'] == entry['strike'])]
        legs_exit = {}
        for cp in ('Call', 'Put'):
            leg = exit_ch[exit_ch['call_put'] == cp]
            if leg.empty:
                break
            r = leg.iloc[0]
            legs_exit[cp] = {'mid': (r['bid'] + r['ask']) / 2,
                             'spread': r['ask'] - r['bid']}
        if len(legs_exit) < 2:
            skipped += 1
            continue

        direction = 1.0 if sig > 0 else -1.0
        entry_prem = entry['Call']['mid'] + entry['Put']['mid']
        exit_prem = legs_exit['Call']['mid'] + legs_exit['Put']['mid']
        strad_delta = entry['Call']['delta'] + entry['Put']['delta']
        pnl_gross = direction * ((exit_prem - entry_prem)
                                 - strad_delta * (spot_n - spot_t))
        cost = 0.5 * (entry['Call']['spread'] + entry['Put']['spread']
                      + legs_exit['Call']['spread'] + legs_exit['Put']['spread'])
        trades.append({'date': str(d.date()), 'direction': direction,
                       'ret_gross': pnl_gross / entry_prem,
                       'ret_net': (pnl_gross - cost) / entry_prem})

    tdf = pd.DataFrame(trades)
    out = {'ticker': ticker, 'theta': theta, 'n_signals': int(len(signals)),
           'n_trades': int(len(tdf)), 'n_skipped': skipped}
    if len(tdf):
        snaps_per_year = 365.25 / np.mean(np.diff(
            [d for d in panel['dates']]).astype('timedelta64[D]').astype(float))
        for kind in ('gross', 'net'):
            r = tdf[f'ret_{kind}']
            sharpe = (r.mean() / r.std() * np.sqrt(snaps_per_year)
                      if r.std() > 0 else 0.0)
            out[kind] = {'mean_ret_per_trade': float(r.mean()),
                         'hit_rate': float((r > 0).mean()),
                         'total_ret': float(r.sum()),
                         'sharpe_annualized': float(sharpe)}
        out['trades'] = trades
    logger.info(f"{ticker}: {out['n_trades']} trades "
                f"(skipped {skipped}), net {out.get('net', {})}")
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    args = parser.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'strategy_us')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.on_gpu
                          else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    proc = UsOptionsProcessor(config['data_us']['data_dir'])
    table = proc.build()

    results = []
    for t in MAG7:
        try:
            results.append(run_ticker(t, table, proc, config, device, dtype,
                                      logger))
        except FileNotFoundError:
            logger.info(f'{t}: no checkpoint, skipped')

    pooled_net = [tr['ret_net'] for r in results for tr in r.get('trades', [])]
    pooled = {}
    if pooled_net:
        r = pd.Series(pooled_net)
        pooled = {'n_trades': len(r), 'mean_ret_per_trade': float(r.mean()),
                  'hit_rate': float((r > 0).mean()),
                  'total_ret': float(r.sum()),
                  'sharpe_annualized_per_trade_series':
                      float(r.mean() / r.std() * np.sqrt(156)) if r.std() > 0 else 0.0}

    with open(os.path.join(log_dir, 'strategy_us_results.json'), 'w') as f:
        json.dump({'per_ticker': [{k: v for k, v in r.items() if k != 'trades'}
                                  for r in results],
                   'pooled_net': pooled}, f, indent=2)

    # PnL curves (net, cumulative per-trade returns by date, pooled)
    all_tr = [dict(tr, ticker=r['ticker']) for r in results
              for tr in r.get('trades', [])]
    if all_tr:
        adf = pd.DataFrame(all_tr).sort_values('date')
        adf['cum_net'] = adf['ret_net'].cumsum()
        adf['cum_gross'] = adf['ret_gross'].cumsum()
        fig, ax = plt.subplots(figsize=(10, 5))
        x = pd.to_datetime(adf['date'])
        ax.plot(x, adf['cum_gross'], label='gross (mid fills)')
        ax.plot(x, adf['cum_net'], label='net of half-spread costs')
        ax.axhline(0, color='k', lw=0.7)
        ax.set_ylabel('cumulative return (per unit straddle premium)')
        ax.set_title('Mag 7 pooled straddle-timing strategy (test period)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'strategy_us_pnl.png'), dpi=110)
        plt.close(fig)

    logger.info('Strategy backtest complete.')


if __name__ == '__main__':
    main()

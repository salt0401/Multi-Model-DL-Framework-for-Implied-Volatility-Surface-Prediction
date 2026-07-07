"""Download Mag 7 EOD option chains from the DoltHub post-no-preference/options
free SQL API, plus split-UNADJUSTED spots / dividends / VIX via yfinance.

Empirically verified properties of the source (2026-07-06):
- No auth; (date, act_symbol) point queries return a full chain in one page
  (~140 rows; paginate defensively if a page returns exactly PAGE_CAP rows).
- Snapshots ~Mon/Wed/Fri in recent years, weekly Saturdays in the 2019 era —
  the true calendar is discovered by probing AAPL on every calendar date.
- META traded as FB before 2022-06: both symbols fetched, FB mapped to META
  downstream (kept separate on disk for provenance).

Resume-safe: dataset/us_options/manifest.json records every (ticker, date)
already fetched (including empties), so re-running only fills gaps.
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

API = 'https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master'
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'us_options')
MANIFEST = os.path.join(OUT_DIR, 'manifest.json')
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'FB']
START = dt.date(2019, 1, 1)
PAGE_CAP = 200          # defensive pagination threshold
MAX_WORKERS = 3
RETRIES = 4


def _query(sql, timeout=60):
    url = API + '?q=' + urllib.parse.quote(sql)
    last_err = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
            if data.get('query_execution_status') in ('Success', 'RowLimit'):
                return data.get('rows', [])
            last_err = data.get('query_execution_message', 'unknown API error')
        except Exception as e:  # noqa: BLE001 - network layer, retry then surface
            last_err = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f'query failed after {RETRIES} tries: {last_err[:200]}')


def fetch_chain(ticker, date_str):
    """Full chain for one (ticker, date); paginates past PAGE_CAP."""
    rows, offset = [], 0
    while True:
        sql = (f"SELECT * FROM option_chain WHERE date='{date_str}' "
               f"AND act_symbol='{ticker}' ORDER BY expiration, strike, call_put "
               f"LIMIT {PAGE_CAP} OFFSET {offset}")
        page = _query(sql)
        rows.extend(page)
        if len(page) < PAGE_CAP:
            return rows
        offset += PAGE_CAP


def build_calendar(manifest):
    """Probe AAPL on every calendar date since START to find snapshot dates."""
    known = manifest.setdefault('calendar', {})
    end = dt.date.today()
    dates = [START + dt.timedelta(days=i) for i in range((end - START).days + 1)]
    todo = [d for d in dates if d.isoformat() not in known]
    print(f'Calendar probe: {len(todo)} dates to check ({len(known)} cached)')

    def probe(d):
        ds = d.isoformat()
        rows = _query("SELECT COUNT(*) c FROM option_chain "
                      f"WHERE date='{ds}' AND act_symbol='AAPL'")
        return ds, int(rows[0]['c']) if rows else 0

    done = 0
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool:
        for ds, count in pool.map(probe, todo):
            known[ds] = count
            done += 1
            if done % 200 == 0:
                print(f'  probed {done}/{len(todo)}')
                save_manifest(manifest)
    save_manifest(manifest)
    active = sorted(ds for ds, c in known.items() if c > 0)
    print(f'Calendar: {len(active)} snapshot dates '
          f'({active[0]} .. {active[-1]})' if active else 'Calendar: EMPTY')
    return active


def save_manifest(manifest):
    tmp = MANIFEST + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f)
    os.replace(tmp, MANIFEST)


def fetch_all_chains(manifest, calendar):
    fetched = manifest.setdefault('fetched', {})
    jobs = [(t, ds) for t in TICKERS for ds in calendar
            if f'{t}|{ds}' not in fetched]
    print(f'Chain fetch: {len(jobs)} (ticker, date) jobs remaining')

    buffers = {t: [] for t in TICKERS}

    def flush(ticker):
        if not buffers[ticker]:
            return
        path = os.path.join(OUT_DIR, f'chains_{ticker}.csv')
        df = pd.DataFrame(buffers[ticker])
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        buffers[ticker] = []

    def job(args):
        ticker, ds = args
        return ticker, ds, fetch_chain(ticker, ds)

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(MAX_WORKERS) as pool:
        for ticker, ds, rows in pool.map(job, jobs):
            buffers[ticker].extend(rows)
            fetched[f'{ticker}|{ds}'] = len(rows)
            done += 1
            if done % 100 == 0:
                for t in TICKERS:
                    flush(t)
                save_manifest(manifest)
                rate = done / (time.time() - t0)
                eta = (len(jobs) - done) / max(rate, 1e-9) / 60
                print(f'  {done}/{len(jobs)} chains ({rate:.1f}/s, ETA {eta:.0f} min)')
    for t in TICKERS:
        flush(t)
    save_manifest(manifest)


def fetch_spots():
    """Split-UNADJUSTED closes (match as-traded strikes) + dividends + VIX/IRX."""
    import yfinance as yf
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    frames = []
    for sym in symbols:
        h = yf.Ticker(sym).history(start=str(START), auto_adjust=False)
        h = h.reset_index()[['Date', 'Close', 'Dividends']]
        h.columns = ['date', 'close_unadj', 'dividend']
        h['ticker'] = sym
        h['date'] = pd.to_datetime(h['date']).dt.tz_localize(None)
        frames.append(h)
        print(f'  {sym}: {len(h)} spot rows')
    pd.concat(frames).to_csv(os.path.join(OUT_DIR, 'spots.csv'), index=False)

    for sym, name in [('^VIX', 'vix_us.csv'), ('^IRX', 'irx.csv')]:
        h = yf.Ticker(sym).history(start=str(START), auto_adjust=False)
        h = h.reset_index()[['Date', 'Close']]
        h.columns = ['date', 'Close']
        h['date'] = pd.to_datetime(h['date']).dt.tz_localize(None)
        h.to_csv(os.path.join(OUT_DIR, name), index=False)
        print(f'  {sym}: {len(h)} rows')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            manifest = json.load(f)

    if '--spots-only' not in sys.argv:
        calendar = build_calendar(manifest)
        fetch_all_chains(manifest, calendar)
    if '--chains-only' not in sys.argv:
        fetch_spots()
    print('Done.')


if __name__ == '__main__':
    main()

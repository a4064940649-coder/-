#!/usr/bin/env python3
"""raw/ 各來源讀成統一長表（只當 CSV 讀；視為不可信）。改寫自 o_v6 srcload.py，路徑改為本題 raw/，另加市場別證據來源。
價格長表欄位：date, code, open, high, low, close, vol（股）, value（元；沒有就 NaN）, chg（漲跌價差；沒有就 NaN）。
成交量 0 或價格 ≤ 0 的列視為沒有成交。快取到 data/src_<name>.parquet。"""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v7common as C

COLS = ['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'value', 'chg']
A0, B0 = pd.Timestamp('2019-01-01'), pd.Timestamp('2026-12-31')

def _clean(d):
    d = d[COLS].copy()
    d['code'] = d['code'].astype(str).str.strip()
    for c in COLS[2:]: d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[(d['vol'] > 0) & (d['close'] > 0) & (d['open'] > 0)]
    d = d[(d.date >= A0) & (d.date <= B0)]
    d = d.drop_duplicates(['date', 'code'], keep='last')
    return d.sort_values(['date', 'code']).reset_index(drop=True)

def hschi():
    out = []
    for f in sorted(glob.glob(str(C.RAW / 'hschi' / 'stock_dayK' / '*.csv'))):
        x = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
        if 'stock_id' not in x.columns: continue
        x = x.rename(columns={'stock_id': 'code', 'max': 'high', 'min': 'low', 'Trading_Volume': 'vol', 'Trading_money': 'value', 'spread': 'chg'})
        x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    return _clean(pd.concat(out, ignore_index=True))

def tasi():
    out = []
    for f in sorted((C.TASI / 'kbar').glob('kbar*.csv')):
        h = f.open('rb').readline().strip()
        if h != b'"stoc","date","max","min","ope","clo","vol"': continue
        x = pd.read_csv(f, dtype={'stoc': str})
        x = x.rename(columns={'stoc': 'code', 'max': 'high', 'min': 'low', 'ope': 'open', 'clo': 'close'})
        x['date'] = pd.to_datetime(x['date'])
        x = x[x['date'] == pd.Timestamp(f.name[4:14])]
        out.append(x)
    d = pd.concat(out, ignore_index=True)
    d['vol'] = pd.to_numeric(d['vol'], errors='coerce') * 1000.0      # 張 → 股
    d['value'] = np.nan; d['chg'] = np.nan
    return _clean(d)

def bacardi(market='TWSE'):
    out = []
    for f in sorted((C.RAW / 'bacardi').glob('prices_*.csv')):
        x = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
        x = x[x['market'] == market]
        x = x.rename(columns={'stock_id': 'code', 'volume': 'vol', 'turnover': 'value', 'change': 'chg'})
        x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    return _clean(pd.concat(out, ignore_index=True))

def history():
    x = pd.read_csv(C.HIST, dtype={'code': str}, parse_dates=['date']).rename(columns={'volume': 'vol'})
    x['value'] = np.nan; x['chg'] = np.nan
    return _clean(x)

def leochen():
    out = []
    for f in sorted((C.RAW / 'leochen1_stock_v' / 'data' / 'history').glob('*.csv')):
        x = pd.read_csv(f, encoding='utf-8-sig')
        x.columns = ['date', 'open', 'high', 'low', 'close', 'vol']
        x['code'] = f.stem; x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    d = pd.concat(out, ignore_index=True); d['value'] = np.nan; d['chg'] = np.nan
    return _clean(d)

def tasi_pbr():
    """Tasi 淨值（證交所／櫃買每日本益比、淨值比檔；檔名 YYYYMMDD市淨值.csv＝上市、櫃淨值＝上櫃）→ (date, code, mkt)。只當市場別證據（名單不完整）。"""
    out = []
    for f in sorted((C.TASI / '淨值').glob('*.csv')):
        m = 'TWSE' if '市淨值' in f.name else ('TPEX' if '櫃淨值' in f.name else None)
        if m is None: continue
        try:
            x = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
        except Exception:
            continue
        col = [c for c in x.columns if '代號' in c]
        if not col: continue
        out.append(pd.DataFrame({'date': pd.Timestamp(f.name[:8]), 'code': x[col[0]].astype(str).str.strip(), 'mkt': m}))
    return pd.concat(out, ignore_index=True).drop_duplicates()

def isin_snaps():
    """GitHub 封存的 ISIN（isin.twse.com.tw C_public mode 2＝上市、mode 4＝上櫃）快照 → (version, code, name, group, market, listed)。"""
    rows = []
    for f in sorted((C.RAW / 'isin' / 'snap').glob('isin_0*.txt')):
        d = json.load(open(f, encoding='utf-8'))
        for g in d['list']:
            for x in g['list']:
                rows.append((pd.Timestamp(d['version'].replace('/', '-')), x['code'], x['name'], g['name'], x['market'], x['listedDate'], f.name))
    return pd.DataFrame(rows, columns=['version', 'code', 'name', 'group', 'market', 'listed', 'file'])

def tasi_rev():
    """Tasi 營收（MOPS 月營收彙總檔，每月一檔 YYYYMM.csv）：「全部國內上市公司合計」之前的列＝上市、之後到「全部國內上櫃公司合計」＝上櫃 → (ym, code, mkt, name)。"""
    out = []
    for f in sorted((C.TASI / '營收').glob('*.csv')):
        lines = f.read_text(encoding='utf-8-sig', errors='replace').splitlines()[1:]
        mkt = 'TWSE'
        for ln in lines:
            a = ln.split(',')
            if not a or not a[0]: continue
            if a[0].startswith('全部國內上市公司合計'): mkt = 'TPEX'; continue
            if a[0].startswith('全部國內上櫃公司合計'): break
            out.append((f.stem, a[0].strip(), mkt, a[1].strip() if len(a) > 1 else ''))
    d = pd.DataFrame(out, columns=['ym', 'code', 'mkt', 'name'])
    # 檔案加入 git 的日期（raw/tasi_file_add_dates.csv）：月檔在次月 45 天內加入才算當時的快照（PIT）；2024-06-14 一次補入的 2021-01～2024-05 不是 PIT
    ad = pd.read_csv(C.RAW / 'tasi_file_add_dates.csv', header=None, names=['f', 'added'])
    ad = ad[ad.f.str.startswith('營收/')].assign(ym=lambda x: x.f.str[3:9]).groupby('ym').added.min()
    d['added'] = pd.to_datetime(d.ym.map(ad))
    end = (pd.PeriodIndex(d.ym, freq='M') + 1).to_timestamp()
    d['pit'] = (d.added - end).dt.days.between(-1, 45)
    return d

LOADERS = {'hschi': hschi, 'tasi': tasi, 'bacardi': lambda: bacardi('TWSE'), 'bacardi_tpex': lambda: bacardi('TPEx'),
           'leochen': leochen, 'history': history, 'tasi_pbr': tasi_pbr, 'isin': isin_snaps, 'tasi_rev': tasi_rev}

def get(name, refresh=False):
    f = C.DATA / f'src_{name}.parquet'
    if f.exists() and not refresh: return pd.read_parquet(f)
    d = LOADERS[name]()
    C.DATA.mkdir(exist_ok=True); d.to_parquet(f)
    return d

if __name__ == '__main__':
    for k in LOADERS:
        d = get(k, refresh=True)
        print(k, len(d), d.iloc[:, 0].min(), d.iloc[:, 0].max(), d['code'].nunique(), flush=True)

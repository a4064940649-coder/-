#!/usr/bin/env python3
"""把 raw/ 下的各官方來源讀成統一長表（只當 CSV 讀；視為不可信）：
欄位 date, code, open, high, low, close, vol（股）, value（元；沒有就 NaN）, chg（漲跌價差＝收盤−參考價；沒有就 NaN）。
成交量 0 或價格 ≤ 0 的列視為當日沒有成交（丟掉）。快取到 data/src_<name>.parquet（不是面板，只是解析結果）。"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import v6common as C

COLS = ['date', 'code', 'open', 'high', 'low', 'close', 'vol', 'value', 'chg']
A0, B0 = pd.Timestamp('2019-01-01'), pd.Timestamp('2026-12-31')

def _clean(d, name):
    d = d[COLS].copy()
    d['code'] = d['code'].astype(str).str.strip()
    for c in COLS[2:]: d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[(d['vol'] > 0) & (d['close'] > 0) & (d['open'] > 0)]
    d = d[(d.date >= A0) & (d.date <= B0)]
    n0 = len(d)
    d = d.drop_duplicates(['date', 'code'], keep='last')
    d.attrs['n_dup_dropped'] = n0 - len(d)
    return d.sort_values(['date', 'code']).reset_index(drop=True)

def finmind_dir(path, pattern='*.csv', name='finmind'):
    out = []
    for f in sorted(glob.glob(str(Path(path) / pattern))):
        x = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
        if 'stock_id' not in x.columns: continue
        x = x.rename(columns={'stock_id': 'code', 'max': 'high', 'min': 'low', 'Trading_Volume': 'vol', 'Trading_money': 'value', 'spread': 'chg'})
        x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    return _clean(pd.concat(out, ignore_index=True), name)

def tasi():
    out = []; skipped = []
    for f in sorted(C.TASI.glob('kbar*.csv')):
        h = f.open('rb').readline().strip()
        if h != b'"stoc","date","max","min","ope","clo","vol"':
            skipped.append(f.name); continue          # 空檔（""）或表頭毀損
        x = pd.read_csv(f, dtype={'stoc': str})
        x = x.rename(columns={'stoc': 'code', 'max': 'high', 'min': 'low', 'ope': 'open', 'clo': 'close'})
        x['date'] = pd.to_datetime(x['date'])
        x['file_date'] = pd.Timestamp(f.name[4:14])
        out.append(x)
    d = pd.concat(out, ignore_index=True)
    bad = int((d['date'] != d['file_date']).sum())
    d = d[d['date'] == d['file_date']]
    d['vol'] = pd.to_numeric(d['vol'], errors='coerce') * 1000.0      # 張 → 股（單位於 explore 交叉驗證）
    d['value'] = np.nan; d['chg'] = np.nan
    r = _clean(d, 'tasi'); r.attrs['n_filedate_mismatch'] = bad
    r.attrs['n_files_skipped'] = len(skipped)
    (C.DATA / 'tasi_skipped_files.txt').write_text('\n'.join(skipped))
    return r

def bacardi():
    out = []
    for f in sorted((C.RAW / 'bacardi').glob('prices_*.csv')):
        x = pd.read_csv(f, dtype={'stock_id': str}, encoding='utf-8-sig')
        x = x[x['market'] == 'TWSE']
        x = x.rename(columns={'stock_id': 'code', 'volume': 'vol', 'turnover': 'value', 'change': 'chg'})
        x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    return _clean(pd.concat(out, ignore_index=True), 'bacardi')

def history():
    x = pd.read_csv(C.HIST, dtype={'code': str}, parse_dates=['date'])
    x = x.rename(columns={'volume': 'vol'})
    x['value'] = np.nan; x['chg'] = np.nan
    return _clean(x, 'history')

def henry():
    p = C.HENRY
    fs = sorted(glob.glob(str(p / '**' / '*.csv'), recursive=True))
    out = []
    for f in fs:
        try:
            x = pd.read_csv(f, dtype={'stock_id': str})
        except Exception:
            continue
        if 'stock_id' not in x.columns or 'Trading_Volume' not in x.columns: continue
        x = x.rename(columns={'stock_id': 'code', 'max': 'high', 'min': 'low', 'Trading_Volume': 'vol', 'Trading_money': 'value', 'spread': 'chg'})
        x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    return _clean(pd.concat(out, ignore_index=True), 'henry') if out else None

def leochen():
    out = []
    for f in sorted((C.RAW / 'leochen1_stock_v' / 'data' / 'history').glob('*.csv')):
        x = pd.read_csv(f, encoding='utf-8-sig')
        x.columns = ['date', 'open', 'high', 'low', 'close', 'vol']
        x['code'] = f.stem; x['date'] = pd.to_datetime(x['date'])
        out.append(x)
    d = pd.concat(out, ignore_index=True)
    d['value'] = np.nan; d['chg'] = np.nan
    return _clean(d, 'leochen')

LOADERS = {'hschi': lambda: finmind_dir(C.RAW / 'hschi' / 'stock_dayK', name='hschi'),
           'hawan': lambda: finmind_dir(C.RAW / 'hawan' / 'prices', '*_price.csv', name='hawan'),
           'tasi': tasi, 'bacardi': bacardi, 'leochen': leochen, 'history': history, 'henry': henry}

def get(name, refresh=False):
    f = C.DATA / f'src_{name}.parquet'
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    d = LOADERS[name]()
    if d is None: return None
    d.to_parquet(f)
    return d

if __name__ == '__main__':
    for k in LOADERS:
        d = get(k, refresh=True)
        if d is None: print(k, 'none'); continue
        print(k, len(d), d.date.min().date(), d.date.max().date(), d.code.nunique(), d.attrs)

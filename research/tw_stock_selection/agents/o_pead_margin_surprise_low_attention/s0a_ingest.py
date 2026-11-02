#!/usr/bin/env python3
"""第 0 步 A：把不可信 raw 資料（GitHub 公開封存 jb7k6295r7-bot/tw-stock-data）轉成本題 data/ 的 parquet。
只讀 raw/，只寫本資料夾 data/。用法：python3 -I -B s0a_ingest.py [per|fin|daily]
- per：證交所 BWIBBU／櫃買每日本益比（2012-01 起）→ data/per_long.parquet
- fin：MOPS XBRL 彙整的單季財報（2013Q1 起）→ data/fin_long.parquet（Q4 單季＝全年−前三季累計）
- daily：上市＋上櫃官方日 K（2015-01 起）→ data/daily_long.parquet
"""
import sys, os, glob, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
RAW = Path('/home/user/-/research/tw_stock_selection/raw/jb7k6295r7-bot_tw-stock-data')
OUT = HERE / 'data'


def ingest_per():
    fr = []
    for sub, mkt in [('data/early/per', 'twse'), ('data/universe/per', 'twse'),
                     ('data/early/otcper', 'tpex'), ('data/universe/otcper', 'tpex')]:
        fs = sorted(glob.glob(str(RAW / sub / '*.csv')))
        for f in fs:
            try:
                d = pd.read_csv(f, dtype={'stock_id': str, 'fs_quarter': str, 'dividend_year': str})
            except Exception as e:  # 空檔
                continue
            if len(d) == 0:
                continue
            d['mkt'] = mkt
            fr.append(d[['date', 'stock_id', 'mkt', 'close', 'per', 'pbr', 'fs_quarter']])
    d = pd.concat(fr, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    for c in ['close', 'per', 'pbr']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.rename(columns={'stock_id': 'code'})
    d = d.drop_duplicates(['date', 'code', 'mkt'], keep='last')
    d.to_parquet(OUT / 'per_long.parquet', index=False)
    print('per rows', len(d), d['date'].min(), d['date'].max(), d.groupby('mkt').size().to_dict())


def ingest_fin():
    fs = sorted(glob.glob(str(RAW / 'data/mops/fin_hist/*.csv')))
    d = pd.concat([pd.read_csv(f, dtype={'stock_id': str}) for f in fs], ignore_index=True)
    d = d.rename(columns={'stock_id': 'code'})
    # 同一公司同一季有合併（cr）與個體（ir）報表時，用合併；er 最後
    pri = {'cr': 0, 'ir': 1, 'er': 2}
    d['_p'] = d['report'].map(pri).fillna(3)
    d = d.sort_values(['code', 'period', '_p']).drop_duplicates(['code', 'period'], keep='first').drop(columns='_p')
    d['y'] = d['period'].str[:4].astype(int); d['q'] = d['period'].str[-1].astype(int)
    d = d.sort_values(['code', 'y', 'q']).reset_index(drop=True)
    # 單季值：Q1＝累計；Q2～Q4＝本季累計−上季累計（同一年、上季存在才算）；原檔 *_q 有值者與推算值比對
    out = []
    for k in ['eps', 'rev', 'gp', 'opi', 'ni', 'nip']:
        ytd = d[f'{k}_ytd']
        prev = d.groupby(['code', 'y'])[f'{k}_ytd'].shift(1)
        prevq = d.groupby(['code', 'y'])['q'].shift(1)
        calc = np.where(d['q'] == 1, ytd, np.where(prevq == d['q'] - 1, ytd - prev, np.nan))
        d[f'{k}_calc'] = calc
        given = d[f'{k}_q']
        both = given.notna() & pd.notna(calc)
        if both.any():
            tol = 0.011 if k == 'eps' else 1.0
            agree = (np.abs(given[both] - d.loc[both, f'{k}_calc']) <= np.maximum(tol, 0.001 * np.abs(given[both]))).mean()
        else:
            agree = np.nan
        out.append((k, int(both.sum()), float(agree)))
        d[f'{k}_s'] = given.where(given.notna(), d[f'{k}_calc'])
    print('單季給定值 vs 累計差推算一致率', out)
    keep = ['code', 'period', 'y', 'q', 'industry', 'report', 'taxonomy', 'eps_s', 'eps_ytd', 'rev_s', 'gp_s', 'opi_s', 'ni_s', 'nip_s',
            'equity_parent', 'assets']
    d[keep].to_parquet(OUT / 'fin_long.parquet', index=False)
    json.dump({'single_q_consistency': out, 'rows': int(len(d))}, open(OUT / 'fin_ingest.json', 'w'), ensure_ascii=False, indent=1)
    print('fin rows', len(d))


def _git_blobs(sub):
    """從 raw 的 git pack 直接串流讀取（不寫工作目錄；磁碟空間不足時的作法）。"""
    import subprocess, io
    ls = subprocess.run(['git', '-C', str(RAW), 'ls-tree', 'HEAD', sub + '/'], capture_output=True, text=True, check=True).stdout
    items = [(l.split()[2], l.split('\t')[1]) for l in ls.strip().split('\n') if l.endswith('.csv')]
    p = subprocess.Popen(['git', '-C', str(RAW), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    for sha, path in items:
        p.stdin.write((sha + '\n').encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        if len(hdr) < 3 or hdr[1] != 'blob':
            continue
        n = int(hdr[2]); data = p.stdout.read(n); p.stdout.read(1)
        yield path, io.BytesIO(data)
    p.stdin.close(); p.wait()


def ingest_daily():
    fr = []
    for path, fh in _git_blobs('data/universe/daily'):
        try:
            d = pd.read_csv(fh, dtype={'stock_id': str}, usecols=['date', 'stock_id', 'market', 'open', 'high', 'low', 'close',
                                                                   'volume', 'amount', 'change', 'shares'])
        except Exception:
            continue
        d = d[d['stock_id'].astype(str).str.fullmatch(r'\d{4}')]
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount', 'change', 'shares']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.rename(columns={'stock_id': 'code', 'market': 'mkt'})
    d = d.drop_duplicates(['date', 'code'], keep='last')
    d.to_parquet(OUT / 'daily_long.parquet', index=False, compression='zstd')
    print('daily rows', len(d), d['date'].min(), d['date'].max(), d.groupby('mkt').size().to_dict())


def ingest_rev():
    fs = sorted(glob.glob(str(RAW / 'data/mops/revenue_hist/*.csv')))
    fr = []
    for f in fs:
        if f.endswith('_esb.csv'):
            continue
        d = pd.read_csv(f, dtype={'stock_id': str})
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d = d.rename(columns={'stock_id': 'code', 'period': 'ym', 'market': 'mkt', '當月營收': 'rev', '去年當月營收': 'rev_ly',
                          '去年同月增減(%)': 'yoy_pct', '產業別': 'ind'})
    for c in ['rev', 'rev_ly', 'yoy_pct']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d[['code', 'ym', 'mkt', 'ind', 'rev', 'rev_ly', 'yoy_pct']].drop_duplicates(['code', 'ym'], keep='last')
    d.to_parquet(OUT / 'rev_long.parquet', index=False)
    print('rev rows', len(d), d['ym'].min(), d['ym'].max(), d.groupby('mkt').size().to_dict())


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    for a in sys.argv[1:]:
        {'per': ingest_per, 'fin': ingest_fin, 'daily': ingest_daily, 'rev': ingest_rev}[a]()

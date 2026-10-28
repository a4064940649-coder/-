#!/usr/bin/env python3
"""複製員 o6：把各原始來源讀成長表（date, code, o, h, l, c, vol, money），只留 2023-12-01～2025-10-31。
不讀研究員程式；只讀 raw/ 原始檔。輸出 replicate_o6/data/src_<name>.parquet。"""
from pathlib import Path
import numpy as np, pandas as pd, glob, os

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'raw'
OUT = HERE / 'data'; OUT.mkdir(exist_ok=True)
A, B = '2023-12-01', '2025-10-31'

def fin(df, name):
    df = df[(df['date'] >= A) & (df['date'] <= B)].copy()
    df['date'] = pd.to_datetime(df['date'])
    for c in ('o', 'h', 'l', 'c', 'vol', 'money'):
        if c in df: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.drop_duplicates(['date', 'code'], keep='last').reset_index(drop=True)
    df.to_parquet(OUT / f'src_{name}.parquet')
    print(name, len(df), df['date'].min(), df['date'].max(), df['code'].nunique())

def hschi():
    out = []
    for f in sorted(glob.glob(str(RAW / 'hschi' / 'stock_dayK' / '*.csv'))):
        d = pd.read_csv(f, dtype={'stock_id': str, 'date': str}, encoding='utf-8-sig')
        d = d.rename(columns={'stock_id': 'code', 'open': 'o', 'max': 'h', 'min': 'l', 'close': 'c', 'Trading_Volume': 'vol', 'Trading_money': 'money'})
        out.append(d[['date', 'code', 'o', 'h', 'l', 'c', 'vol', 'money']])
    fin(pd.concat(out), 'hschi')

def bacardi():
    out = []
    for f in sorted(glob.glob(str(RAW / 'bacardi' / 'prices_*.csv'))):
        d = pd.read_csv(f, dtype={'stock_id': str, 'date': str}, encoding='utf-8-sig')
        d = d.rename(columns={'stock_id': 'code', 'open': 'o', 'high': 'h', 'low': 'l', 'close': 'c', 'volume': 'vol', 'turnover': 'money'})
        out.append(d[['date', 'code', 'market', 'o', 'h', 'l', 'c', 'vol', 'money']])
    fin(pd.concat(out), 'bacardi')

def leochen():
    out = []
    for f in sorted(glob.glob(str(RAW / 'leochen1_stock_v' / 'data' / 'history' / '*.csv'))):
        d = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
        d.columns = ['date', 'o', 'h', 'l', 'c', 'vol']
        d['code'] = Path(f).stem
        out.append(d)
    fin(pd.concat(out), 'leochen')

def tasi():
    out = []
    for f in sorted(glob.glob(str(RAW / 'tasi' / 'kbar' / 'kbar*.csv'))):
        try:
            d = pd.read_csv(f, dtype=str, encoding='utf-8-sig')
        except Exception as e:
            print('skip', f, e); continue
        if not {'stoc', 'date', 'max', 'min', 'ope', 'clo', 'vol'} <= set(d.columns):
            print('skip cols', f, list(d.columns)[:8]); continue
        d = d.rename(columns={'stoc': 'code', 'ope': 'o', 'max': 'h', 'min': 'l', 'clo': 'c'})
        d['code'] = d['code'].astype(str).str.strip()
        d['code'] = np.where(d['code'].str.len() < 4, d['code'].str.zfill(4), d['code'])
        out.append(d[['date', 'code', 'o', 'h', 'l', 'c', 'vol']])
    fin(pd.concat(out), 'tasi')

if __name__ == '__main__':
    hschi(); bacardi(); leochen(); tasi()

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Load margin sources as long frames / wide matrices (cached)."""
import os, glob
import numpy as np, pandas as pd
from r_lib import RAW, CACHE, BB_MARGIN, MHIST, VF_QFII, COMMON


def _c(name, fn):
    p = f'{CACHE}/{name}.pkl'
    if os.path.exists(p):
        return pd.read_pickle(p)
    x = fn()
    pd.to_pickle(x, p)
    return x


def tej():
    def f():
        t = pd.read_parquet(f'{RAW}/yh_tej_margin_2005_2024.parquet')
        t['date'] = pd.to_datetime(t['date']).astype('datetime64[ns]')
        return t
    return _c('tej', f)


def bb():
    def f():
        b = pd.read_parquet(BB_MARGIN)
        b['date'] = pd.to_datetime(b['Date']).astype('datetime64[ns]')
        b = b.rename(columns={'Ticker': 'code'})
        return b
    return _c('bb', f)


def mhist():
    def f():
        h = pd.read_csv(MHIST, dtype={'code': str})
        h['date'] = pd.to_datetime(h['date']).astype('datetime64[ns]')
        return h
    return _c('mhist', f)


def snaps(kind):
    def f():
        out = []
        for fn in sorted(glob.glob(f'{RAW}/joeyc/{kind}/*.csv.gz')):
            d = pd.read_csv(fn, dtype={'code': str})
            d['date'] = pd.Timestamp(os.path.basename(fn)[:10])
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return _c('snap_' + kind.split('-')[0], f)


def vf():
    def f():
        v = pd.read_parquet(VF_QFII)
        v['date'] = pd.to_datetime(v['date']).astype('datetime64[ns]')
        return v
    return _c('vf', f)


def finbrain():
    def f():
        b = pd.read_parquet(f'{RAW}/finbrain_margin_buy.parquet')
        s = pd.read_parquet(f'{RAW}/finbrain_margin_sell.parquet')
        for x in (b, s):
            x.index = pd.to_datetime(x.index).astype('datetime64[ns]')
            x.columns = x.columns.astype(str)
        return b, s
    return _c('finbrain', f)


def wide(df, val, codecol='code'):
    d = df[df[codecol].astype(str).str.match(COMMON)]
    return d.pivot_table(index='date', columns=codecol, values=val, aggfunc='last')

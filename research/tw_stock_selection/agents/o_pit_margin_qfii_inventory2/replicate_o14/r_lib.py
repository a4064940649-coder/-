"""Independent replication library for o14 (MARGN / QFIIS inventory).
Only reads raw/ of the researcher folder plus local read-only sources.
"""
import os, glob, gzip, io
import numpy as np, pandas as pd

AG = '/home/user/-/research/tw_stock_selection/agents'
RAW = f'{AG}/o_pit_margin_qfii_inventory2/raw'
HERE = f'{AG}/o_pit_margin_qfii_inventory2/replicate_o14'
CACHE = f'{HERE}/cache'
LISTED = f'{AG}/o_v7_listing_exec_engine/data/listed_twse.parquet'
PANEL_VALUE = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/value.parquet'
BB_MARGIN = f'{AG}/o_sbl_foreign_level/raw/bb_margin.parquet'
VF_QFII = f'{AG}/o_sbl_foreign_level/raw/vf_twse_foreign.parquet'
MHIST = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/margin_hist.csv.gz'
TEJM = f'{AG}/o_pit_shares_industry_2006_2013/raw/tej_ff5/shares_monthly.csv'

SEG = {'S1': ('2006-01-02', '2012-12-31'), 'S2': ('2013-01-02', '2018-12-28'),
       'S3': ('2019-01-02', '2026-10-31')}
END = pd.Timestamp('2026-10-31')
COMMON = r'^[1-9]\d{3}$'
os.makedirs(CACHE, exist_ok=True)


def mi_index():
    m = pd.read_parquet(f'{RAW}/joeyc_mi_index_4digit_2014_2026.parquet')
    m['date'] = pd.to_datetime(m['date']).astype('datetime64[ns]')
    return m


def denominators():
    """Return (D, first, last, cal_all) where D is bool date x code (2006-01-02..2026-10-31),
    first/last = first/last in-market date per code (combined sources), cal_all full calendar."""
    p = f'{CACHE}/denom.pkl'
    if os.path.exists(p):
        return pd.read_pickle(p)
    L = pd.read_parquet(LISTED)
    L.index = pd.to_datetime(L.index).astype('datetime64[ns]')
    L = L.loc[:, L.columns.str.match(COMMON)]
    Lpre = L.loc[:'2013-12-31']
    m = mi_index()
    m = m[m.code.str.match(COMMON)]
    M = pd.crosstab(m['date'], m['code']).astype(bool)
    codes = sorted(set(Lpre.columns[Lpre.any()]) | set(M.columns))
    full = pd.concat([Lpre.reindex(columns=codes, fill_value=False),
                      M.reindex(columns=codes, fill_value=False)]).fillna(False).astype(bool)
    full = full[~full.index.duplicated()].sort_index()
    # first listing date: min of listed_twse (all years) and combined
    fL = L.idxmax().where(L.any())
    first = full.idxmax().where(full.any())
    fL = fL.reindex(codes)
    first = pd.concat([first, fL], axis=1).min(axis=1)
    last = full[::-1].idxmax().where(full.any())
    D = full.loc['2006-01-02':END]
    out = (D, first, last, full.index)
    pd.to_pickle(out, p)
    return out


def value_panel(cal):
    """Value panel: <2014 shared panel value.parquet, >=2014 MI_INDEX value."""
    p = f'{CACHE}/value.pkl'
    if os.path.exists(p):
        return pd.read_pickle(p)
    V = pd.read_parquet(PANEL_VALUE)
    V.index = pd.to_datetime(V.index).astype('datetime64[ns]')
    V = V.loc[:'2013-12-31']
    m = mi_index()
    m = m[m.code.str.match(COMMON)]
    MV = m.pivot_table(index='date', columns='code', values='value', aggfunc='last').astype(float)
    W = pd.concat([V, MV]).sort_index()
    W = W[~W.index.duplicated(keep='last')]
    pd.to_pickle(W, p)
    return W


def pool300(D, W, roll=60, minp=20, n=300):
    """Top-n by t-1 rolling mean value among D-True stocks on t."""
    cnt = W.notna().sum(axis=1)
    Wc = W[cnt >= 300]
    R = Wc.rolling(roll, min_periods=minp).mean()
    R = R.shift(1)  # t-1 value on row t (calendar of valid days)
    R = R.reindex(D.index, method='ffill')
    R = R.reindex(columns=D.columns)
    R = R.where(D)
    rk = R.rank(axis=1, ascending=False, method='first')
    return (rk <= n) & D


def seg_mask(idx, s):
    a, b = SEG[s]
    return (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))


def cov(have, den):
    den = den.values if hasattr(den, 'values') else den
    h = have.values if hasattr(have, 'values') else have
    n = den.sum()
    return float((h & den).sum() / n) if n else float('nan'), int(n), int((den & ~h).sum())

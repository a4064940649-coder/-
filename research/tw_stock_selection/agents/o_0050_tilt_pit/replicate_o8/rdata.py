"""Independent replication o8: data loading (written from frozen spec only)."""
import json, numpy as np, pandas as pd

AG = '/home/user/-/research/tw_stock_selection/agents'
O8 = AG + '/o_0050_tilt_pit'
V7 = AG + '/o_v7_listing_exec_engine/data'
OUT = O8 + '/replicate_o8'


def load_panel():
    cl = pd.read_parquet(V7 + '/close_raw.parquet')
    op = pd.read_parquet(V7 + '/open.parquet')
    rt = pd.read_parquet(V7 + '/ret.parquet')
    va = pd.read_parquet(V7 + '/value.parquet')
    # exfix: V7 ret is 0 on flag-1 ex-right/ex-div days; rebuild TR from reference price
    import glob
    fl = pd.read_parquet(V7 + '/flag.parquet').reindex(index=cl.index, columns=cl.columns)
    R = O8 + '/raw/'
    a = pd.concat([pd.read_csv(f, dtype={'code': str}) for f in sorted(glob.glob(R + 'jc_exright/data/twse-exright/*/*.csv.gz'))])
    b = pd.read_csv(R + 'screener/exdiv.csv', dtype={'code': str})
    c5 = pd.read_csv(R + 'screener/exdiv_5y.csv.gz', dtype={'code': str})
    x = pd.concat([q[['date', 'code', 'ref_price']] for q in (a, b, c5)]).drop_duplicates(['date', 'code'])
    x['date'] = pd.to_datetime(x['date'])
    refw = x.pivot_table(index='date', columns='code', values='ref_price', aggfunc='first')
    refw.index = refw.index.astype(cl.index.dtype)
    refw = refw.reindex(index=cl.index, columns=cl.columns)
    m = (fl == 1) & (rt == 0)
    rfix = cl / refw - 1
    pchg = cl / cl.ffill().shift(1) - 1
    fb = pchg.where((pchg <= 0) & (pchg > -0.25), 0.0)  # no ref: price-only drop (div), else 0 (capital reduction etc.)
    newr = rfix.where(rfix.notna(), fb)
    rt = rt.where(~m, newr)
    global EXFIX_STATS
    EXFIX_STATS = dict(n_flag1_zero=int(m.sum().sum()), n_ref=int((m & refw.notna()).sum().sum()))
    dp = pd.read_parquet(AG + '/o_infra_v3_rebuild/delist_prices.parquet')
    dp['date'] = pd.to_datetime(dp['date'])
    for fld, df in (('close', cl), ('open', op), ('ret', rt), ('value', va)):
        w = dp.pivot_table(index='date', columns='code', values=fld, aggfunc='last')
        w.index = w.index.astype(df.index.dtype)
        w = w.reindex(index=df.index)
        for c in w.columns:
            if c not in df.columns:
                df[c] = np.nan
            m = df[c].isna() & w[c].notna()
            df.loc[m, c] = w.loc[m, c]
    # ret on first day after a gap filled from close if missing
    cf = cl.ffill()
    rr = cl / cf.shift(1) - 1
    rt = rt.where(rt.notna() | cl.isna(), rr)
    return cl, op, rt, va


def load_shares():
    s = pd.read_parquet(O8 + '/data/shares_pit.parquet')
    return s


def tw50_members():
    j = json.load(open(O8 + '/raw/agents/tw50_members.json'))
    steps = sorted(j['steps_backward'], key=lambda s: s['eff'])
    sched = []  # (eff, members_after)
    for i, s in enumerate(steps):
        after = steps[i + 1]['members_before'] if i + 1 < len(steps) else j['current_2026-10-09']
        sched.append((pd.Timestamp(s['eff']), list(after)))
    first_eff = pd.Timestamp(steps[0]['eff'])
    return sched, first_eff, steps[0]['members_before']


def bench0050():
    b = pd.read_csv(O8 + '/raw/bench/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    return b


def taiex_days():
    t = pd.read_csv(O8 + '/raw/bench/taiex_twii.csv', parse_dates=['date'])
    return pd.DatetimeIndex(t['date'])


def industries():
    sl = pd.read_csv(O8 + '/raw/screener/stock_list.csv', dtype=str)
    return dict(zip(sl.code, sl.industry))


def is_fin(code, ind):
    i = ind.get(code)
    if i is not None:
        return i == '金融保險業'
    return code.startswith('28') or code in ('5880', '6005')


def revenue():
    R = pd.read_parquet(O8 + '/raw/agents/yoy_long_R_FIX.parquet')
    Y = pd.read_parquet(O8 + '/raw/agents/yoy_rev5y_FIX.parquet')
    R.index = pd.PeriodIndex(R.index, freq='M')
    Y.index = pd.PeriodIndex(Y.index, freq='M')
    R.columns = R.columns.astype(str); Y.columns = Y.columns.astype(str)
    cols = R.columns.union(Y.columns)
    idx = R.index.union(Y.index)
    out = R.reindex(index=idx, columns=cols)
    Yr = Y.reindex(index=idx, columns=cols)
    out = Yr.where(Yr.notna(), out)  # prefer 5y FIX when available
    return out

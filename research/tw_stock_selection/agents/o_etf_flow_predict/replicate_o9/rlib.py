# Independent replication library for o_etf_flow_predict (o9). Written from frozen spec only.
import json, datetime as dt
import numpy as np, pandas as pd

AG = '/home/user/-/research/tw_stock_selection/agents'
PANEL = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
SCR = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data'
EVENTS = AG + '/o_0050_tilt_pit/raw/agents/etf_rebalance_events.csv'
MEMBERS = AG + '/o_0050_tilt_pit/raw/agents/tw50_members.json'
SHARES = AG + '/o_0050_tilt_pit/data/shares_pit.parquet'

_cache = {}


def panel(name):
    if name not in _cache:
        df = pd.read_parquet(f'{PANEL}/{name}.parquet')
        df.index = pd.to_datetime(df.index)
        _cache[name] = df
    return _cache[name]


def tdays():
    if 'td' not in _cache:
        c = panel('close_raw')
        _cache['td'] = c.index[c.notna().sum(1) >= 300]
    return _cache['td']


def shares(lag=0):
    k = ('sh', lag)
    if k not in _cache:
        s = pd.read_parquet(SHARES)
        s.index = pd.to_datetime(s.index)
        _cache[('sh', 0)] = s
        if lag:
            _cache[k] = s.shift(lag)
    return _cache[k]


def third_friday(y, m):
    d = dt.date(y, m, 1)
    off = (4 - d.weekday()) % 7
    return pd.Timestamp(d + dt.timedelta(days=off + 14))


def last_td_le(d):
    td = tdays()
    i = td.searchsorted(pd.Timestamp(d), side='right') - 1
    return td[i]


def cutoff(eff_third_fri):
    c = eff_third_fri - pd.Timedelta(days=25)
    assert c.weekday() == 0
    return last_td_le(c)


def universe_mcap(C, lag=0, fix4763=True, liq=False):
    """market cap at C for universe; returns Series sorted desc, ranks"""
    c = panel('close_raw').loc[C]
    s = shares(lag)
    s_row = s.loc[:C].iloc[-1].copy()
    if fix4763 and '4763' in s_row.index and C < pd.Timestamp('2025-05-02'):
        s_row['4763'] = s_row['4763'] * 0.1
    codes = [x for x in c.index if len(x) == 4 and x.isdigit() and not x.startswith('0') and not x.startswith('91')]
    mc = (c[codes] * s_row.reindex(codes)).dropna()
    mc = mc[mc > 0]
    if liq:
        v = liq_ok(C)
        mc = mc[mc.index.isin(v[v].index)]
    rk = mc.rank(ascending=False, method='first').astype(int)
    return mc, rk


def adv60(C):
    v = panel('value').loc[tdays()]
    return v.loc[:C].tail(60).mean()


def liq_ok(C):
    a = adv60(C)
    p = panel('close_raw').loc[C]
    return (a >= 1e7) & (p >= 10)


def tw50_members_at(C):
    d = json.load(open(MEMBERS))
    steps = sorted(d['steps_backward'], key=lambda s: s['eff'])  # ascending
    cur = d[[k for k in d if k.startswith('current')][0]]
    # members after step i = members_before of step i+1; after last = current
    lists = []
    for i, s in enumerate(steps):
        after = steps[i + 1]['members_before'] if i + 1 < len(steps) else cur
        lists.append((pd.Timestamp(s['eff']), after))
    mem = None
    for eff, after in lists:
        if eff <= C:
            mem = after
    if mem is None:
        mem = steps[0]['members_before']
    return set(mem)


def events():
    e = pd.read_csv(EVENTS, dtype=str)
    return e


def tw50_reviews(start='2015-09', end='2026-09'):
    out = []
    for p in pd.period_range(start, end, freq='M'):
        if p.month in (3, 6, 9, 12):
            tf = third_friday(p.year, p.month)
            out.append((str(p), tf, cutoff(tf)))
    return out


def predict_tw50(C, lag=0, fix4763=True, liq=False):
    mc, rk = universe_mcap(C, lag, fix4763, liq)
    mem = tw50_members_at(C)
    adds = sorted([x for x in rk.index if x not in mem and rk[x] <= 40], key=lambda x: rk[x])
    dels = [m for m in mem if (m not in rk.index) or rk[m] >= 61]
    keep = [m for m in mem if m not in dels]
    extra = len(keep) + len(adds) - 50
    if extra > 0:
        keep_sorted = sorted(keep, key=lambda x: rk[x])
        dels += keep_sorted[-extra:]
    return adds, sorted(dels), rk, mem


def tstat(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))

# J 窗邊界的幾種讀法，對協定 resid_post2019（V7 100萬 FIX 一般成本 JMED，2019-05 起去均值）比對
import sys, os, bisect, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L
AG = os.path.join(L.HERE, '..', '..')
V7 = pd.read_parquet(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/navs_exec7.parquet'))
E = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
PL = pd.read_csv(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/plans_j7.csv'))
PL = PL[(PL.strat == 'AB_V7_B10') & (PL.seed == 0)]

def windows_calendar(idx, cal):
    """cal：交易日序列。回傳 {(m, j): (a, b)}"""
    months = sorted({d.to_period('M') for d in cal})
    j0 = {}
    for m in months:
        p = cal.searchsorted(pd.Timestamp(m.year, m.month, 10))
        if p < len(cal) and cal[p].to_period('M') == m: j0[m] = p
    ms = sorted(j0); W = {}
    for k, m in enumerate(ms):
        for j in range(13):
            pa = j0[m] + j
            if pa >= len(cal): continue
            pb = j0[ms[k + 1]] + j if k + 1 < len(ms) else len(cal) - 1
            pb = min(pb, len(cal) - 1)
            if pb > pa: W[(str(m), j)] = (cal[pa], cal[pb])
    return W

def windows_plans():
    j0 = sorted(pd.to_datetime(PL[PL.j == 0].signal)); W = {}
    for j in range(13):
        s = sorted(pd.to_datetime(PL[PL.j == j].signal))
        for k, a in enumerate(s):
            m = str(j0[bisect.bisect_right(j0, a) - 1].to_period('M'))
            b = s[k + 1] if k + 1 < len(s) else V7.index[-1]
            W[(m, j)] = (a, b)
    return W

def jmed(nav, fmt, W):
    e = E.reindex(nav.index).ffill(); rows = {}
    for (m, j), (a, b) in W.items():
        s = nav[fmt.format(j=j)]
        rows.setdefault(m, {})[j] = 100 * ((s.loc[b] / s.loc[a] - 1) - (e.loc[b] / e.loc[a] - 1))
    return pd.DataFrame(rows).T.sort_index()

def check(M, tag):
    j = M.median(axis=1)
    ms = [m for m in j.index if '2019-05' <= m <= '2026-09']
    v = j.loc[ms].values; d = (v - v.mean()) - L.RESID
    bad = [(m, round(x, 3)) for m, x in zip(ms, d) if abs(x) > 1e-6]
    print(tag, 'n', len(ms), 'max', np.abs(d).max().round(4), 'n_bad', len(bad), bad[:8])

idx = V7.index
fmt = 'B10|J{j}|K5T|normal|100萬|FIX'
check(jmed(V7, fmt, windows_calendar(idx, idx)), 'nav-calendar')
cal2 = idx[idx.isin(E.index)]
check(jmed(V7, fmt, windows_calendar(idx, cal2)), 'nav∩0050')
check(jmed(V7, fmt, windows_plans()), 'plans')
# 名單交易日曆：所有名單 signal/exec 日
pc = pd.DatetimeIndex(sorted(set(pd.to_datetime(PL.signal)) | set(pd.to_datetime(PL['exec']))))
cal3 = idx[idx.isin(pc) | ~idx.isin(pc)]

def show(M, tag):
    j = M.median(axis=1)
    ms = [m for m in j.index if '2019-05' <= m <= '2026-09']
    v = j.loc[ms].values; d = (v - v.mean()) - L.RESID
    off = np.median(d)
    print(tag, [(m, round(x - off, 3)) for m, x in zip(ms, d) if abs(x - off) > 1e-3])
show(jmed(V7, fmt, windows_calendar(idx, idx)), 'nav-calendar')
show(jmed(V7, fmt, windows_plans()), 'plans')
import json
JE = json.load(open(os.path.join(AG, 'o_forward_sequential_v2/raw/v7/jscan_end_days.json')))

def windows_plans2(endkey='V7_B10'):
    j0 = sorted(pd.to_datetime(PL[PL.j == 0].signal)); A = {}
    for j in range(13):
        for a in sorted(pd.to_datetime(PL[PL.j == j].signal)):
            m = j0[bisect.bisect_right(j0, a) - 1].to_period('M')
            A[(m, j)] = a
    W = {}
    for (m, j), a in A.items():
        n = m + 1
        if (n, j) in A: b = A[(n, j)]
        elif str(n) == '2026-10': b = pd.Timestamp(JE[f'{endkey}|J{j}'])
        else: continue
        W[(str(m), j)] = (a, b)
    return W
show(jmed(V7, fmt, windows_plans2()), 'plans2')
check(jmed(V7, fmt, windows_plans2()), 'plans2')

def windows_plans3(endkey='V7_B10'):
    j0 = sorted(pd.to_datetime(PL[PL.j == 0].signal)); W = {}
    for j in range(13):
        s = sorted(pd.to_datetime(PL[PL.j == j].signal))
        for k, a in enumerate(s):
            m = str(j0[bisect.bisect_right(j0, a) - 1].to_period('M'))
            b = s[k + 1] if k + 1 < len(s) else pd.Timestamp(JE[f'{endkey}|J{j}'])
            W[(m, j)] = (a, b)
    return W
show(jmed(V7, fmt, windows_plans3()), 'plans3')
check(jmed(V7, fmt, windows_plans3()), 'plans3')

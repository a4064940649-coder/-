#!/usr/bin/env python3
"""六個事前列出的版本（A1、A2、B1、B2、C1、C2）與共用評估函式。"""
import math
import numpy as np, pandas as pd
import oa_lib as O
K, V = O.K, O.V
IDX, COLS, CI = O.IDX, O.COLS, O.CI

VERSIONS = {
    'A1': '換股成交日開盤前 MSCI 標準納入已公告且尚未生效 → 該股跳過，由下一名遞補',
    'A2': 'A1 加上 MSCI 小型納入',
    'B1': '持股遇到 MSCI 標準納入生效日 E → E 收盤賣出（跌停鎖死延後一天），E+1 開盤用當月名單下一名（非本次納入股、非持股、非開盤漲停）遞補',
    'B2': 'B1 改為 E 收盤賣出後持有 0050 到下一換股日',
    'C1': 'A1＋B1',
    'C2': 'A2＋B1，只對 pressure（K_class ÷ 訊號日 20 日均成交值）≥ 挑選期中位數的事件啟動',
}
COST = {'normal': (K.rates(0.001), (K.FEE, K.FEE + 0.001)),
        'stress': ((K.FEE + 0.004, K.FEE + K.TAX + 0.004), (K.FEE + 0.001, K.FEE + 0.002))}

INC = O.inclusion_events()
RT = O.review_table().set_index('review')
# 事前可知的壓力：訊號日（該審核月的換股訊號日）20 日均成交值
def _p_sig(r):
    if r.review not in RT.index: return np.nan
    t = IDX.get_loc(RT.signal[r.review]); v = O.V20[t, CI[r.code]]
    return O.K_ADD[r.cls] / v if np.isfinite(v) and v > 0 else np.nan
INC['pressure_sig'] = [_p_sig(r) for r in INC.itertuples()]
_sel = INC[(INC.period == 'select') & INC.cls.isin(['STD', 'SC'])]
_liq = [bool(np.isfinite(O.V60[r.iE - 1, CI[r.code]]) and O.V60[r.iE - 1, CI[r.code]] >= 1e7) for r in _sel.itertuples()]
P_MED = float(_sel[_liq].pressure_sig.median())
GATE = set(INC[INC.cls.isin(['STD', 'SC']) & (INC.pressure_sig >= P_MED)].event_id)

S = O.strategy(True)
T0, T1_FULL, SD = S.bounds()
_EW = {}
def ew_nav(n, t1):
    key = t1
    if key not in _EW:
        pl = S.plan(n, 'official')
        _EW[key], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, T0, t1)
    return _EW[key]

def run(ver, n, cost='normal', t1=None, inc=None):
    inc = INC if inc is None else inc
    t1 = T1_FULL if t1 is None else t1
    (b, s), b50 = COST[cost]
    gate = GATE if ver == 'C2' else None
    if ver in ('A1', 'C1'): pl, sk = O.plan_mod(S, n, inc, ('STD',))
    elif ver in ('A2',): pl, sk = O.plan_mod(S, n, inc, ('STD', 'SC'))
    elif ver == 'C2': pl, sk = O.plan_mod(S, n, inc, ('STD', 'SC'), gate)
    else: pl, sk = O.plan_mod(S, n)
    acts, hits = {}, pd.DataFrame()
    if ver in ('B1', 'C1', 'C2'): acts, hits = O.build_actions(pl, inc, ('STD',), 'repl', gate)
    if ver == 'B2': acts, hits = O.build_actions(pl, inc, ('STD',), 'e50')
    nav, log = O.simulate_mod(pl, b, s, T0, t1, acts, b50)
    return dict(nav=nav, log=log, skips=sk, hits=hits, plan=pl)

def base(n, cost='normal', t1=None):
    t1 = T1_FULL if t1 is None else t1
    (b, s), b50 = COST[cost]
    pl, _ = O.plan_mod(S, n)
    nav, _ = O.simulate_mod(pl, b, s, T0, t1)
    return nav, pl

def period_returns(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    x = nav.reindex(D)
    return (x.shift(-1) / x - 1).iloc[:-1]

def affected_periods(res, base_plan):
    """受影響的換股期（訊號日）：名單和原版不同，或期內有 E 日賣出。"""
    aff = set()
    for p, q in zip(res['plan'], base_plan):
        if not np.array_equal(np.sort(p[1]), np.sort(q[1])): aff.add(p[3])
    sigs = [p[3] for p in res['plan']]; ex = [p[0] for p in res['plan']]
    for x in res['log']:
        if x[0] == 'sell_E':
            t = IDX.get_loc(pd.Timestamp(x[1]))
            k = max(i for i, e in enumerate(ex) if e <= t); aff.add(sigs[k])
    return aff

def paired(nav_m, nav_b, sd, aff, a, b):
    pm, pb = period_returns(nav_m, sd), period_returns(nav_b, sd)
    d = (pm - pb).dropna()
    d = d[(d.index >= pd.Timestamp(a)) & (d.index <= pd.Timestamp(b))]
    da = d[d.index.isin(list(aff))]
    def tt(x):
        return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std(ddof=1) > 0 else None
    return {'n_affected_months': int(len(da)), 'mean_affected%': round(float(da.mean()) * 100, 3) if len(da) else None, 't_affected': tt(da),
            'sum_affected%': round(float(da.sum()) * 100, 2), 'mean_all%': round(float(d.mean()) * 100, 4), 'n_all': int(len(d)),
            'affected_list': {str(k.date()): round(float(v) * 100, 2) for k, v in da.items()}}

def seg(nav, ew, a, b):
    r = V.seg_stats(nav, ew, a, b, SD, ad=a)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}

#!/usr/bin/env python3
"""breadth_derisk 實驗共用：建立原版計畫、B、各規則曝險、跑淨值、算各期統計。只 import 本資料夾的 bd_core。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import bd_core as K

BEARS = {'2008': ('2008-05-19', '2008-11-20'), '2011': ('2011-01-28', '2011-12-19'), '2015': ('2015-04-27', '2015-08-24'),
         '2018': ('2018-01-23', '2018-12-26'), '2020-03': ('2020-01-14', '2020-03-19'), '2022': ('2022-01-04', '2022-10-25')}
KS = (10, 20, 30)
WGRID = [round(x, 2) for x in np.arange(0, 1.0001, 0.05)]

BSER = K.breadth_series('2005-01-01')

def d3_flag(d):
    hist = BSER.loc[BSER.index < d, 'B'].iloc[-36:]
    if len(hist) < 36: return False
    return BSER.loc[d, 'B'] < np.percentile(hist.to_numpy(), 20)

def build(Y, start, end=None):
    """回傳 dict：universe → (Strategy, plan_n10)。"""
    out = {}
    for u, fx in (('ALL', None), ('EXFIN', K.FIN_CODES)):
        S = K.Strategy(Y, start, end=end, fin_exclude=fx)
        out[u] = (S, S.plan(10))
    return out

def rule_plans(S, pl):
    """各規則的 [(exec_t, picks, exposure, sig_day)]。"""
    S52 = S.S52
    R = {'BASE': [(p[0], p[1], 1.0, p[3]) for p in pl]}
    for k in KS:
        R[f'D1_K{k}'] = [(p[0], p[1], min(1.0, BSER.loc[p[3], 'B'] / k), p[3]) for p in pl]
    d2 = []
    for p in pl:
        t = K.IDX.get_loc(p[3]); s = S52[t][p[1]]
        keep = p[1][np.isfinite(s) & (s >= 0.95)]
        d2.append((p[0], keep, len(keep) / 10.0, p[3]))
    R['D2'] = d2
    R['D3'] = [(p[0], p[1], 0.5 if d3_flag(p[3]) else 1.0, p[3]) for p in pl]
    for w in WGRID:
        R[f'MIX_{w:.2f}'] = [(p[0], p[1], w, p[3]) for p in pl]
    return R

def run_nav(plan, slip, alt, t0, t1):
    bb, ss = K.rates(slip); b5, s5 = K.rates50(slip)
    nav, sw, lg = K.simulate_x([(p[0], p[1], p[2]) for p in plan], bb, ss, t0, t1, alt=alt, b50=b5, s50=s5)
    return nav, sw

def seg(nav, a, b):
    x = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    if len(x) < 2: return None
    # 以 a 之前最後一個收盤為起點
    prev = nav[nav.index < pd.Timestamp(a)]
    base = prev.iloc[-1] if len(prev) else x.iloc[0]
    eq = pd.concat([pd.Series([base]), x.reset_index(drop=True)]) / base
    return float(eq.iloc[-1] - 1), float((eq / eq.cummax() - 1).min())

def period_stats(nav, sw, plan, a, b, sd, base_nav):
    o = K.stats(nav, a, b, sd, K.E50, base_nav)
    o['t_vs_base'] = o.pop('t_EW', None)
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    def mr(x):
        xd = x.reindex(D); r = (xd.shift(-1) / xd - 1).iloc[:-1]; return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]
    o['diff_vs_base_m'] = float((mr(nav) - mr(base_nav)).mean())
    o['Calmar'] = o['CAGR'] / abs(o['maxDD']) if o['maxDD'] < 0 else float('nan')
    ins = [p for p in plan if pd.Timestamp(a) <= p[3] <= pd.Timestamp(b)]
    o['avg_exposure_target'] = float(np.mean([p[2] for p in ins]))
    o['months_exposure_lt1'] = int(sum(p[2] < 0.999 for p in ins)); o['n_months'] = len(ins)
    m = (sw.index >= pd.Timestamp(a)) & (sw.index <= pd.Timestamp(b))
    o['avg_stock_weight_daily'] = float(sw[m].mean())
    return o

def bear_stats(nav):
    out = {}
    for k, (a, b) in BEARS.items():
        if nav.index[0] <= pd.Timestamp(a) and nav.index[-1] >= pd.Timestamp(b):
            r = seg(nav, pd.Timestamp(a) + pd.Timedelta(days=1), b)   # 高點日收盤→低點日收盤
            out[k] = {'ret': round(r[0] * 100, 1), 'mdd': round(r[1] * 100, 1)}
    return out

def frontier_match(front, mdd):
    """front：[(w, CAGR, MDD)] w 遞增。回傳 MDD 等於 mdd 時（線性內插）的 (w, CAGR)。"""
    best = (None, None)
    for (w0, c0, m0), (w1, c1, m1) in zip(front[:-1], front[1:]):
        if (m0 - mdd) * (m1 - mdd) <= 0 and m0 != m1:
            f = (mdd - m0) / (m1 - m0)
            c = c0 + f * (c1 - c0)
            if best[1] is None or c > best[1]: best = (w0 + f * (w1 - w0), c)   # 多個交點時取年化最高者
    return best


def evaluate(Y, start, end, periods, navstore, tag):
    """跑全部規則 × 版本 × 0050/現金 × 成本，回傳 {universe: {rule|alt|slip: {period: stats}}}。"""
    B = build(Y, start, end)
    res = {}; plans_out = {}
    for u, (S, pl) in B.items():
        t0 = K.IDX.get_loc(S.sig_days[0]); t1 = K.IDX.get_loc(S.end_day); sd = S.sig_days + [S.end_day]
        R = rule_plans(S, pl)
        base_nav, _ = run_nav(R['BASE'], 0.001, 'cash', t0, t1)
        base_nav_s, _ = run_nav(R['BASE'], 'stress', 'cash', t0, t1)
        ru = {}
        for rn, plan in R.items():
            is_mix = rn.startswith('MIX_')
            combos = [('cash', 0.001), ('0050', 0.001)]
            if not is_mix or rn == 'MIX_0.80':
                combos += [('cash', 'stress'), ('0050', 'stress')]
            if rn == 'BASE': combos = [('cash', 0.001), ('cash', 'stress')]
            for alt, slip in combos:
                nav, sw = run_nav(plan, slip, alt, t0, t1)
                bn = base_nav if slip == 0.001 else base_nav_s
                key = f'{rn}|{alt}|{slip}'
                navstore[f'{tag}|{u}|{key}'] = nav
                o = {p: period_stats(nav, sw, plan, a, b, sd, bn) for p, (a, b) in periods.items()}
                o['bears'] = bear_stats(nav)
                ru[key] = o
        res[u] = ru
        plans_out[u] = {rn: {str(p[3].date()): round(float(p[2]), 4) for p in plan} for rn, plan in R.items() if not rn.startswith('MIX_')}
        plans_out[u]['n_picks_D2'] = {str(p[3].date()): int(len(p[1])) for p in R['D2']}
    return res, plans_out

def add_matched(res, periods):
    """每個規則在每期：同 MDD 下 0050 前緣與現金前緣的年化（線性內插）。"""
    for u, ru in res.items():
        for p in periods:
            for alt in ('cash', '0050'):
                front = [(w, ru[f'MIX_{w:.2f}|{alt}|0.001'][p]['CAGR'], ru[f'MIX_{w:.2f}|{alt}|0.001'][p]['maxDD']) for w in WGRID]
                for key, o in ru.items():
                    if key.startswith('MIX_') and key != 'MIX_0.80|0050|0.001' and key != 'MIX_0.80|cash|0.001': continue
                    if not key.endswith('|0.001'): continue
                    w, c = frontier_match(front, o[p]['maxDD'])
                    o[p][f'matchDD_{alt}_w'] = None if w is None else round(w, 3)
                    o[p][f'matchDD_{alt}_CAGR'] = c

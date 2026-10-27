#!/usr/bin/env python3
"""迭代建模（只用 SEL）：plan.json 事前列出的 12 版＋7 個必報對照組，一般／壓力成本；套用事前選版規則。寫 iterations.json。
所有淨值只模擬到 2019-01 訊號日收盤（T_SPLIT），計畫只含 2019-01 以前的訊號日。"""
import sys, json, time, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, O = L.K, L.V, L.O
T0_ = time.time()
def log(*a): print(f'[{time.time()-T0_:5.0f}s]', *a, flush=True)
PL = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(PL['PLAN'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == PL['PLAN_sha256']
SPLIT, t1 = L.SPLIT, L.T_SPLIT
sd = [d for d in L.SD if d <= SPLIT]
w = lambda n: L.wr(n, sd)
E = w(K.E50); TS = w(O.TSMC_IDX); Z = TS - E
VERS = {'V01': (50, 5, '55', 'ew', 0.3), 'V02': (30, 5, '55', 'ew', 0.3), 'V03': (80, 5, '55', 'ew', 0.3), 'V04': (50, 3, '55', 'ew', 0.3),
        'V05': (50, 8, '55', 'ew', 0.3), 'V06': (50, 5, '52', 'ew', 0.3), 'V07': (50, 5, '55', 'vw', 0.3), 'V08': (50, 5, '55', 'ew', 0.2),
        'V09': (50, 5, '55', 'ew', 0.5), 'V10': (80, 8, '55', 'ew', 0.3), 'V11': (30, 3, '55', 'ew', 0.3), 'V12': (50, 5, '52', 'vw', 0.3)}
BASE = (50, 5, '55', 'ew', 0.3)
def complexity(v): return sum(a != b for a, b in zip(v, BASE))

mp = L.main_plan(SPLIT)
navM, tovM = L.sim(L.wplan_main(mp), L.NORMAL, t1)
navMs, _ = L.sim(L.wplan_main(mp), L.STRESS3, t1)
ewMain, _ = K.simulate([(p[0], p[2]) for p in mp], 0, 0, L.T0, t1)
xM = w(navM) - E; xMs = w(navMs) - E
P_SEL = ('2006-04-01', str(L.LAST_SEL.date()), '2006-04-01', str(SPLIT.date()))
def seg(nav, ew): 
    st = O.segstat(nav, ew, sd, *P_SEL); return {k: v for k, v in st.items() if k != 'yearly'}
sM = seg(navM, ewMain)
log('MAIN SEL', sM['CAGR'], sM['t_0050'], sM['TE_ann'])

LEGS, POOLEW = {}, {}
def leg(P, n, score, wm):
    k = (P, n, score, wm)
    if k not in LEGS: LEGS[k] = L.plan_leg(P, n, score, wm, upto=SPLIT)
    return LEGS[k]
def poolew(P):
    if P not in POOLEW: POOLEW[P] = L.pool_ew(leg(P, 1, '55', 'ew'), t1)
    return POOLEW[P]

def evaluate(name, wp, legnav=None, legpool=None, extra=None):
    nav, tov = L.sim(wp, L.NORMAL, t1); navs, _ = L.sim(wp, L.STRESS3, t1)
    st = seg(nav, ewMain); sts = seg(navs, ewMain)
    xc = w(nav) - E
    r = {'SEL_normal': st, 'SEL_stress3': {k: sts[k] for k in ('CAGR', 't_0050', 'mean_ex_0050_m', 'TE_ann')}, 'turnover': round(tov, 4),
         'decomp_vs_MAIN': L.decomp(xc, xM), 'paired_vs_MAIN': L.mstat(w(nav) - w(navM)), 'TE_minus_MAIN_pp': round(st['TE_ann'] - sM['TE_ann'], 2),
         'mean_ex_minus_MAIN_pp': round(st['mean_ex_0050_m'] - sM['mean_ex_0050_m'], 3), 'beta_ex0050_on_2330m0050': L.beta(xc, Z),
         'corr_ex0050_with_MAIN': L.corr(xc, xM)}
    if extra: r.update(extra)
    log(name, st['CAGR'], st['t_0050'], st['mean_ex_0050_m'], st['TE_ann'])
    return r, nav

IT = {'PLAN_sha256': PL['PLAN_sha256'], 'scope': 'SEL only', 'n_versions': 12, 'MAIN_SEL': sM, 'MAIN_SEL_stress3': {k: seg(navMs, ewMain)[k] for k in ('CAGR', 't_0050', 'mean_ex_0050_m', 'TE_ann')},
      'MAIN_turnover': round(tovM, 4), 'versions': {}, 'controls': {}}
for k, (P, n, sc, wm, wl) in VERS.items():
    lp = leg(P, n, sc, wm)
    navL, tovL = L.sim(L.wplan_leg(lp), L.NORMAL, t1); navLs, _ = L.sim(L.wplan_leg(lp), L.STRESS3, t1)
    ew = poolew(P)
    xLP = w(navL) - w(ew); xLPs = w(navLs) - w(ew); xL = w(navL) - E
    n2330 = float(np.mean([L.J2330 in set(p[1].tolist()) for p in lp]))
    legst = {'L_vs_poolEW_net': L.mstat(xLP), 'L_vs_poolEW_stress3': L.mstat(xLPs), 'L_vs_0050': L.mstat(xL), 'poolEW_vs_0050': L.mstat(w(ew) - E),
             'L_turnover': round(tovL, 4), 'share_months_2330': round(n2330, 3), 'corr_Lex0050_MAINex0050': L.corr(xL, xM),
             'beta_Lex0050_on_2330m0050': L.beta(xL, Z), 'beta_LexPool_on_2330m0050': L.beta(xLP, Z)}
    r, _ = evaluate(k, L.wplan_mix(mp, lp, wl), extra={'spec': {'P': P, 'n': n, 'score': sc, 'weight': wm, 'w_leg': wl}, 'leg': legst, 'complexity': complexity(VERS[k])})
    IT['versions'][k] = r
# 對照組
lp1 = leg(50, 1, 'v60', 'ew'); lpP = leg(50, 5, '55', 'ew')
IT['controls']['TSMC30'], _ = evaluate('TSMC30', O.tsmc_plan(mp, 0.3))
IT['controls']['MAIN70_TOP1VALUE30'], _ = evaluate('TOP1', L.wplan_mix(mp, lp1, 0.3), extra={'top1_2330_share': round(float(np.mean([p[1][0] == L.J2330 for p in lp1])), 3)})
IT['controls']['MAIN70_POOLEW30'], _ = evaluate('POOLEW', L.wplan_pool(lpP, 0.3, mp))
for lab, cost in (('normal', L.NORMAL), ('stress3', L.STRESS3)):
    base = navM if lab == 'normal' else navMs
    nv = L.sleeve_mix(base, 0.3, t1, cost)
    st = seg(nv, ewMain)
    if lab == 'normal':
        r = {'SEL_normal': st, 'decomp_vs_MAIN': L.decomp(w(nv) - E, xM), 'paired_vs_MAIN': L.mstat(w(nv) - w(navM)),
             'TE_minus_MAIN_pp': round(st['TE_ann'] - sM['TE_ann'], 2), 'mean_ex_minus_MAIN_pp': round(st['mean_ex_0050_m'] - sM['mean_ex_0050_m'], 3),
             'corr_ex0050_with_MAIN': L.corr(w(nv) - E, xM), 'note': '袖子層混合（面板沒有 0050 開盤價）'}
        IT['controls']['MAIN70_0050_30'] = r
    else:
        IT['controls']['MAIN70_0050_30']['SEL_stress3'] = {k: st[k] for k in ('CAGR', 't_0050', 'mean_ex_0050_m', 'TE_ann')}
log('0050 mix', IT['controls']['MAIN70_0050_30']['SEL_normal']['t_0050'])
navL1, _ = L.sim(L.wplan_leg(lpP), L.NORMAL, t1)
IT['controls']['L_ALONE_V01leg'] = {'SEL_normal': seg(navL1, poolew(50))}

# 選版規则
cand = {k: v for k, v in IT['versions'].items() if v['mean_ex_minus_MAIN_pp'] >= -0.15}
if cand:
    best = max(v['SEL_normal']['t_0050'] for v in cand.values())
    tied = [k for k, v in cand.items() if best - v['SEL_normal']['t_0050'] < 0.01]
    pick = sorted(tied, key=lambda k: (IT['versions'][k]['complexity'], k))[0]
    rule_note = '符合月均超額條件者中 t 最高'
else:
    pick = 'V01'; rule_note = '沒有任何版本符合「月均超額不比主策略低 0.15pp 以上」→ 依事前規則凍結 V01 並註明'
IT['selection'] = {'eligible': sorted(cand), 'picked': pick, 'note': rule_note,
                   'table': {k: {'t_0050': v['SEL_normal']['t_0050'], 'mean_ex_minus_MAIN_pp': v['mean_ex_minus_MAIN_pp'], 'TE_minus_MAIN_pp': v['TE_minus_MAIN_pp'],
                                 'L_vs_pool_t': v['leg']['L_vs_poolEW_net']['t'], 'L_vs_pool_mean_pp': v['leg']['L_vs_poolEW_net']['mean_pp']} for k, v in IT['versions'].items()}}
IT['written_at'] = pd.Timestamp.now(tz='UTC').isoformat()
(HERE / 'iterations.json').write_text(L.dumps(IT), encoding='utf-8')
log('pick', pick, rule_note)
for k, v in IT['selection']['table'].items(): print(k, v)

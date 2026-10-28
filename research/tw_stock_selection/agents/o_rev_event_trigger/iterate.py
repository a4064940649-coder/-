#!/usr/bin/env python3
"""迭代（只用挑選期：錨定月 2006-04～2018-12，淨值只模擬到 2019-01 錨定月的同型訊號日）。
版本照 plan.json；依事前規則選版，寫 iterations.json。"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F
L, K, V = F.L, F.K, F.V
PL = json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(PL['PLAN'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == PL['PLAN_sha256'], 'plan.json 被改過'
Q = json.loads((F.DATA / 'q_night.json').read_text(encoding='utf-8'))['q']
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
st = F.setup()
M_END = pd.Period('2018-12', 'M'); M_NEXT = pd.Period('2019-01', 'M')
SEEDS = range(40)

def sel_days(j):
    if j == 'OFF':
        off, _ = F.official_days()
        days = [(m, d) for m, d in off if m <= M_END]; end = dict(off)[M_NEXT]
    else:
        days, _ = F.version_days(j); end = dict(days)[M_NEXT]; days = [(m, d) for m, d in days if m <= M_END]
    return days, end

def run_version(name, j, mode, news=False, kind='mc', nonab=True):
    days, end = sel_days(j)
    o = {'name': name, 'n_windows': len(days), 'first_signal': str(days[0][1].date()), 'eval_end': str(end.date())}
    W = F.windows([d for _, d in days], end)
    if nonab:
        S, pl, t0, t1, sd = F.build(days, end, None)
        nn, lg = F.sim(pl, t0, t1, 'normal'); ns, _ = F.sim(pl, t0, t1, 'stress'); ew = F.ew_nav(pl, t0, t1)
        o['nonAB'] = {'normal': F.block(nn, W, ('SEL',), ew)['SEL'], 'stress': F.block(ns, W, ('SEL',))['SEL'], 'turnover': lg['turnover']}
    else:
        nn = None
    if j == 'OFF':
        rets = {'normal': [], 'stress': []}; seeds_t = []; ewr = []
        for s in SEEDS:
            S, pl, t0, t1, sd = F.build(days, end, V.make_unavail('mc', s))
            for c in rets:
                n_, _ = F.sim(pl, t0, t1, c); rets[c].append(n_.pct_change())
                if c == 'normal': seeds_t.append(F.block(n_, W, ('SEL',))['SEL']['t_0050'])
            ewr.append(F.ew_nav(pl, t0, t1).pct_change())
        mean = {c: (1 + pd.concat(v, axis=1).mean(axis=1).fillna(0)).cumprod() for c, v in rets.items()}
        ew = (1 + pd.concat(ewr, axis=1).mean(axis=1).fillna(0)).cumprod()
    else:
        R = F.ab_navs(days, end, mode=mode, seeds=SEEDS if kind == 'mc' else [0], q=Q, news=news, costs=('normal', 'stress'))
        mean, ew = R['mean'], R['ew']
        seeds_t = [F.block(n_, W, ('SEL',))['SEL']['t_0050'] for n_ in R['seeds']['normal']]
        o['mean_unavail_in_top300'] = R['nold']
    o['AB'] = {'normal': F.block(mean['normal'], W, ('SEL',), ew)['SEL'], 'stress': F.block(mean['stress'], W, ('SEL',))['SEL'],
               'seed_t': {'median': float(np.median(seeds_t)), 'min': float(np.min(seeds_t)), 'max': float(np.max(seeds_t)), 'n': len(seeds_t)}}
    if nn is not None:
        o['lookahead'] = F.paired_cal(nn, mean['normal'], 'SEL')
        o['lookahead_win'] = F.paired_win(nn, W, mean['normal'], W, 'SEL')
    return o, mean, W

IT = []; NAVS = {}; WS = {}
specs = [('it1_J+1', 1, 'day', False, 'mc'), ('it2_J0', 0, 'day', False, 'mc'), ('it3_J0N', 0, 'night', False, 'mc'),
         ('it4_J0_ABN', 0, 'day', True, 'mc'), ('it5_OFF', 'OFF', 'day', False, 'mc'), ('it6_J0_ABworst', 0, 'day', False, 'worst')]
for name, j, mode, news, kind in specs:
    o, mean, W = run_version(name, j, mode, news, kind, nonab=name not in ('it4_J0_ABN', 'it6_J0_ABworst'))
    NAVS[name] = mean; WS[name] = W
    IT.append(o); log(name, 'AB t', round(o['AB']['normal']['t_0050'], 3), 'look', o.get('lookahead', {}).get('mean_pp'))
for o in IT:
    n = o['name']
    if n == 'it1_J+1': continue
    o['paired_vs_J1_cal'] = F.paired_cal(NAVS[n]['normal'], NAVS['it1_J+1']['normal'], 'SEL')
    o['paired_vs_J1_win'] = F.paired_win(NAVS[n]['normal'], WS[n], NAVS['it1_J+1']['normal'], WS['it1_J+1'], 'SEL')
    o['paired_vs_J1_cal_stress'] = F.paired_cal(NAVS[n]['stress'], NAVS['it1_J+1']['stress'], 'SEL')
# J0N 對 J0（同一成交日，差別只在營收腿用 D 當晚值）
o3 = [o for o in IT if o['name'] == 'it3_J0N'][0]
o3['paired_vs_J0_cal'] = F.paired_cal(NAVS['it3_J0N']['normal'], NAVS['it2_J0']['normal'], 'SEL')

# ---------------- 事前規則 ----------------
cand = {'J+1': 'it1_J+1', 'J0': 'it2_J0', 'J0N': 'it3_J0N'}
get = {o['name']: o for o in IT}
look = {k: abs(get[v]['lookahead']['mean_pp']) for k, v in cand.items()}
clean = [k for k in cand if look[k] < 0.03] or [min(look, key=look.get)]
tsel = {k: get[cand[k]]['AB']['normal']['t_0050'] for k in cand}
prov = max(clean, key=lambda k: tsel[k])
step3 = None; final = prov
if prov != 'J+1':
    pt = get[cand[prov]]['paired_vs_J1_cal']['t']
    step3 = {'provisional': prov, 'paired_vs_J1_cal_t': pt}
    if abs(pt) < 2: final = 'J+1'
dec = {'lookahead_pp_abs': look, 'clean_candidates': clean, 't_select_AB': tsel, 'provisional': prov, 'step3': step3, 'selected': final,
       'n_versions_this_topic': len(IT), 'note': 'it4／it6 是可得性模型的敏感度，不是候選；it5 官方混合日是參考'}
res = {'plan_sha256': PL['PLAN_sha256'], 'q_night': Q, 'iterations': IT, 'decision': dec}
(HERE / 'iterations.json').write_text(json.dumps(F.jsonable(res), ensure_ascii=False, indent=1), encoding='utf-8')
log('decision', dec)

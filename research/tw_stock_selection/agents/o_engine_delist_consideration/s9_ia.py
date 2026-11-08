#!/usr/bin/env python3
"""登記重算（紙上 ia_core 口徑，只跑一次）：C0、U_opt、U_opt_F2 × 一般／壓力 × freeze／nofreeze／dzero／cons_main／cons_lo／cons_hi／cons_c50。
斷言 PREREG 存在、雜湊相符、RESULTS_IA 尚不存在。輸出 results.json 的 RESULTS_IA；淨值存 scratchpad/o17_delist/navs_ia.parquet。
python3 -I -B s9_ia.py"""
import sys, json, math, pickle, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cz_lib as Z
RJ = HERE / 'results.json'
R = json.loads(RJ.read_text(encoding='utf-8'))
assert 'PREREG' in R, 'PREREG 不存在'
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_IA' not in R, '只跑一次'
for f, h in R['PREREG']['file_sha256'].items():
    p = HERE / f if (HERE / f).exists() else None
    if p is not None: assert Z.sha(p) == h, f
import cz_engine as CE
K = CE.K
assert Z.sha(Z.AG / 'independent_audit' / 'ia_core.py') == R['PREREG']['file_sha256']['ia_core.py']
assert Z.sha(Z.WORK / 'plans_ia.pkl') == R['PREREG']['file_sha256']['plans_ia.pkl']
sys.path.insert(0, str(Z.AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
END = pd.Timestamp('2026-10-12')
E = pd.read_csv(HERE / 'events_v2.csv', dtype={'code': str, 'successor': str})
PL = pickle.loads((Z.WORK / 'plans_ia.pkl').read_bytes())
COST = {'normal': K.rates(0.001), 'stress': K.rates('stress')}
d = R['PREREG']['events_v2']['summary']['d']['d_used']
ST = {'cons_main': CE.build_settlement(E, 'main', d=d), 'cons_lo': CE.build_settlement(E, 'lo', d=d),
      'cons_hi': CE.build_settlement(E, 'hi', d=d), 'cons_c50': CE.build_settlement(E, 'main', d=d, cash_0050=True)}
E50 = F.e50()

def win(nav, SD):
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None), 'Y2016_2018': ('2016-01', '2018-12'), 'Y2010_2013': ('2010-01', '2013-12')}
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 1e-15 else None
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def stats(W):
    o = {}
    for k, (a, b) in PERS.items():
        m = np.ones(len(W), bool)
        if a: m &= W.index >= pd.Period(a, 'M')
        if b: m &= W.index <= pd.Period(b, 'M')
        x, y = W.r.to_numpy()[m], W.r50.to_numpy()[m]
        o[k] = dict(n=int(m.sum()), ann=ann(x), ann_0050=ann(y), gex=ann(x) - ann(y), t=tt(x - y), mean_ex_pp=float(np.mean(x - y) * 100))
    s = (W.r - W.r50).groupby(W.yr.to_numpy()).sum(); b3 = list(s.sort_values(ascending=False).index[:3])
    m = ~np.isin(W.yr.to_numpy(), b3)
    o['ALL']['t_drop_best3'] = tt(W.r.to_numpy()[m] - W.r50.to_numpy()[m]); o['ALL']['best3'] = [int(x) for x in b3]
    return o

NAV = {}; LOGS = {}; OUT = {}
for k, v in PL.items():
    plan = [(p[0], p[1]) for p in v['plan']]; SD = pd.DatetimeIndex(v['sd'])
    for c, (bb, ss) in COST.items():
        for eg in ('freeze', 'nofreeze', 'dzero') + tuple(ST):
            if eg in ST: nav, lg = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode='consideration', settle=ST[eg])
            else: nav, lg = CE.simulate(plan, bb, ss, v['t0'], v['t1'], mode=eg)
            NAV[f'{k}|{c}|{eg}'] = nav
            if c == 'normal':
                LOGS[f'{k}|{eg}'] = dict(n_frozen_events=len(lg['frozen']), frozen_max_share=max([x[2] for x in lg['frozen']] or [0.0]),
                                         frozen_max_date=(max(lg['frozen'], key=lambda x: x[2])[0] if lg['frozen'] else None),
                                         settled=lg.get('settled', []))
            OUT[f'{k}|{c}|{eg}'] = stats(win(nav, SD))
    print(k, 'done', flush=True)
# 現行正式＝freeze；Δ
def delta(a, b):
    o = {}
    for p in ('ALL', 'SEG1', 'SEG2', 'SEG3'):
        o[f'{p}_ann'] = a[p]['ann'] - b[p]['ann']; o[f'{p}_gex'] = a[p]['gex'] - b[p]['gex']; o[f'{p}_t'] = (a[p]['t'] or 0) - (b[p]['t'] or 0)
    o['t_drop_best3'] = (a['ALL']['t_drop_best3'] or 0) - (b['ALL']['t_drop_best3'] or 0)
    return o
DL = {}
for key in OUT:
    k, c, eg = key.split('|')
    if eg != 'freeze': DL[key + '_vs_freeze'] = delta(OUT[key], OUT[f'{k}|{c}|freeze'])
# C0 判定
def judge(k, costs=('normal', 'stress')):
    rows = []
    for c in costs:
        for eg in ST:
            dd = DL[f'{k}|{c}|{eg}_vs_freeze']
            fields = (['ALL_ann', 'ALL_t'] if c == 'stress' else ['ALL_ann', 'ALL_t', 'SEG1_gex', 'SEG2_gex', 'SEG3_gex', 't_drop_best3'])
            for f in fields:
                lim = 0.05 if f.endswith('_t') or f == 't_drop_best3' else 0.3
                rows.append(dict(cost=c, engine=eg, field=f, delta=dd[f], limit=lim, exceed=bool(abs(dd[f]) > lim)))
    worst = max(rows, key=lambda r: abs(r['delta']) / r['limit'])
    return dict(rows=rows, annotate=any(r['exceed'] for r in rows), worst=worst)
JUD = {k: judge(k) for k in PL}
pd.DataFrame(NAV).to_parquet(Z.WORK / 'navs_ia.parquet')
R['RESULTS_IA'] = dict(utc=pd.Timestamp.now('UTC').isoformat(), stats=OUT, deltas=DL, judgement=JUD, logs=LOGS,
                       navs_file='scratchpad/o17_delist/navs_ia.parquet', navs_sha256=Z.sha(Z.WORK / 'navs_ia.parquet'))
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k in PL:
    print(k, {eg: round(OUT[f'{k}|normal|{eg}']['ALL']['ann'], 3) for eg in ('freeze', 'nofreeze', 'dzero') + tuple(ST)},
          {eg: round(OUT[f'{k}|normal|{eg}']['ALL']['t'], 4) for eg in ('freeze', 'nofreeze', 'dzero') + tuple(ST)}, 'annotate', JUD[k]['annotate'], JUD[k]['worst'])

#!/usr/bin/env python3
"""登記前煙霧測試（只做管線核對與名單層面計數，不算任何「新組合」的報酬）。
  1. 名單：C0、ALLPIT_REAL_{opt,p90,carry}、ALLPIT_REAL_opt_KYX 的名單 sha 必須等於題 1 RESULTS_PAPER.lists_sha；DSPR 只報異動槽數。
  2. C0 freeze 一般／壓力逐日重現 main_v2/navs_gapfix.parquet（< 1e−12）。
  3. ALLPIT_REAL_opt × cons_main（題 1 已公開的新正式數字）重現：全期年化、t、三段（一般與壓力；差 < 1e−9）。
  4. 下界能否觸發：各變體名單「持有跨越下市」事件（code, lt）× 引擎規則；v3_lo／v3_main／v3_hi 規則是否相同、與 cons_main 是否相同。
python3 -I -B smoke.py（在 scratchpad 執行）"""
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import ilib as I; sys.path.pop(0)
T0 = time.time()
R1 = json.loads((I.T1 / 'results.json').read_text(encoding='utf-8'))
LS1 = R1['RESULTS_PAPER']['lists_sha']; PV = R1['PREREG']['paper']['variants']
VAR = {'C0': (PV['C0']['kind'], (), 'union')}
for f in ('opt', 'p90', 'carry'): VAR[f'ALLPIT_REAL_{f}'] = (PV[f'ALLPIT_REAL_{f}']['kind'], (), 'union')
VAR['ALLPIT_REAL_opt_KYX'] = (PV['ALLPIT_REAL_opt_KYX']['kind'], tuple(PV['ALLPIT_REAL_opt_KYX']['excl']), 'union')
VAR['ALLPIT_REAL_opt_DSPR'] = (PV['ALLPIT_REAL_opt']['kind'], (), 'replace')
B = {k: I.build(kind, excl=ex, dsp_mode=dm) for k, (kind, ex, dm) in VAR.items()}
out = dict(lists_sha={k: I.lists_sha(b) for k, b in B.items()})
out['lists_sha_match_T1'] = {k: (out['lists_sha'][k] == LS1[k]) for k in B if k in LS1}
PK = {k: {p[3]: set(I.COLS[j] for j in p[1]) for p in b['plan']} for k, b in B.items()}
out['DSPR_changed_slots_vs_ALLPIT_REAL_opt'] = int(sum(len(PK['ALLPIT_REAL_opt'][d] - PK['ALLPIT_REAL_opt_DSPR'][d]) for d in PK['C0']))
out['DSPR_changed_days'] = sorted(str(d.date()) for d in PK['C0'] if PK['ALLPIT_REAL_opt'][d] != PK['ALLPIT_REAL_opt_DSPR'][d])
out['changed_slots_vs_C0'] = {k: int(sum(len(PK['C0'][d] - PK[k][d]) for d in PK['C0'])) for k in B}
print('lists', round(time.time() - T0), out['lists_sha_match_T1'], flush=True)
# 2. C0 重現
OFF = pd.read_parquet(I.AG / 'main_v2' / 'navs_gapfix.parquet')
rep = {}
for c in ('normal', 'stress'):
    nav, _ = I.sim(B['C0'], c, 'freeze'); ref = OFF[f'EXFIN|n10|{c}'].dropna()
    rep[c] = float(np.nanmax(np.abs(nav.reindex(ref.index).to_numpy() - ref.to_numpy())))
    if c == 'normal': W0 = {c: I.O.win(nav, B['C0']['sd'])}
    else: W0[c] = I.O.win(nav, B['C0']['sd'])
out['C0_reproduce_maxabs'] = rep
# 3. 題 1 新正式數字重現
S1 = R1['RESULTS_PAPER']['stats']; rp = {}
for c in ('normal', 'stress'):
    nav, _ = I.sim(B['ALLPIT_REAL_opt'], c, 'cons_main'); s = I.O.stats_pair(I.O.win(nav, B['C0']['sd']), W0[c])
    ref = S1[f'ALLPIT_REAL_opt|{c}|cons_main|vs_C0']
    rp[c] = dict(ann=s['ALL']['ann'], ann_T1=ref['ALL']['ann'], t=s['ALL']['t_vs_0050'], t_T1=ref['ALL']['t_vs_0050'],
                 maxabs=max(abs(s[p][f] - ref[p][f]) for p in ('ALL', 'SEG1', 'SEG2', 'SEG3') for f in ('ann', 'gex', 't_vs_0050')))
out['ALLPIT_REAL_opt_cons_main_reproduce'] = rp
print('reproduce', round(time.time() - T0), rep, rp, flush=True)
# 4. 持有跨越下市事件 × 規則
def pj(r): return json.dumps([{k: (round(v, 9) if isinstance(v, float) else v) for k, v in p.items()} for p in r['parts']], sort_keys=True, default=str)
RK = {eg: {(r['code'], r['lt']): r for rs in I.ST[eg].values() for r in rs} for eg in I.ST}
HE = {}
for k, b in B.items():
    if k in ('ALLPIT_REAL_opt_DSPR',): continue
    ev = sorted(set((c, lt) for c, lt, _, _ in I.held_events(b['plan'], I.ST['v3_main'], b['t1'])) | set((c, lt) for c, lt, _, _ in I.held_events(b['plan'], I.ST['cons_main'], b['t1'])))
    rows = []
    for c, ltd in ev:
        lt = int(I.K.IDX.get_loc(pd.Timestamp(ltd)))
        g = {eg: RK[eg].get((c, lt)) for eg in I.ST}
        rows.append(dict(code=c, lt=ltd, v3=I.v3_status(c), type_cons=(g['cons_main'] or {}).get('type'), type_v3=(g['v3_main'] or {}).get('type'),
                         v3_lo_eq_hi=(g['v3_lo'] is not None and g['v3_hi'] is not None and pj(g['v3_lo']) == pj(g['v3_hi'])),
                         v3_eq_cons=(g['v3_main'] is not None and g['cons_main'] is not None and pj(g['v3_main']) == pj(g['cons_main']))))
    HE[k] = dict(n=len(rows), rows=rows, n_v3_bound_sensitive=sum(not r['v3_lo_eq_hi'] for r in rows), n_v3_ne_cons=sum(not r['v3_eq_cons'] for r in rows),
                 frozen_no_rule=I.held_nodelist_rule(b['plan'], b['t1']))
out['held_events_paper'] = HE
out['runtime_s'] = round(time.time() - T0, 1)
(HERE / 'smoke.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for k, h in HE.items(): print(k, h['n'], 'bound-sensitive', h['n_v3_bound_sensitive'], 'v3≠cons', h['n_v3_ne_cons'], 'nofrule', len(h['frozen_no_rule']))
print('done', out['runtime_s'], out['DSPR_changed_slots_vs_ALLPIT_REAL_opt'], out['changed_slots_vs_C0'])

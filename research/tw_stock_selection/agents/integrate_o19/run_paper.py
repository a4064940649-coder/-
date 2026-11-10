#!/usr/bin/env python3
"""integrate_o19 紙上整合（PREREG 之後只跑一次）：o18 正式管線（只讀 import o18lib），處置官方清單置換為
official（現行）／strict（I1）／posthoc（I2）／I3a／I3b／I3c；C0、ALLPIT_REAL_{opt,p90,carry} × {一般, 壓力}；
I5 固定比例混合 80/20、50/50（第二條腿 I1、I3a），每窗再平衡。
python3 -I -B run_paper.py"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
assert 'RESULTS_PAPER' not in R, '只跑一次'
for f, h in R['PREREG']['file_sha256'].items(): assert hashlib.sha256((AG / f).read_bytes()).hexdigest() == h, f
T0 = time.time()
T1 = AG / 'o_delist_rev_backfill_allpit'; AU = AG / 'o_disposal_list_pit_audit_pkl18'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']
DU0 = O.DU.copy()
def ld(p): return pd.read_csv(p, dtype={'code': str}, parse_dates=['start', 'end'])[['code', 'start', 'end']].copy()
DUV = {'official': DU0[['code', 'start', 'end']].copy(), 'strict': ld(AU / 'data/DISP_OFFICIAL_v2.csv'), 'posthoc': ld(AU / 'data/DISP_OFFICIAL_v2_posthoc.csv'),
       'I3a': ld(HERE / 'data/DU_I3a.csv'), 'I3b': ld(HERE / 'data/DU_I3b.csv'), 'I3c': ld(HERE / 'data/DU_I3c.csv')}
k0 = set(map(tuple, DU0[['code', 'start', 'end']].astype(str).values)); assert k0 == set(map(tuple, DUV['strict'].astype(str).values))
VARS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
def lsha(b):
    PK = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}
    return hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK.items()}, sort_keys=True).encode()).hexdigest(), PK
B, LSH, PKS = {}, {}, {}
for dv, d in DUV.items():
    O.DU = d; O._dc.clear()
    for k, kind in VARS.items():
        b = O.build(kind); B[(dv, k)] = b; LSH[(dv, k)], PKS[(dv, k)] = lsha(b)
O.DU = DU0; O._dc.clear()
for k in VARS: assert LSH[('official', k)] == R1['lists_sha'][k], k
diff = {}
for dv in DUV:
    for k in VARS:
        a, b = PKS[('official', k)], PKS[(dv, k)]
        diff[f'{dv}|{k}'] = [dict(signal=str(d.date()), out=sorted(set(a[d]) - set(b.get(d, []))), into=sorted(set(b.get(d, [])) - set(a[d]))) for d in a if set(a[d]) != set(b.get(d, []))]
SD0 = B[('official', 'C0')]['sd']
NAV = {}
for (dv, k), b in B.items():
    for c in ('normal', 'stress'):
        if dv != 'official' and not diff[f'{dv}|{k}'] and dv != 'strict':
            NAV[(dv, k, c)] = NAV[('official', k, c)]; continue
        NAV[(dv, k, c)] = O.sim(b, c, 'freeze' if k == 'C0' else 'cons_main')[0]
W = {key: O.win(nav, SD0) for key, nav in NAV.items()}
ST = {}
for (dv, k, c) in W:
    s = O.stats_pair(W[(dv, k, c)], W[('official', k, c)]); s['ALL']['mdd_daily'] = float((NAV[(dv, k, c)] / NAV[(dv, k, c)].cummax() - 1).min() * 100)
    s['vs_C0'] = {p: dict(diff_pp=v['diff_vs_ref_pp'], t=v['t_diff_vs_ref'], corr=v['corr_ref']) for p, v in O.stats_pair(W[(dv, k, c)], W[('official', 'C0', c)]).items() if p in O.PERS}
    ST[f'{dv}|{k}|{c}'] = s
# 混合（I5）
BL = {}
for k in VARS:
    for c in ('normal', 'stress'):
        M = W[('official', k, c)]
        for leg in ('strict', 'I3a'):
            L = W[(leg, k, c)]
            J = M.join(L[['r']], rsuffix='_l', how='inner')
            for w in (0.8, 0.5):
                Wb = J[['r50', 'yr']].copy(); Wb['r'] = w * J['r'] + (1 - w) * J['r_l']
                s = O.stats_pair(Wb[['r', 'r50', 'yr']], M)
                BL[f'{k}|{c}|leg={leg}|w={w}'] = s
# 重現正式
rep = {}
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    for c in ('normal', 'stress'):
        a = ST[f'official|{k}|{c}']; b = R1['stats'][f'{k}|{c}|cons_main|vs_C0']
        rep[f'{k}|{c}'] = max(abs(a[p][f] - b[p][f]) for p in ('ALL', 'SEG1', 'SEG2', 'SEG3') for f in ('ann', 'gex', 't_vs_0050'))
smp = {}
for dv in DUV:
    for k in VARS:
        a = ST[f'{dv}|{k}|normal']['ALL']; se = abs(a['gex'] / a['t_vs_0050'])
        smp[f'{dv}|{k}'] = dict(gex=a['gex'], lo=a['gex'] - 1.96 * se, hi=a['gex'] + 1.96 * se)
res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), lists_sha={f'{a}|{b}': v for (a, b), v in LSH.items()},
           changed_slots={k: v for k, v in diff.items()}, n_changed_signals={k: len(v) for k, v in diff.items()},
           reproduce_official_maxabs=rep, stats=ST, blends=BL, sampling_interval=smp)
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_PAPER' not in R; R['RESULTS_PAPER'] = res
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for key, s in ST.items():
    a = s['ALL']; print(key, 'ann %.3f t %.3f gex %.3f d %.4f(t %s) corr %s seg %.2f/%.2f/%.2f' % (a['ann'], a['t_vs_0050'], a['gex'], a['diff_vs_ref_pp'], a['t_diff_vs_ref'], a['corr_ref'], s['SEG1']['gex'], s['SEG2']['gex'], s['SEG3']['gex']))
print('changed', res['n_changed_signals']); print('rep', rep, res['runtime_s'])

#!/usr/bin/env python3
"""S4a 紙上重算（PREREG＋ADDENDUM 之後，只跑一次）：o18 正式管線（只讀 import o_delist_rev_backfill_allpit/o18lib），
處置 union 遮罩的官方清單改讀 DISP_OFFICIAL_v2（嚴格＝寬鬆）與 _posthoc。C0、ALLPIT_REAL_{opt,p90,carry} × cons_main × {一般, 壓力}。
另：最差格的單邊滑價兩平點（超額歸零、t<1）、抽樣區間（±1.96 SE）。
python3 -I -B s10_paper.py"""
import sys, json, hashlib, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'; R = json.loads(RJ.read_text(encoding='utf-8'))
for k in ('PREREG', 'PREREG_ADDENDUM'):
    assert hashlib.sha256(json.dumps(R[k], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R[k + '_sha256'], k
assert 'RESULTS_PAPER' not in R, '只跑一次'
for f, h in R['PREREG_ADDENDUM']['files'].items(): assert hashlib.sha256((HERE / f).read_bytes()).hexdigest() == h, f
T0 = time.time()
T1 = AG / 'o_delist_rev_backfill_allpit'
sys.path.insert(0, str(T1)); import o18lib as O; sys.path.pop(0)
R1 = json.loads((T1 / 'results.json').read_text(encoding='utf-8'))['RESULTS_PAPER']
DU0 = O.DU.copy()
def load_du(f):
    d = pd.read_csv(HERE / f, dtype={'code': str}, parse_dates=['start', 'end'])
    return d[['code', 'start', 'end']].copy()
def set_du(d):
    O.DU = d; O._dc.clear()
def lists_sha(b):
    PK = {p[3]: [O.COLS[j] for j in p[1]] for p in b['plan']}
    return hashlib.sha256(json.dumps({str(d.date()): v for d, v in PK.items()}, sort_keys=True).encode()).hexdigest(), PK
VARS = {'C0': ('base', None, None, False, None), 'ALLPIT_REAL_opt': ('opt', '2006-01-01', 'UB', True, 'B1'),
        'ALLPIT_REAL_p90': ('p90', '2006-01-01', 'UB', True, 'B1'), 'ALLPIT_REAL_carry': ('carry', '2006-01-01', 'UB', True, 'B1')}
DUV = {'strict': load_du('data/DISP_OFFICIAL_v2.csv'), 'posthoc': load_du('data/DISP_OFFICIAL_v2_posthoc.csv')}
# 嚴格版與現行覆蓋完全相同（PREREG identity_shortcut 的前提）
k0 = set(map(tuple, DU0[['code', 'start', 'end']].astype(str).values)); k1 = set(map(tuple, DUV['strict'].astype(str).values))
assert k0 == k1, '嚴格版覆蓋與現行不同'
B, LSH, PKS = {}, {}, {}
for dv, d in DUV.items():
    set_du(d)
    for k, kind in VARS.items():
        b = O.build(kind); B[(dv, k)] = b; LSH[(dv, k)], PKS[(dv, k)] = lists_sha(b)
set_du(DU0)
for k in VARS: assert LSH[('strict', k)] == R1['lists_sha'][k], ('嚴格版名單與官方不同', k)
diff_slots = {}
for k in VARS:
    a, b = PKS[('strict', k)], PKS[('posthoc', k)]
    diff_slots[k] = [dict(signal=str(d.date()), out=sorted(set(a[d]) - set(b.get(d, []))), into=sorted(set(b.get(d, [])) - set(a[d]))) for d in a if set(a[d]) != set(b.get(d, []))]
SD0 = B[('strict', 'C0')]['sd']
NAV, ST = {}, {}
for (dv, k), b in B.items():
    if dv == 'posthoc' and not diff_slots[k]: continue
    for c in ('normal', 'stress'):
        eng = 'freeze' if k == 'C0' else 'cons_main'
        nav, lg = O.sim(b, c, eng); NAV[(dv, k, c)] = nav
W = {key: O.win(nav, SD0) for key, nav in NAV.items()}
for key in W:
    dv, k, c = key
    s = O.stats_pair(W[key], W[('strict', 'C0', c)]); s['ALL']['mdd_daily'] = float((NAV[key] / NAV[key].cummax() - 1).min() * 100)
    ST['|'.join(key)] = s
# 重現官方紙上（題 1 RESULTS_PAPER）
rep = {}
for k in ('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'):
    for c in ('normal', 'stress'):
        a = ST[f'strict|{k}|{c}']; b = R1['stats'][f'{k}|{c}|cons_main|vs_C0']
        rep[f'{k}|{c}'] = max(abs(a[p][f] - b[p][f]) for p in ('ALL', 'SEG1', 'SEG2', 'SEG3') for f in ('ann', 'gex', 't_vs_0050'))
# 兩平點：最差格（全期年化最低者）單邊滑價網格
worst = min(('ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry'), key=lambda k: ST[f'strict|{k}|normal']['ALL']['ann'])
be = {}
for nm, kk in (('C0', 'C0'), ('worst_ALLPIT', worst)):
    grid = []
    for slip in (0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008):
        bb, ss = O.K.rates(slip); pl = [(p[0], p[1]) for p in B[('strict', kk)]['plan']]
        nav = O.CE.simulate(pl, bb, ss, B[('strict', kk)]['t0'], B[('strict', kk)]['t1'], mode='freeze' if kk == 'C0' else 'consideration', settle=None if kk == 'C0' else O.ST['cons_main'])[0]
        s = O.stats_pair(O.win(nav, SD0), W[('strict', 'C0', 'normal')])['ALL']
        grid.append(dict(slip=slip, ann=s['ann'], gex=s['gex'], t=s['t_vs_0050']))
    g = pd.DataFrame(grid)
    def cross(col, lvl):
        for i in range(1, len(g)):
            a, b = g[col].iloc[i - 1], g[col].iloc[i]
            if (a - lvl) * (b - lvl) <= 0 and a != b: return float(g.slip.iloc[i - 1] + (lvl - a) / (b - a) * (g.slip.iloc[i] - g.slip.iloc[i - 1]))
        return None
    be[nm] = dict(variant=kk, grid=grid, slip_excess_zero=cross('gex', 0.0), slip_t_below_1=cross('t', 1.0), slip_t_below_196=cross('t', 1.96))
# 抽樣區間（全期超額 ± 1.96 SE；SE≈gex/t）與假設區間
smp = {}
for k in VARS:
    a = ST[f'strict|{k}|normal']['ALL']
    se = abs(a['gex'] / a['t_vs_0050']) if a['t_vs_0050'] else np.nan
    smp[k] = dict(gex=a['gex'], t=a['t_vs_0050'], lo=a['gex'] - 1.96 * se, hi=a['gex'] + 1.96 * se)
assm = dict(ann=[min(ST[f'{dv}|{k}|normal']['ALL']['ann'] for k in VARS if k != 'C0' for dv in ('strict', 'posthoc') if f'{dv}|{k}|normal' in ST),
                 max(ST[f'{dv}|{k}|normal']['ALL']['ann'] for k in VARS if k != 'C0' for dv in ('strict', 'posthoc') if f'{dv}|{k}|normal' in ST)],
            note='放回 3 種 × {嚴格＝寬鬆, posthoc}')
res = dict(utc=pd.Timestamp.now('UTC').isoformat(), runtime_s=round(time.time() - T0, 1), lists_sha={f'{a}|{b}': v for (a, b), v in LSH.items()},
           strict_equals_official_lists=True, reproduce_official_maxabs=rep, posthoc_changed_slots=diff_slots, stats=ST, breakeven=be, sampling_interval=smp, assumption_interval=assm)
R = json.loads(RJ.read_text(encoding='utf-8')); assert 'RESULTS_PAPER' not in R; R['RESULTS_PAPER'] = res
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for key, s in ST.items():
    a = s['ALL']; print(key, 'ann %.3f t %.3f gex %.3f seg %.2f/%.2f/%.2f' % (a['ann'], a['t_vs_0050'], a['gex'], s['SEG1']['gex'], s['SEG2']['gex'], s['SEG3']['gex']))
print('rep', rep); print('diff', {k: len(v) for k, v in diff_slots.items()}, diff_slots); print(json.dumps(be, default=str)[:1500]); print(smp)

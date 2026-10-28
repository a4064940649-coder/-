#!/usr/bin/env python3
"""獨立複製：照 o_disposal_rejudge/v3run.py 的 R_DEL|E1 路徑重建主策略腿（只讀 import 上游模組），存 nav（normal / stress3）、EW、訊號日與持股。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import numpy as np, pandas as pd
import drj_core as R
res = json.loads((AG / 'o_disposal_rejudge' / 'results.json').read_text(encoding='utf-8'))
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3
K, V = R.K, R.V
O3.apply_patch(require_freeze=True)
E = O3.load_events(require_freeze=True)
XR = O3.exit_rules(E, 'main')
D = R.load_list(); DS = R.Disp(D)
YL0, Y50 = R.revenue()
YRD = O3.load_rdel()
S = V.V2('gapfix', YRD, Y50, exfin=True)
base = R.make_strategy('E1', DS)._mask
orig = S.cross_section
def cs(d, _o=orig, _m=base):
    old = K.disposal_mask; K.disposal_mask = _m
    try: return _o(d)
    finally: K.disposal_mask = old
S.cross_section = cs
t0, t1, sd = S.bounds()
pl = S.plan(10, 'official')
out = {}
for lab, (b, s) in (('normal', K.rates(0.001)), ('stress3', K.rates(0.004)), ('nodisc', (0.001425 + 0.001, 0.001425 + 0.003 + 0.001))):
    nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=XR)
    out[lab] = nav
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
out['ew'] = ew
navs = pd.DataFrame(out)
e50 = R.E50.reindex(navs.index)
navs['e50'] = e50
navs.to_parquet(HERE / 'my_navs.parquet')
cg, mdd = R.cagr_mdd(navs['normal'], R.SEL_A, R.HOLD_B)
tv = R.tvs(navs['normal'], e50, sd, R.SEL_A, R.HOLD_B)
print('ALL CAGR', cg, 'mdd', mdd, 't', tv)
assert abs(cg - 26.945918746008978) < 1e-9 and abs(tv['t'] - 2.5486212673374764) < 1e-9
# 2330 GROW 連乘
j2330 = K.CI['2330']
g = pd.Series(K.GROW[:, j2330], index=K.IDX).reindex(navs.index)
plan = [(str(K.IDX[p[0]].date()), str(pd.Timestamp(p[3]).date()), [K.COLS[j] for j in p[1]]) for p in pl]
meta = dict(sd=[str(pd.Timestamp(x).date()) for x in sd], plan=plan)
(HERE / 'my_plan.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
g.to_frame('g2330').to_parquet(HERE / 'my_2330.parquet')
print(len(sd), sd[:2], sd[-2:], len(pl))

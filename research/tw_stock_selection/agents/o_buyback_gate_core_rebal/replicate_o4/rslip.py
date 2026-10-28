#!/usr/bin/env python3
"""滑價網格：照 o_disposal_rejudge/v3run.py 的 R_DEL|E1 路徑重建主策略腿（只讀 import 上游模組），存 nav（normal / stress3）、EW、訊號日與持股。"""
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
for s_ in np.round(np.arange(0.0005, 0.00401, 0.00025), 6):
    b, s = K.rates(float(s_))
    nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=XR)
    out[f'{s_:.5f}'] = nav
pd.DataFrame(out).to_parquet(HERE / 'my_navs_slip.parquet')
print('done', len(out))

#!/usr/bin/env python3
"""重建主策略 V3 日淨值（o_disposal_rejudge/v3run.py 的 R_DEL|E1、排除金融、n10；只讀 import drj_core 與 o3_core），
一般成本（滑價 0.1%）、壓力 stress3（每邊滑價 0.4%＝一般再加 0.3%）、舊壓力（買賣各 0.5%）。另存同池等權（零成本）、換股計畫。
輸出 main_navs.parquet、main_plan.json。斷言 ALL 年化與對 0050 t 與 v3_results.json 一致。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import numpy as np, pandas as pd
import drj_core as R
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K
npatch = O3.apply_patch(require_freeze=True)
E = O3.load_events(require_freeze=True)
XR = O3.exit_rules(E, 'main')
D = R.load_list(); DS = R.Disp(D)
YRD = O3.load_rdel(); _, Y50 = R.revenue()
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
navs = {}; tov = {}
for nm, (b, s) in {'normal': K.rates(0.001), 'stress3': K.rates(0.004), 'stress_old': K.rates('stress'), 'gross': (0.0, 0.0)}.items():
    nav, lg = O3.simulate([(p[0], p[1]) for p in pl], b, s, t0, t1, xrules=XR)
    navs['MAIN|' + nm] = nav; tov[nm] = lg['turnover']
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
navs['POOL_EW|gross'] = ew
idx = navs['MAIN|normal'].index
navs['E50'] = K.E50.reindex(idx)
navs['A2330'] = pd.Series(np.cumprod(K.GROW[:, K.CI['2330']]), index=K.IDX).reindex(idx)
N = pd.DataFrame(navs)
N.to_parquet(HERE / 'main_navs.parquet')
ref = json.loads((AG / 'o_disposal_rejudge' / 'v3_results.json').read_text(encoding='utf-8'))['runs']['R_DEL|E1']['stats']['ALL']
cg, mdd = R.cagr_mdd(N['MAIN|normal'], R.SEL_A, R.HOLD_B)
tt = R.tvs(N['MAIN|normal'], N['E50'], sd, R.SEL_A, R.HOLD_B)['t']
print('ALL', cg, mdd, tt, 'ref', ref['CAGR'], ref['maxDD'], ref['vs0050']['t'])
assert abs(cg - ref['CAGR']) < 1e-6 and abs(tt - ref['vs0050']['t']) < 1e-6
plan = [dict(exec=str(K.IDX[p[0]].date()), sig=str(p[3].date()), picks=[K.COLS[j] for j in p[1]],
             open_raw=[float(K.O.iat[p[0], j]) for j in p[1]], v60=[float(K.V.iloc[max(p[0]-60,0):p[0], j].mean()) for j in p[1]]) for p in pl]
json.dump(dict(sig_days=[str(d.date()) for d in sd], turnover=tov, n_patched=npatch, plan=plan, t0=str(K.IDX[t0].date()), t1=str(K.IDX[t1].date())),
          open(HERE / 'main_plan.json', 'w'), ensure_ascii=False)
print('turnover', tov)

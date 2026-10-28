#!/usr/bin/env python3
"""成本損益兩平用：主策略 V3 在多個單邊滑價與全額手續費下的日淨值（同 build_main.py 的規格）。
原說明：重建主策略 V3 日淨值（o_disposal_rejudge/v3run.py 的 R_DEL|E1、排除金融、n10；只讀 import drj_core 與 o3_core），
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
navs = {}
FEE = K.FEE
for slip in (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006):
    nav, _ = O3.simulate([(p[0], p[1]) for p in pl], FEE + slip, FEE + K.TAX + slip, t0, t1, xrules=XR)
    navs[f'slip{slip:.4f}'] = nav
nav, _ = O3.simulate([(p[0], p[1]) for p in pl], 0.001425 + 0.001, 0.001425 + K.TAX + 0.001, t0, t1, xrules=XR)
navs['fullfee_slip0.0010'] = nav
pd.DataFrame(navs).to_parquet(HERE / 'main_navs_slip.parquet')
print(pd.DataFrame(navs).iloc[-1])

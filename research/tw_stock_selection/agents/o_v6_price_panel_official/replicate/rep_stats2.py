#!/usr/bin/env python3
"""補充（區分「族群不同」與「實作不同」）：用複製員的 SPA 實作（replicate_o5/spa.py、rlib.sb_indices），
但族群改用 o_main_spa_breaks 凍結的 F_long（data/family_excess.parquet＋family_members.csv，2,319 條），
o_v4 的 AB_V4、AB_V4_B10、V4 換成複製員引擎的 V6 版、其餘 11 條換成研究員 navs_v6。輸出 replicate/rep_stats2.json。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
H = Path(__file__).resolve().parent; TOP = H.parent; AG = TOP.parent
R5 = AG / 'o_main_spa_breaks' / 'replicate_o5'; SB = AG / 'o_main_spa_breaks' / 'data'
sys.path.insert(0, str(R5)); import rlib, spa as SP; sys.path.pop(0)
REP = pd.read_parquet(H / 'out_rep.parquet'); NV = pd.read_parquet(TOP / 'navs_v6.parquet')
D = pd.read_parquet(SB / 'family_excess.parquet'); MM = pd.read_csv(SB / 'family_members.csv')
mem = MM[MM.cov_W_long & MM.F_long].member_id.tolist()
r50 = rlib.win_ret(rlib.e0050()); pref = 'o_v4_revenue_patch_official/navs_v4.parquet::'
X = D.copy()
for m in [c for c in mem if c.startswith(pref)]:
    k = m[len(pref):]
    if k in ('AB_V4|normal', 'AB_V4_B10|normal', 'V4|normal'):
        nav = REP[{'AB_V4|normal': 'AB_V6', 'AB_V4_B10|normal': 'AB_V6_B10', 'V4|normal': 'V6'}[k]]
    else:
        k6 = k.replace('V4', 'V6'); k6 = 'O3REP_V6' + k6[5:] if k6.startswith('O3REP') else k6; nav = NV[k6]
    X[m] = rlib.win_ret(nav) - r50
A = X.loc[:, mem].fillna(0.0).to_numpy(); out = {}
for L in (3, 6, 12):
    idx = rlib.sb_indices(A.shape[0], L, 10000, 20261028 + L)
    for foc in ('AB_V4|normal', 'AB_V4_B10|normal'):
        o = SP.spa(A, mem.index(pref + foc), L, idx, stepm=False, unstd=False)
        out[f'{foc.replace("V4", "V6")}|L{L}'] = {k: o[k] for k in ('p_SPA_c_focus', 'rank', 't_main', 'K')}
        print(foc, L, out[f'{foc.replace("V4", "V6")}|L{L}'], flush=True)
(H / 'rep_stats2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))

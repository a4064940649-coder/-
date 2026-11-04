#!/usr/bin/env python3
"""步驟 4（凍結）：依 ITERATIONS 的事前選版結果寫 PREREG（規則、參數、程式 sha256）。之後才可跑最終 OC（seed 集 B）。
用法：python3 -I -B s4_freeze.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g8lib as G

D = G.check_reg('PREREG_DESIGN')
R = G.load_res(); IT = R['ITERATIONS']
sel = IT['selected']
p = IT['params']
if sel is None:
    rule = dict(version=None, upgrade=None, note='沒有版本通過誤升級條件 → 24 個月內 S1 翻案路徑不可行')
else:
    pp = p[sel]
    rule = dict(version=sel, rule=pp['rule'], up_looks=pp.get('up_looks'), b=pp.get('b'), e_thr=pp.get('e_thr'),
                boundary=f"z_n ≥ b·√(24/n)，n∈{pp.get('up_looks')}，b＝{pp.get('b'):.6f}" if pp['rule'] in ('R-FIX', 'R-OBF') else 'E_k ≥ 20')
pre = dict(
    utc=G.now_utc(),
    selected=sel, upgrade_rule=rule,
    sigma_design=G.SIGMA_D, c=p['v1']['c'], c_d=p['v1']['c_d'],
    down_rule=dict(looks='有效錨定月 n＝12、18、24（之後每 6 個月）', stat='z_n＝S_n／(σ_eff√n)', boundary=f"z_n ≤ −{p['v1']['c_d']:.6f}"),
    refit=D['refit'],
    iteration_summary={v: dict(pass_=IT['versions'][v]['pass_'], worst_p1=IT['versions'][v]['worst_p1'], by_dgp=IT['versions'][v]['by_dgp']) for v in IT['versions']},
    n_versions_tried=4,
    file_sha256={f: G.sha(G.HERE / f) for f in ('g8lib.py', 's1_explore.py', 's2_prereg_design.py', 's3_iterate.py', 's4_freeze.py')},
    data_sha256={'resid_post2019（去均值、round 1e-10）': G.canon_sha(np.load(G.SCR / 'resid_post2019.npy').tolist()),
                 'resid_2024': G.canon_sha(np.load(G.SCR / 'resid_2024.npy').tolist())},
)
sh = G.register('PREREG', pre)
print('PREREG', sh, sel, rule)

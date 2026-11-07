#!/usr/bin/env python3
"""步驟 4：登記 PREREG（集 A 的 b_a 與完整決策規格；之後才跑集 B／C、歷史示範與判定器正式測試）。
用法：python3 -I -B s4_prereg.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8alib as L

D0 = L.check_reg('PREREG_DESIGN', files=False)
res = L.load_res(); assert 'ITERATIONS' in res
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
assert L.sha(HERE / 'iterations.json') == res['ITERATIONS']['sha256']
spec = dict(
 prereg_design_sha256=res['PREREG_DESIGN_sha256'],
 params=dict(sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_V2, b_v2=L.B_V2, delta_w=L.DW, sig_max=L.SIG_MAX,
             b_a={v: it['versions'][v]['b_a'] for v in ('A1', 'A2', 'A3')}),
 calibration_summary={v: dict(driving_cells=it['versions'][v]['driving_cells'][:4], max_q_by_dgp=it['versions'][v]['max_q_by_dgp']) for v in ('A1', 'A2', 'A3')},
 versions=D0['versions'], b_rule=D0['b_rule'], qualification=D0['qualification'], selection=D0['selection'], oc_sets=D0['oc_sets'], seeds=D0['seeds'],
 predictions=D0['predictions'], governance=D0['governance'], fallback=D0['fallback'], final_gate=D0['final_gate'],
 note='A2 的格表條件（所在 σ 格 v2a OC 最大誤升級 ≤ 0.05）以集 B／C 的最終主口徑表為準；集 A 只用 σ_checked ≤ 10.4。',
 file_sha256={f: L.sha(HERE / f) for f in ('g8alib.py', 'refimpl.py', 'synth.py', 's3_iterate.py', 'iterations.json', 'd5_resid_sel.json')},
)
h = L.register('PREREG', spec)
print('PREREG', h, L.load_res()['PREREG_utc'], spec['params']['b_a'])

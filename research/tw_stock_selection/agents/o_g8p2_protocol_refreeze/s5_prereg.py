#!/usr/bin/env python3
"""步驟 5：PREREG（凍結四個版本的完整參數、Δ_w、正式 OC 規格與選版規則；之後才可跑集 B／C、歷史示範與判定器測試）。
用法：python3 -I -B s5_prereg.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g8lib2 as L

L.check_reg('PREREG_DESIGN', files=False)
R = L.load_res()
IT, EX = R['ITERATIONS'], R['EXPLORE']
VERS = {
    'V1': dict(stat='MAXSD', delta=0.0, upgrade=True, c_d=IT['c_d']['V1'], b=IT['b']['V1']['b']),
    'V2': dict(stat='HAC_T', delta=0.0, upgrade=True, c_d=IT['c_d']['V2'], b=IT['b']['V2']['b']),
    'V3': dict(stat='MAXSD', delta=L.DELTA_V3, upgrade=True, c_d=IT['c_d']['V3'], b=IT['b']['V3']['b']),
    'V4': dict(stat='LIT_ONLY', delta=0.0, upgrade=False, c_d=IT['c_d']['V4'], b=None),
}
P = dict(
    versions=VERS, delta_w=EX['delta_w'], sigma_d=L.SIGMA_D, c=L.C_INFL, sig_grid=L.SIG_GRID, sig_label=L.SIG_LABEL, s_grid=L.S_GRID,
    alphas=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    oc_final=dict(seed_B=2026110711, P_B=100_000, seed_C=2026110821, P_C=400_000,
                  cells=dict(main_false_up='α＝0 × D1/D2/D3 × 9 σ × 5 s（mode 0，μ＝s − Δ_w）',
                             strict='α＝0 × D1/D2/D3 × 9 σ × 5 s（mode 1，μ＝s；與 G8P1 對照共用路徑）',
                             g8p1='同 strict 路徑，G8P1 原樣',
                             down='α ∈ {−1, −0.5, 0} × 9 σ × D1/D2/D3，s＝0（s 序 0）與 s＝Δ_w（s 序 5）',
                             power='α ∈ {+0.5, +1, +2} × 9 σ × D1/D2/D3，s＝Δ_w（s 序 5）與 s＝0（s 序 0）',
                             sens='D3x、D4：α＝0 × 9 σ × 5 s（主口徑）；G8P1 在 D3x、D4（μ＝s）',
                             rerun='主口徑誤升級 |p−0.05| < 2SE_B 的（版本, 格）→ 集 C、P＝400,000 重跑，以 C 為準'),
                  seed_rule='cell_rng(base, DGP, σ 序, s 序, α 序, mode)；同一格各版本共用路徑'),
    criteria=R['PREREG_DESIGN']['criteria'], selection=R['PREREG_DESIGN']['selection'],
    simplicity_order=['V1', 'V3', 'V2'],
    code_sha256={f: L.sha(HERE / f) for f in ('g8lib2.py', 's3_explore.py', 's4_iterate.py', 's5_prereg.py', 'ro_snapshot.py')},
    iterations_sha256=L.canon_sha(IT), explore_sha256=L.canon_sha(EX),
    file_sha256={f: L.sha(HERE / f) for f in ('g8lib2.py',)},
    note='PREREG 之後不得改任何版本參數；集 B／C 只用來判定合格與選版。歷史 V7X 線套用只作 in-sample 規則示範（不是證據）。',
)
h = L.register('PREREG', P)
print('PREREG sha256', h, L.load_res()['PREREG_utc'])
print(json.dumps(VERS, ensure_ascii=False, indent=1))

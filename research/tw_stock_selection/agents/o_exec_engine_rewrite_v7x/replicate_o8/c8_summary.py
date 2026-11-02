#!/usr/bin/env python3
"""彙整：R0 對 V7 原表、R4 正式表、修正拆解、R4c、敏感度。輸出 out/summary.json。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import d8, s8

O = HERE / 'out'
T = json.load(open(d8.O7 / 'results.json'))['RESULTS_EXEC']['table']
TI = ('30萬', '100萬', '300萬', '1000萬'); AC = ('FIX', 'GROW')
CELLS = [(m, c, ti, a) for m in ('K5T', 'TICK') for c in ('normal', 'stress') for ti in TI for a in AC]


def ld(tag):
    p = O / f'run_{tag}.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else None


def per_j(run, cell, key):
    return np.array([run['stats']['|'.join((f'J{j}',) + cell)][key] for j in range(13)])


def jm(run, cell): return s8.jmedian([run['stats']['|'.join((f'J{j}',) + cell)] for j in range(13)])


out = {}
R0 = ld('R0')
if R0:
    rec = {}
    for cell in CELLS:
        mine = jm(R0, cell)
        v = [T['|'.join(('B10', f'J{j}') + cell)] for j in range(13)]
        rec['|'.join(cell)] = dict(full=mine['CAGR'] - float(np.median([x['CAGR'] for x in v])),
                                   sel=mine['CAGR_SEL'] - float(np.median([x['CAGR_SEL'] for x in v])),
                                   mine_full=mine['CAGR'], v7_full=float(np.median([x['CAGR'] for x in v])))
    out['R0_vs_V7'] = rec
    out['R0_sel_maxabs'] = max(abs(r['sel']) for r in rec.values())
    nn = {k: r['full'] for k, r in rec.items() if '|normal|' in k}
    ss = {k: r['full'] for k, r in rec.items() if '|stress|' in k}
    out['R0_full_normal_maxabs'] = max(abs(x) for x in nn.values())
    out['R0_full_normal_gt010'] = {k: round(x, 3) for k, x in nn.items() if abs(x) > 0.10}
    out['R0_full_stress_maxabs'] = max(abs(x) for x in ss.values())
    out['R0_full_stress_n_gt010'] = sum(abs(x) > 0.10 for x in ss.values())
    out['R0_acc'] = {k: v for k, v in R0['acc'].items() if 'K5T|normal|30萬|FIX' in k or 'K5T|normal|1000萬|GROW' in k}

R0n = ld('R0_nan')
if R0n and R0:
    out['diag_nanv60'] = {'|'.join(c): jm(R0n, c)['CAGR'] - float(np.median([T['|'.join(('B10', f'J{j}') + c)]['CAGR'] for j in range(13)]))
                          for c in CELLS if c[1] == 'normal'}

R4 = ld('R4')
if R4:
    tab = {}
    for ti in TI:
        for a in AC:
            c = ('K5T', 'normal', ti, a); m = jm(R4, c); st = jm(R4, ('K5T', 'stress', ti, a)); tk = jm(R4, ('TICK', 'normal', ti, a))
            win = (m['ex'] > 0) and (m['t'] >= 1.5) and (m['ex_Y19'] >= 0)
            tab[f'{ti}|{a}'] = dict(CAGR=m['CAGR'], p10=m['p10'], ex=m['ex'], t=m['t'], ex_SEL=m['ex_SEL'], ex_Y19=m['ex_Y19'], ex_VAL=m['ex_VAL'],
                                    HOLD=m['HOLD'], maxDD=m['maxDD'], stress=st['CAGR'], TICK_CAGR=tk['CAGR'], TICK_exY19=tk['ex_Y19'],
                                    E50=m['E50'], E50_HOLD=m['E50_HOLD'], verdict='贏' if win else ('輸' if m['ex'] < 0 else '無可辨識差異'))
    out['R4_table'] = tab
    out['R4_qoa_1000GROW'] = {k: v for k, v in R4['acc'].items() if 'K5T|normal|1000萬|GROW' in k}
    out['R4_n_disp_mdelay'] = dict(n_disp=float(np.median([v['n_disp'] for k, v in R4['acc'].items() if 'K5T|normal' in k])),
                                   n_mdelay=float(np.median([v['n_mdelay'] for k, v in R4['acc'].items() if 'K5T|normal' in k])))
    # 拆解
    runs = {k: ld(k) for k in ('R0', 'R1', 'R2', 'R3', 'R4', 'R4c')}
    dec = {}
    for nm, (a_, b_) in dict(F1=('R1', 'R0'), F2=('R2', 'R1'), F3=('R3', 'R2'), F4=('R4', 'R3'), ALL=('R4', 'R0'), F2c=('R4c', 'R4')).items():
        if runs[a_] is None or runs[b_] is None: continue
        for model in ('K5T', 'TICK'):
            for cost in ('normal', 'stress'):
                for ti in TI:
                    for a in AC:
                        c = (model, cost, ti, a)
                        d = per_j(runs[a_], c, 'CAGR') - per_j(runs[b_], c, 'CAGR')
                        dec[f'{nm}|{model}|{cost}|{ti}|{a}'] = float(np.median(d))
    out['decomp'] = dec
    pos = {k: v for k, v in dec.items() if v > 0.05 and not k.startswith(('ALL', 'F2c'))}
    out['falsify_positive_gt005'] = pos
    if runs['R4c']:
        out['R4c'] = {f'{ti}|{a}': dict(CAGR=jm(runs['R4c'], ('K5T', 'normal', ti, a))['CAGR'], ex=jm(runs['R4c'], ('K5T', 'normal', ti, a))['ex'],
                                        t=jm(runs['R4c'], ('K5T', 'normal', ti, a))['t'], ex_Y19=jm(runs['R4c'], ('K5T', 'normal', ti, a))['ex_Y19'])
                      for ti in TI for a in AC}
    # F3 敏感度（30萬 FIX，對 R4）
    for z in ('R4_z025_K5T|normal|30萬|FIX', 'R4_z100_K5T|normal|30萬|FIX'):
        r = ld(z)
        if r:
            c = ('K5T', 'normal', '30萬', 'FIX')
            out[f'sens_{z}'] = float(np.median(per_j(r, c, 'CAGR') - per_j(R4, c, 'CAGR')))
    # F3 30萬 FIX 分期
    if runs['R3'] and runs['R2']:
        c = ('K5T', 'normal', '30萬', 'FIX')
        out['F3_30FIX_SEL'] = float(np.median(per_j(runs['R3'], c, 'CAGR_SEL') - per_j(runs['R2'], c, 'CAGR_SEL')))
        out['F3_30FIX_Y19'] = float(np.median(per_j(runs['R3'], c, 'CAGR_Y19') - per_j(runs['R2'], c, 'CAGR_Y19')))

json.dump(out, open(O / 'summary.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ('R0_vs_V7', 'decomp', 'R0_acc', 'R4_qoa_1000GROW')}, indent=1, ensure_ascii=False))
if 'decomp' in out:
    for nm in ('F1', 'F2', 'F3', 'F4', 'ALL', 'F2c'):
        row = [f"{out['decomp'].get(f'{nm}|K5T|normal|{ti}|{a}', float('nan')):+.2f}" for ti in TI for a in AC]
        print(nm, ' '.join(row))

#!/usr/bin/env python3
"""迭代（只用挑選期 2014-01～2018-12）：6 版（λ∈{0.5,1,1.5} × 無上限／上限 ×2 下限 ×0.25）× T1／T2，13 個 j，一般成本。
選版規則（事前）：排除安慰劑 j 中位數 t < −1 的版本，取挑選期 j 中位數 IR（對 B、扣一般成本）最高者；T1、T2 各選一版。寫 iterations.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import tcore as T, tlib as L
SEL_A, SEL_B = '2014-01-01', '2018-12-31'
JS = list(range(13))
VERS = [(0.5, False), (1.0, False), (1.5, False), (0.5, True), (1.0, True), (1.5, True)]
EX = json.loads((L.HERE / 'explore.json').read_text(encoding='utf-8'))
IT = {'period': f'{SEL_A}～{SEL_B}', 'rule': '排除安慰劑 t_jmedian < −1；取 j 中位數 IR（T−B，一般成本）最高', 'versions': {}}
for kind in ('T1', 'T2'):
    for lam, cap in VERS:
        key = f'{kind}_lam{lam}_{"cap" if cap else "nocap"}'
        per = {}
        for j in JS + ['off']:
            ds = T.sig_days(j, SEL_A, SEL_B)
            assert max(ds[:-1]) <= pd.Timestamp(SEL_B)
            pre = T.precompute(ds, kind)
            PB = {d: (u, w) for d, u, w, tz in pre}
            PT = {}
            for d, U, wb, tz in pre:
                wt = wb.copy()
                for nm, (idx, z) in tz.items(): wt[idx] = T.tilt(wb[idx], z, lam, cap)
                PT[d] = (U, wt)
            rT, iT = T.run(ds, PT); rB, iB = T.run(ds, PB); b = T.bench_win(ds)
            sB = T.stats(rT - rB); s50 = T.stats(rT - b)
            per[str(j)] = {'vsB': sB, 'vs0050': s50, 'turnover': float(iT['turnover'][1:].mean()), 'cost_pp': float(iT['cost'][1:].mean() * 100),
                           'active_share': float(np.mean([np.abs(PT[d][1] - PB[d][1]).sum() / 2 for d in PT])), 'CAGR': T.cagr(rT), 'CAGR_B': T.cagr(rB)}
        jm = lambda f: float(np.median([f(per[str(j)]) for j in JS]))
        IT['versions'][key] = {
            'lam': lam, 'cap': cap, 'kind': kind,
            'jmedian': {'vsB_mean_pp': jm(lambda p: p['vsB']['mean_pp']), 'vsB_t': jm(lambda p: p['vsB']['t']), 'vsB_IR': jm(lambda p: p['vsB']['IR']),
                        'vsB_TE': jm(lambda p: p['vsB']['TE']), 'vs0050_mean_pp': jm(lambda p: p['vs0050']['mean_pp']), 'vs0050_t': jm(lambda p: p['vs0050']['t']),
                        'turnover': jm(lambda p: p['turnover']), 'cost_pp': jm(lambda p: p['cost_pp']), 'active_share': jm(lambda p: p['active_share'])},
            'official_day': per['off'], 'placebo_t_jmedian': EX['E5_placebo_gross_vs_B'][key]['t_jmedian'],
            'per_j_vsB_mean_pp': {j: per[str(j)]['vsB']['mean_pp'] for j in JS}}
        print(key, {k: round(v, 3) for k, v in IT['versions'][key]['jmedian'].items()}, flush=True)
for kind in ('T1', 'T2'):
    cand = {k: v for k, v in IT['versions'].items() if v['kind'] == kind and v['placebo_t_jmedian'] >= -1}
    best = max(cand, key=lambda k: cand[k]['jmedian']['vsB_IR'])
    IT[f'selected_{kind}'] = best
IT['n_versions_tested'] = len(IT['versions'])
IT['multiple_testing_note'] = '6 版 × 2 宇宙 × 13 j；另有 1 次早期計時試算（T1 λ1 無上限 j0，數字同本表）'
(L.HERE / 'iterations.json').write_text(L.dumps(IT), encoding='utf-8')
print('selected', IT['selected_T1'], IT['selected_T2'])

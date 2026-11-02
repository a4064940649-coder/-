#!/usr/bin/env python3
"""成本審查（只讀 out/）：壓力成本下的對隨機 t、周轉（實際 vs 隨機）、滑價兩平點、K5T 可執行格。
python3 -I -B review_cost.py → results.json['review_cost']"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from s4_stats import load, tstat, JK, summarize
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
low = R['main']['P1']['lower']
OUT = {}
for cfg in ['a', 'b']:
    z = load(f'run_{cfg}_MAIN_zero')
    if z is not None:
        s0 = summarize(z); OUT[f'{cfg}_zero_cost_gross'] = {k: s0[k] for k in ['t_jmed', 't_p10', 't_off', 'mean_pp_jmed', 'mean_pp_off']}
    st = load(f'run_{cfg}_MAIN_stress')
    if st is not None:
        s = summarize(st); OUT[f'{cfg}_stress'] = {k: s[k] for k in ['t_jmed', 't_p10', 't_off', 'mean_pp_jmed', 'vs0050_t_jmed', 'vs0050_mean_pp_jmed']}
    D = load(f'run_{cfg}_MAIN')
    # 單邊周轉（靜態權重差，窗間）：實際 vs 隨機平均名單
    tov_a, tov_r = [], []
    for k in JK:
        Wa = D[k]['Wa'].astype(float); tov_a.append(np.mean(0.5 * np.abs(np.diff(Wa, axis=0)).sum(axis=1)))
        Wr = D[k]['Wr'].astype(float); sat_r = np.mean(np.clip(Wr - D[k]['Rrep'].astype(float), 0, None).sum(axis=1)); tov_r.append(sat_r)
    OUT[f'{cfg}_random_satellite_weight_jmed（隨機每月幾乎全換，單邊周轉≈此值）'] = float(np.median(tov_r))
    OUT[f'{cfg}_turnover_actual_oneway_jmed'] = float(np.median(tov_a))
    # 兩平：均值超額 / (每月單邊周轉 × 2) ≈ 可額外承受的單邊滑價（相對隨機時，隨機的周轉更高，這裡保守只用實際周轉）
    m = np.median([(D[k]['ra'] - D[k]['Rr'].mean(axis=0)).mean() for k in JK])
    OUT[f'{cfg}_breakeven_extra_slip_oneway_vs_random'] = float(m / (2 * np.median(tov_a))) if np.median(tov_a) > 0 else None
    ex = HERE / 'out' / f'exec_{cfg}.json'
    if ex.exists(): OUT[f'{cfg}_K5T'] = json.loads(ex.read_text(encoding='utf-8'))
R['review_cost'] = OUT
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=float))

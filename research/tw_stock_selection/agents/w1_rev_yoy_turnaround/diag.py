#!/usr/bin/env python3
"""看過主結果後才加的拆解診斷（不影響判定）：版本 C 的超額來自營收條件還是季線（價格趨勢）條件。
執行：python3 -I diag.py（會先重跑 run_yoy.py，輸出不變），結果寫入 results_diag.json。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_yoy as R  # noqa: E402

base = R.sigs['B']; y20 = R.at_m(R.Y0 > 20); y0p = R.at_m(R.Y0 > 0)
VAR = {'C_noMA60(B且YoY>20)': R.at_m(R.SIG_C0), 'B且MA60(無>20%)': base & R.above, 'YoY>20且MA60(無連3增)': y20 & R.above,
       'MA60_only(股池內站上季線)': np.ones_like(base) & R.above, 'YoY>0且MA60': y0p & R.above, 'C(主設定，對照)': R.sigs['C']}
out = {}
t0, t1 = R.t0, R.t1
for name, sg in VAR.items():
    R.sigs['_x'] = sg
    ev, pools, info, nblk = R.build('_x', 1)
    o = {'n_pick_mean_select': round(float(info.n_pick[info.d < pd.Timestamp('2019-05-01')].mean()), 1), 'n_pick_mean_valid': round(float(info.n_pick[info.d >= pd.Timestamp('2019-05-01')].mean()), 1)}
    for cost in ('gross', 0.001):
        b, s = (0.0, 0.0) if cost == 'gross' else R.L.cost_rates(cost)
        nav, tov = R.sim_w(ev, b, s, t0, t1)
        o[f'cost_{cost}'] = {'tov': round(tov, 3), **{k: R.daily_stats(nav, R.W[k], R.E50, R.EW) for k in ('select_ex_IFRS', 'valid_2019-05_2026-10', 'y2024-01_2025-12', 'A_clean_to_2013-01sig', 'B_survivor_2014-05_2019-04')}}
    out[name] = o
    w = o['cost_0.001']
    print(name, o['n_pick_mean_select'], w['tov'], {k: (w[k]['CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m']) for k in ('select_ex_IFRS', 'valid_2019-05_2026-10', 'A_clean_to_2013-01sig', 'B_survivor_2014-05_2019-04')})
(HERE / 'results_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

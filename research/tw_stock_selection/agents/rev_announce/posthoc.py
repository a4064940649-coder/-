#!/usr/bin/env python3
"""rev_announce 第 3 步（事後拆解，不參與判定）：P2 配對差拆成 2021～2025 與 2026；2024-01～2026-10 年化（對照 main_refine 第 5 節）；
B 版只看 2026（15 日新制＋公告日資料）的影響。輸出 results.json 的 POSTHOC。"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent / 'main_refine'))
import rlib as R  # noqa: E402

res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
ND = pd.read_parquet(HERE / 'navs.parquet')
W = {'2021_2025': ('2021-01-01', '2025-12-31'), '2026': ('2026-01-01', '2026-10-31'), '2024_2026': ('2024-01-01', '2026-10-31'),
     '2006_2026': ('2006-04-01', '2026-10-31')}
out = {}
for n in (10, 30):
    for v in ('A_exFIN', 'B_pit_mc', 'B_pit_worst', 'AB', 'AB_worst'):
        k = f'{v}_n{n}'; o = {}
        for w, (a, b) in W.items():
            p = R.paired(ND[f'{k}|slip0.001'], ND[f'BASE_n{n}|slip0.001'], a, b)
            s1 = R.seg_stats(ND[f'{k}|slip0.001'], a, b); s0 = R.seg_stats(ND[f'BASE_n{n}|slip0.001'], a, b)
            o[w] = {'diff_pct_m': p['diff_pct_m'], 't': p['t_paired_m'], 'n_months': p['n_months'], 'CAGR': s1['CAGR'], 'BASE_CAGR': s0['CAGR'],
                    'maxDD': s1['maxDD'], 'BASE_maxDD': s0['maxDD']}
        out[k] = o
        print(k, {w: (x['diff_pct_m'], x['t'], round(x['CAGR'] * 100, 1), round(x['BASE_CAGR'] * 100, 1)) for w, x in o.items()})
res['POSTHOC'] = {'說明': '事後拆解，不參與判定', 'paired_by_window': out}
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

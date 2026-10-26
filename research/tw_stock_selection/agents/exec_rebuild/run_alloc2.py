#!/usr/bin/env python3
"""主研究員補測（第 1 輪後）：依長歷史檢驗把處置部位上限降為 20% 後的配置。
沿用 run_alloc.simulate 的引擎（M2 主策略 NAV、處置 10 槽固定金額、100 萬、滑價 0.1%），只改權重。
輸出 results_alloc2.json。"""
import json, sys, io, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
src = (Path(__file__).resolve().parent / 'run_alloc.py').read_text(encoding='utf-8')
head = src.split('res = {}')[0]                      # 只取定義，不重跑原網格
exec(compile(head, 'run_alloc_head', 'exec'))
NEW = {'選股30 100%': ('n30', {'main': 1.0, 'disp': 0.0, 'b50': 0.0}),
       '選股10 100%': ('n10', {'main': 1.0, 'disp': 0.0, 'b50': 0.0}),
       '選股30 80%＋處置 20%': ('n30', {'main': .8, 'disp': .2, 'b50': 0.0}),
       '選股10 80%＋處置 20%': ('n10', {'main': .8, 'disp': .2, 'b50': 0.0}),
       '0050 30%＋選股30 50%＋處置 20%': ('n30', {'main': .5, 'disp': .2, 'b50': .3}),
       '0050 50%＋選股30 50%': ('n30', {'main': .5, 'disp': 0.0, 'b50': .5})}
res = {}
for name, (nk, w) in NEW.items():
    for slip in (0.001, 'stress'):
        nav, _ = simulate(f'M2_{nk}_slip{slip}', w, 1e6, '0050', 'slots', slip)
        win = L.all_windows(nav, L.E50_NAV)
        res[f'{name}｜slip{slip}'] = {'windows': win, 'final_value_NTD': round(float(nav.iloc[-1] * 1e6))}
        print(name, slip, {k: (x['CAGR'], x['maxDD'], x.get('t_vs_0050_m')) for k, x in win.items() if x}, flush=True)
(OUT / 'results_alloc2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

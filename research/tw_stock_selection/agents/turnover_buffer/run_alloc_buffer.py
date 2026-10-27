#!/usr/bin/env python3
"""選定 K 的主策略放進 run_alloc2 的配置（100 萬、閒置持 0050、處置 10 槽、滑價 0.1%／壓力）。
沿用 exec_rebuild/run_alloc.py 的 simulate（exec 其定義段，不改原檔），只把主策略 NAV 換成本資料夾的緩衝版。
K = 原版時結果必須與 exec_rebuild/results_alloc2.json 相同（重現檢查）。另報網格內挑選期最佳但未達門檻的 K（僅參考）。
輸出 results_alloc_buffer.json。"""
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blib as B  # noqa: F401  （設定 sys.path：momentum_family、exec_rebuild）
ER = HERE.parent / 'exec_rebuild'
src = (ER / 'run_alloc.py').read_text(encoding='utf-8')
head = src.split('res = {}')[0]
exec(compile(head, 'run_alloc_head', 'exec'))          # 定義 L, D, MN, simulate …（只讀）
BN = pd.read_parquet(HERE / 'buffer_nav.parquet')
RB = json.loads((HERE / 'results_buffer.json').read_text(encoding='utf-8'))['B_MAIN_P1xREV']
ORIG2 = json.loads((ER / 'results_alloc2.json').read_text(encoding='utf-8'))
t0, t1 = L.pos(L.dates[0]), L.pos(L.dates[-1])
idx = L.IDX[t0:t1 + 1]

CONF = {'選股{n} 80%＋處置 20%': {'main': .8, 'disp': .2, 'b50': 0.0},
        '0050 30%＋選股{n} 50%＋處置 20%': {'main': .5, 'disp': .2, 'b50': .3},
        '選股{n} 100%': {'main': 1.0, 'disp': 0.0, 'b50': 0.0}}
runs = {}
for n in (30, 10):
    sel = RB[f'SELECTION_n{n}']
    runs[f'n{n}_K{sel["chosen_K"]}_chosen'] = (n, sel['chosen_K'])
    if sel['best_K'] != sel['chosen_K']:
        runs[f'n{n}_K{sel["best_K"]}_bestNotChosen_reference'] = (n, sel['best_K'])
    if n != sel['chosen_K']:
        runs[f'n{n}_K{n}_orig'] = (n, n)
res = {'_meta': {'note': '主策略部位 = turnover_buffer 的 B_MAIN_P1xREV 緩衝版 NAV；其餘同 run_alloc2（100 萬、idle 0050、處置 10 槽）。chosen = 事前規則選定；reference = 未達 +0.3 門檻、僅供參考。',
                 'runs': {k: list(v) for k, v in runs.items()}}}
for lab, (n, K) in runs.items():
    for slip in (0.001, 'stress'):
        key_main = f'M2_n{n}_slip{slip}'
        MN = MN.copy()
        MN[key_main] = BN[f'B_MAIN_P1xREV_n{n}_K{K}_{slip}'].reindex(idx).to_numpy()
        globals()['MN'] = MN
        for cname, w in CONF.items():
            if n == 10 and cname.startswith('0050'): continue
            name = cname.format(n=n)
            nav, _ = simulate(key_main, w, 1e6, '0050', 'slots', slip)
            win = L.all_windows(nav, L.E50_NAV)
            k = f'{lab}｜{name}｜slip{slip}'
            res[k] = {'windows': win, 'final_value_NTD': round(float(nav.iloc[-1] * 1e6))}
            ok = ORIG2.get(f'{name}｜slip{slip}')
            if K == n and ok:
                res[k]['reproduces_results_alloc2'] = all(abs(ok['windows'][w_]['CAGR'] - win[w_]['CAGR']) < 1e-4 for w_ in win if win[w_])
            f = win['full_2021-03_2026-10']; v = win['valid_2024-01_2026-10']
            print(k, 'full', f['CAGR'], f['maxDD'], f['t_vs_0050_m'], '| valid', v['CAGR'], v['maxDD'], v['t_vs_0050_m'], res[k].get('reproduces_results_alloc2', ''), flush=True)
(HERE / 'results_alloc_buffer.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

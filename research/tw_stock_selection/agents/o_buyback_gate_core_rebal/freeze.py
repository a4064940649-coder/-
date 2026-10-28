# -*- coding: utf-8 -*-
"""凍結：依 plan.json 的選版規則（只用 SEL）選出各 w 的頻率、追蹤誤差預算表的格子，寫 PREREG（sha256＋時間戳）進 results.json。"""
import sys, json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np
import rlib as L
a, b = L.PER['SEL']
it = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
g1 = [x for x in it['iterations'] if x['it'] == 'it1'][0]['cells']
sel = {}
for w in (20, 30, 40):
    cells = {k.split('|')[1]: v for k, v in g1.items() if k.startswith(f'{w}|')}
    best = max(v['paired_vs_M']['ann_diff_pp'] for v in cells.values())
    near = [r for r, v in cells.items() if v['paired_vs_M']['ann_diff_pp'] >= best - 0.05]
    order = {r: i for i, r in enumerate(L.RULES)}
    pick = sorted(near, key=lambda r: (cells[r]['n_trades'], order[r]))[0]
    sel[w] = dict(rule=pick, near_ties=near, SEL_ann_diff_vs_M=cells[pick]['paired_vs_M']['ann_diff_pp'], SEL_t=cells[pick]['paired_vs_M']['t'])
# 追蹤誤差預算表（SEL）
grid = {}
for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    for ru in (['M'] if w == 0 else L.RULES):
        r = L.sim(w, ru, a, b); s = L.stats(r['nav'])
        grid[f'{int(round(w*100))}|{ru}'] = dict(CAGR=s['CAGR'], TE=s['TE'], MDD=s['MDD'], longest_lag_m=s['longest_lag_m'], n_trades=r['n_trades'])
budget = {}
for T in (10, 12, 15, 18):
    ok = {k: v for k, v in grid.items() if v['TE'] <= T}
    if not ok: budget[T] = None; continue
    k = max(ok, key=lambda k: ok[k]['CAGR'])
    budget[T] = dict(cell=k, **ok[k])
PRE = dict(
    plan_sha256=json.loads((HERE / 'plan.json').read_text(encoding='utf-8'))['sha256'],
    gate_prereg_sha256=json.loads((HERE / 'gate_prereg.json').read_text(encoding='utf-8'))['sha256'],
    selected={str(k): v['rule'] for k, v in sel.items()},
    primary=dict(w0050=0.30, rule=sel[30]['rule'], text=L.PLAN['rules'][sel[30]['rule']]),
    selection_detail=sel,
    budget_cells={str(T): (v['cell'] if v else None) for T, v in budget.items()},
    judgment=L.PLAN['judgment'], cost=L.PLAN['cost'], periods={k: [str(x.date()) for x in v] for k, v in L.PER.items()},
    start_convention=L.PLAN['start_convention'], execution=L.PLAN['execution'],
    oos_runs='POST、VAL、HOLD 各跑一次；不得改規格。對照：每月同 w、主策略 100%、0050、同池等權、2330 30% 每月。',
    files_sha256={f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in ('rlib.py', 'main_navs.parquet', 'main_plan.json', 'build_main.py', 'iterations.json', 'explore.json')},
)
txt = json.dumps(PRE, ensure_ascii=False, sort_keys=True, default=float)
res_p = HERE / 'results.json'
res = json.loads(res_p.read_text(encoding='utf-8')) if res_p.exists() else {}
assert 'PREREG' not in res, 'PREREG 已存在，不可覆寫'
res['PREREG'] = json.loads(txt)
res['PREREG_sha256'] = hashlib.sha256(txt.encode()).hexdigest()
res['PREREG_time_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
res['SEL_grid_budget'] = grid
res['SEL_budget'] = {str(k): v for k, v in budget.items()}
res_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps(dict(sel=sel, budget=budget, sha=res['PREREG_sha256'], t=res['PREREG_time_utc']), ensure_ascii=False, indent=1, default=float))

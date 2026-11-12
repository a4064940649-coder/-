#!/usr/bin/env python3
"""第 1 步探索之二（只用挑選期 ≤2014-12-31）：敏感度格在挑選期的年化變動與下單日、順延鏈（只讀 run_engine 的彙總 pickle，不載入引擎）。併入 explore.json['E_grid_SEL14']。
python3 -I -B explore_grid.py"""
import json, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/csr')
CELLS = ['normal|30萬|std', 'normal|100萬|std', 'normal|300萬|std', 'normal|1000萬|std', 'stress|100萬|std', 'normal|30萬|wholelot']
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']; GRID = [(mo, mc) for mo in (0.5, 1.0, 2.0) for mc in (0.5, 1.0, 2.0)]
M = pickle.load(open(SCR / 'center_meta.pkl', 'rb')); G = pickle.load(open(SCR / 'grid_meta.pkl', 'rb')); T = pickle.load(open(SCR / 'tick_meta.pkl', 'rb'))
med = lambda x: float(np.median(x))
out = {}
for ck in CELLS:
    base = {v: med([M['pstats'][f'{v}|{j}|{ck}']['SEL14']['cagr'] for j in range(13)]) for v in VARS}
    def ent(src, pref):
        cg = {v: med([src['pstats'][f'{pref}|{v}|{j}|{ck}']['SEL14']['cagr'] for j in range(13)]) for v in VARS}
        sm = [src['summ'][f'{pref}|REAL_opt|{j}|{ck}']['SEL14'] for j in range(13)]
        return dict(dCAGR_SEL14={v: cg[v] - base[v] for v in VARS}, dCAGR_worst=min(cg.values()) - min(base.values()),
                    fill_days_share=med([s['n_fill_days'] / s['n_tdays'] for s in sm]), chain_td_p50=med([s['chain_td_p50'] or 0 for s in sm]),
                    chain_td_max=med([s['chain_td_max'] or 0 for s in sm]), drag_cost_pp=med([s['drag_cost'] / s['yrs'] * 100 for s in sm]), drag_odd_pp=med([s['drag_odd_amt'] / s['yrs'] * 100 for s in sm]))
    for (mo, mc) in GRID:
        if (mo, mc) == (1.0, 1.0): continue
        out[f'{ck}|mo={mo}|mc={mc}'] = ent(G, f'{mo}|{mc}|1.0')
    out[f'{ck}|tick=0.5'] = ent(T, '1.0|1.0|0.5')
    sm = [M['summ'][f'REAL_opt|{j}|{ck}']['SEL14'] for j in range(13)]
    out[f'{ck}|center'] = dict(CAGR_SEL14=base, fill_days_share=med([s['n_fill_days'] / s['n_tdays'] for s in sm]), chain_td_p50=med([s['chain_td_p50'] or 0 for s in sm]),
                               chain_td_max=med([s['chain_td_max'] or 0 for s in sm]), drag_cost_pp=med([s['drag_cost'] / s['yrs'] * 100 for s in sm]), drag_odd_pp=med([s['drag_odd_amt'] / s['yrs'] * 100 for s in sm]))
X = json.loads((HERE / 'explore.json').read_text(encoding='utf-8')); X['E_grid_SEL14'] = out
(HERE / 'explore.json').write_text(json.dumps(X, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
for k, v in out.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else {c: round(d, 3) for c, d in b.items()}) for a, b in v.items()})

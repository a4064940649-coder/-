#!/usr/bin/env python3
"""引擎閘門：同一份 V7 名單（AB_V7_B10、j0、seed 0）分別用 V7X 原資料層與本題合併面板資料層模擬，逐日淨值須一致（上市欄位口徑相同）。
用法：python3 -I -B s0b_gate_engine.py orig  → out/gate_orig.parquet；python3 -I -B s0b_gate_engine.py mine → 比對並寫 data/gate_engine.json"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; V7X = HERE.parent / 'o_exec_engine_rewrite_v7x'
FORMAL = dict(f1=True, f2=True, f3=0.5, f4=True, stress_odd_extra=True, rank_right=True)
CELLS = [('100萬', 1e6, 'GROW', 'normal'), ('1000萬', 1e7, 'FIX', 'normal'), ('300萬', 3e6, 'GROW', 'stress')]
if sys.argv[1] == 'orig':
    sys.path.insert(0, str(V7X))
    import x_data as D, x_exec as X
    P = D.plans('AB_V7_B10', js=[0], seeds=[0])[(0, 0)]
    out = {}
    for nm, t, m, c in CELLS:
        cf = dict(tier=t, mode=m, model='K5T', cost=c); cf.update(FORMAL)
        out[f'{nm}|{m}|{c}'] = X.Sim(P, cf).run()
    pd.DataFrame(out).to_parquet(HERE / 'out' / 'gate_orig.parquet')
else:
    sys.path.insert(0, str(HERE))
    import otlib as L
    D, X = L.D, L.X
    Pd = pd.read_csv(L.O7D / 'plans_j7.csv', dtype={'picks': str}); Pd = Pd[(Pd.strat == 'AB_V7_B10') & (Pd.seed == 0) & (Pd.j == 0)]
    P = sorted([(int(e), np.array([D.CI[c] for c in pk.split(',')], int), D.TPOS[pd.Timestamp(s)]) for s, e, pk in zip(Pd.signal, Pd.exec_idx, Pd.picks)], key=lambda x: x[0])
    O = pd.read_parquet(HERE / 'out' / 'gate_orig.parquet'); res = {}
    for nm, t, m, c in CELLS:
        cf = dict(tier=t, mode=m, model='K5T', cost=c); cf.update(FORMAL)
        nav = X.Sim(P, cf).run(); k = f'{nm}|{m}|{c}'
        a, b = nav.align(O[k], join='inner')
        res[k] = dict(max_rel_diff=float(np.nanmax(np.abs(a / b - 1))), end_mine=float(a.iloc[-1]), end_orig=float(b.iloc[-1]))
    (HERE / 'data' / 'gate_engine.json').write_text(json.dumps(res, indent=1))
    print(res)

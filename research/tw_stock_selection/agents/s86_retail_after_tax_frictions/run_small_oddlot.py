#!/usr/bin/env python3
"""S86 補充：小資 50 萬情境「幾乎全用零股」的合計（零股全期可買；每一筆個股成交每邊再加 0.3% 或 0.6%）。輸出 results_small_oddlot.json。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s86lib as L
from s86lib import stlib, COSTS
P, U, SG = L.setup((0, 1, 2, 3))
DY, DYM, NQ, dy50, e50ev, dinfo = L.div_arrays(P)
D0, DE = P.IDX[P.t0], P.IDX[-1]; D19 = P.IDX[P.IDX < pd.Timestamp('2019-01-01')][-1]
BASE = dict(sat_w=0.7, sizing='W10', lock=False, DY=DY, dy50=dy50)
FR = dict(div_tax=0.05, nhi_rate=0.0211, nhi_thr=20000.0, delay=20, fee_rate=L.FEE_FULL * 0.6)
out = {}
for cap in (3e5, 5e5):
    for nm, cost, kw in (('零股制度回推_不加摩擦', COSTS['strict'], {}), ('加稅費延後6折', COSTS['strict'], FR),
                         ('加稅費延後6折_全部零股+0.3%', dict(cmult=1.0, xslip=0.006, oddx=0.0), FR), ('加稅費延後6折_全部零股+0.6%', dict(cmult=1.0, xslip=0.009, oddx=0.0), FR)):
        v = [L.simulate_s86(P, SG[s], U, cost=cost, capital=cap, odd_from='all', **BASE, **kw)[0] for s in (0, 1, 2, 3)]
        out[f'{int(cap/1e4)}萬|{nm}'] = dict(c19=[float(stlib.cagr(n, D19, DE)) for n in v], full=[float(stlib.cagr(n, D0, DE)) for n in v])
    b = L.bench50(P, dy50, cap=cap, **FR)[0]
    out[f'{int(cap/1e4)}萬|0050_稅費後'] = dict(c19=[float(stlib.cagr(b, D19, DE))], full=[float(stlib.cagr(b, D0, DE))])
json.dump(out, open(Path(__file__).resolve().parent / 'results_small_oddlot.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items(): print(k, round(np.mean(v['c19']) * 100, 2), round(np.mean(v['full']) * 100, 2), [round(x * 100, 2) for x in v['c19']])

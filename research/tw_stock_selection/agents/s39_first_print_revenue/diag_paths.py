#!/usr/bin/env python3
"""S39 第 4 步（描述）：20 條路徑中位移 1～3（100 萬）各版與現行的交易差異明細，解釋路徑差從哪幾筆來。輸出 results.json 的 paths_trade_diff 鍵。python3 -I -B diag_paths.py"""
import sys, json, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s39lib
from s39lib import s17lib, s10lib, s4lib, stlib, COSTS

H = Path(__file__).resolve().parent
RES = H / 'results.json'
res = json.loads(RES.read_text())
P = s39lib.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
F = s17lib.rev_family(P, defs=['STD'])
s17lib.e50_fix_suspension(P)
fp = s39lib.load_fp()
RT = {m: dict(S=s39lib.surge_variant(P, F, fp, m), av=F['av']) for m in ('CUR', 'FPM', 'FPA')}
TA = P.IDX[P.IDX < pd.Timestamp('2024-07-01')][-1]
out = {}
for s in (0, 1, 2, 3):
    lgs = {}
    for v in RT:
        S1 = s10lib.make_signal_s10(P, U, RT[v], shift=s)
        S2, ekw = s10lib.variant_S(S1, 'E0')
        nav, lg, _ = s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=1e6, sat_w=0.7, sizing='W10', lock=False, **ekw)
        lg = lg[lg['ent_t'] > P.IDX.get_loc(TA)].copy()
        lg['code'] = [P.COLS[j] for j in lg['j']]
        lgs[v] = lg
    for v in ('FPM', 'FPA'):
        ka = set(zip(lgs[v]['code'], lgs[v]['ent_t'], lgs[v]['exit_t'])); kb = set(zip(lgs['CUR']['code'], lgs['CUR']['ent_t'], lgs['CUR']['exit_t']))
        rows = []
        for side, d, ks in ((v, lgs[v], ka - kb), ('CUR', lgs['CUR'], kb - ka)):
            for y in d.itertuples():
                if (y.code, y.ent_t, y.exit_t) in ks:
                    rows.append(dict(side=side, code=y.code, ent=str(P.IDX[y.ent_t].date()), exit=str(P.IDX[y.exit_t].date()), why=y.why, ret=round(float(y.proceeds / y.cost - 1), 4)))
        out[f'+{s}|{v}'] = rows
    gc.collect()
res['paths_trade_diff_100萬'] = out
RES.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
for k, v in out.items():
    print(k)
    for r in v:
        print('  ', r)

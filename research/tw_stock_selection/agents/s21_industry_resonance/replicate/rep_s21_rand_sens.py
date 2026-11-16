#!/usr/bin/env python3
"""隨機換股對照的設計敏感度（原研究未登記、未寫明隨機設計細節）：3 種隨機設計 × 20 次，主跑位移 0、100 萬。
輸出 replicate/results_rand_sens.json。python3 -I -B rep_s21_rand_sens.py"""
import sys, json, time, gc
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s21_lib as L
from rep_s21_lib import s20lib, s17lib, s10lib, s4lib, stlib, COSTS
H = Path(__file__).resolve().parent
P = L.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
G = s20lib.growth_features(P); RT = dict(S=G['S'], av=G['av']); del G, INNOV
s17lib.e50_fix_suspension(P)
PI = L.PITInd(P)
lastb = lambda d: P.IDX[P.IDX < pd.Timestamp(d)][-1]
DE, TB, D19 = P.IDX[-1], lastb('2022-01-01'), lastb('2019-01-01')
CELL = dict(sat_w=0.7, sizing='W10', lock=False)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
V, rows = L.build_variants(P, U, RT, PI, S0, 0, ['CUR', 'SWAP_P2'])
spec = {x['k']: (len(set(x['f1']) - set(x['new']['SWAP_P2'])), len(set(x['new']['SWAP_P2']) - set(x['f1']))) for x in rows}
del V
def run(Vr, cap, fresh=False):
    S2, ekw = L.variant_S(S0, *Vr['RAND'])
    if fresh:
        S2 = L.fresh(S2, None)
    return s10lib.simulate_s10(P, S2, U, cost=COSTS['strict'], capital=cap, **CELL, **ekw)[0]
navc = s10lib.simulate_s10(P, L.fresh(L.variant_S(S0, S0['sig'], S0['rank'])[0], None), U, cost=COSTS['strict'], capital=1e6, **CELL)[0]
ratio = float(navc.loc[TB] / 1e6)
FZ = dict(fresh_valid=0.3719730443528215, since2019=0.39333332459046755)
out = {}
for mode, seed in (('same_n_r20', 1), ('fill10_r20', 2), ('same_n_any', 3)):
    rng = np.random.default_rng(seed); D = []
    for i in range(20):
        Vr, _ = L.build_variants(P, U, RT, PI, S0, 0, ['RAND'], rng=rng, rand_spec=spec, rand_mode=mode)
        D.append(dict(fresh_valid=stlib.cagr(run(Vr, 1e6 / ratio, True), TB, DE), since2019=stlib.cagr(run(Vr, 1e6), D19, DE)))
        del Vr; gc.collect()
    d = pd.DataFrame(D)
    out[mode] = {c: dict(median=float(d[c].median()), p10=float(d[c].quantile(.1)), p90=float(d[c].quantile(.9)), FZ_beats=int((FZ[c] > d[c]).sum())) for c in d}
    print(mode, json.dumps(out[mode]), flush=True)
out['seeds'] = {'same_n_r20': 1, 'fill10_r20': 2, 'same_n_any': 3}
(H / 'results_rand_sens.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""P4b 全期（2006-04～2026-10-12）淨值：主策略隔日開盤 vs 買進改隔日收盤（賣出仍開盤）。凍結後執行。輸出 p4_full.json。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import olib as L
PR, SH, PT = L.assert_prereg()
K, V = L.K, L.V
IDX = L.IDX
YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
t0, t1, sd = S.bounds(); pl = S.plan(10, 'official'); pp = [(p[0], p[1]) for p in pl]
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
nons3 = {p[0] for p in pl if not (L.SRC3[0] <= IDX[p[0]] <= L.SRC3[1])}
out = {}
for cn, (bb, ss) in (('normal', K.rates(0.001)), ('stress', K.rates('stress'))):
    base = L.simulate_exec(pp, bb, ss, t0, t1)
    for nm, ad in (('buyclose_ex_src3', nons3), ('buyclose_all', None)):
        nav = L.simulate_exec(pp, bb, ss, t0, t1, 'close', 'open', apply_days=ad)
        o = {}
        for k, (a, b, adl) in list(V.SEGS.items()) + [('VAL', ('2019-05-01', '2026-10-12', '2019-05-01'))]:
            s0 = V.seg_stats(base, ew, a, b, sd, ad=adl, yearly=(k == 'ALL')); s1 = V.seg_stats(nav, ew, a, b, sd, ad=adl, yearly=(k == 'ALL'))
            pv, _ = L.paired(nav, base, sd, a, b)
            o[k] = {'open': {x: s0[x] for x in ('CAGR', 'maxDD', 't_0050', 't_EW')}, 'buyclose': {x: s1[x] for x in ('CAGR', 'maxDD', 't_0050', 't_EW')}, 'paired': pv}
            if k == 'ALL': o[k]['yearly_open'] = s0['yearly']; o[k]['yearly_buyclose'] = s1['yearly']
        out[f'{cn}|{nm}'] = o
        print(cn, nm, {k: (round(v['open']['CAGR'], 1), round(v['buyclose']['CAGR'], 1), round(v['open']['t_0050'], 2), round(v['buyclose']['t_0050'], 2), round(v['buyclose']['maxDD'], 1), round(v['paired']['t'], 2)) for k, v in o.items()})
(L.HERE / 'p4_full.json').write_text(L.jdump(out), encoding='utf-8')

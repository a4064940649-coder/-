#!/usr/bin/env python3
"""獨立複製 o6：缺口格清冊（只 import v4lib 引擎，不讀研究員本題程式）。
缺口格 = 前 300 池 ok_other 為真、但非 AB y3 缺值的股月；訊號日 2013-01-01～2021-12-31。
輸出 r_gap_cells.csv。"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_rev_gap_news_2013' / '..' / 'o_v4_revenue_patch_official'))
import numpy as np, pandas as pd
import v4lib as L
K = L.K
XR = L.setup_px()
YFX, Y5F = L.load_rfix()
GAP = L.gap_dict()
DMT = L.disp_mask_fn('trade')
S = L.V4(YFX, Y5F, DMT)
mc = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype=str)
MC = dict(zip(mc.code, mc.delist_date))
b10 = pd.read_csv(AG / 'o_v4_revenue_patch_official' / 'plan_V4_B10.csv', dtype=str)
B10 = {(r.signal, c) for r in b10.itertuples() for c in r.picks.split(',')}
v4p = pd.read_csv(AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv', dtype=str)
V4P = {(r.signal, c) for r in v4p.itertuples() for c in r.picks.split(',')}
rows = []
sds = [d for d in S.sig_days if pd.Timestamp('2013-01-01') <= d <= pd.Timestamp('2021-12-31')]
for d in sds:
    cs = S.cross_section(d)
    g = cs['ok_other'] & ~np.isfinite(cs['y3'])
    ds = d.strftime('%Y-%m-%d')
    for j in np.where(g)[0]:
        c = L.COLS[j]
        if c in MC and isinstance(MC[c], str) and pd.Timestamp(d) <= pd.Timestamp(MC[c]):
            ty = 'merger'
        elif c in GAP:
            ty = 'deterior'
        else:
            ty = 'nondelist'
        rows.append(dict(sig=ds, code=c, type=ty, in_B10=(ds, c) in B10, in_V4=(ds, c) in V4P))
G = pd.DataFrame(rows)
G.to_csv(HERE / 'r_gap_cells.csv', index=False)
print('sig days', len(sds), 'cells', len(G), 'stocks', G.code.nunique())
print(G.type.value_counts().to_dict())
print('2013-2020', (G.sig < '2021').sum(), 'nondelist', ((G.sig < '2021') & (G.type == 'nondelist')).sum())
print(G.sig.str[:4].value_counts().sort_index().to_dict())
print('B10 picks', G.in_B10.sum(), G[G.in_B10].type.value_counts().to_dict())
print(G[G.in_B10].code.value_counts().head(5).to_dict())

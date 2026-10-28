#!/usr/bin/env python3
"""研究員自查（不是正式獨立複製）：不 import proxy.py，用 pandas merge 重寫 N1 覆蓋率，對 gate 的 23.9% 比對。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rglib as G
K, V = G.K, G.V
b = G.base()
X = pd.read_csv(HERE / 'gap_cells_2013_2021.csv', dtype={'code': str})
M = pd.read_parquet(HERE / 'data' / 'mentions_P3.parquet'); M['ts'] = pd.to_datetime(M.ts)
d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); d['ym'] = d.month.str[:7]
d['pri'] = (d.market != 'sii').astype(int); d = d.sort_values(['ym', 'code', 'pri']).drop_duplicates(['ym', 'code'])
r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}).drop_duplicates(['ym', 'code'], keep='last')
lvl = {(c, y): v for c, y, v in zip(d.code, d.ym, d.revenue_k_twd) if v > 0}
for c, y, v in zip(r5.code, r5.ym, r5.revenue):
    if (c, y) not in lvl and v > 0: lvl[(c, y)] = v
YL, Y5 = b['YFX'], b['Y5F']
rows = []
for s, c in zip(X.sig, X.code):
    D = pd.Timestamp(s); lim = D + pd.Timedelta(hours=13, minutes=30); m = pd.Period(D, 'M')
    Y = Y5 if D >= V.SWITCH else YL
    nnews = 0; nval = 0
    for p in (m - 1, m - 2, m - 3):
        o = Y.at[p, c] if (c in Y.columns and p in Y.index) else np.nan
        if np.isfinite(o): nval += 1; continue
        g = M[(M.code == c) & (M.ym == str(p)) & (M.ts <= lim)].sort_values('ts')
        if len(g) == 0: continue
        ok = g.yoy.notna().any()
        if not ok and g.rev_k.notna().any():
            base = lvl.get((c, str(p - 12)))
            if base is None:
                gb = M[(M.code == c) & (M.ym == str(p - 12)) & (M.ts <= lim) & M.rev_k.notna()].sort_values('ts')
                base = gb.rev_k.iloc[0] if len(gb) else None
            ok = base is not None and base > 0
        if ok: nnews += 1; nval += 1
    rows.append(nnews >= 1 and nval >= 1)
r = np.array(rows)
gate = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))['GATE']['G1']['cov_all_732']
T = pd.read_csv(HERE / 'data' / 'proxy_table_gap_cells.csv', dtype={'code': str})
agree = float((r == np.isfinite(T.proxy_N1.to_numpy())).mean())
print(json.dumps({'rep_cov_N1': float(r.mean()), 'gate_cov_N1': gate, 'diff_pp': float((r.mean() - gate) * 100), 'cellwise_agree': agree}, indent=1))

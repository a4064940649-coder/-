import json, pandas as pd, numpy as np
B = '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/data/'
g = pd.read_csv(B + 'gap_cells_2006_2012.csv', dtype={'code': str})
o = dict(gap_cells=len(g), gap_days=g.sig.nunique(), gap_codes=g.code.nunique(), risk=int(g.risk.sum()), risk_by_s52pct=int((g.s52_pct >= 0.7).sum()))
f = pd.read_csv(B + 'fv_cells.csv', dtype={'code': str, 'ym': str})
f = f[f.ym >= '2013-01']
inc = (f.rel_first.abs() >= 0.005) & (f.rel_first.abs() <= 0.5)
cor = inc & (f.consensus | f.internal_ok)
o.update(fv_n=len(f), raw_rate=inc.mean() * 100, cor_rate=cor.mean() * 100, n_cor=int(cor.sum()))
t = pd.read_csv(B + 'top30_cells.csv', dtype={'code': str, 'ym': str})
k = set(zip(t.code, t.ym)); m = np.array([(c, y) in k for c, y in zip(f.code, f.ym)])
o.update(top30_n=int(m.sum()), top30_raw=inc[m].mean() * 100, top30_cor=cor[m].mean() * 100, k_top30_cor=int(cor[m].sum()),
         ratio_raw=inc[m].mean() / inc.mean(), ratio_cor=cor[m].mean() / cor.mean())
r = pd.read_csv(B + 'fv_replace.csv', dtype={'code': str}); o['n_replace'] = len(r)
print(json.dumps(o, indent=1))
json.dump(o, open('s4_counts.json', 'w'), indent=1)

#!/usr/bin/env python3
"""探索總表：彙整 explore_gap5.py、explore_fv.py 的輸出，加上缺口股的個股層級描述統計（挑選期 2006-04～2012-12，
個股訊號窗報酬對同池等權；不是策略報酬）。輸出 explore.json。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v5lib as L
L4, K, V = L.L4, L.K, L.V
L4.check_freeze(); L4.setup_px()
YFX, Y5F = L4.load_rfix()
S = L4.V4(YFX, Y5F, L4.disp_mask_fn('trade'))
G = pd.read_csv(HERE / 'data' / 'gap_cells_2006_2012.csv', dtype={'code': str})
# 個股層級：訊號日收盤 → 下一訊號日收盤（自算還原價，停牌／下市以最後收盤計），減同日 ok_other∪有營收池的等權
A = K.MYADJ.ffill()
sd = S.sig_days
nxt = {d: sd[i + 1] for i, d in enumerate(sd[:-1])}
rows = []
for d, g in G.groupby('sig'):
    d = pd.Timestamp(d)
    if d not in nxt: continue
    cs = S.cross_section(d)
    r = (A.loc[nxt[d]] / A.loc[d] - 1).to_numpy()
    pool = cs['ok_other']
    ew = np.nanmean(np.where(pool, r, np.nan))
    for c, p, dl in zip(g.code, g.s52_pct, g.delisted):
        rows.append((d, c, p, dl, r[K.CI[c]] - ew))
X = pd.DataFrame(rows, columns=['d', 'code', 's52_pct', 'delisted', 'ex'])
def bym(x):
    m = x.groupby('d').ex.mean()
    return {'n_cells': int(len(x)), 'n_months': int(len(m)), 'mean_pp': float(m.mean() * 100), 't_by_month': float(L4.tstat(m.to_numpy()))}
mech = {'all': bym(X), 's52_pct>=0.7': bym(X[X.s52_pct >= 0.7]), 's52_pct<0.7': bym(X[X.s52_pct < 0.7]),
        'delisted': bym(X[X.delisted]), 'not_delisted': bym(X[~X.delisted])}
E = json.load(open(HERE / 'explore_fv.json')); Gs = json.load(open(HERE / 'data' / 'gap_summary.json'))
out = {'gap_2006_2012': Gs, 'gap_stock_level_next_window_excess_vs_pool_EW（描述，不是策略報酬）': mech,
       'first_value': {k: v for k, v in E.items() if k != 'examples_disagree_top10'},
       'first_value_examples_top10': E['examples_disagree_top10'], 'sources': json.load(open(HERE / 'sources_log.json'))}
json.dump(out, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(mech, indent=1))

"""Tier gross active contributions (static close->close weights), top contributing stock, ex-top-stock t."""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/replicate_o8')
import numpy as np, pandas as pd
from rcore import *

W = World(); W.build_pre()
S = pd.read_parquet(OUT + '/series_exfix.parquet')


def tt(x):
    x = np.asarray(x, float); return x.mean() * 100, x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))


res = {}
for ver in ('T1', 'T2'):
    acc = {}
    for j in range(13):
        g = S[(S.ver == ver) & (S.j == str(j)) & (S.per >= '2019-01') & (S.per <= '2026-09')].sort_values('per')
        rows = []; stock = {}
        for _, r in g.iterrows():
            w = W.weights(r.d, pd.Period(r.per), ver)
            rr = ((1 + W.rt.loc[r.d:r.d1].iloc[1:].fillna(0)).prod() - 1).reindex(w['B'].index).fillna(0)
            act = (w['T'] - w['B']) * rr
            tiers = pd.Series(w['tiers'])
            row = act.groupby(tiers).sum().to_dict(); row['tot'] = act.sum()
            rows.append(row)
            for c, v in act.items():
                stock.setdefault(c, {})[len(rows) - 1] = v
        df = pd.DataFrame(rows).fillna(0)
        st = pd.DataFrame(stock).reindex(range(len(rows))).fillna(0)
        top = st.sum().idxmax()
        a = (g['T'] - g['B']).values
        o = {f'{k}': tt(df[k]) for k in df.columns}
        o['top'] = top; o['top_sum_pp'] = float(st[top].sum() * 100)
        o['ex2308_vsB'] = tt(a - st['2308'].values if '2308' in st else a)
        o['ex_top_vsB'] = tt(a - st[top].values)
        acc[j] = o
        print(ver, j, top, {k: tuple(round(x, 3) for x in v) for k, v in o.items() if isinstance(v, tuple)}, flush=True)
    keys = [k for k in acc[0] if isinstance(acc[0][k], tuple)]
    res[ver] = {k: [float(np.median([acc[j][k][0] for j in acc if k in acc[j]])), float(np.median([acc[j][k][1] for j in acc if k in acc[j]]))] for k in keys}
    res[ver]['tops'] = [acc[j]['top'] for j in acc]
    print(ver, json.dumps(res[ver]))
json.dump(res, open(OUT + '/tier.json', 'w'), indent=1)

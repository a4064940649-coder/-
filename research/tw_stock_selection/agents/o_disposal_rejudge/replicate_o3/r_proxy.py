import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_disposal_rejudge/replicate_o3")
import numpy as np, pandas as pd
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
adj = pd.read_parquet(P + 'adj.parquet'); idx = adj.index; T = len(idx)
tw = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].sort_index().reindex(idx).ffill()
A6 = adj / adj.shift(6) - 1; m6 = tw / tw.shift(6) - 1
att = (A6 >= 0.32) & (A6.sub(m6, axis=0) >= 0.20)
trig = (att & att.shift(1, fill_value=False) & att.shift(2, fill_value=False)).to_numpy()
ev = []
for c, col in enumerate(adj.columns):
    i = 0; x = trig[:, c]
    while i < T:
        if x[i]:
            s, e = i + 1, i + 10
            if e < T: ev.append((col, idx[i], idx[s], idx[e]))
            i = e + 1
        else: i += 1
px = pd.DataFrame(ev, columns=['code', 'ann', 'start', 'end'])
L = pd.read_csv('my_list.csv', dtype={'code': str}, parse_dates=['ann', 'start', 'end'])
real = L[L.srcl == 'twse']
def near(a, b, days=5):
    g = b.groupby('code')['start'].apply(lambda s: s.values)
    out = []
    for c, s in zip(a.code, a.start):
        v = g.get(c); out.append(v is not None and (np.abs((v - np.datetime64(s)) / np.timedelta64(1, 'D')) <= days).any())
    return np.array(out)
for lo, hi in (('2006-01-01', '2021-08-31'), ('2021-09-01', '2026-10-12')):
    p = px[(px.start >= lo) & (px.start <= hi)]; r = real[(real.start >= lo) & (real.start <= hi)]
    print(lo, hi, 'proxy', len(p), 'real', len(r), 'precision', round(near(p, real).mean(), 2), 'recall', round(near(r, px).mean(), 2))
fail = [2005, 2006, 2008, 2010, 2015, 2016, 2019]
add = px[px.start.dt.year.isin(fail)]
add = add[~near(add, L)].copy(); add['nth'] = 1; add['srcl'] = 'proxy'
print('proxy added', len(add))
AP = pd.concat([L, add], ignore_index=True)
AP.to_csv('my_list_AP.csv', index=False)

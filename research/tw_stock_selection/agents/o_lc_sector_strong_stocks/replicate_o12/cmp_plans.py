import sys, pickle; sys.path.insert(0, '.')
import pandas as pd, numpy as np
import rlib as R
p = pd.read_csv('../out/plans_SEL_CTRL_J0_J6_J12.csv.gz', dtype={'code': str, 'top_sectors': str})
for j in (0, 6, 12):
    D = pickle.load(open(f'out/plans_J{j}.pkl', 'rb'))
    for spec, key in (('SEL', 'IND_n2_cap_c15'), ('CTRL_SEL', 'CTRL_n2_cap_c15')):
        q = p[(p.spec == spec) & (p.j == j)]
        res = []
        for (t, P), info in zip(D['plans'], D['diag']):
            ds = str(R.IDX[t].date())
            g = q[q.signal == ds]
            if g.empty: res.append((ds, np.nan, np.nan, '')); continue
            th = dict(zip(g.code, g.weight))
            mine = {R.COLS[k]: 0.7 * v for k, v in P['B'].items()}
            for k, v in P[key].items(): mine[R.COLS[k]] = mine.get(R.COLS[k], 0) + 0.3 * v
            ks = set(th) | set(mine)
            l1 = sum(abs(th.get(k, 0) - mine.get(k, 0)) for k in ks)
            sec_ok = (g.top_sectors.iloc[0] == '|'.join(info['sel'])) if spec == 'SEL' else True
            res.append((ds, l1, sec_ok, g.top_sectors.iloc[0]))
        df = pd.DataFrame(res, columns=['d', 'l1', 'sec_ok', 'ts']).set_index('d')
        df.index = pd.to_datetime(df.index)
        pit = df['2014':]
        print(j, spec, 'n', len(df), 'L1 med all', round(df.l1.median(), 4), 'PIT med', round(pit.l1.median(), 4), 'PIT >0.05:', int((pit.l1 > 0.05).sum()), 'sec mismatch PIT', int((~pit.sec_ok.astype(bool)).sum()), 'all', int((~df.sec_ok.astype(bool)).sum()))
        if spec == 'SEL' and j == 6:
            print(pit[pit.l1 > 0.05].head(20))

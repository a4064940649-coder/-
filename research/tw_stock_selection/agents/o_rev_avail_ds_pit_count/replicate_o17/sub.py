import sys
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import numpy as np, pandas as pd
import v2_core as V
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_rev_avail_ds_pit_count/replicate_o17')
import r1
YL, Y5, Y5f = V.load_revenue()
m = V.V2('gapfix', YL, Y5, exfin=True, latest=False)
pl = m.plan(10, 'official')
rows = []
for e, pick, elidx, d in pl:
    if V.deadline_case(d) == 'D<S': continue
    cs = m.cross_section(d); el = cs['el']
    prv = pd.Series(np.where(el, cs['y3'], np.nan)).rank(pct=True).to_numpy()
    rm = str(pd.Period(d, 'M') - 1); S = d.date().isoformat(); ps = set(pick.tolist())
    for j in elidx:
        rows.append(dict(sig=S, code=V.COLS[j], sel=j in ps, prv=prv[j], st=r1.classify(V.COLS[j], rm, S)[0]))
D = pd.DataFrame(rows); D['av'] = D.st == 'AVAIL'
D['per'] = np.select([D.sig < '2014', D.sig < '2019', D.sig < '2024-06'], ['2006-2013', '2014-2018', '2019-2024-05'], '2024-06+')
for p, g in D.groupby('per'):
    pool = g.groupby('sig').av.mean().mean(); sel = g[g.sel].av.mean()
    print(p, f'sel {sel:.1%} pool(day-avg) {pool:.1%} pool(slot) {g.av.mean():.1%}', f'prv>=0.9 {g[g.prv>=0.9].av.mean():.1%} <0.5 {g[g.prv<0.5].av.mean():.1%}')
print('ALL prv>=0.9', f'{D[D.prv>=0.9].av.mean():.1%}', '<0.5', f'{D[D.prv<0.5].av.mean():.1%}')
D.to_csv('rep_pool_avail.csv', index=False)

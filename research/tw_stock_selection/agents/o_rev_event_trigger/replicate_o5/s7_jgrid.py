import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import rcore as C, numpy as np, pandas as pd, json
res = {}
for j in range(int(sys.argv[1]), 13):
    sd, end, am, sk = C.build_days(j)
    lst = []
    for s in range(10):
        S = C.mk(sd, end, unavail=C.make_unavail(am, s, mode='day'))
        n, ew, sdd, _ = C.run(S); lst.append(n['normal'])
    nv = C.mean_nav(lst)
    x = C.winx(nv, sdd); amx = pd.PeriodIndex([am[d] for d in x.index], freq='M')
    res[j] = dict(t=C.tstat(x.values), t19=C.tstat(x[amx >= pd.Period('2019-01', 'M')].values), skipped=len(sk))
    print(j, res[j], flush=True)
import sys as _s
if int(_s.argv[1]) > 0: raise SystemExit
print('median', np.median(t), 'p10', np.percentile(t, 10), 'max', t.max(), int(t.argmax()))
print('post2019 J0-4', np.mean([res[j]['t19'] for j in range(5)]), 'J8-12', np.mean([res[j]['t19'] for j in range(8, 13)]))
json.dump(res, open('jgrid.json', 'w'), indent=1)

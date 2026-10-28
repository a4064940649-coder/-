import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import time, json, pickle; T=time.time()
import rcore as C
import pandas as pd, numpy as np
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]
out = {}; meta = {}
def variant(name, j, mode, q=None, nseed=40, slips=()):
    sd, end, am, sk = C.build_days(j)
    navs = {'normal': [], 'stress': []}; ews = []; sn = {s: [] for s in slips}; per_seed = []
    for s in range(nseed):
        S = C.mk(sd, end, unavail=C.make_unavail(am, s, mode=mode, q=q))
        t0, t1, sdd = S.bounds(); pl = S.plan(10, 'official')
        for c in navs:
            nav, _ = C.L.sim_plan(pl, t0, t1, c, C.XR); navs[c].append(nav)
        for x in slips:
            nav, _ = C.L.sim_plan(pl, t0, t1, ('slip', x), C.XR); sn[x].append(nav)
        ew, _ = C.K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ews.append(ew)
        per_seed.append(navs['normal'][-1])
        if mode == 'none': break
    res = {c: C.mean_nav(v) for c, v in navs.items()}; res['EW'] = C.mean_nav(ews)
    for x in slips: res[f'slip{x}'] = C.mean_nav(sn[x])
    out[name] = res; meta[name] = dict(sd=sdd, am=am, end=end, seeds=per_seed)
    print(name, round(time.time()-T), flush=True)
variant('Jp1_AB', 1, 'day', slips=SLIPS)
variant('Jp1_N', 1, 'none')
variant('J0_AB', 0, 'day')
variant('J0_N', 0, 'none')
variant('J0_W', 0, 'worst', nseed=1)
variant('J0N_AB', 0, 'night', q=6/1821)
pickle.dump((out, meta), open('runs.pkl', 'wb'))
print('done', time.time()-T)

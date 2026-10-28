import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import time; T=time.time()
import rcore as C
import pandas as pd, numpy as np
ref = pd.read_parquet(C.AG/'o_v4_revenue_patch_official'/'navs_v4.parquet')
# official via my mk with official days
am = {d: pd.Period(d,'M') for d in C.OFF_DAYS+[C.OFF.end_day]}
S = C.mk(C.OFF_DAYS, C.OFF.end_day)
nv, ew, sd, pl = C.run(S, ('normal','stress'))
print('t plan', time.time()-T)
r = ref['V4|normal'].dropna()
print('V4 max rel diff', float((nv['normal'].reindex(r.index)/r-1).abs().max()))
st = C.seg_stats(nv['normal'], ew, sd, am, '2006-04','2026-10', C.OFF.end_day); print('V4', st)
x = C.L.seg(nv['normal'], ew, sd, '2006-04-01','2026-10-12'); print('L.seg', x['CAGR'], x['t_0050'])
# AB official 40 seeds with V.make_unavail
lst=[]; ews=[]
for s in range(40):
    S = C.mk(C.OFF_DAYS, C.OFF.end_day, unavail=C.V.make_unavail('mc', s))
    n_, e_, sd_, _ = C.run(S); lst.append(n_['normal']); ews.append(e_)
ab = C.mean_nav(lst); abe = C.mean_nav(ews)
r = ref['AB_V4|normal'].dropna()
print('AB_V4 max rel diff', float((ab.reindex(r.index)/r-1).abs().max()))
print('AB_V4', C.seg_stats(ab, abe, sd, am, '2006-04','2026-10', C.OFF.end_day))
print('cal t', C.tstat(C.cal_x(ab)))
ab.to_frame('AB_V4').assign(V4=nv['normal']).to_parquet('nav_off.parquet')
print('time', time.time()-T)

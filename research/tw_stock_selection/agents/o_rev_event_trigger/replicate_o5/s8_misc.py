import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import rcore as C, pandas as pd, numpy as np
off = pd.read_parquet('nav_off.parquet')['AB_V4']
print('AB_V4 SOP', C.cagr(off, '2023-12-29', '2026-10-12') * 100, C.cagr(C.E50, '2023-12-29', '2026-10-12') * 100)
sd, end, am, _ = C.build_days(1)
S = C.mk(sd, end, unavail=C.make_unavail(am, 0)); t0, t1, sdd = S.bounds(); pl = S.plan(10, 'official')
nav, lg = C.L.sim_plan(pl, t0, t1, 'normal', C.XR); print('J+1 turnover', lg['turnover'])
od = C.OFF_DAYS + [C.OFF.end_day]; oam = {d: pd.Period(d, 'M') for d in od}
lst = []
for s in range(40):
    S = C.mk(C.OFF_DAYS, C.OFF.end_day, unavail=C.make_unavail(oam, s)); n, ew, sdx, pl2 = C.run(S); lst.append(n['normal'])
    if s == 0: print('off turnover', C.L.sim_plan(pl2, *S.bounds()[:2], 'normal', C.XR)[1]['turnover'])
nv = C.mean_nav(lst)
print('official mix day-mode SEL', C.seg_stats(nv, ew, sdx, oam, '2006-04', '2018-12', C.OFF.end_day)['t'], 'ALL', C.seg_stats(nv, ew, sdx, oam, '2006-04', '2026-10', C.OFF.end_day)['t'])

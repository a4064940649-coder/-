import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import rcore as C, pandas as pd
sd, end, am, _ = C.build_days(1)
S = C.mk(sd, end, unavail=C.make_unavail(am, 0)); t0, t1, _ = S.bounds(); pl = S.plan(10, 'official')
for a, b in [('2006-04-01', '2026-10-13'), ('2024-01-01', '2026-10-13')]:
    r = C.V.top_contrib(pl, t1, a, b); print(a, r['top_code'], r['top_name'], round(r['top_share'], 3), round(r['ex_top_drop_pp'], 2))

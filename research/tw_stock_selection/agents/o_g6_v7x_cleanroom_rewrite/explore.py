# 探索（只用開發切片 2006-04～2008-12 的描述統計，不算報酬）：名單規模、零股占比、參與率、LIM 次數
import numpy as np, pandas as pd, json, math
import xdata, xengine, xlists, xstats
L = xlists.Lists(); W = xengine.World()
endd = xstats.last_le(L.dates, '2008-12-31')
ex = {'scope': '2006-04-10～2008-12-31、j0～2、seed 0～2（開發切片）；只計數，不算報酬', 'by_j': {}}
for j in range(3):
    days = [x for x in L.jdays(j) if x[1] < endd]
    ds = sum(1 for rm, si in days if L.dates[si] == pd.Timestamp(L.cal[L.cal.rev_month == rm].deadline_D.iloc[0]))
    plan = L.plan(days, seed=0)
    sim = xengine.Sim(W, plan, endd, dict(record=True, merger='orig')); sim.run()
    tr = sim.trades
    odd = sum(1 for (t, jj, n, c) in tr if abs(n) % 1000 != 0); lots = sum(1 for (t, jj, n, c) in tr if abs(n) >= 1000)
    part = []
    for (t, jj, n, c) in tr:
        L_ = abs(n) // 1000
        oa = xengine.SHARE[W.DEC[t, jj] - 1] * W.V[t, jj]
        if L_ > 0 and oa > 0: part.append(L_ * 1000 * W.PC[t, jj] / oa)
    lim = sum(1 for (si, e, pk) in plan for jj in pk if W.O[e, jj] >= W.UP[e, jj] - 1e-9)
    ex['by_j'][f'J{j}'] = {'signal_days': len(days), 'D_eq_S_days': ds, 'slots': sum(len(p[2]) for p in plan),
        'fills_with_odd_part': odd, 'fills_with_lots': lots, 'participation_Q_over_OA_median': float(np.median(part)) if part else None,
        'participation_gt10pct_share': float(np.mean(np.array(part) > 0.10)) if part else None, 'held_limitup_at_exec': lim,
        'n_fills': len(tr)}
json.dump(ex, open('explore.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(ex, ensure_ascii=False))

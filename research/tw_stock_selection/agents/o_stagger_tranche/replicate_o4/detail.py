import sys, pathlib, json, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
from math import erf, sqrt
OUT = {}
def pone(t): return float(1 - 0.5 * (1 + erf(t / sqrt(2))))
off = [(m, C.OFFICIAL[m]) for m in C.ANCHORS]
navs = {}
pl = C.make_plan(off, 10, None); navs['OFF'], lgO = C.tsim([pl], *C.NORMAL, T0, C.TEND)
navs['J1'], lgJ, plJ = run([1])
navs['V08'], lg8, pl8 = run([1, 6, 11, 16], 'a0', 'b0', 'c1')
for nm, cst in (('gross', C.ZERO), ('flat', C.FLAT), ('p03', C.P03)):
    navs['V08_' + nm] = run([1, 6, 11, 16], 'a0', 'b0', 'c1', cost=cst)[0]
# EW：同池等權（各批的合格池，零成本）
def ew_for(js, c):
    pls = plans_for(js)
    # 以池為 picks
    pools = []
    for pl_ in pls:
        q = []
        for (e, pk, m, s) in pl_:
            uid = None
            r = [v for kk, v in C._CS.items() if kk[0] == m and kk[1] == s]
            q.append((e, np.where(r[0]['el'])[0], m, s))
        pools.append(q)
    return C.tsim(pools, 0, 0, T0, C.TEND, c1_days=C1DAYS if c == 'c1' else None)[0]
ew8 = ew_for([1, 6, 11, 16], 'c1'); ew8c0 = ew_for([1, 6, 11, 16], 'c0')
plO = C.make_plan(off, 10, None)
ewO = C.tsim([[(e, np.where(C._CS[(m, s, None)]['el'])[0]) for (e, pk, m, s) in plO]], 0, 0, T0, C.TEND)[0]
PA = {'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12')}
def pstats(nav, ew=None):
    x = win(nav); r = {}
    for p, (a, b) in PA.items():
        y = x if p == 'ALL' else selstats(x, p)
        cg, mdd = C.cagr_mdd(nav, a, b)
        # 窗版年化：該期第一個官方日收盤到最後一窗終點
        ws = y.index[0]; we = [d for d in SD if d > y.index[-1]][0]
        yrs = (we - ws).days / 365.25
        cgw = (nav.loc[we] / nav.loc[ws]) ** (1 / yrs) * 100 - 100
        e50 = C.E50.reindex(nav.index)
        r[p] = dict(CAGR_date=cg, CAGR_win=cgw, mdd=mdd, t=C.tstat(y), n=len(y), ex_mean=float(y.mean() * 100), ex_sd=float(y.std() * 100),
                    E50_date=C.cagr_mdd(e50, a, b)[0], E50_win=(e50.loc[we] / e50.loc[ws]) ** (1 / yrs) * 100 - 100)
        if ew is not None:
            xe = C.excess(nav, SD, ew); ye = xe if p == 'ALL' else selstats(xe, p); r[p]['t_EW'] = C.tstat(ye)
    ct = C.cal_t(nav); r['t_cal_ALL'] = C.tstat(ct); r['t_cal_SEL'] = C.tstat(ct[ct.index <= pd.Period('2018-12', 'M')])
    r['TE_ALL'] = float(x.std() * math.sqrt(12) * 100)
    return r
for k, ew in (('OFF', ewO), ('J1', None), ('V08', ew8)):
    OUT[k] = pstats(navs[k], ew)
OUT['V08']['t_EW_c0pool'] = C.tstat(C.excess(navs['V08'], SD, ew8c0))
for k in ('V08_gross', 'V08_flat', 'V08_p03'): OUT[k] = pstats(navs[k])
OUT['turnover'] = dict(OFF=lgO['turnover'], J1=lgJ['turnover'], V08=float(np.mean([x[2] for x in lg8['tov'] if True][4:])))
# 配對差
def paired(a, b, p):
    d = C.winret(navs[a], SD) - C.winret(navs[b], SD)
    if p == 'SEL': d = selstats(d, 'SEL')
    elif p == 'POST': d = d[d.index >= pd.Timestamp('2019-01-01')]
    return dict(mean=float(d.mean() * 100), t=C.tstat(d), n=len(d))
OUT['paired'] = {f'{a}-{b}|{p}': paired(a, b, p) for a, b in (('V08', 'J1'), ('V08', 'OFF'), ('J1', 'OFF')) for p in ('SEL', 'POST', 'ALL')}
# 壓力下的配對差（V08 vs J+1 同成本）
for nm, cst in (('flat', C.FLAT), ('p03', C.P03), ('gross', C.ZERO)):
    nj = run([1], cost=cst)[0]; navs['J1_' + nm] = nj
    d = selstats(C.winret(navs['V08_' + nm], SD) - C.winret(nj, SD)); OUT['paired'][f'V08-J1|SEL|{nm}'] = float(d.mean() * 100)
# 端點
def endpoint(nav):
    x = win(nav); o = {'t': C.tstat(x), 'drop_last1': C.tstat(x.iloc[:-1])}
    srt = x.sort_values(ascending=False)
    for k in (1, 3, 5, 10): o[f'drop_best{k}'] = C.tstat(x.drop(srt.index[:k]))
    yr = x.groupby(x.index.year).mean(); by = int(yr.idxmax()); o['best_year'] = by; o['drop_best_year'] = C.tstat(x[x.index.year != by])
    yr2 = x.groupby([pd.Period(d, 'M').year for d in x.index]).sum(); o['best_year_sum'] = int(yr2.idxmax())
    o['p_one'] = pone(o['t'])
    return o
OUT['endpoint'] = {k: endpoint(navs[k]) for k in ('OFF', 'J1', 'V08')}
# B 判定
xJ = selstats(win(navs['J1'])); x8 = selstats(win(navs['V08']))
dd = selstats(C.winret(navs['V08'], SD) - C.winret(navs['J1'], SD))
best = dd.idxmax()
OUT['B5'] = dict(best_month=str(best.date()), B1=float(dd.drop(best).mean() * 100), B2=float(x8.drop(best).std() / xJ.drop(best).std() - 1) * 100,
                 B2_TE_only_V08=float(x8.drop(best).std() / xJ.std() - 1) * 100)
# 容量
def capacity(lg, plans, a, b):
    rows = []
    for (i, t, newj, T, tot) in lg['buys']:
        # 訊號日 t 的 60 日均成交值：成交日 t 的訊號日＝該批 plan 中 e==t 的 s
        rows.append((i, t, newj, T, tot))
    smap = {}
    for i, pl_ in enumerate(plans):
        for (e, pk, m, s) in pl_: smap[(i, e)] = (m, s)
    ratios = []
    for (i, t, newj, T, tot) in rows:
        m, s = smap[(i, t)]
        if not (pd.Period(a, 'M') <= m <= pd.Period(b, 'M')): continue
        v60 = C.V60[C.POS[s]]
        for j in newj: ratios.append(1e7 * T / tot / v60[j])
    r = np.array(ratios)
    return dict(n=len(r), share_gt1pct=float((r > 0.01).mean() * 100), median_pct=float(np.median(r) * 100))
OUT['capacity'] = {}
for nm, js, c in (('J1', [1], 'c0'), ('V01', [1, 11], 'c0'), ('V02', [1, 8, 15], 'c0'), ('V08', [1, 6, 11, 16], 'c1'), ('V03', [1, 6, 11, 16], 'c0'), ('V07', [1, 11], 'c1')):
    nv, lgx, pls = run(js, 'a0', 'b0', c)
    OUT['capacity'][nm] = dict(SEL=capacity(lgx, pls, '2006-04', '2018-12'), ALL=capacity(lgx, pls, '2006-04', '2026-10'))
json.dump(OUT, open(pathlib.Path(__file__).parent / 'detail.json', 'w'), indent=1, default=float)
pd.DataFrame(navs).to_parquet(pathlib.Path(__file__).parent / 'detail_navs.parquet')
print(json.dumps(OUT, indent=1, default=lambda v: round(float(v), 4)))

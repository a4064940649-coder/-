#!/usr/bin/env python3
"""主欄、各欄位、逐 seed、端點、統計、配對、機制、事件數。輸出 s1_main.json 與 navs（scratchpad）。"""
import sys, json, math, pickle
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5')
import numpy as np, pandas as pd
from multiprocessing import Pool
import rcore5 as R

HERE = '/home/user/-/research/tw_stock_selection/agents/o_v5_pit_revenue/replicate_o5/'
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o5/'
import os; os.makedirs(SP, exist_ok=True)
D = R.official_dates()
REVS = {'V4': R.RAW, 'V5': R.Rev('fv'), 'V5L': R.Rev('fv_latestbase')}

def ab_plans(args):
    rev, seed, bound, worst = args
    return R.build_plan(D, REVS[rev], U=R.unavail_mc(seed, worst=worst), bound=bound)

def runs(plans, cost='normal', slip=0.001):
    b, s = R.rates(cost, slip)
    return [R.run(p, b, s) for p in plans]

def summ(nav, plan, ew=None):
    W = R.wins(plan); ex = R.excess(nav, W); d0 = W[0][0]
    o = dict(cagr=R.cagr(nav, d0) * 100, t=R.tstat(ex), mdd=R.mdd(nav, d0) * 100, mean=ex.mean() * 100, sd=ex.std() * 100, n=len(ex))
    if ew is not None: o['t_ew'] = R.tstat(R.wret(nav, W) - R.wret(ew, W)); o['ew_cagr'] = R.cagr(ew, d0) * 100
    return o

if __name__ == '__main__':
    out = {}
    P = {}
    P['V4'] = R.build_plan(D, R.RAW); P['V5'] = R.build_plan(D, REVS['V5'])
    P['V5_B10'] = R.build_plan(D, REVS['V5'], bound='B10'); P['V5_B05'] = R.build_plan(D, REVS['V5'], bound='B05')
    with Pool(4) as pool:
        for key, rev, bound, worst in [('AB_V5', 'V5', None, False), ('AB_V4', 'V4', None, False), ('AB_V5_B10', 'V5', 'B10', False),
                                       ('AB_V5_B05', 'V5', 'B05', False), ('AB_V5L', 'V5L', None, False)]:
            P[key] = pool.map(ab_plans, [(rev, s, bound, worst) for s in range(40)])
        P['ABW_V5'] = [ab_plans(('V5', 0, None, True))]
    pl0 = P['V5']
    # EW（同池等權、零成本）
    ew = {}
    ew['V5pool_X'] = R.run(pl0, 0.0, 0.0, use_pool=True)
    ew['V5pool_noX'] = R.run(pl0, 0.0, 0.0, use_pool=True, X=False)
    NAV = {}
    for k in ['V4', 'V5', 'V5_B10', 'V5_B05']:
        NAV[k] = R.run(P[k], *R.rates()); NAV[k + '|stress'] = R.run(P[k], *R.rates('stress'))
    SEED = {}
    for k in ['AB_V5', 'AB_V4', 'AB_V5_B10', 'AB_V5_B05', 'AB_V5L', 'ABW_V5']:
        SEED[k] = runs(P[k]); NAV[k] = R.avg_nav(SEED[k])
    SEED['AB_V5|stress'] = runs(P['AB_V5'], 'stress'); NAV['AB_V5|stress'] = R.avg_nav(SEED['AB_V5|stress'])
    SEED['AB_V5|fullfee'] = runs(P['AB_V5'], 'fullfee'); NAV['AB_V5|fullfee'] = R.avg_nav(SEED['AB_V5|fullfee'])
    out['cols'] = {k: summ(v, pl0) for k, v in NAV.items()}
    out['ew'] = {k: summ(NAV['AB_V5'], pl0, v) for k, v in ew.items()}
    # 分段
    W = R.wins(pl0)
    out['seg'] = {k: R.seg(NAV['AB_V5'], W, a, b) for k, (a, b) in R.SEGS.items()}
    # 逐 seed
    def perseed(navs):
        ts = []; tdb = []
        for n in navs:
            ex = R.excess(n, W).to_numpy(); ts.append(R.tstat(ex)); tdb.append(R.tstat(np.sort(ex)[:-1]))
        ts = np.array(ts); tdb = np.array(tdb)
        return dict(med=float(np.median(ts)), min=float(ts.min()), max=float(ts.max()), p10=float(np.percentile(ts, 10)),
                    n_lt230=int((ts < 2.3).sum()), n_dropbest_lt230=int((tdb < 2.3).sum()), ts=ts.round(6).tolist())
    out['perseed'] = {k: perseed(SEED[k]) for k in ['AB_V5', 'AB_V5|stress', 'AB_V5_B10', 'AB_V4']}
    # 端點
    nav = NAV['AB_V5']; ex = R.excess(nav, W); v = ex.to_numpy(); srt = np.sort(v)[::-1]
    ep = dict(t=R.tstat(v), drop_last=R.tstat(v[:-1]))
    for q in (1, 3, 5, 10): ep[f'drop_best{q}'] = R.tstat(np.sort(v)[:-q])
    yr = ex.groupby(ex.index.year).sum(); by = int(yr.idxmax())
    ep['best_year'] = by; ep['drop_best_year'] = R.tstat(ex[ex.index.year != by])
    ep['drop2006'] = R.tstat(ex[ex.index.year != 2006]); ep['drop2007'] = R.tstat(ex[ex.index.year != 2007])
    for st in ('2007-01-01', '2008-01-01'):
        x = ex[ex.index >= st]
        prev = nav[nav.index < st].index[-1]
        ep['start' + st[:7]] = dict(t=R.tstat(x), n=len(x), cagr_from_prevclose=R.cagr(nav, prev) * 100, cagr_from_firstwin=R.cagr(nav, x.index[0]) * 100)
    ep['best_month'] = str(ex.idxmax().date()); ep['best_month_pp'] = float(ex.max() * 100)
    s = nav[nav.index <= R.END]; e5 = R.E50[s.index]
    me = s.groupby(s.index.to_period('M')).last(); be = e5.groupby(e5.index.to_period('M')).last()
    cm = (me.pct_change() - be.pct_change()).dropna()
    ep['cal_t'] = R.tstat(cm[cm.index >= pd.Period('2006-05', 'M')]); ep['cal_n'] = int((cm.index >= pd.Period('2006-05', 'M')).sum())
    # 第一個月含 2006-04-11→04-30
    first = (me.iloc[0] / nav[pl0 and R.IDX[pl0[0]['i_d']]] - 1) - (be.iloc[0] / R.E50[R.IDX[pl0[0]['i_d']]] - 1)
    cm2 = pd.concat([pd.Series([first], index=[me.index[0]]), cm])
    ep['cal_t_incl_apr'] = R.tstat(cm2)
    out['endpoint'] = ep
    out['nw'] = [R.nw_t(v, L) for L in (3, 6, 12)]
    out['boot_p'] = R.boot_p(v)
    out['bonf'] = {m: m * R.p1(R.tstat(v)) for m in (10, 20, 50)}
    # 配對
    def pair(a, b, segs=('S1', 'S2', 'S3', 'S4')):
        d = R.wret(a, W) - R.wret(b.reindex(a.index).ffill() if not b.index.equals(a.index) else b, W)
        o = dict(mean_pp=d.mean() * 100, t=R.tstat(d) if d.std() > 0 else None, n=len(d), nz=int((d.abs() > 1e-12).sum()),
                 sd_pp=d.std() * 100, mde_pp=2.8016 * d.std() * 100 / math.sqrt(len(d)),
                 dcagr_pp=(R.cagr(a, W[0][0]) - R.cagr(b.reindex(a.index).ffill(), W[0][0])) * 100)
        for k in segs:
            x0, x1 = R.SEGS[k]; dd = d[(d.index >= x0) & (d.index <= x1)]
            o[k] = dict(mean_pp=dd.mean() * 100, t=R.tstat(dd) if dd.std() > 0 else None, nz=int((dd.abs() > 1e-12).sum()))
        return o
    out['pair'] = {'AB_V5-AB_V4': pair(NAV['AB_V5'], NAV['AB_V4']), 'AB_V5_B10-AB_V5': pair(NAV['AB_V5_B10'], NAV['AB_V5']),
                   'V5-V4': pair(NAV['V5'], NAV['V4'])}
    mv = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')
    mv.index = pd.DatetimeIndex(mv.index)
    for c in ['EXFIN|n10|normal', 'AB_EXFIN|n10|mc']:
        out['pair']['AB_V5-main_v2[' + c + ']'] = pair(NAV['AB_V5'], mv[c].reindex(R.IDX).ffill().reindex(NAV['AB_V5'].index))
    # 首次值替換改變的前 10 名股月
    def changed(pa, pb):
        return [(str(R.IDX[a['i_d']].date()), sorted(set(a['picks']) - set(b['picks']))) for a, b in zip(pa, pb) if set(a['picks']) != set(b['picks'])]
    ch_non = changed(P['V5'], P['V4'])
    ch_ab = [changed(a, b) for a, b in zip(P['AB_V5'], P['AB_V4'])]
    out['fv_changed'] = dict(nonAB=len(ch_non), AB_mean_stockmonths=float(np.mean([sum(len(x[1]) for x in c) for c in ch_ab])),
                             AB_mean_months=float(np.mean([len(c) for c in ch_ab])),
                             AB_months=sorted({x[0] for c in ch_ab for x in c}))
    # B10 選進的缺口股股月
    def gapsel(pl): return [(R.IDX[p['i_d']], j) for p in pl for j in p['picks'] if j in p['gap']]
    g0 = gapsel(P['V5_B10']); gab = [gapsel(p) for p in P['AB_V5_B10']]
    out['b10_gap_sel'] = dict(nonAB=len(g0), nonAB_2006_2012=sum(1 for d, j in g0 if d.year <= 2012),
                              AB_mean=float(np.mean([len(x) for x in gab])), AB_mean_2006_2012=float(np.mean([sum(1 for d, j in x if d.year <= 2012) for x in gab])),
                              AB_union=len(set().union(*[set(x) for x in gab])), AB_union_2006_2012=sum(1 for d, j in set().union(*[set(x) for x in gab]) if d.year <= 2012))
    # 機制檢定：個股 成交日開盤 → 下一成交日開盤（停牌以最後收盤；不含 X）
    def stock_ret(i0, i1, j):
        if not (TR[i0, j] and np.isfinite(R.On[i0, j])): return np.nan
        g = R.Cn[i0, j] / R.On[i0, j] * np.prod(R.GROW[i0 + 1:i1, j])
        if TR[i1, j] and np.isfinite(R.On[i1, j]): g *= R.GROW[i1, j] * R.On[i1, j] / R.Cn[i1, j]
        return g - 1
    TR = R.TRADED
    def mech(plans_new, plans_old):
        rows = {}
        for pn, po in zip(plans_new, plans_old):
            en = [p['i_e'] for p in pn]
            nxt = dict(zip(en, en[1:] + [R.TPOS[R.END]]))
            mo = {p['i_d']: p for p in po}
            for p in pn:
                q = mo.get(p['i_d'])
                if q is None: continue
                a = set(p['picks']) - set(q['picks']); b = set(q['picks']) - set(p['picks'])
                if not a and not b: continue
                i0, i1 = p['i_e'], nxt[p['i_e']]
                ra = np.nanmean([stock_ret(i0, i1, j) for j in a]) if a else np.nan
                rb = np.nanmean([stock_ret(i0, i1, j) for j in b]) if b else np.nan
                rows.setdefault(R.IDX[p['i_d']], []).append(ra - rb)
        s = pd.Series({k: np.nanmean(v) for k, v in rows.items()}).sort_index().dropna()
        o = dict(n=len(s), mean_pp=s.mean() * 100, t=R.tstat(s) if len(s) > 1 else None, mde_pp=2.8016 * s.std() * 100 / math.sqrt(len(s)) if len(s) > 1 else None,
                 months=[str(x.date()) for x in s.index] if len(s) < 10 else None)
        for k in ('S1', 'S2', 'S3'):
            x0, x1 = R.SEGS[k]; ss = s[(s.index >= x0) & (s.index <= x1)]
            o[k] = dict(n=len(ss), mean_pp=ss.mean() * 100 if len(ss) else None)
        return o
    out['mech'] = {'B10_vs_AB_V5': mech(P['AB_V5_B10'], P['AB_V5']), 'FV_AB_V5_vs_AB_V4': mech(P['AB_V5'], P['AB_V4'])}
    json.dump(out, open(HERE + 's1_main.json', 'w'), ensure_ascii=False, indent=1, default=float)
    pickle.dump(dict(NAV=NAV, P=P, ew=ew), open(SP + 's1.pkl', 'wb'))
    print(json.dumps({k: out[k] for k in out if k != 'perseed'}, ensure_ascii=False, indent=1, default=float))
    print({k: {kk: vv for kk, vv in v.items() if kk != 'ts'} for k, v in out['perseed'].items()})

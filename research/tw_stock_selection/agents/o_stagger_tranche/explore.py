#!/usr/bin/env python3
"""探索（只用挑選期：錨定月 2006-04～2018-12；所有模擬只跑到 2019-01 錨定訊號日收盤，不碰 2019+）。"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import st_core as C
T0 = time.time()
res = json.loads((C.HERE / 'results.json').read_text(encoding='utf-8'))
assert 'PLAN_sha256' in res and 'PREREG' not in res
st = C.setup(); K, V, R = C.K, C.V, C.R
SEL_END = pd.Period('2018-12', 'M'); NEXT = pd.Period('2019-01', 'M')
JS = list(range(-8, 13))

def days_sel(j):
    days, _ = C.version_days(j)
    d = [(m, x) for m, x in days if m <= SEL_END]
    end = C.day_for(NEXT, j)
    k = 1
    while end is None:
        end = C.day_for(NEXT + k, j); k += 1
    return d, end

def off_sel():
    days, _ = C.official_list()
    d = [(m, x) for m, x in days if m <= SEL_END]
    nxt = [x for m, x in days if m > SEL_END][0]
    return d, nxt

def run(days, end, cost, seed=0, ab=True, w52=0.5, ew=True):
    pl = C.plan_single(days, seed=seed, ab=ab, w52=w52)
    t0 = C.IDX.get_loc(days[0][1]); t1 = C.IDX.get_loc(end)
    nav, _, lg = C.tsim([pl], *cost, t0, t1, xrules=st['XR'], log=True)
    W = C.windows_from(days, end)
    e = C.ew_nav(pl, t0, t1) if ew else None
    return nav, W, pl, e, lg

def ex_series(nav, W):
    return (C.wret(nav, W) - C.wret(C.E50.reindex(nav.index), W))

out = {'note': '全部只到挑選期終點（2019-01 錨定訊號日收盤）。t＝按錨定月分群。', 'scan': {}, 'scan_gross': {}}
EXN = {}; EXG = {}; EXP = {}; EXRV = {}
for j in JS:
    d, end = days_sel(j)
    navn, W, pl, ew, lg = run(d, end, C.NORMAL)
    navg, _, _, _, _ = run(d, end, C.GROSS, ew=False)
    bn = C.block(navn, W, C.SEL_KEYS, ew=ew); bg = C.block(navg, W, ['SEL'])
    ep = C.endpoint(navn, W, 'SEL')
    out['scan'][f'J{j:+d}'] = dict(n_windows=len(W), CAGR=bn['SEL']['CAGR'], maxDD=bn['SEL']['maxDD'], t_0050=bn['SEL']['t_0050'], t_cal=ep['t_cal'],
                                   t_EW=bn['SEL']['t_EW'], ex_mean_pp=bn['SEL']['ex_mean_pp'], ex_sd_pp=bn['SEL']['ex_sd_pp'],
                                   gross_ex_mean_pp=bg['SEL']['ex_mean_pp'], gross_t=bg['SEL']['t_0050'],
                                   S1=dict(CAGR=bn['S1']['CAGR'], t=bn['S1']['t_0050'], ex=bn['S1']['ex_mean_pp']),
                                   S2a=dict(CAGR=bn['S2a']['CAGR'], t=bn['S2a']['t_0050'], ex=bn['S2a']['ex_mean_pp']),
                                   Y2008=dict(ex=bn['Y2008']['ex_mean_pp']),
                                   turnover=C.turnover_month(lg, navn, C.IDX.get_loc(d[0][1])),
                                   n_old_rev=int(np.mean([C.cross(x, m)['nold'] for m, x in d])))
    EXN[j] = ex_series(navn, W); EXG[j] = ex_series(navg, W)
    # 機制拆解：只用價格腿、只用營收腿（扣成本前）
    nP, WP, _, _, _ = run(d, end, C.GROSS, w52=1.0, ew=False)
    nR, WR, _, _, _ = run(d, end, C.GROSS, w52=0.0, ew=False)
    EXP[j] = ex_series(nP, WP); EXRV[j] = ex_series(nR, WR)
    print(j, {k: round(v, 3) if isinstance(v, float) else v for k, v in out['scan'][f'J{j:+d}'].items() if k in ('CAGR', 't_0050', 'ex_mean_pp', 'gross_ex_mean_pp', 't_cal')},
          round(time.time() - T0), 's', flush=True)

# j=0 的 AB 抽樣 seed 1～4
out['J0_seeds'] = {}
for s in range(5):
    d, end = days_sel(0)
    nav, W, _, _, _ = run(d, end, C.NORMAL, seed=s, ew=False)
    b = C.block(nav, W, ['SEL'])['SEL']
    out['J0_seeds'][s] = dict(CAGR=b['CAGR'], t=b['t_0050'], ex=b['ex_mean_pp'])
# 官方日曆：V3 正式（未做 AB）與官方 AB（seed 0～4）
d, end = off_sel()
nav, W, pl, ew, _ = run(d, end, C.NORMAL, ab=False)
b = C.block(nav, W, C.SEL_KEYS, ew=ew); ep = C.endpoint(nav, W, 'SEL')
out['OFF'] = dict(CAGR=b['SEL']['CAGR'], maxDD=b['SEL']['maxDD'], t_0050=b['SEL']['t_0050'], t_cal=ep['t_cal'], ex_mean_pp=b['SEL']['ex_mean_pp'],
                  ex_sd_pp=b['SEL']['ex_sd_pp'], S1=b['S1']['t_0050'], S2a=b['S2a']['t_0050'])
out['OFF_AB'] = {}
for s in range(5):
    nav, W, _, _, _ = run(d, end, C.NORMAL, seed=s, ab=True, ew=False)
    b = C.block(nav, W, ['SEL'])['SEL']; ep = C.endpoint(nav, W, 'SEL')
    out['OFF_AB'][s] = dict(CAGR=b['CAGR'], t_0050=b['t_0050'], t_cal=ep['t_cal'], ex_mean_pp=b['ex_mean_pp'])

# 分布摘要
ts = np.array([out['scan'][f'J{j:+d}']['t_0050'] for j in JS]); tc = np.array([out['scan'][f'J{j:+d}']['t_cal'] for j in JS])
cg = np.array([out['scan'][f'J{j:+d}']['CAGR'] for j in JS])
def dist(x): return dict(min=float(x.min()), p10=float(np.percentile(x, 10)), p25=float(np.percentile(x, 25)), median=float(np.median(x)),
                         p75=float(np.percentile(x, 75)), p90=float(np.percentile(x, 90)), max=float(x.max()))
def pct_of(v, x): return float((x < v).mean() + 0.5 * (x == v).mean())
offab_t = float(np.mean([v['t_0050'] for v in out['OFF_AB'].values()]))
out['dist_SEL'] = dict(t_own=dist(ts), t_cal=dist(tc), CAGR=dist(cg),
                       OFF_t=out['OFF']['t_0050'], OFF_pctile_t=pct_of(out['OFF']['t_0050'], ts), OFF_AB_t_mean=offab_t, OFF_AB_pctile_t=pct_of(offab_t, ts),
                       J1_pctile_t=pct_of(out['scan']['J+1']['t_0050'], ts))

# (2) 營收新鮮度：錨定月配對
Mn = pd.DataFrame(EXN); Mg = pd.DataFrame(EXG)
def fresh(M):
    a = M[[1, 2, 3, 4, 5]].mean(axis=1); b = M[list(range(-8, 0))].mean(axis=1)
    dd = (a - b).dropna()
    o = dict(fresh_mean_pp=float(a.mean() * 100), stale_mean_pp=float(b.mean() * 100), diff_pp=float(dd.mean() * 100), t=C.tstat(dd), n=int(len(dd)))
    for k in ('S1', 'S2a', 'Y2008'):
        x = C.sel(dd, k); o[k] = dict(diff_pp=float(x.mean() * 100), t=C.tstat(x), n=int(len(x)))
    # 只看 J+1 對 J−1
    d1 = (M[1] - M[-1]).dropna(); o['J+1_minus_J-1'] = dict(pp=float(d1.mean() * 100), t=C.tstat(d1))
    d2 = (M[1] - M[12]).dropna(); o['J+1_minus_J+12'] = dict(pp=float(d2.mean() * 100), t=C.tstat(d2))
    return o
out['freshness'] = {'gross': fresh(Mg), 'normal': fresh(Mn),
                    'curve_gross_pp': {int(j): float(Mg[j].mean() * 100) for j in JS}, 'curve_normal_pp': {int(j): float(Mn[j].mean() * 100) for j in JS}}
Mp = pd.DataFrame(EXP); Mr = pd.DataFrame(EXRV)
out['legs'] = {'price_only': fresh(Mp), 'revenue_only': fresh(Mr),
               'curve_price_only_pp': {int(j): float(Mp[j].mean() * 100) for j in JS}, 'curve_revenue_only_pp': {int(j): float(Mr[j].mean() * 100) for j in JS},
               't_price_only': {int(j): C.tstat(Mp[j].dropna()) for j in JS}, 't_revenue_only': {int(j): C.tstat(Mr[j].dropna()) for j in JS}}
# (4) 時點雜訊：同一錨定月不同 j 的月超額橫斷面標準差
disp = Mn.std(axis=1, ddof=1) * 100
out['timing_dispersion_pp'] = {k: float(C.sel(disp, k).mean()) for k in ('SEL', 'S1', 'S2a', 'Y2008')}
out['timing_dispersion_pp']['S1_ex2008'] = float(disp[(disp.index >= pd.Period('2006-01', 'M')) & (disp.index <= pd.Period('2013-12', 'M')) & (disp.index.year != 2008)].mean())
# (5) 容量：k＝1～4（a0，一般成本），只算挑選期
PL = {1: [1], 2: [1, 11], 3: [1, 8, 15], 4: [1, 6, 11, 16]}
out['capacity_SEL'] = {}
for k, js in PL.items():
    pls = C.plan_tranches(js, 'a0', 'b0', max_m=SEL_END)
    t0 = min(pl[0][0] for pl in pls) - 1; t1 = max(pl[-1][0] for pl in pls) + 15
    nav, navs, lg = C.tsim(pls, *C.NORMAL, t0, t1, xrules=st['XR'], log=True)
    out['capacity_SEL'][k] = {'js': js, **C.capacity(lg['trades'], nav, t0), **{f'thr{q}': C.capacity(lg['trades'], nav, t0, thr=q)['share_gt'] for q in (0.005, 0.02, 0.05)}}
    print('cap', k, out['capacity_SEL'][k], flush=True)
(C.HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
pd.DataFrame({'normal': Mn.stack(), 'gross': Mg.stack()}).to_csv(C.HERE / 'explore_rows.csv')
print(json.dumps({k: out[k] for k in ('dist_SEL', 'freshness', 'timing_dispersion_pp', 'OFF', 'OFF_AB', 'J0_seeds')}, ensure_ascii=False, indent=1, default=float)[:6000])
print(json.dumps(out['legs'], ensure_ascii=False, default=float)[:3000])
print('done', round(time.time() - T0), 's')

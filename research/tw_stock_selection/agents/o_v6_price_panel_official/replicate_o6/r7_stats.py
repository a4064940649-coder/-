#!/usr/bin/env python3
"""複製員 o6：由 out_<panel>/navs.parquet 計算正式數字（跨循環、分段、HOLD、共同基期、逐 seed、端點、損益兩平、V6−V4 配對）。
用法：O6R_PANEL=compat python3 -I -B r7_stats.py"""
import os, sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import r4_env
PANEL = os.environ.get('O6R_PANEL', 'compat')
L = r4_env.load(PANEL); K = L.K
OUT = HERE / f'out_{PANEL}'
NV = pd.read_parquet(OUT / 'navs.parquet')
PJ = json.load(open(OUT / 'plans.json'))
sd = list(pd.DatetimeIndex(PJ['sig_days']))
V4 = pd.read_parquet(L.AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
sd4 = list(pd.to_datetime(pd.read_csv(L.AG / 'o_v4_revenue_patch_official' / 'plan_V4.csv')['signal'])) + [pd.Timestamp('2026-10-12')]
A_, B_ = '2006-04-01', '2026-10-12'
E50 = K.E50
res = {}
def r(x, k=4): return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), k)

def seg(name, a=A_, b=B_, ewk=None, nav=None, sdx=None):
    nav = NV[name] if nav is None else nav
    ew = NV[ewk or (name.split('|')[0] + '|EW')] if ewk != 'none' else NV['V6|EW']
    s = L.seg(nav.dropna(), ew.reindex(nav.dropna().index), sdx or sd, a, b)
    return {k: r(v) for k, v in s.items()}

# 1. 跨循環
cc = {}
for k in ('V6|normal', 'AB_V6|normal', 'V6_B10|normal', 'AB_V6_B10|normal', 'AB_V6|stress', 'AB_V6_B10|stress', 'ABW_V6|normal',
          'ABW_V6_B10|normal', 'V6|fullfee', 'V6_B10|stress', 'V6|stress', 'V6_B10|fullfee'):
    cc[k] = seg(k)
res['cross_cycle'] = cc
d4 = {}
for a6, a4 in (('AB_V6|normal', 'AB_V4|normal'), ('AB_V6_B10|normal', 'AB_V4_B10|normal'), ('V6|normal', 'V4|normal'), ('V6_B10|normal', 'V4_B10|normal'),
               ('AB_V6|stress', 'AB_V4|stress')):
    s4 = L.seg(V4[a4], V4[a4.split('|')[0] + '|EW'], sd4, A_, B_)
    d4[a6] = dict(V6=cc[a6]['CAGR'], V4=r(s4['CAGR']), V4_t=r(s4['t_0050']), dCAGR_pp=r(cc[a6]['CAGR'] - s4['CAGR']), dt=r(cc[a6]['t_0050'] - s4['t_0050']))
res['vs_V4'] = d4

# 2. 分段 AB_V6
res['segments_AB_V6'] = {p: seg('AB_V6|normal', a, b) for p, (a, b) in L.PER.items()}
res['segments_AB_V6_B10'] = {p: seg('AB_V6_B10|normal', a, b) for p, (a, b) in L.PER.items()}

# 3. HOLD 日曆歸屬：V6 格點上窗終點 ≥ 2024-01-02
def wins(nav, grid):
    g = pd.DatetimeIndex([d for d in grid if d in nav.index])
    v = nav.reindex(g)
    return pd.DataFrame({'start': g[:-1], 'end': g[1:], 'r': v.to_numpy()[1:] / v.to_numpy()[:-1] - 1})
def hold_cal(nav, grid, cut='2024-01-02', ctrl_end=None):
    w = wins(nav, grid); b = wins(E50.reindex(nav.index), grid)
    w['b'] = b['r'].to_numpy(); w['x'] = w['r'] - w['b']; w['lx'] = np.log((1 + w['r']) / (1 + w['b']))
    h = w[w['end'] >= pd.Timestamp(cut)]
    return w, h
for nm in ('AB_V6|normal', 'AB_V6_B10|normal'):
    w, h = hold_cal(NV[nm], sd)
    res[f'hold_calendar|{nm}'] = dict(n=len(h), t=r(L.tstat(h['x'].to_numpy())), mean_pp=r(h['x'].mean() * 100), cumlog=r(h['lx'].sum()),
                                      first=str(h['start'].iloc[0].date()), last_end=str(h['end'].iloc[-1].date()))
    hs = w[w['start'] >= pd.Timestamp('2024-01-01')]
    res[f'hold_start|{nm}'] = dict(n=len(hs), t=r(L.tstat(hs['x'].to_numpy())), mean_pp=r(hs['x'].mean() * 100), cumlog=r(hs['lx'].sum()))
# 共同基期 2023-12-29 → 2026-10-12
def cg(nav, a='2023-12-29', b='2026-10-12'):
    nav = nav.dropna(); a, b = pd.Timestamp(a), pd.Timestamp(b)
    return r(((nav.loc[b] / nav.loc[a]) ** (365.25 / (b - a).days) - 1) * 100)
res['common_base'] = {'AB_V6': cg(NV['AB_V6|normal']), 'AB_V4': cg(V4['AB_V4|normal']), '0050': cg(E50),
                      'AB_V6_B10': cg(NV['AB_V6_B10|normal']), 'AB_V4_B10': cg(V4['AB_V4_B10|normal'])}

# 4. 逐 seed
ST = json.load(open(OUT / 'seed_t.json'))
def sdist(x):
    x = np.array(x); return dict(median=r(np.median(x)), min=r(x.min()), max=r(x.max()), lt2=int((x < 2.0).sum()), lt23=int((x < 2.3).sum()))
res['seeds'] = {f'{k}|{c}': sdist(v) for k, d in ST.items() for c, v in d.items()}

# 5. 端點
def xs(nav, sdx=sd, a=A_, b=B_): return L.excess(nav, E50, sdx, a, b)
def cal_t(nav, a='2006-04', b='2026-10'):
    me = nav.groupby(nav.index.to_period('M')).last(); be = E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    x = (me.pct_change() - be.pct_change()).dropna()
    return L.tstat(x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))].to_numpy())
def endpoint(nav):
    x = xs(nav); o = {'t': L.tstat(x), 'n': len(x), 'drop_last1': L.tstat(x.iloc[:-1])}
    srt = x.sort_values(ascending=False)
    for q in (1, 3, 5, 10): o[f'drop_best{q}'] = L.tstat(srt.iloc[q:])
    o['drop_first12'] = L.tstat(x.iloc[12:])
    o['drop_first_year_2006'] = L.tstat(x[x.index.year > 2006])
    o['drop_first_365d'] = L.tstat(x[x.index >= x.index[0] + pd.Timedelta(days=365)])
    o['start_2008'] = L.tstat(x[x.index >= pd.Timestamp('2008-01-01')])
    o['start_2007'] = L.tstat(x[x.index >= pd.Timestamp('2007-01-01')])
    o['cal_t'] = cal_t(nav)
    o['cal_t_from2006_05'] = cal_t(nav, '2006-05')
    return {k: r(v) if not isinstance(v, int) else v for k, v in o.items()}
res['endpoint'] = {k: endpoint(NV[k]) for k in ('AB_V6|normal', 'AB_V6_B10|normal')}

# 6. 損益兩平
SL = sorted(float(c.split('slip')[1]) for c in NV.columns if c.startswith('AB_V6|slip'))
def be(name):
    pts = [(s, L.seg(NV[f'{name}|slip{s}'], NV[f'{name}|EW'], sd, A_, B_)['t_0050']) for s in SL]
    def cross(th):
        for (s1, a), (s2, b) in zip(pts, pts[1:]):
            if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
        return None
    return {'slip_t2.3_pct': r(cross(2.3) * 100) if cross(2.3) else None, 'slip_t2.0_pct': r(cross(2.0) * 100) if cross(2.0) else None,
            'grid': [(s, r(t, 3)) for s, t in pts]}
res['breakeven'] = {k: be(k) for k in ('AB_V6', 'AB_V6_B10')}

# 7. V6 − V4 配對（V4 格點 245 窗）
sys.path.insert(0, str(L.AG / 'o_main_spa_breaks'))
import fam_common as FC
sys.path.pop(0)
def pair(a6, a4):
    d = (FC.win_ret(NV[a6]) - FC.win_ret(V4[a4])).dropna()
    nz = d[d.abs() > 1e-12]
    big = d.abs().idxmax()
    o = dict(n=len(d), mean_pp=r(d.mean() * 100), t=r(L.tstat(d.to_numpy())), n_nonzero=len(nz),
             first_nz=str(nz.index[0].date()) if len(nz) else None, last_nz=str(nz.index[-1].date()) if len(nz) else None,
             max_win=str(big.date()), max_pp=r(d.loc[big] * 100), t_drop_max=r(L.tstat(d.drop(big).to_numpy())))
    return o
res['paired_V6_minus_V4'] = {k: pair(f'{k.replace("V4", "V6")}|normal', f'{k}|normal') for k in ('AB_V4', 'AB_V4_B10', 'V4', 'V4_B10')}
json.dump(res, open(OUT / 'stats.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:12000])

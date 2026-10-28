#!/usr/bin/env python3
"""反方審查（資料與統計角度）：o_v6_price_panel_official。只讀研究員檔案，不改任何東西。
python3 -I -B review_stat_o6.py"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

H = Path(__file__).resolve().parent
AG = H.parent
END = pd.Timestamp('2026-10-12')
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
TCAL = pd.DatetimeIndex(sorted(set(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date'])))
NV = pd.read_parquet(H / 'navs_v6.parquet'); NJ = pd.read_parquet(H / 'navs_jscan_v6.parquet')
N4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
plan = pd.read_csv(H / 'plan_V6.csv')
G6 = pd.DatetimeIndex(list(pd.to_datetime(plan['signal'])) + [END])

def sample(nav, days):
    s = nav.dropna(); s = s[~s.index.duplicated(keep='last')]
    v = s.reindex(s.index.union(days)).ffill().reindex(days)
    v[days < s.index[0]] = np.nan
    return v
def wr(nav, g):
    v = sample(nav, g).to_numpy(); return pd.Series(v[1:] / v[:-1] - 1, index=g[:-1])
def t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))
def nw_t(x, lag):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x); e = x - x.mean()
    g0 = e @ e / n; s = g0
    for l in range(1, lag + 1): s += 2 * (1 - l / (lag + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / math.sqrt(s / n))
def xs(nav, g, a='2006-04-01', b=END):
    x = wr(nav, g) - wr(e50, g); return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()

out = {}
# ---------- A. 標題 t 重算＋自相關／HAC ----------
A = {}
for k in ('V6|normal', 'AB_V6|normal', 'AB_V6_B10|normal', 'AB_V6|stress'):
    x = xs(NV[k], G6)
    ac = [float(pd.Series(x.to_numpy()).autocorr(l)) for l in (1, 2, 3)]
    lx = np.log1p(wr(NV[k], G6)) - np.log1p(wr(e50, G6)); lx = lx[lx.index >= '2006-04-01'].dropna()
    A[k] = {'n': len(x), 't': t(x), 'NW3': nw_t(x, 3), 'NW6': nw_t(x, 6), 'NW12': nw_t(x, 12), 'acf1_3': ac, 't_logexcess': t(lx),
            'mean_pp': float(x.mean() * 100)}
out['A_headline_t'] = A

# ---------- B. 換股日：官方格點 vs j 格點 ----------
def deadline(m):
    sp = {'2019-02': '2019-02-15', '2023-04': '2023-04-12'}
    ts = pd.Timestamp(sp.get(str(m), f'{m.year}-{m.month:02d}-10'))
    return TCAL[TCAL.searchsorted(ts)]
off = pd.to_datetime(plan['signal'])
jpos = []
for d in off:
    D = deadline(pd.Period(d, 'M')); jpos.append(int(TCAL.get_loc(d) - TCAL.get_loc(D)) if d in TCAL else None)
B = {'official_equals_j_counts': pd.Series(jpos).value_counts().sort_index().to_dict()}
# 每個 j 的格點（簡化：不做 GOOD 日過濾，對 t 影響小；只當交叉評價用）
def jgrid(j):
    days = [TCAL[TCAL.get_loc(deadline(pd.Period(d, 'M'))) + j] for d in off]
    return pd.DatetimeIndex(days + [END])
cross = {}
for col in ('V6|normal', 'AB_V6|normal'):
    base = col.split('|')[0]
    for j in (0, 1, 2, 3):
        gj = jgrid(j); nj = NJ[f'{base}|normal|J{j}']
        cross[f'{base}|J{j}'] = {'jnav_on_jgrid_t': t(xs(nj, gj)), 'jnav_on_official_grid_t': t(xs(nj, G6)),
                                 'official_nav_on_jgrid_t': t(xs(NV[col], gj)), 'official_nav_on_official_grid_t': t(xs(NV[col], G6))}
    # 日報酬層級（與格點無關）：日超額的 t（日資料 NW 20）
    for j in (0, 1):
        r = NJ[f'{base}|normal|J{j}'].pct_change(); ro = NV[col].pct_change(); r5 = e50.reindex(ro.index).ffill().pct_change()
        m = (ro.index >= '2006-04-11') & (ro.index <= END)
        cross[f'{base}|daily_J{j}'] = {'official_daily_excess_NW20': nw_t((ro - r5)[m].dropna(), 20),
                                      f'J{j}_daily_excess_NW20': nw_t((r.reindex(ro.index) - r5)[m].dropna(), 20),
                                      'official_minus_J_daily_mean_bp': float((ro - r.reindex(ro.index))[m].mean() * 1e4),
                                      'official_minus_J_daily_t': nw_t((ro - r.reindex(ro.index))[m].dropna(), 20)}
B['cross_grid'] = cross
out['B_rebalance_day'] = B

# ---------- C. 集中度 ----------
Cc = {}
for k in ('AB_V6|normal', 'AB_V6_B10|normal'):
    x = xs(NV[k], G6); yr = x.groupby(x.index.year).sum().sort_values(ascending=False)
    o = {'top_years_sum_pp': {int(a): round(b * 100, 1) for a, b in yr.head(5).items()}, 'total_sum_pp': float(x.sum() * 100)}
    for nd in (1, 2, 3):
        drop = list(yr.index[:nd]); o[f't_drop_top{nd}_years'] = t(x[~x.index.year.isin(drop)])
    xx = x.sort_values(ascending=False)
    o['share_of_total_from_top10_windows'] = float(xx.head(10).sum() / x.sum())
    o['share_of_total_from_top20_windows'] = float(xx.head(20).sum() / x.sum())
    o['t_2009_on'] = t(x[x.index >= '2009-01-01']); o['t_2010_on'] = t(x[x.index >= '2010-01-01'])
    o['t_pre2019'] = t(x[x.index < '2019-01-01']); o['t_2019_on'] = t(x[x.index >= '2019-01-01'])
    Cc[k] = o
out['C_concentration'] = Cc

# ---------- D. 保留期：只靠一窗？選擇偏誤收縮後？ ----------
rm = wr(NV['AB_V6|normal'], G6); r5 = wr(e50, G6); ends = pd.Series(G6[1:], index=G6[:-1])
lx = (np.log1p(rm) - np.log1p(r5)); ax = rm - r5
pre = ((ends < pd.Timestamp('2024-01-02')) & (rm.index >= pd.Timestamp('2006-04-01'))).to_numpy()
hold = (ends >= pd.Timestamp('2024-01-02')).to_numpy()
def sb_paths(n_src, n_out, L, Bn, seed):
    rng = np.random.default_rng(seed); p = 1 / L
    idx = np.empty((Bn, n_out), np.int64); idx[:, 0] = rng.integers(0, n_src, Bn)
    for tt in range(1, n_out):
        new = rng.random(Bn) < p
        idx[:, tt] = np.where(new, rng.integers(0, n_src, Bn), (idx[:, tt - 1] + 1) % n_src)
    return idx
src = lx[pre].to_numpy(); act = float(lx[hold].sum()); nh = int(hold.sum())
idx = sb_paths(len(src), nh, 6, 10000, 5156)
D = {'actual_cumlog': act, 'n_hold': nh, 'pre_mean_log_pp': float(src.mean() * 100), 'p_lower_L6_repro': float((src[idx].sum(1) <= act).mean())}
for f in (1.0, 0.5, 0.0):     # 把挑選期均值收縮 f 倍（0＝零超額虛無）
    s2 = src - src.mean() * (1 - f)
    D[f'p_lower_L6_pre_mean_x{f}'] = float((s2[idx].sum(1) <= act).mean())
# 拿掉 2023-12-11 窗
h2 = hold & (rm.index != pd.Timestamp('2023-12-11'))
act2 = float(lx[h2].sum()); idx2 = sb_paths(len(src), int(h2.sum()), 6, 10000, 5156)
D['drop_2023_12_11'] = {'cumlog': act2, 'p_lower_L6': float((src[idx2].sum(1) <= act2).mean())}
D['window_2023_12_11_log_excess'] = float(lx.loc['2023-12-11'])
D['hold_arith_t_vs_0050'] = t(ax[hold]); D['hold_mean_pp'] = float(ax[hold].mean() * 100)
# 保留期 j 分布：各 j 的 t_HOLD_cal（研究員 RESULTS_JSCAN）
R = json.loads((H / 'results.json').read_text(encoding='utf-8'))
D['jscan_t_HOLD_cal_AB'] = [round(v['t_HOLD_cal'], 3) for v in R['RESULTS_JSCAN']['AB_V6|normal']['by_j'].values()]
out['D_hold'] = D

# ---------- E. 台積電缺口：去台積電 0050 是事後換基準；超前是否顯著 ----------
dec = R['RESULTS_HOLD_SPA']['tsmc_decomp']['AB_V6|calendar']
adj = pd.read_parquet(H / 'data' / 'adj.parquet', columns=['2330'])['2330']
ex = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/ex_tsmc.csv', parse_dates=['date']).set_index('date')['w2330'] / 100
r2330 = wr(adj, G6)
w = sample(ex, G6[:-1]); w.index = G6[:-1]
w_mean_used = dec['w2330_mean']
wfill = w.fillna(w_mean_used)          # 粗略：缺實際權重的窗用研究員報的保留期平均權重
rex = (r5 - wfill * r2330) / (1 - wfill)
dd = (rm - rex)[hold]
E = {'researcher': {k: dec[k] for k in ('shortfall_vs_0050_log', 'tsmc_gap_log', 'main_vs_ex_tsmc_log', 'gap_over_shortfall', 'w2330_mean')},
     'approx_main_vs_ex_tsmc_t': t(dd), 'approx_main_vs_ex_tsmc_mean_pp': float(dd.mean() * 100),
     'n_hold_windows_with_actual_weight': int(w[hold].notna().sum())}
# 若落後量 0.057 的抽樣 se（log 超額 34 窗）
E['hold_cumlog_se'] = float(lx[hold].std(ddof=1) * math.sqrt(nh))
E['ratio_3_9_with_shortfall_at_minus_1se'] = float(dec['tsmc_gap_log'] / max(dec['shortfall_vs_0050_log'] - E['hold_cumlog_se'], 1e-9)) if dec['shortfall_vs_0050_log'] - E['hold_cumlog_se'] > 0 else 'shortfall 在 1 個 se 內就變號（比值無定義）'
out['E_tsmc'] = E

# ---------- F. V6 − V4：資料修補是否單向對策略有利 ----------
F = {}
for k in ('AB_V6|normal', 'V6|normal', 'AB_V6_B10|normal'):
    k4 = k.replace('V6', 'V4'); a = NV[k]; b = N4[k4].dropna()
    ya = a.pct_change(); yb = b.pct_change().reindex(ya.index)
    yy = {}
    for y in (2024, 2025, 2026):
        m = ya.index.year == y
        yy[y] = round(float((1 + ya[m].fillna(0)).prod() - (1 + yb[m].fillna(0)).prod()) * 100, 2)
    F[k] = {'year_diff_pp_V6_minus_V4': yy}
out['F_v6_minus_v4'] = F

# ---------- G. 2024-01 上市過濾用到未來（2024-02-15～22 有值）----------
P4c = None
try:
    cr6 = pd.read_parquet(H / 'data' / 'close_raw.parquet'); src6 = pd.read_parquet(H / 'data' / 'src.parquet')
    gap = cr6.loc['2024-01-02':'2024-02-14']
    lc = pd.read_parquet(H / 'data' / 'src_leochen.parquet')
    lc = lc[(lc['date'] >= '2024-01-02') & (lc['date'] <= '2024-02-14')]
    in_lc = set(lc['code'].astype(str))
    has = gap.notna().any()
    codes_gap = set(has[has].index)
    hs = pd.read_parquet(H / 'data' / 'src_hschi.parquet'); hs = hs[(hs['date'] >= '2024-01-02') & (hs['date'] <= '2024-02-14')]
    hs_codes = set(hs['code'].astype(str)) & set(cr6.columns)
    G = {'n_codes_with_price_in_gap': len(codes_gap), 'n_admitted_not_in_leochen': len(codes_gap - in_lc),
         'admitted_not_in_leochen': sorted(codes_gap - in_lc)[:30],
         'hschi_codes_in_panel_cols_but_dropped_in_gap': sorted(hs_codes - codes_gap)[:40], 'n_hschi_dropped': len(hs_codes - codes_gap)}
    # 被丟掉的是否之後在 V6 還有價（若之後沒有＝下市股被前視過濾掉）
    after = cr6.loc['2024-02-15':'2024-06-30']
    G['dropped_with_no_price_after_0215'] = sorted([c for c in hs_codes - codes_gap if not after[c].notna().any()])
except Exception as e:
    G = {'error': repr(e)}
out['G_gap_listing_filter'] = G

# ---------- H. 多重檢定：只就資料版本與 j-scan 而言 ----------
J = R['RESULTS_JSCAN']
out['H_multiplicity'] = {'official_t_AB': A['AB_V6|normal']['t'], 'j_t_max_AB': J['AB_V6|normal']['t_max'], 'j_t_median_AB': J['AB_V6|normal']['t_median'],
                         'official_t_V6': A['V6|normal']['t'], 'j_t_max_V6': J['V6|normal']['t_max'],
                         'official_t_B10': A['AB_V6_B10|normal']['t'], 'j_t_max_B10': J['AB_V6_B10|normal']['t_max'],
                         'replicator_spa_L6': R['SUMMARY_posthoc']['replication']['spa_p_focus_L6']}

def conv(o):
    if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [conv(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return float(o)
    return o
print(json.dumps(conv(out), ensure_ascii=False, indent=1, default=str))

# ---------- I. 上市過濾造成的保留期股池空洞（hschi 有官方價、V6 卻空白）----------
def hole_check():
    cr6 = pd.read_parquet(H / 'data' / 'close_raw.parquet'); v6 = pd.read_parquet(H / 'data' / 'value.parquet')
    hs = pd.read_parquet(H / 'data' / 'src_hschi.parquet'); hs = hs[(hs['date'] >= '2024-01-02') & (hs['date'] <= '2025-03-31') & (hs['vol'] > 0)]
    hs = hs[hs['code'].isin(cr6.columns)]
    hc = hs.pivot_table(index='date', columns='code', values='close'); hv = hs.pivot_table(index='date', columns='code', values='value')
    w = cr6.loc['2024-01-02':'2025-03-31']
    hole = hc.notna() & w.reindex(index=hc.index, columns=hc.columns).isna()
    # 只算「V6 在 2025-04 之後或 2023 年有值」的代碼（＝面板視為上市的股票）
    known = (cr6.loc['2025-04-01':].notna().any() | cr6.loc['2023-01-01':'2023-12-29'].notna().any())
    hole = hole.loc[:, [c for c in hole.columns if known.get(c, False)]]
    nd = hole.sum(); nd = nd[nd > 0].sort_values(ascending=False)
    res = {'n_codes': int(len(nd)), 'n_stock_days': int(nd.sum()), 'top_codes_days': {k: int(v) for k, v in nd.head(25).items()}}
    # 若補回，會進前 300（V60 中位成交值排名）的月份數
    vv = v6.loc['2023-10-01':'2025-03-31'].copy()
    hv2 = hv.reindex(columns=nd.index)
    allv = vv.copy()
    for c in nd.index:
        allv[c] = allv[c].fillna(hv2[c].reindex(allv.index)) if c in allv else hv2[c].reindex(allv.index)
    cl = cr6.loc['2023-10-01':'2025-03-31'].copy()
    for c in nd.index: cl[c] = cl[c].fillna(hc[c].reindex(cl.index))
    ent = {}
    sig = [d for d in pd.to_datetime(plan['signal']) if pd.Timestamp('2024-01-01') <= d <= pd.Timestamp('2025-03-31')]
    for d in sig:
        v60 = allv.loc[:d].tail(60).mean(); v60 = v60[(v60 >= 1e7) & (cl.loc[:d].ffill().iloc[-1] >= 10)]
        top = set(v60.sort_values(ascending=False).head(300).index)
        ent[str(d.date())] = sorted(top & set(nd.index))
    res['would_enter_top300_by_signal'] = {k: v for k, v in ent.items()}
    res['avg_n_hole_in_top300'] = float(np.mean([len(v) for v in ent.values()]))
    return res
try:
    out2 = {'I_hold_universe_hole': hole_check()}
except Exception as e:
    out2 = {'I_hold_universe_hole': {'error': repr(e)}}
print(json.dumps(conv(out2), ensure_ascii=False, indent=1, default=str))

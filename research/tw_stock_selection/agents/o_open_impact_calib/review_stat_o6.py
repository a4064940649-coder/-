#!/usr/bin/env python3
"""反方審查（資料與統計）：o_open_impact_calib。只讀研究員的檔案，不改動。
用法：python3 -I -B review_stat_o6.py
A. 擴散虛無：Y＝Σ|r|/Σσ√(q/V) 是否只是量時鐘下的隨機漫步常數 √(2/π)·k（k＝實現波動÷σ60）。
B. 校準宇宙的篩選偏誤：被 full12／開盤一致／根數篩掉的前 300 股日，波動是否較大。
C. 5 分 K 覆蓋率（存活者）：前 300 池股日中有 5 分 K 的比例。
D. 容量表 t 的年份依賴、制度分段、Newey-West、Holm 多重檢定。
E. 擁擠：以 BOT50 安慰劑做差中差。
"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

out = {}
# ---------------- A ----------------
X = pd.read_parquet(HERE / 'data' / 'stockdays.parquet'); R = pd.read_parquet(HERE / 'data' / 'bars.parquet')
import calib_lib as C
EXP = json.load(open(HERE / 'explore.json'))
BAD = set(EXP['step0_inventory']['5m_vs_official_volume_ratio_by_day']['anomalous_days_lt075_or_gt1'])
U = C.universe(X) & ~X.date.dt.strftime('%Y-%m-%d').isin(BAD)
for per in ('CAL', 'VAL'):
    m = (U & (X.period == per)).to_numpy()
    ys, xs, r2, x2, n = [], [], [], [], 0
    perbar = {}
    for b in C.BARS:
        r = R.loc[m, b].to_numpy(); q = X.loc[m, 'q' + b].to_numpy(); s = X.loc[m, 'sig60_l1'].to_numpy(); V = X.loc[m, 'V60sh'].to_numpy()
        ok = np.isfinite(r) & (q > 0) & np.isfinite(V) & (V > 0)
        r, q, s, V = r[ok], q[ok], s[ok], V[ok]
        ys.append(np.abs(r)); xs.append(s * np.sqrt(q / V)); r2.append(r * r); x2.append(s * s * q / V)
        perbar[b] = dict(Y=float(np.abs(r).sum() / (s * np.sqrt(q / V)).sum()), k=float(math.sqrt((r * r).sum() / (s * s * q / V).sum())))
    y = np.concatenate(ys); x = np.concatenate(xs); rr = np.concatenate(r2); xx = np.concatenate(x2)
    Y = y.sum() / x.sum(); k = math.sqrt(rr.sum() / xx.sum())
    # 標準化殘差 z＝r/(σ√(q/V))：若為量時鐘擴散，E|z|/√E z² 為分布形狀常數（常態＝0.798）
    z = np.sqrt(rr / xx)
    out[f'A_{per}'] = dict(Y_ratio=Y, k_rms=k, Y_over_k=Y / k, sqrt_2_over_pi=math.sqrt(2 / math.pi),
                           mean_absz_over_rms_z=float(z.mean() / math.sqrt((z * z).mean())), n_bars=int(len(y)), per_bar=perbar)

# ---------------- B ----------------
F = pd.read_parquet(HERE / 'data' / 'feat_all_2023.parquet'); F = F.sort_values(['code', 'date'])
for c in ('V60', 'sig60', 'vrank'): F[c + '_l1'] = F.groupby('code')[c].shift(1)
F['period'] = C.period(F.date)
pool = F[(F.vrank_l1 <= 300) & (F.fin == False) & F.period.isin(['CAL', 'VAL']) & ~F.date.dt.strftime('%Y-%m-%d').isin(BAD)]
key = X[['code', 'date']].assign(has5m=True)
P = pool.merge(key, on=['code', 'date'], how='left'); P['has5m'] = P.has5m.fillna(False).astype(bool)
P = P.merge(X[['code', 'date', 'full12', 'nbar', 'y_open', 'o_off', 'h_off', 'l_off', 'prevc', 'vol_off']].rename(columns={'prevc': 'pc5'}), on=['code', 'date'], how='left')
P['inU'] = P.merge(X.assign(U=U)[['code', 'date', 'U']], on=['code', 'date'], how='left').U.fillna(False).astype(bool).to_numpy()
P['rng_sig'] = (P.h_off - P.l_off) / P.pc5 / P.sig60_l1
cov = {}
for per, g in P.groupby('period'):
    cov[per] = dict(pool_stockdays=int(len(g)), share_with_5m=float(g.has5m.mean()), share_in_calib_universe=float(g.inU.mean()),
                    range_over_sigma_median_in=float(g[g.inU].rng_sig.median()), range_over_sigma_median_out_with5m=float(g[g.has5m & ~g.inU].rng_sig.median()),
                    share_excluded_given_5m=float((~g[g.has5m].inU).mean()))
out['B_C_coverage_selection'] = cov
# 缺 5 分 K 的池股：是否之後下市／不在 Yahoo manifest
miss_codes = P[~P.has5m].code.value_counts()
out['C_missing_5m_codes_top'] = {str(k): int(v) for k, v in miss_codes.head(15).items()}
out['C_n_codes_pool'] = int(P.code.nunique()); out['C_n_codes_missing_any'] = int(miss_codes.size)
last = F.groupby('code').date.max()
out['C_missing_codes_last_date_before_2026_06'] = int((last.reindex(miss_codes.index) < pd.Timestamp('2026-06-01')).sum())

# ---------------- D ----------------
import ecore6 as E
A = pd.read_parquet(HERE / 'data' / 'cap_avg_navs.parquet')
B = pd.read_parquet(HERE / 'data' / 'bench.parquet'); E50 = B['E50']
P4 = pd.read_csv(HERE / 'data' / 'plan_v4.csv'); sd = [pd.Timestamp(s) for s in P4.signal] + [pd.Timestamp(E.B_)]

def nw_t(x, L=3):
    x = np.asarray(x, float); x = x - x.mean() + x.mean(); n = len(x); m = x.mean(); e = x - m
    g0 = (e * e).sum() / n; s = g0
    for l in range(1, L + 1): s += 2 * (1 - l / (L + 1)) * (e[l:] * e[:-l]).sum() / n
    return float(m / math.sqrt(s / n))

D = {}
cells = [c for c in A.columns if c.split('|')[1] in ('M3V_point', 'STRESS')]
refs = {'AB_V4|normal(零衝擊)': B['AB_V4|normal'], 'AB_V4_B10|normal(零衝擊)': B['AB_V4_B10|normal']}
allser = {c: A[c] for c in cells}; allser.update(refs)
pvals = {}
for c, nav in allser.items():
    nav = nav.dropna()
    x = E.excess(nav, E50.reindex(nav.index), sd)
    yr = x.groupby(x.index.year).sum()
    t = E.tstat(x)
    d = dict(t=t, t_NW3=nw_t(x.values, 3), n=int(len(x)), mean_pp=float(x.mean() * 100),
             t_ex2021=E.tstat(x[x.index.year != 2021]), t_ex2021_2020=E.tstat(x[~x.index.year.isin([2020, 2021])]),
             t_post_continuous_2020_03_23=E.tstat(x[x.index >= pd.Timestamp('2020-03-23')]),
             t_pre_2020_03_23=E.tstat(x[x.index < pd.Timestamp('2020-03-23')]),
             t_calib_window_2024_02_2026_04=E.tstat(x[(x.index >= pd.Timestamp('2024-02-15')) & (x.index <= pd.Timestamp('2026-04-01'))]),
             top3_years_share_of_sum=float(yr.nlargest(3).sum() / yr.sum()), top3_years=[int(v) for v in yr.nlargest(3).index],
             t_drop_top3_years=E.tstat(x[~x.index.year.isin(yr.nlargest(3).index)]))
    D[c] = d
    from math import erf
    pvals[c] = 1 - 0.5 * (1 + erf(t / math.sqrt(2)))   # 單尾常態近似
out['D_cells'] = D
# Holm：主欄 16 格（AB/B10 × 4 級距 × 2 版，κ 點）單尾 5%
main = [c for c in cells if c.split('|')[1] == 'M3V_point']
ps = sorted((pvals[c], c) for c in main)
holm = {}
for i, (p, c) in enumerate(ps):
    holm[c] = dict(p_one_sided=p, holm_threshold=0.05 / (len(ps) - i), reject=p < 0.05 / (len(ps) - i))
# Holm 是逐步：第一個不拒絕之後全部不拒絕
stop = False
for p, c in ps:
    if stop or not holm[c]['reject']: stop = True; holm[c]['reject'] = False
out['D_holm_main16'] = holm
out['D_holm_n_reject'] = int(sum(v['reject'] for v in holm.values()))

# ---------------- E ----------------
it = json.load(open(HERE / 'iterations.json'))
v7 = [v for r in it['rounds'] for v in r['versions'] if v['version'] == 'v7'][0]
Eo = {}
for k in ('crowd_CAL', 'crowd_VAL', 'crowd_ALL_descriptive'):
    c = v7[k]; t10 = np.array(c['TOP10']['per_day_pp']); b50 = np.array(c['BOT50']['per_day_pp']); d = t10 - b50
    rng = np.random.default_rng(1); bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    Eo[k] = dict(TOP10=float(t10.mean()), BOT50=float(b50.mean()), DiD=float(d.mean()), DiD_t=float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))),
                 DiD_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], n_days=int(len(d)), share_days_DiD_pos=float((d > 0).mean()))
out['E_crowding_DiD_pp'] = Eo

json.dump(out, open(HERE / 'review_stat_o6.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

#!/usr/bin/env python3
"""R52：獨立重現 S52（資料窺探校正後顯著性）。只依 SPEC.json／PREREG.json／FINDINGS.md 的文字規格重寫，
不讀 S52 的 run.py／posthoc.py／gen_family.py／results*.json。spa_lib.py 唯讀 import（mcp_tests、sb_counts）。
執行：python3 -I -B replicate.py   （從任何目錄皆可）
"""
import sys, json, math, os, resource
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]                       # /home/user/-
RS = ROOT / 'research/tw_stock_selection'
SPA_DIR = RS / 'agents/o_main_spa_breaks'
sys.path.insert(0, str(SPA_DIR))
import spa_lib                                # 唯讀

NAV = ROOT / 'xq_scripts/site/navs_official.csv.gz'
FAM = SPA_DIR / 'data/family_excess.parquet'
BENCH = SPA_DIR / 'data/bench.parquet'
FOCUS = 'E0|W10_70'
MY_SEED = 20521152                            # R52 自選種子（不同於 S52 的 5252）
B, L_MAIN = 5000, 6
N_MAIN = 3006
EMC = 0.5772156649

nav = pd.read_csv(NAV, parse_dates=['date']).set_index('date').sort_index()
nav = nav[nav.index >= '2015-07-01']
R = {}

# ---------- 1. 全期日曆月底 ----------
def monthly_points(s, start='2015-07-01'):
    s = s[s.index >= start]
    me = s.groupby(s.index.to_period('M')).tail(1)   # 每月最後交易日（最後一月為部分月）
    return pd.concat([s.iloc[:1], me])                # 第一個月以起始日淨值為基底

def stats_block(ps, pb):
    rs, rb = ps.pct_change().dropna(), pb.pct_change().dropna()
    ex = (rs - rb).values; T = len(ex)
    m, sd = ex.mean(), ex.std(ddof=1)
    sr = m / sd
    yrs = T / 12
    cagr_s = (ps.iloc[-1] / ps.iloc[0]) ** (1 / yrs) - 1
    cagr_b = (pb.iloc[-1] / pb.iloc[0]) ** (1 / yrs) - 1
    g3 = float(stats.skew(ex)); g4 = float(stats.kurtosis(ex, fisher=False))
    rf = 1.01 ** (1 / 12) - 1
    xs = rs.values - rf
    # NW(6) t
    e = ex - m; g0 = (e * e).mean(); lr = g0
    for i in range(1, 7):
        lr += 2 * (1 - i / 7) * (e[i:] * e[:-i]).sum() / T
    psr0 = stats.norm.cdf(sr * math.sqrt(T - 1) / math.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return dict(months=T, cagr_strat=cagr_s, cagr_0050=cagr_b, cagr_diff_pp=(cagr_s - cagr_b) * 100,
                ex_mean_ann_pp=m * 12 * 100, ex_sd_ann_pp=sd * math.sqrt(12) * 100,
                IR=sr * math.sqrt(12), sr_month=sr, t_iid=sr * math.sqrt(T), t_nw6=m / math.sqrt(lr / T),
                sharpe_strat_rf1=xs.mean() / xs.std(ddof=1) * math.sqrt(12), skew=g3, kurt=g4,
                ac1=float(np.corrcoef(ex[1:], ex[:-1])[0, 1]), PSR0=float(psr0)), ex

pts = monthly_points(nav)
full, ex_full = stats_block(pts[FOCUS], pts['0050'])
R['monthly_full'] = full
p19 = nav[nav.index >= '2018-12-01']
p19 = pd.concat([p19[p19.index < '2019-01-01'].iloc[-1:], monthly_points(nav, '2019-01-01').iloc[1:]])
R['monthly_2019'], _ = stats_block(p19[FOCUS], p19['0050'])

# ---------- 3. 廣族群 ----------
fam = pd.read_parquet(FAM)
fam = fam[fam.index >= '2015-07-01']
n_win = len(fam)
fam = fam.loc[:, fam.notna().all()]
n_nonan = fam.shape[1]
fam = fam.T.drop_duplicates().T
n_dedup = fam.shape[1]
fam = fam.loc[:, fam.std(ddof=1) > 0]
n_final = fam.shape[1]
D = fam.values.astype(np.float64)
sr_fam = D.mean(0) / D.std(0, ddof=1)
V = float(np.var(sr_fam, ddof=1))
R['broad_family'] = dict(windows=n_win, members_nonan=n_nonan, members_dedup=n_dedup, members_final=n_final,
                         sd_SR_ann=math.sqrt(V * 12), max_SR_ann=float(sr_fam.max() * math.sqrt(12)))

# DSR（月頻、焦點＝日曆月底序列）
T = len(ex_full); sr = ex_full.mean() / ex_full.std(ddof=1)
g3 = float(stats.skew(ex_full)); g4 = float(stats.kurtosis(ex_full, fisher=False))
def sr0_of(N, v=V): return math.sqrt(v) * ((1 - EMC) * stats.norm.ppf(1 - 1 / N) + EMC * stats.norm.ppf(1 - 1 / (N * math.e)))
def psr(s, s0): return stats.norm.cdf((s - s0) * math.sqrt(T - 1) / math.sqrt(1 - g3 * s + (g4 - 1) / 4 * s ** 2))
def dsr_block(N, v=V):
    s0 = sr0_of(N, v)
    need = optimize.brentq(lambda s: psr(s, s0) - 0.95, s0, s0 + 5)
    return dict(N=N, SR0_ann=s0 * math.sqrt(12), DSR=float(psr(sr, s0)), need_SR_ann=need * math.sqrt(12))
lib = spa_lib.dsr(ex_full, sr_fam, N_MAIN)    # 交叉核對：spa_lib.dsr 應與自寫一致
R['DSR'] = {str(N): dsr_block(N) for N in [N_MAIN, 100, 500, 1000, 3000, 2990, 3021]}
R['DSR']['spa_lib_crosscheck_N3006'] = dict(SR0_ann=lib['SR0'] * math.sqrt(12), DSR=lib['DSR'])
R['DSR']['focus_SR_ann'] = sr * math.sqrt(12)
R['DSR']['lit_V0.5'] = dsr_block(N_MAIN, 0.5 / 12)

# ---------- 4. Harvey-Liu 打折 ----------
t_focus = sr * math.sqrt(T)
p_focus = 2 * (1 - stats.norm.cdf(t_focus))
t_fam = sr_fam * math.sqrt(n_win)
p_fam = 2 * (1 - stats.norm.cdf(np.abs(t_fam)))
def adj(N):
    q = (np.arange(1, N) - 0.5) / (N - 1)
    others = np.quantile(p_fam, q)
    p = np.concatenate([others, [p_focus]])
    o = np.argsort(p, kind='mergesort'); ps = p[o]
    r = int(np.where(o == N - 1)[0][0])       # 焦點排名（0 起）
    bonf = min(N * p_focus, 1.0)
    j = np.arange(1, N + 1)
    holm = np.maximum.accumulate(np.minimum((N - j + 1) * ps, 1.0))[r]
    cN = (1.0 / j).sum()
    bhy_raw = np.minimum(N * cN / j * ps, 1.0)
    bhy = np.minimum.accumulate(bhy_raw[::-1])[::-1][r]
    out = {}
    for k, pa in (('bonf', bonf), ('holm', holm), ('bhy', bhy)):
        ta = stats.norm.ppf(1 - pa / 2)
        sra = ta / math.sqrt(T / 12)
        out[k] = dict(p_adj=float(pa), t_adj=float(ta), haircut=float(1 - sra / (sr * math.sqrt(12))),
                      ex_after_pp=float(full['ex_mean_ann_pp'] * sra / (sr * math.sqrt(12))))
    return out
R['haircut'] = {'t_focus': t_focus, 'p_focus': p_focus, 'bonf_full_haircut_minN': int(math.ceil(1 / p_focus))}
for N in [N_MAIN, 100, 500, 1000, 3000]:
    R['haircut'][str(N)] = adj(N)
# Bonferroni／Holm 打折 100% 的最小 N（Holm 需掃描）
minN_holm = next(N for N in range(2, 400) if adj(N)['holm']['p_adj'] >= 1.0)
R['haircut']['holm_full_haircut_minN'] = minN_holm

# ---------- 5. SPA（廣族群＋焦點窗超額） ----------
bench = pd.read_parquet(BENCH); bench = bench[bench.index >= '2015-07-01']
idx = nav.index
starts = idx.searchsorted(fam.index)
ends = np.append(starts[1:], idx.searchsorted(pd.Timestamp('2026-10-12')))
r_f = nav[FOCUS].values[ends] / nav[FOCUS].values[starts] - 1
r_b = nav['0050'].values[ends] / nav['0050'].values[starts] - 1
R['SPA_bench_check_maxdiff'] = float(np.abs(r_b - bench['r0050'].reindex(fam.index).values).max())
x_win = r_f - r_b
DD = np.column_stack([D, x_win])
foc = DD.shape[1] - 1
R['SPA'] = {}
for tag, seed, L in (('my_seed_L6', MY_SEED, 6), ('my_seed2_L6', MY_SEED + 1, 6), ('s52_seed_L6', 5252, 6),
                     ('my_seed_L3', MY_SEED, 3), ('my_seed_L12', MY_SEED, 12)):
    W = spa_lib.sb_counts(DD.shape[0], L, B, seed)
    o = spa_lib.mcp_tests(DD, W, L, var='pr', focus=foc)
    R['SPA'][tag] = {k: o[k] for k in ('n', 'N', 'p_SPA_c_focus', 'p_RC_focus', 'p_SPA_c', 'p_StepM_focus',
                                       'p_single_focus', 'focus_t', 'focus_rank', 'max_t')}
    del W
R['SPA']['focus_window_t_iid'] = float(x_win.mean() / x_win.std(ddof=1) * math.sqrt(len(x_win)))

# ---------- 6. 月切點敏感度 ----------
def cut_points(d, mode):
    months = pd.period_range('2015-07', '2026-10', freq='M')
    out = [idx[0]]
    for p in months:
        tgt = pd.Timestamp(p.year, p.month, d)
        if mode == 'last_le':
            k = idx.searchsorted(tgt, side='right') - 1
        else:                                   # first_ge
            k = idx.searchsorted(tgt, side='left')
            if k >= len(idx): continue
        if k >= 0 and idx[k] > out[-1]: out.append(idx[k])
    return out
cut = {}
for mode in ('last_le', 'first_ge'):
    for partial in (False, True):
        ts = []
        for d in range(1, 29):
            pts_d = cut_points(d, mode)
            if partial and pts_d[-1] < idx[-1]: pts_d = pts_d + [idx[-1]]
            s = nav.loc[pts_d]
            e = (s[FOCUS].pct_change() - s['0050'].pct_change()).dropna().values
            ts.append(e.mean() / e.std(ddof=1) * math.sqrt(len(e)))
        cut[f'{mode}_partial{int(partial)}'] = dict(t_min=min(ts), t_max=max(ts), t_median=float(np.median(ts)),
                                                    argmin_day=int(np.argmin(ts)) + 1, argmax_day=int(np.argmax(ts)) + 1,
                                                    t_by_day=[round(x, 4) for x in ts])
R['cut_sensitivity'] = cut
R['peak_rss_MB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
json.dump(clean(R), open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1)

# ---------- compare.csv ----------
H = R['haircut']; Sm = R['SPA']['my_seed_L6']; C = cut['last_le_partial0']
rows = [
 ('1 月數', 136, full['months'], 0), ('1 策略年化%', 34.32, full['cagr_strat'] * 100, 0.05),
 ('1 0050年化%', 21.09, full['cagr_0050'] * 100, 0.05), ('1 月超額年化（百分點）', 11.68, full['ex_mean_ann_pp'], 0.05),
 ('1 資訊比率', 0.721, full['IR'], 0.01), ('1 t（iid）', 2.43, full['t_iid'], 0.01), ('1 PSR', 0.997, full['PSR0'], 0.01),
 ('2 N（BACKLOG 自數）', 3006, 3006, 30),
 ('3 廣族群成員數', 2410, n_final, 0), ('3 年化 SR 標準差', 0.60, R['broad_family']['sd_SR_ann'], 0.01),
 ('3 SR0 年化（N=3006）', 2.13, R['DSR']['3006']['SR0_ann'], 0.01), ('3 DSR（N=3006）', 0.0, R['DSR']['3006']['DSR'], 0.01),
 ('3 需要 SR 年化（N=3006）', 2.60, R['DSR']['3006']['need_SR_ann'], 0.01),
 ('4 BHY 折扣（N=3006）', 0.957, H['3006']['bhy']['haircut'], 0.01), ('4 BHY 剩餘超額（百分點）', 0.50, H['3006']['bhy']['ex_after_pp'], 0.05),
 ('4 Bonferroni 打折100%最小N', 66, H['bonf_full_haircut_minN'], 0),
 ('5 SPA_c 焦點 p（自選種子）', 0.068, Sm['p_SPA_c_focus'], 0.02),
 ('6 切點 t 最小', 2.45, C['t_min'], 0.01), ('6 切點 t 最大', 3.18, C['t_max'], 0.01),
]
df = pd.DataFrame(rows, columns=['item', 's52', 'r52', 'tol'])
df['diff'] = df['r52'] - df['s52']
df['pass'] = df['diff'].abs() <= df['tol'] + 1e-9
df.to_csv(HERE / 'compare.csv', index=False, encoding='utf-8-sig')
print(df.to_string()); print('peak MB', R['peak_rss_MB'])

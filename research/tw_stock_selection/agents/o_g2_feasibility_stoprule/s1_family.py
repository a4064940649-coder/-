#!/usr/bin/env python3
"""探索（1）族群：2,334 條淨 IR 分布、各門檻實際通過數、有效 N、去重、NPMLE 去卷積、主策略與 V7X 的後驗、P4 理想 vs 可執行折價。
輸出：explore.json 的 family 段、results.json 的 FAMILY 段、data/family_ir.parquet（每成員彙總）。"""
import json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__file__))
import g2lib as G

r = G.check_prereg()
X, meta = G.family()
N = X.shape[1]; n = X.shape[0]
assert N == 2334, N
out = {'N': N, 'n_windows': n}

# ---------- 每成員統計 ----------
rows = []
R50 = G.R50.to_numpy()
for k, c in enumerate(X.columns):
    x = X[c].to_numpy()
    d = {'member_id': c}
    for p in ['ALL', 'Y19', 'SEG1', 'SEG2', 'SEG3']:
        m = G.pmask(p)
        d[f'IR_{p}'] = G.ir(x[m]); d[f't_{p}'] = G.tstat(x[m])
        d[f'gx_{p}'] = G.geo_ex(x[m] + R50[m], R50[m])
    # 拿掉最好 3 個日曆年
    s = pd.Series(x, index=G.WIN).dropna()
    yr = s.groupby(s.index.year).apply(lambda g: np.prod(1 + g.to_numpy() + R50[G.WIN.get_indexer(g.index)]) - np.prod(1 + R50[G.WIN.get_indexer(g.index)]))
    best = yr.sort_values(ascending=False).index[:3]
    d['t_drop3'] = G.tstat(s[~s.index.year.isin(best)].to_numpy())
    xs = s.to_numpy(); d['SR_m'] = float(xs.mean() / xs.std(ddof=1)) if xs.std(ddof=1) > 0 else np.nan
    d['n'] = len(xs)
    rows.append(d)
F = pd.DataFrame(rows).merge(meta, on='member_id')
F['se_iid'] = np.sqrt(12) * np.sqrt((1 + F.SR_m ** 2 / 2) / F.n)
EXC = re.compile(r'(2330|tsmc|TSMC|lev2|L2_|BENCH_)')
F['lev_tsmc'] = F.member_id.str.contains(EXC)

# ---------- SPA_c（族群最大值）與 DSR ----------
Xf = X.fillna(0.0).to_numpy()
W = G.SP.sb_counts(n, 6, 10000, 20261102)
Mb = G.SP.boot_means(W, Xf)
om2 = G.SP.lrv(Xf, 'pr', 6); om = np.sqrt(om2); sn = np.sqrt(n)
dbar = Xf.mean(0); tst = sn * dbar / om
thr = -np.sqrt(om2 / n * 2 * np.log(np.log(n)))
gc = np.where(dbar >= thr, dbar, 0.0)
Z = (Mb - gc[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]
Tb = np.maximum(Z.max(axis=1), 0.0)
np.save(G.SCR / 'Tb_family_c_L6.npy', Tb)
F['t_pr'] = tst
F['p_SPA_c_focus'] = [float((Tb >= t).mean()) if t > 0 else 1.0 for t in tst]
sr_all = Xf.mean(0) / Xf.std(0, ddof=1)
sr_all = sr_all[np.isfinite(sr_all)]
et = G.SP.eff_trials(Xf[:, np.std(Xf, 0) > 0])
dsr_raw, dsr_lj = [], []
for k in range(N):
    x = Xf[:, k]
    if np.std(x) == 0: dsr_raw.append(np.nan); dsr_lj.append(np.nan); continue
    dsr_raw.append(G.SP.dsr(x, sr_all, N)['DSR']); dsr_lj.append(G.SP.dsr(x, sr_all, et['M_eff_LiJi'])['DSR'])
F['DSR_raw'] = dsr_raw; F['DSR_LiJi'] = dsr_lj
d0 = G.SP.dsr(Xf[:, 0], sr_all, N)
out['SPA_family'] = {'B': 10000, 'L': 6, 'seed': 20261102, 'Tb_q95': float(np.quantile(Tb, 0.95)), 'Tb_q50': float(np.quantile(Tb, 0.5)), 'max_t': float(tst.max()),
                     'argmax': X.columns[int(np.argmax(tst))]}
out['DSR_family'] = {'var_SR_monthly': d0['var_SR'], 'SR0_monthly_rawN': d0['SR0'], 'SR0_annual_rawN': d0['SR0'] * np.sqrt(12),
                     'SR0_monthly_LiJi': G.SP.dsr(Xf[:, 0], sr_all, et['M_eff_LiJi'])['SR0']}
out['eff_trials'] = et

# ---------- 門檻實際通過數 ----------
gates = {
 'G2a_t_ALL_ge3': F.t_ALL >= 3, 'G2b_t_Y19_ge2': F.t_Y19 >= 2,
 'G4a_3seg_pos': (F.gx_SEG1 > 0) & (F.gx_SEG2 > 0) & (F.gx_SEG3 > 0), 'G4b_drop3_ge2': F.t_drop3 >= 2,
 'G3a_SPA_lt05': F.p_SPA_c_focus < 0.05, 'G3b_DSR_ge095': F.DSR_raw >= 0.95}
cnt = {k: int(v.sum()) for k, v in gates.items()}
cnt['G2a&G2b'] = int((gates['G2a_t_ALL_ge3'] & gates['G2b_t_Y19_ge2']).sum())
allg = np.logical_and.reduce(list(gates.values()))
cnt['joint_G2G3G4ab'] = int(allg.sum())
cnt['joint_excl_lev_tsmc'] = int((allg & ~F.lev_tsmc).sum())
cnt['G2a&G2b_excl_lev_tsmc'] = int((gates['G2a_t_ALL_ge3'] & gates['G2b_t_Y19_ge2'] & ~F.lev_tsmc).sum())
out['gate_counts_actual'] = cnt
out['G2a_G2b_members'] = F.loc[gates['G2a_t_ALL_ge3'] & gates['G2b_t_Y19_ge2'], ['member_id', 'IR_ALL', 't_ALL', 't_Y19', 'p_SPA_c_focus', 'DSR_raw', 'lev_tsmc']].round(4).to_dict('records')
out['G2a_members'] = F.loc[gates['G2a_t_ALL_ge3'], ['member_id', 't_ALL', 't_Y19', 'lev_tsmc']].round(3).to_dict('records')

def dist(s):
    s = s.dropna(); q = [0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {**{f'p{int(a*100)}': float(s.quantile(a)) for a in q}, 'max': float(s.max()), 'mean': float(s.mean()), 'n': int(len(s))}
out['IR_dist'] = {p: dist(F[f'IR_{p}']) for p in ['ALL', 'Y19', 'SEG1', 'SEG2', 'SEG3']}
out['IR_dist_excl_lev_tsmc'] = {p: dist(F.loc[~F.lev_tsmc, f'IR_{p}']) for p in ['ALL', 'Y19']}
out['n_lev_tsmc'] = int(F.lev_tsmc.sum())
out['share_IR_ALL_ge'] = {str(a): float((F.IR_ALL >= a).mean()) for a in [0.3, 0.5, 0.66]}

# ---------- 去重 ----------
C = np.corrcoef(Xf.T)
def dedup(order_idx, groups):
    keep = []
    for g in groups:
        kept = []
        for i in g:
            if np.std(Xf[:, i]) == 0: continue
            if all(C[i, j] <= 0.95 for j in kept): kept.append(i)
        keep += kept
    return sorted(keep)
fold_groups = [list(np.where(F.folder.to_numpy() == f)[0]) for f in pd.unique(F.folder)]
keep_f = dedup(None, fold_groups)
keep_g = dedup(None, [list(range(N))])
F['keep_folder'] = False; F.loc[keep_f, 'keep_folder'] = True
F['keep_global'] = False; F.loc[keep_g, 'keep_global'] = True
et_f = G.SP.eff_trials(Xf[:, keep_f])
out['dedup'] = {'n_keep_folder': len(keep_f), 'n_keep_global': len(keep_g), 'eff_trials_folder_dedup': et_f,
                'IR_dist_folder_dedup': dist(F.loc[F.keep_folder, 'IR_ALL']), 'IR_dist_global_dedup': dist(F.loc[F.keep_global, 'IR_ALL'])}

# ---------- NPMLE 去卷積 ----------
grid = np.round(np.arange(-2.0, 2.5001, 0.01), 2)
# bootstrap se（敏感度）
Wb = W[:2000]
Mb2 = (Wb @ Xf / n)
sdb = np.sqrt(((Wb @ (Xf ** 2) / n) - Mb2 ** 2).clip(1e-12))
irb = Mb2 / sdb * np.sqrt(12)
F['se_boot'] = irb.std(axis=0)
dec = {}
for lab, mask, secol in [('main_folder_dedup_iid', F.keep_folder & F.IR_ALL.notna(), 'se_iid'), ('all_iid', F.IR_ALL.notna(), 'se_iid'),
                         ('global_dedup_iid', F.keep_global & F.IR_ALL.notna(), 'se_iid'), ('folder_dedup_boot', F.keep_folder & F.IR_ALL.notna() & np.isfinite(F.se_boot) & (F.se_boot > 0.01), 'se_boot'),
                         ('folder_dedup_excl_lev_tsmc', F.keep_folder & F.IR_ALL.notna() & ~F.lev_tsmc, 'se_iid')]:
    sub = F[mask]
    w, ll, it = G.npmle(sub.IR_ALL.to_numpy(), sub[secol].to_numpy(), grid)
    mean_t = float((w * grid).sum()); var_t = float((w * grid ** 2).sum() - mean_t ** 2)
    imax = sub.IR_ALL.idxmax()
    post_max = G.posterior(F.loc[imax, 'IR_ALL'], F.loc[imax, secol], grid, w)
    post_max['member'] = F.loc[imax, 'member_id']
    # 每成員後驗平均 → 後驗真實 IR 的分布（另一種「真實 IR > 0.3 的比例」）
    Lk = np.exp(-0.5 * ((sub.IR_ALL.to_numpy()[:, None] - grid[None, :]) / sub[secol].to_numpy()[:, None]) ** 2) * w[None, :]
    Lk = Lk / Lk.sum(1, keepdims=True); pm = Lk @ grid
    dec[lab] = {'n': int(len(sub)), 'loglik': ll, 'iters': it, 'prior_mean': mean_t, 'prior_sd': float(np.sqrt(max(var_t, 0))),
                'share_true_gt_0.3': float(w[grid > 0.3 + 1e-9].sum()), 'share_true_ge_0.5': float(w[grid >= 0.5 - 1e-9].sum()),
                'share_true_ge_0.66': float(w[grid >= 0.66 - 1e-9].sum()),
                'support': {f'{g:.2f}': float(v) for g, v in zip(grid, w) if v > 1e-3},
                'posterior_of_max': post_max, 'postmean_member_max': float(pm.max()), 'postmean_share_gt_0.3': float((pm > 0.3).mean())}
    if lab == 'main_folder_dedup_iid':
        np.save(G.SCR / 'npmle_w_main.npy', w); W_MAIN = w
out['deconvolution'] = dec
# 參數式常態 EB（敏感度；常態先驗會把左偏分布的右尾灌大，只作對照）
pe = {}
for lab, mask in [('folder_dedup', F.keep_folder & F.IR_ALL.notna()), ('all', F.IR_ALL.notna())]:
    sub = F[mask]; m0 = float(sub.IR_ALL.mean()); tau2 = float(max(sub.IR_ALL.var() - (sub.se_iid ** 2).mean(), 0))
    from scipy import stats as _st
    pe[lab] = {'mean': m0, 'tau': tau2 ** 0.5, 'share_true_ge_0.5': float(1 - _st.norm.cdf((0.5 - m0) / tau2 ** 0.5)), 'share_true_gt_0.3': float(1 - _st.norm.cdf((0.3 - m0) / tau2 ** 0.5)),
               'AB_V4_post_mean': float(m0 + tau2 / (tau2 + 0.2227 ** 2) * (0.5559 - m0))}
    # 只用 IR>0 的右半部擬合（右尾專用；半常態鏡射）
pos = F[F.keep_folder & (F.IR_ALL > F.IR_ALL[F.keep_folder].median())]
out['deconvolution_normal_EB_sensitivity'] = pe

# ---------- 主策略與 V7X 後驗 ----------
ab = F.set_index('member_id').loc['o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal']
paths = G.v7x_paths('100萬', 'normal')
xv = G.win_ex(G.javg_nav(paths)).to_numpy()
srv = np.nanmean(xv) / np.nanstd(xv, ddof=1)
se_v = np.sqrt(12) * np.sqrt((1 + srv ** 2 / 2) / np.isfinite(xv).sum())
post = {'AB_V4_ideal': {**G.posterior(ab.IR_ALL, ab.se_iid, grid, W_MAIN), 't_ALL': float(ab.t_ALL), 't_Y19': float(ab.t_Y19), 'IR_Y19': float(ab.IR_Y19)},
        'V7X_javg_100w_GROW_normal': {**G.posterior(G.ir(xv), se_v, grid, W_MAIN), 't_ALL': G.tstat(xv), 't_Y19': G.tstat(xv[G.pmask('Y19')]),
                                      'IR_Y19': G.ir(xv[G.pmask('Y19')])}}
for lab in ['all_iid', 'folder_dedup_excl_lev_tsmc']:
    pass
out['posterior_main'] = post
out['AB_V4_rank_t'] = int((F.t_ALL > ab.t_ALL).sum() + 1)

# ---------- P4：理想 vs 可執行 ----------
ideal = pd.read_parquet(G.AG / 'o_v7_listing_exec_engine' / 'navs_jscan_v7.parquet')
p4 = {}
for cell in ['100萬', '300萬', '1000萬']:
    ex = G.v7x_paths(cell, 'normal'); di = []
    for j in range(13):
        xi = G.win_ex(ideal[f'V7_B10|normal|J{j}']).to_numpy(); xe = G.win_ex(ex[j]).to_numpy()
        for per in ['ALL', 'Y19']:
            m = G.pmask(per)
            di.append({'j': j, 'per': per, 'IR_ideal': G.ir(xi[m]), 'IR_exec': G.ir(xe[m]), 'diff': G.ir(xi[m]) - G.ir(xe[m]),
                       'mean_gap_pp_month': float(np.nanmean(xi[m] - xe[m]) * 100)})
    dd = pd.DataFrame(di)
    p4[cell] = {per: {'median_IR_ideal': float(g.IR_ideal.median()), 'median_IR_exec': float(g.IR_exec.median()), 'median_diff': float(g['diff'].median()),
                      'min_diff': float(g['diff'].min()), 'max_diff': float(g['diff'].max()), 'median_gap_pp_month': float(g.mean_gap_pp_month.median())}
                for per, g in dd.groupby('per')}
p4['subsample_small_highturnover'] = '無法檢定：族群中沒有同時具備理想引擎與 V7X 可執行版本的中小型或高周轉成員（上櫃題 15 條只有可執行版）'
out['P4_ideal_vs_exec'] = p4

F.to_parquet(G.DATA / 'family_ir.parquet')
ex = json.loads((G.HERE / 'explore.json').read_text()) if (G.HERE / 'explore.json').exists() else {}
ex['family'] = out
(G.HERE / 'explore.json').write_text(json.dumps(ex, ensure_ascii=False, indent=1, default=float))
G.save_key('FAMILY', out)
print(json.dumps({k: out[k] for k in ['gate_counts_actual', 'share_IR_ALL_ge', 'DSR_family', 'SPA_family']}, ensure_ascii=False, default=float, indent=0))
print(json.dumps(out['IR_dist']['ALL'], default=float)); print(json.dumps(out['dedup'], default=float)[:800])
for k, v in dec.items(): print(k, {a: v[a] for a in ['n', 'prior_mean', 'prior_sd', 'share_true_gt_0.3', 'share_true_ge_0.5', 'share_true_ge_0.66', 'postmean_member_max']}, v['posterior_of_max'])
print(json.dumps(post, default=float, ensure_ascii=False)); print(json.dumps(p4, default=float, ensure_ascii=False)[:1500])

#!/usr/bin/env python3
"""S52 主分析：焦點月統計、PSR／DSR、Harvey-Liu 打折（Bonferroni／Holm／BHY）、Hansen SPA／White RC（近親族群與廣族群）。
規格 SPEC.json、判定 PREREG.json（sha256 驗證後才執行）。SPA 核心沿用 agents/o_main_spa_breaks/spa_lib.py（唯讀 import）。
用法：python3 -I -B run.py"""
import sys, json, math, time, resource, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent.parent.parent
t0 = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f in ('SPEC', 'PREREG'):
    assert sha(HERE / f'{f}.json') == (HERE / f'{f}.sha256').read_text().split()[0], f'{f}.json 與登記的 sha256 不符'
SPEC = json.loads((HERE / 'SPEC.json').read_text())
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import spa_lib                                              # noqa: E402

N_MAIN = SPEC['N']['total']                                 # 3006
N_SENS = [100, 500, 1000, 3000]
EMC = 0.5772156649015329
B = SPEC['methods']['SPA']['B']; SEED = SPEC['methods']['SPA']['seed']
LS = [SPEC['methods']['SPA']['L_main']] + SPEC['methods']['SPA']['L_sens']
RF_M = 1.01 ** (1 / 12) - 1
FOCUS = 'OFFICIAL:E0|W10_70'

# ---------------- 1. 焦點月統計
M = pd.read_csv(HERE / 'family_monthly_nav.csv.gz', index_col=0, parse_dates=True)
R = M.pct_change().iloc[1:]
R.index = R.index.to_period('M')
WIN = {'full': R.index >= pd.Period('2015-07', 'M'), 'since2019': R.index >= pd.Period('2019-01', 'M')}
assert int(WIN['full'].sum()) == 136 and int(WIN['since2019'].sum()) == 94


def desc(x):
    x = np.asarray(x, float); T = len(x)
    mu, sd = x.mean(), x.std(ddof=1)
    rho1 = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    # Newey-West（Bartlett，落後 6）t
    e = x - mu; s = (e @ e) / T
    for i in range(1, 7):
        s += 2 * (1 - i / 7) * (e[i:] @ e[:-i]) / T
    return dict(T=T, mean_m=float(mu), sd_m=float(sd), ann_arith=float(mu * 12), ann_vol=float(sd * math.sqrt(12)),
                SR_m=float(mu / sd), SR_ann=float(mu / sd * math.sqrt(12)), t_iid=float(mu / sd * math.sqrt(T)),
                t_nw6=float(mu / math.sqrt(s / T)), skew=float(stats.skew(x)), kurt=float(stats.kurtosis(x, fisher=False)), rho1=rho1)


focus_stats = {}
for w, msk in WIN.items():
    rs, rb = R.loc[msk, FOCUS].to_numpy(), R.loc[msk, '0050'].to_numpy()
    ex = rs - rb
    yrs = len(rs) / 12
    cg = lambda r: float(np.prod(1 + r) ** (1 / yrs) - 1)
    focus_stats[w] = dict(months=len(rs), period=f'{R.index[msk][0]}～{R.index[msk][-1]}', cagr_strategy=cg(rs), cagr_0050=cg(rb),
                          cagr_diff_pp=(cg(rs) - cg(rb)) * 100, excess_vs_0050=desc(ex), strategy_vs_rf1=desc(rs - RF_M), b0050_vs_rf1=desc(rb - RF_M),
                          corr_with_0050=float(np.corrcoef(rs, rb)[0, 1]))
print('焦點', json.dumps({w: (v['excess_vs_0050']['ann_arith'], v['excess_vs_0050']['SR_ann'], v['excess_vs_0050']['t_iid']) for w, v in focus_stats.items()}), flush=True)

# ---------------- 2. 廣族群（o_main_spa_breaks 窗頻對 0050 超額）
FE = pd.read_parquet(AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet')
BEN = pd.read_parquet(AG / 'o_main_spa_breaks' / 'data' / 'bench.parquet')
off = pd.read_csv(ROOT / 'xq_scripts' / 'site' / 'navs_official.csv.gz', index_col=0, parse_dates=True)
END = pd.Timestamp('2026-10-12')


def broad_matrix(start):
    sub = FE[FE.index >= pd.Timestamp(start)]
    sub = sub.loc[:, sub.notna().all()]
    n_complete = sub.shape[1]
    sub = sub.T.round(12).drop_duplicates().T                 # 完全相同的欄去重
    zero = sub.std(ddof=1) < 1e-10                            # 超額恆為 0（＝0050 本身或全押 0050 的成員），SR 無定義 → 剔除並記數
    n_zero = int(zero.sum()); sub = sub.loc[:, ~zero]
    grid = list(sub.index) + [END]
    nav_s = off['E0|W10_70'].reindex(grid); nav_b = off['0050'].reindex(grid)
    assert nav_s.notna().all() and nav_b.notna().all(), '窗格點不在正式淨值交易日內'
    rs = (nav_s.to_numpy()[1:] / nav_s.to_numpy()[:-1] - 1); rb = (nav_b.to_numpy()[1:] / nav_b.to_numpy()[:-1] - 1)
    bdiff = float(np.abs(rb - BEN['r0050'].reindex(sub.index).to_numpy()).max())
    fx = pd.Series(rs - rb, index=sub.index, name=FOCUS)
    return sub, fx, dict(start=str(sub.index[0].date()), n_windows=int(sub.shape[0]), n_complete=int(n_complete), n_dedup=int(sub.shape[1]), n_zero_var_dropped=n_zero,
                         max_abs_diff_r0050_vs_bench=bdiff)


BROAD = {}
for w, st in (('full', '2015-07-01'), ('since2019', '2019-01-01')):
    sub, fx, meta = broad_matrix(st)
    X = sub.to_numpy()
    sr = X.mean(0) / X.std(0, ddof=1)
    tt = sr * math.sqrt(X.shape[0])
    fsr = float(fx.mean() / fx.std(ddof=1))
    eff = spa_lib.eff_trials(X)
    meta.update(V_SR_window=float(np.var(sr, ddof=1)), SD_SR_window=float(np.std(sr, ddof=1)), SR_ann_quantiles={q: float(np.quantile(sr, q) * math.sqrt(12)) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 1.0)},
                SR_mean_ann=float(sr.mean() * math.sqrt(12)), focus_SR_window_ann=fsr * math.sqrt(12), focus_t_window=float(fsr * math.sqrt(len(fx))),
                focus_rank_by_SR=int(1 + (sr > fsr).sum()), share_members_t_gt_2=float((tt > 2).mean()), share_members_SR_gt_focus=float((sr > fsr).mean()), eff_trials=eff)
    BROAD[w] = dict(sub=sub, fx=fx, sr=sr, t=tt, meta=meta)
    print('廣族群', w, json.dumps({k: v for k, v in meta.items() if k != 'SR_ann_quantiles'}, ensure_ascii=False), flush=True)

# ---------------- 3. 近親族群（月頻）
fam_cols = [c for c in R.columns if c.split(':')[0] in ('S45', 'S46', 'S47', 'S48')]
NEAR = {}
for w, msk in WIN.items():
    ex = R.loc[msk, [FOCUS] + fam_cols].sub(R.loc[msk, '0050'], axis=0)
    dd = ex.T.round(6).drop_duplicates().T                    # 現行格四處重複（＝正式）只留正式
    assert dd.columns[0] == FOCUS
    X = dd.to_numpy(); sr = X.mean(0) / X.std(0, ddof=1)
    NEAR[w] = dict(ex=dd, sr=sr, meta=dict(n_cells_raw=len(fam_cols), n_dedup_incl_focus=int(dd.shape[1]), V_SR_month=float(np.var(sr[1:], ddof=1)),
                                          SD_SR_month=float(np.std(sr[1:], ddof=1)), focus_rank_by_SR=int(1 + (sr[1:] > sr[0]).sum()),
                                          SR_ann_min=float(sr.min() * math.sqrt(12)), SR_ann_median=float(np.median(sr) * math.sqrt(12)), SR_ann_max=float(sr.max() * math.sqrt(12)),
                                          eff_trials=spa_lib.eff_trials(X)))
    print('近親', w, json.dumps(NEAR[w]['meta'], ensure_ascii=False), flush=True)

# ---------------- 4. PSR／DSR


def psr(sr, sr_star, T, g3, g4):
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))))


def sr0(V, N):
    N = max(float(N), 2.0)
    return math.sqrt(V) * ((1 - EMC) * stats.norm.ppf(1 - 1 / N) + EMC * stats.norm.ppf(1 - 1 / (N * math.e)))


def min_sr(srz, T, g3, g4, target=0.95):
    f = lambda s: psr(s, srz, T, g3, g4) - target
    lo, hi = srz, srz + 5.0
    return float(optimize.brentq(f, lo, hi)) if f(lo) < 0 < f(hi) else float('nan')


DSR = {}
for w in WIN:
    e = focus_stats[w]['excess_vs_0050']; T, SRm, g3, g4 = e['T'], e['SR_m'], e['skew'], e['kurt']
    Vs = {'broad_empirical': BROAD[w]['meta']['V_SR_window'], 'near_family': NEAR[w]['meta']['V_SR_month'], 'literature_BLdP2014': 0.5 / 12}
    effN = {'LiJi': BROAD[w]['meta']['eff_trials']['M_eff_LiJi'], 'Nyholt': BROAD[w]['meta']['eff_trials']['M_eff_Nyholt']}
    Ns = [('main_3006', N_MAIN)] + [(str(n), n) for n in N_SENS] + [('LiJi_eff', effN['LiJi']), ('Nyholt_eff', effN['Nyholt'])]
    tab = {}
    for vk, V in Vs.items():
        tab[vk] = {}
        for nk, N in Ns:
            z = sr0(V, N)
            ms = min_sr(z, T, g3, g4)
            tab[vk][nk] = dict(N=float(N), SR0_ann=z * math.sqrt(12), DSR=psr(SRm, z, T, g3, g4), min_SR_ann_for_DSR95=ms * math.sqrt(12),
                               min_ann_excess_pp_at_same_vol=ms * e['sd_m'] * 12 * 100)
    DSR[w] = dict(T=T, SR_m=SRm, SR_ann=e['SR_ann'], skew=g3, kurt=g4, PSR_0=psr(SRm, 0.0, T, g3, g4),
                  min_SR_ann_for_PSR95_vs0=min_sr(0.0, T, g3, g4) * math.sqrt(12), V_SR_used=Vs, table=tab,
                  strategy_vs_rf1=dict(SR_ann=focus_stats[w]['strategy_vs_rf1']['SR_ann'],
                                       PSR_0=psr(focus_stats[w]['strategy_vs_rf1']['SR_m'], 0.0, T, focus_stats[w]['strategy_vs_rf1']['skew'], focus_stats[w]['strategy_vs_rf1']['kurt']),
                                       b0050_SR_ann=focus_stats[w]['b0050_vs_rf1']['SR_ann']))
    print('DSR', w, 'PSR0', round(DSR[w]['PSR_0'], 4), {vk: round(tab[vk]['main_3006']['DSR'], 4) for vk in tab}, flush=True)

# ---------------- 5. Harvey-Liu 打折


def p2(t):
    return float(2 * stats.norm.sf(abs(t)))


def t_of_p(p):
    return float(stats.norm.isf(p / 2)) if p < 1 else 0.0


def others_p(pemp, n_other):
    q = (np.arange(n_other) + 0.5) / n_other
    return np.quantile(np.sort(pemp), q)


def holm_bhy(p_focus, p_others):
    allp = np.concatenate([[p_focus], p_others]); N = len(allp)
    order = np.argsort(allp, kind='stable'); ps = allp[order]
    r = int(np.where(order == 0)[0][0])                         # 焦點 0-based 名次
    holm = max(min((N - j) * ps[j], 1.0) for j in range(r + 1))
    cN = float(np.sum(1.0 / np.arange(1, N + 1)))
    bhy_all = np.minimum(N * cN / np.arange(1, N + 1) * ps, 1.0)
    bhy = float(np.min(bhy_all[r:]))
    return float(holm), bhy, r + 1


HC = {}
for w in WIN:
    e = focus_stats[w]['excess_vs_0050']; T = e['T']; SRa = e['SR_ann']; tf = e['t_iid']; pf = p2(tf)
    pemp = np.array([p2(t) for t in BROAD[w]['t']])
    out = dict(t_focus=tf, p_single=pf, ann_arith_excess_pp=e['ann_arith'] * 100, SR_ann=SRa, T=T, by_N={})
    for N in [N_MAIN] + N_SENS:
        po = others_p(pemp, N - 1)
        pb = min(N * pf, 1.0); ph, pbhy, rk = holm_bhy(pf, po)
        row = dict(focus_rank_among_N=rk)
        for nm, pa in (('Bonferroni', pb), ('Holm', ph), ('BHY', pbhy)):
            ta = t_of_p(pa); sra = ta / math.sqrt(T / 12); ratio = max(sra / SRa, 0.0) if SRa > 0 else 0.0
            row[nm] = dict(p_adj=pa, t_adj=ta, SR_adj_ann=sra, haircut=1 - ratio, ann_excess_adj_pp=e['ann_arith'] * ratio * 100)
        out['by_N'][str(N)] = row
    # 自相關調整版（Lo 2002 風格：SR 年化因子以 ρ1 修正，僅供參考）
    rho = e['rho1']; q = 12
    eta = q / math.sqrt(q + 2 * sum((q - k) * rho ** k for k in range(1, q)))
    out['autocorr_note'] = dict(rho1=rho, Lo2002_ann_factor=eta, SR_ann_Lo2002=e['SR_m'] * eta, t_nw6=e['t_nw6'])
    out['empirical_p_others'] = dict(source=f'廣族群 {w} 窗 t 的雙尾 p（{len(pemp)} 個去重成員）', share_p_lt_005=float((pemp < 0.05).mean()), median_p=float(np.median(pemp)))
    HC[w] = out
    print('HL', w, json.dumps({k: {m: round(v[m]['ann_excess_adj_pp'], 2) for m in ('Bonferroni', 'Holm', 'BHY')} for k, v in out['by_N'].items()}, ensure_ascii=False), flush=True)

# ---------------- 6. SPA／RC


def run_spa(D, focus_idx, tag, wkey):
    D = np.ascontiguousarray(D, float)
    out = {}
    for L in LS:
        W = spa_lib.sb_counts(D.shape[0], L, B, SEED + L + (0 if wkey == 'full' else 100))
        r = spa_lib.mcp_tests(D, W, L, var='pr', focus=focus_idx)
        out[f'L{L}'] = {k: r[k] for k in ('n', 'N', 'max_t', 'SPA_stat', 'p_RC', 'p_RC_focus', 'p_SPA_l', 'p_SPA_c', 'p_SPA_u', 'p_SPA_l_focus', 'p_SPA_c_focus',
                                          'p_SPA_u_focus', 'focus_rank', 'focus_t', 'p_StepM_focus', 'p_single_focus', 'StepM_5pct_n_reject', 'n_kept_SPA_c')}
        del W
        print('SPA', tag, wkey, f'L{L}', 'focus p_c', round(r['p_SPA_c_focus'], 4), 'max p_c', round(r['p_SPA_c'], 4), 'rank', r['focus_rank'], '/', r['N'], f'{time.time()-t0:.0f}s', rss(), 'MB', flush=True)
    return out


SPA = {'near': {}, 'broad': {}}
for w in WIN:
    SPA['near'][w] = run_spa(NEAR[w]['ex'].to_numpy(), 0, 'near', w)
    D = np.column_stack([BROAD[w]['fx'].to_numpy(), BROAD[w]['sub'].to_numpy()])
    SPA['broad'][w] = run_spa(D, 0, 'broad', w)
    SPA['broad'][w]['best_member'] = str(BROAD[w]['sub'].columns[int(np.argmax(BROAD[w]['t']))])

# ---------------- 7. 判定
PR = json.loads((HERE / 'PREREG.json').read_text())
dec = {}
for w in WIN:
    c1 = DSR[w]['table']['broad_empirical']['main_3006']['DSR']
    c2 = SPA['broad'][w]['L6']['p_SPA_c_focus']
    c3 = HC[w]['by_N'][str(N_MAIN)]['BHY']['p_adj']
    passed = dict(C1_DSR=bool(c1 >= 0.95), C2_SPA=bool(c2 < 0.05), C3_haircut=bool(c3 < 0.05))
    k = sum(passed.values())
    dec[w] = dict(values=dict(DSR=c1, SPA_c_focus_p=c2, BHY_p=c3), passed=passed,
                  label='扣資料窺探後仍顯著' if k == 3 else ('部分通過（證據不一致）' if k >= 1 else '扣資料窺探後不顯著'),
                  G3=dict(SPA_p_lt_005=passed['C2_SPA'], DSR_ge_095=passed['C1_DSR'], G3_pass=bool(passed['C1_DSR'] and passed['C2_SPA'])))
print('判定', json.dumps(dec, ensure_ascii=False), flush=True)

res = dict(spec_sha256=sha(HERE / 'SPEC.json'), prereg_sha256=sha(HERE / 'PREREG.json'), run_sha256=sha(Path(__file__)),
           gen_family_sha256=sha(HERE / 'gen_family.py'), family_monthly_nav_sha256=sha(HERE / 'family_monthly_nav.csv.gz'),
           spa_lib_sha256=sha(AG / 'o_main_spa_breaks' / 'spa_lib.py'), N_main=N_MAIN, N_sources=SPEC['N'],
           focus=focus_stats, broad_family={w: BROAD[w]['meta'] for w in WIN}, near_family={w: NEAR[w]['meta'] for w in WIN},
           near_family_SR_ann={w: dict(zip(NEAR[w]['ex'].columns, (NEAR[w]['sr'] * math.sqrt(12)).round(4).tolist())) for w in WIN},
           DSR=DSR, haircut=HC, SPA=SPA, decision=dec, primary_window=PR['primary_window'],
           bootstrap=dict(B=B, L=LS, seed=SEED, kind='stationary (Politis-Romano 1994)', variance='pr'),
           runtime_s=round(time.time() - t0, 1), maxrss_mb=rss())
(HERE / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
print('done', res['runtime_s'], 's', res['maxrss_mb'], 'MB')

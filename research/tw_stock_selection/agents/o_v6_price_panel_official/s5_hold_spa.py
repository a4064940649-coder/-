#!/usr/bin/env python3
"""步驟 5：保留期、斷點、台積電缺口拆解與 SPA_c 判定欄（PREREG 後一次跑完）。
(a) 保留期預測分布（沿用 o_main_spa_breaks/run_hold 的平穩 bootstrap，L3/6/12，B=10000，種子 5150+L）：
    主口徑＝日曆歸屬（V6 格點，窗終點 ≥ 2024-01-02 → HOLD）；並報起點歸屬（V6 格點）與 V4 格點日曆歸屬。
(b) Chow 2019-05、2024-01（HAC）；(c) 台積電缺口拆解（V6 格點、V6 面板價格、ex_tsmc 實際權重＋市值近似校準）；
(d) SPA_c：族群（F_long，W_long，V4 格點 245 窗）照舊，只把 o_v4_revenue_patch_official 的 14 條成員換成 V6 版；
    焦點 AB_V6、AB_V6_B10（皆為族群成員）、AB_V6 壓力成本（加入族群當第 N+1 條）；L3/6/12，B=10000，種子同 o5。"""
import sys, json, math, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from scipy import stats
import v6common as C
RES = C.check_prereg()
PR = RES['PREREG']
import v6stats as ST
sys.path.insert(0, str(C.SPADIR))
import spa_lib as S, breaks_lib as BL, fam_common as FC     # 只讀 import（o_main_spa_breaks 自寫檢定）
sys.path.pop(0)
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
NV = pd.read_parquet(C.HERE / 'navs_v6.parquet')
e50 = C.e50()
END = pd.Timestamp(ST.B0)
plan6 = pd.read_csv(C.HERE / 'plan_V6.csv')
G6 = pd.DatetimeIndex(list(pd.to_datetime(plan6['signal'])) + [END])
G4 = FC.GRID
def wr(nav, g): return FC.win_ret(nav, g)
out = {'grid_v6_n_windows': len(G6) - 1, 'grid_v4_n_windows': len(G4) - 1}
HP = PR['hold']

# ---------------- (a) 保留期預測分布 ----------------
def sb_paths(n_src, n_out, L, Bn, seed):
    rng = np.random.default_rng(seed); p = 1 / L
    idx = np.empty((Bn, n_out), np.int64); idx[:, 0] = rng.integers(0, n_src, Bn)
    for t in range(1, n_out):
        new = rng.random(Bn) < p
        idx[:, t] = np.where(new, rng.integers(0, n_src, Bn), (idx[:, t - 1] + 1) % n_src)
    return idx
def predictive(ra, rb, pre, hold):
    lx = np.log1p(ra) - np.log1p(rb); ax = ra - rb
    nh = int(hold.sum())
    act = {'cumlog': float(lx[hold].sum()), 'mean_arith_pp': float(ax[hold].mean() * 100), 'n_hold': nh, 'n_pre': int(pre.sum()),
           'hold_first_window': str(ra.index[hold][0].date())}
    src_l = lx[pre].to_numpy(); src_a = ax[pre].to_numpy()
    res = {'actual': act, 'pre_mean_arith_pp': float(src_a.mean() * 100)}
    for L in HP['block_lengths']:
        idx = sb_paths(len(src_l), nh, L, HP['B'], HP['seed'] + L)
        sl = src_l[idx].sum(axis=1); sa = src_a[idx].mean(axis=1)
        res[f'L{L}'] = {'p_lower_cumlog': float((sl <= act['cumlog']).mean()), 'p_lower_mean': float((sa <= act['mean_arith_pp'] / 100).mean()),
                        'q10_cumlog': float(np.quantile(sl, 0.10)), 'q50_cumlog': float(np.quantile(sl, 0.50)), 'q90_cumlog': float(np.quantile(sl, 0.90))}
    a_pre, a_h = src_a, ax[hold].to_numpy()
    om_pre = BL.lrv1(a_pre - a_pre.mean()); om_h = BL.lrv1(a_h - a_h.mean())
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.80)
    res['MDE80_one_sided_pp'] = {'two_sample_HAC': float(z * math.sqrt(om_pre / len(a_pre) + om_h / len(a_h)) * 100),
                                 'hold_only_HAC': float(z * math.sqrt(om_h / len(a_h)) * 100)}
    idx = sb_paths(len(src_a), nh, 6, HP['B'], HP['seed'] + 99)
    sa = src_a[idx].mean(axis=1); res['MDE80_boot_pp'] = float((np.quantile(sa, 0.80) - np.quantile(sa, 0.05)) * 100)
    res['diff_hold_minus_pre_pp'] = float((a_h.mean() - a_pre.mean()) * 100)
    pl = [res[f'L{L}']['p_lower_cumlog'] for L in HP['block_lengths']]
    res['p_lower_range'] = [min(pl), max(pl)]
    res['judgement_L6'] = '與雜訊相容' if 0.10 <= res['L6']['p_lower_cumlog'] <= 0.90 else ('落後超出雜訊' if res['L6']['p_lower_cumlog'] < 0.10 else '超前超出雜訊')
    return res
r50_6 = wr(e50, G6); r50_4 = wr(e50, G4)
hold_out = {}
for name, col, ewc in (('AB_V6', 'AB_V6|normal', 'AB_V6|EW'), ('AB_V6_B10', 'AB_V6_B10|normal', 'AB_V6_B10|EW'), ('AB_V6_stress', 'AB_V6|stress', 'AB_V6|EW')):
    rm6 = wr(NV[col], G6); ends6 = pd.Series(G6[1:], index=G6[:-1])
    pre_c = (ends6 < ST.HOLD_CAL_END) & (rm6.index >= pd.Timestamp(HP['pre'][0])); hold_c = ends6 >= ST.HOLD_CAL_END
    pre_s = (rm6.index >= pd.Timestamp(HP['pre'][0])) & (rm6.index <= pd.Timestamp('2023-12-31')); hold_s = rm6.index >= pd.Timestamp('2024-01-01')
    rm4 = wr(NV[col], G4); ends4 = pd.Series(G4[1:], index=G4[:-1])
    pre_c4 = (ends4 < ST.HOLD_CAL_END) & (rm4.index >= pd.Timestamp(HP['pre'][0])); hold_c4 = ends4 >= ST.HOLD_CAL_END
    o = {'x50_calendar_v6grid': predictive(rm6, r50_6, pre_c.to_numpy(), hold_c.to_numpy()),
         'x50_start_v6grid': predictive(rm6, r50_6, pre_s, hold_s),
         'x50_calendar_v4grid': predictive(rm4, r50_4, pre_c4.to_numpy(), hold_c4.to_numpy())}
    if name != 'AB_V6_stress':
        rew = wr(NV[ewc], G6)
        o['xEW_calendar_v6grid'] = predictive(rm6, rew, pre_c.to_numpy(), hold_c.to_numpy())
    hold_out[name] = o
    log('hold', name, {k: (v['actual']['n_hold'], round(v['actual']['cumlog'], 4), v['p_lower_range']) for k, v in o.items()})
out['hold'] = hold_out

# ---------------- (b) Chow ----------------
ch = {}
for name, col, ewc in (('AB_V6', 'AB_V6|normal', 'AB_V6|EW'), ('AB_V6_B10', 'AB_V6_B10|normal', 'AB_V6_B10|EW')):
    rm = wr(NV[col], G6); m = rm.index >= pd.Timestamp('2006-04-01')
    ser = {'x50_net': (rm - r50_6)[m].to_numpy(), 'xEW_net': (rm - wr(NV[ewc], G6))[m].to_numpy()}
    idx = rm.index[m]; ends = pd.Series(G6[1:], index=G6[:-1])[m]
    ks = {'2019-05': int(np.argmax(idx >= pd.Timestamp('2019-05-01'))), '2024-01(start)': int(np.argmax(idx >= pd.Timestamp('2024-01-01'))),
          '2024-01(calendar)': int(np.argmax(ends.to_numpy() >= np.datetime64(ST.HOLD_CAL_END)))}
    ch[name] = {s: {k: BL.chow(y, kk, hac=True) for k, kk in ks.items()} for s, y in ser.items()}
out['chow'] = ch

# ---------------- (c) 台積電缺口拆解（V6 格點、V6 面板）----------------
adj = pd.read_parquet(C.DATA / 'adj.parquet'); cr = pd.read_parquet(C.DATA / 'close_raw.parquet')
sh = pd.read_csv(C.XTR / 'shares_hist.csv', dtype={'code': str}); sh = sh[sh.ym == sh.ym.max()].set_index('code')['shares']
codes = [c for c in cr.columns if c in sh.index]
g = G6
crg = cr[codes].reindex(cr.index.union(g)).ffill(limit=5).reindex(g)
mcap = crg * sh.reindex(codes).to_numpy()[None, :]
adjg = adj.reindex(adj.index.union(g)).ffill(limit=5).reindex(g)
Rg = (adjg.shift(-1) / adjg - 1).iloc[:-1]
rows = []
for d in g[:-1]:
    mm = mcap.loc[d].dropna(); mm = mm[mm > 0]; top50 = mm.sort_values(ascending=False).iloc[:50]
    rows.append(dict(win=d, w_approx=float(top50.get('2330', np.nan) / top50.sum()), r2330=float(Rg.loc[d].get('2330', np.nan))))
T = pd.DataFrame(rows).set_index('win')
ex = pd.read_csv(C.XTR / 'ex_tsmc.csv', parse_dates=['date']).set_index('date')['w2330'] / 100
exg = ex.reindex(ex.index.union(g)).ffill(limit=5).reindex(g[:-1]); exg[g[:-1] < ex.index[0]] = np.nan
T['w_actual'] = exg.to_numpy()
ov = T[['w_approx', 'w_actual']].dropna(); cal = float((ov.w_actual - ov.w_approx).mean()) if len(ov) else 0.0
T['w'] = T['w_actual'].fillna((T['w_approx'] + cal).clip(0, 0.99))
T['r0050'] = r50_6; T['r_ex'] = (T.r0050 - T.w * T.r2330) / (1 - T.w)
dec = {'calib_actual_minus_approx': cal, 'n_overlap': int(len(ov))}
for name, col in (('AB_V6', 'AB_V6|normal'), ('AB_V6_B10', 'AB_V6_B10|normal')):
    rm = wr(NV[col], G6); ends = pd.Series(G6[1:], index=G6[:-1])
    for lab, h in (('calendar', (ends >= ST.HOLD_CAL_END).to_numpy()), ('start', (rm.index >= pd.Timestamp('2024-01-01')))):
        lw_m = float(np.log1p(rm[h]).sum()); lw_50 = float(np.log1p(T.r0050[h]).sum()); lw_ex = float(np.log1p(T.r_ex[h]).sum())
        sf = lw_50 - lw_m; gap = lw_50 - lw_ex
        dec[f'{name}|{lab}'] = {'n_windows': int(h.sum()), 'log_main': lw_m, 'log_0050': lw_50, 'log_0050_ex_tsmc': lw_ex,
                                'shortfall_vs_0050_log': sf, 'tsmc_gap_log': gap, 'main_vs_ex_tsmc_log': lw_m - lw_ex,
                                'gap_over_shortfall': float(gap / sf) if sf > 0 else None, 'w2330_mean': float(T.w[h].mean()),
                                'arith_excess_sum_pp': float((rm - T.r0050)[h].sum() * 100), 'arith_A_vs_ex_pp': float((rm - T.r_ex)[h].sum() * 100),
                                'arith_G_pp': float((T.w * (T.r2330 - T.r_ex))[h].sum() * 100)}
    # 2024-01 兩窗明細
    dec[f'{name}|windows_2023_12_to_2024_02'] = {str(d.date()): {'main_pct': float(rm[d] * 100), '0050_pct': float(T.r0050[d] * 100), '2330_pct': float(T.r2330[d] * 100),
                                                                'end': str(ends[d].date())} for d in rm.index if pd.Timestamp('2023-12-01') <= d <= pd.Timestamp('2024-02-29')}
out['tsmc_decomp'] = dec

# ---------------- (d) SPA_c 判定欄 ----------------
D = pd.read_parquet(FC.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(FC.DATA / 'family_excess_ew.parquet')
MM = pd.read_csv(FC.DATA / 'family_members.csv'); B = pd.read_parquet(FC.DATA / 'bench.parquet')
N4 = pd.read_parquet(C.V4DIR / 'navs_v4.parquet')
pref = 'o_v4_revenue_patch_official/navs_v4.parquet::'
v4mem = [m for m in MM.member_id if m.startswith(pref)]
# 核對：用 navs_v4 自己重算的超額與族群檔相同
chk = max(float(np.nanmax(np.abs((wr(N4[m[len(pref):]], G4) - B['r0050']).to_numpy() - D[m].to_numpy()))) for m in v4mem)
assert chk < 1e-10, f'族群超額重算不符 {chk}'
def v6name(k):
    k = k.replace('V4', 'V6')
    return 'O3REP_V6' + k[5:] if k.startswith('O3REP') else k
D6 = D.copy()
rep = {}
for m in v4mem:
    k6 = v6name(m[len(pref):]); nav = NV[k6]
    D6[m] = (wr(nav, G4) - B['r0050']).to_numpy()
    rep[m] = k6
# 對同池等權的損失：全部成員改對 AB_V6|EW（原族群是對 AB_V4|EW）
Dew6 = D6.add((B['r0050'] - wr(NV['AB_V6|EW'], G4)).to_numpy(), axis=0)
D6['EXTRA::AB_V6|stress'] = (wr(NV['AB_V6|stress'], G4) - B['r0050']).to_numpy()
mem_long = MM[MM['cov_W_long'] & MM['F_long']].member_id.tolist()
wm = FC.wmask(D.index, *FC.WINDOWS['W_long']); n = int(wm.sum())
spa = {'n_windows': n, 'family_size_F_long': len(mem_long), 'replaced_members': rep, 'family_excess_recompute_maxdiff': chk}
seed_base = PR['spa']['seed_base']
for L in PR['spa']['block_lengths']:
    Wc = S.sb_counts(n, L, PR['spa']['B'], seed_base + 1000 * 0 + L)
    for foc, mem, DD in (('AB_V6', mem_long, D6), ('AB_V6_B10', mem_long, D6), ('AB_V6_stress', mem_long + ['EXTRA::AB_V6|stress'], D6),
                         ('AB_V4(原族群核對)', mem_long, D), ('AB_V6|xEW', mem_long, Dew6)):
        fid = {'AB_V6': pref + 'AB_V4|normal', 'AB_V6_B10': pref + 'AB_V4_B10|normal', 'AB_V6_stress': 'EXTRA::AB_V6|stress',
               'AB_V4(原族群核對)': pref + 'AB_V4|normal', 'AB_V6|xEW': pref + 'AB_V4|normal'}[foc]
        X = DD.loc[wm, mem].fillna(0.0).to_numpy()
        o = S.mcp_tests(X, Wc, L, PR['spa']['variance'], focus=mem.index(fid))
        o.pop('top_idx', None); o.pop('top_t', None); o.pop('argmax', None)
        spa[f'{foc}|L{L}'] = {k: o[k] for k in ('p_SPA_c_focus', 'p_SPA_l_focus', 'p_SPA_u_focus', 'p_SPA_c', 'focus_rank', 'focus_t', 'p_StepM_focus', 'p_single_focus', 'max_t', 'N')}
        log('spa', foc, L, round(o['p_SPA_c_focus'], 4), o['focus_rank'])
X = D6.loc[wm, mem_long].fillna(0.0).to_numpy()
spa['eff_trials_F_long_V6'] = S.eff_trials(X)
out['spa'] = spa
# ---------------- 判定 ----------------
pA = spa['AB_V6|L6']['p_SPA_c_focus']; pB = spa['AB_V6_B10|L6']['p_SPA_c_focus']; pS = spa['AB_V6_stress|L6']['p_SPA_c_focus']
h = hold_out['AB_V6']['x50_calendar_v6grid']
out['decision'] = {
    'SPA_focus_p_L6': {'AB': pA, 'B10': pB, 'stress': pS},
    'grade': '可提議回到第 2 級（需總監整合確認）' if (pA <= 0.10 and pB <= 0.10) else '維持第 3 級',
    'prediction6_falsified(AB≤0.10 且 B10≤0.15)': bool(pA <= 0.10 and pB <= 0.15),
    'hold_p_lower_L6_calendar': h['L6']['p_lower_cumlog'], 'hold_p_lower_range_calendar': h['p_lower_range'],
    'hold_judgement': h['judgement_L6'], 'hold_MDE_pp': h['MDE80_one_sided_pp'], 'hold_MDE_boot_pp': h['MDE80_boot_pp'],
    'tsmc_gap_over_shortfall_calendar': dec['AB_V6|calendar']['gap_over_shortfall'],
}
C.save_key('RESULTS_HOLD_SPA', out)
log('done', out['decision'])

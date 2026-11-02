#!/usr/bin/env python3
"""第 3、4 步：樣本外判定（2006-04～2021-09）、只預收段、2024～2026 描述、穩健性、9 種時點、容量、組合、SPA／DSR。
先斷言 PREREG 存在且雜湊相符；只跑一次。→ results.json 的 RESULTS、trades_oos.csv.gz、navs.parquet"""
import sys, json, math, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from scipy import stats as SS
import core as K

DRY = '--dry' in sys.argv   # 乾跑：全部期間換成挑選期，用來除錯程式，不寫任何檔、不看樣本外
RES = {'PREREG': {'selected_version': 'v1_cap20_shrink_defer', 'exec': {'cap': 0.2, 'min_fill': 0.0, 'defer': True}}} if DRY else K.check_prereg()
PR = RES['PREREG']
SELV = PR['selected_version']; CAPP = PR['exec']['cap']; MINF = PR['exec']['min_fill']; DEF = PR['exec']['defer']
AG = K.AG
U = K.universe()
CUTS = K.run6_cuts(U)
IV = K.disp_intervals(U)
OOS = K.PER['OOS_2006-04_2021-09']; PRE = K.PER['PREPAY_2004_2006-03']; SEL = K.PER['SEL_2021-10_2023']; DSC = K.PER['DESC_2024_2026-10']
FULL = ('2006-04-01', '2026-10-22')
if DRY:
    OOS = PRE = DSC = FULL = SEL
    _ob = K.build
    def _b(cfg=None, u=None, period=None):
        d, st = _ob(cfg, u, period)
        return (K.sub(d, *SEL).reset_index(drop=True) if len(d) else d), st
    K.build = _b
    K.PER = {k: SEL for k in K.PER}
R = {'time_utc': datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z', 'selected_version': SELV}

def per(df, col='net'):
    return {k: K.S(K.sub(df, *v), col) for k, v in K.PER.items()} | {'FULL_2006-04_2026-10': K.S(K.sub(df, *FULL), col)}

def p_one(t, direction=+1):
    if t is None: return None
    return float(1 - SS.norm.cdf(direction * t))

# ======================= 主規則事件（全期） =======================
ALL, st_all = K.build({}, U)
ALL['seg'] = np.select([ALL['date'] < pd.Timestamp('2006-04-01'), ALL['date'] <= pd.Timestamp('2021-09-30'),
                        ALL['date'] <= pd.Timestamp('2023-12-31')], ['PREPAY', 'OOS', 'SEL'], 'DESC')
flags = pd.read_csv(HERE / 'universe_flags.csv', dtype={'code': str})
ALL = ALL.merge(flags[['ordr', 'overlap_proxy']], on='ordr', how='left')
X10 = K.exec_cost_event(ALL, 1e5, cap=CAPP, defer=DEF)
ALL['net_x'] = X10['net_x']; ALL['cost_x'] = X10['cost_x']; ALL['fill10'] = X10['fill']
XS = K.exec_cost_event(ALL, 1e5, cap=CAPP, stress=True, defer=DEF)
ALL['net_x_stress'] = XS['net_x']
for c in (0.012, 0.02): ALL[f'net_c{int(c*1000)}'] = ALL['gross'] - c
R['filter_counts_all'] = st_all
R['P1'] = {'net_flat_0.8_1.2': per(ALL), 'gross': per(ALL, 'gross'), 'net50': per(ALL, 'net50'), 'raw_net': per(ALL, 'raw_net'),
           'net_cost1.2': per(ALL, 'net_c12'), 'net_cost2.0': per(ALL, 'net_c20'), 'net_exec_V7X_10萬': per(ALL, 'net_x'),
           'net_exec_V7X_stress_10萬': per(ALL, 'net_x_stress')}
O = K.sub(ALL, *OOS).copy()
# 缺價最壞界限：OOS 窗內有記錄但無價（或不在面板）的普通股處置記錄，以原始 −30%、等權 0 計入
U2 = U[(U['start'] >= pd.Timestamp(OOS[0])) & (U['start'] <= pd.Timestamp(OOS[1]))]
miss = U2[[not (r.in_D and K.TR[max(r.s_i - 5, 0):min(r.e_i + 1, K.T), K.CI[r.code]].any()) for r in U2.itertuples()]]
B = pd.concat([O[['date', 'net']], pd.DataFrame({'date': miss['start'].values, 'net': -0.30 - 0.008})], ignore_index=True)
R['P1']['worst_case_missing_as_-30pct'] = {'n_missing': int(len(miss)), 'stats': K.S(B)}
R['P1']['OOS_unseen_subset(no_overlap_with_proxy)'] = K.S(O[O['overlap_proxy'] == False])
R['P1']['OOS_overlap_with_proxy(seen)'] = K.S(O[O['overlap_proxy'] == True])
R['P1']['OOS_yearly'] = {int(y): K.S(g) for y, g in O.groupby(O['date'].dt.year)}
R['P1']['FULL_yearly'] = {int(y): K.S(g) for y, g in ALL.groupby(ALL['date'].dt.year)}
s1 = R['P1']['net_flat_0.8_1.2']['OOS_2006-04_2021-09']
P1v = ('成立' if s1['mean_pct'] > 0 and (s1['t_month'] or -9) >= 2.0 else
       '不成立：處置溢酬是 2021 年後的現象' if s1['mean_pct'] <= 0 or (s1['t_month'] or -9) < 1.0 else '不確定（1.0 ≤ t < 2.0）')
R['P1']['verdict'] = P1v

# ======================= P2 =======================
P2 = {}
d = O.copy(); d['f'] = d['regime'] == 'periodic_match'
P2['P2a_periodic_minus_prepay_OOS_gross'] = K.cluster_t(d, 'f', 'gross')
P2['P2a_net'] = K.cluster_t(d, 'f', 'net')
d2 = K.sub(ALL, PRE[0], OOS[1]).copy(); d2['f'] = d2['regime'] == 'periodic_match'
P2['P2a_incl_2004_2006-03_gross'] = K.cluster_t(d2, 'f', 'gross')
P2['prepay_only_OOS'] = K.S(d[~d['f']], 'gross'); P2['periodic_OOS'] = K.S(d[d['f']], 'gross')
d3 = O[(O['date'] >= K.PERIODIC_START) & O['match_min'].isin([5, 20])].copy(); d3['f'] = d3['match_min'] == 20
P2['P2b_20min_minus_5min_gross'] = K.cluster_t(d3, 'f', 'gross')
P2['P2b_net'] = K.cluster_t(d3, 'f', 'net')
P2['min5'] = K.S(d3[~d3['f']], 'gross'); P2['min20'] = K.S(d3[d3['f']], 'gross')
P2['match_min_counts_OOS'] = {str(k): int(v) for k, v in O['match_min'].value_counts(dropna=False).items()}
R['P2'] = P2

# ======================= P3 安慰劑 =======================
PLo = K.placebo(O, U, CUTS, IV)
R['P3'] = {'ev_minus_pl_gross': K.S(PLo, 'ev_minus_pl'), 'coverage': round(float(PLo['pl_raw'].notna().mean()), 3),
           'mode_counts': {str(k): int(v) for k, v in PLo['pl_mode'].value_counts(dropna=False).items()},
           'pl_raw_mean_pct': round(float(PLo['pl_raw'].mean() * 100), 2), 'ev_raw_mean_pct_covered': round(float(PLo.loc[PLo['pl_raw'].notna(), 'raw'].mean() * 100), 2),
           'ev_raw_mean_pct_uncovered': round(float(PLo.loc[PLo['pl_raw'].isna(), 'raw'].mean() * 100), 2) if PLo['pl_raw'].isna().any() else None,
           'by_bucket': {int(b): K.S(g, 'ev_minus_pl') for b, g in PLo.groupby('bucket')}}
p3 = R['P3']['ev_minus_pl_gross']
R['P3']['verdict'] = '制度機制（≥1% 且 t≥2）' if p3.get('mean_pct', -9) >= 1.0 and (p3.get('t_month') or -9) >= 2 else '不成立（最高待觀察，不得寫制度溢酬）'
PLs = K.placebo(K.sub(ALL, *SEL), U, CUTS, IV); PLd = K.placebo(K.sub(ALL, *DSC), U, CUTS, IV)
R['P3']['SEL_ref'] = K.S(PLs, 'ev_minus_pl'); R['P3']['DESC_2024_2026'] = K.S(PLd, 'ev_minus_pl')
pd.concat([PLo.assign(seg='OOS'), PLs.assign(seg='SEL'), PLd.assign(seg='DESC')])[['seg', 'date', 'code', 'raw', 'pl_raw', 'pl_n', 'pl_mode', 'bucket', 'ev_minus_pl']].to_csv(HERE / 'placebo_events.csv.gz', index=False)

# ======================= P4 子樣本 =======================
P4 = {}
o = O.copy()
o['vdrop_big'] = o['vdrop'] <= o['vdrop'].median()
o['lim10'] = o['date'] >= K.LIM_SWITCH
o['q5'] = o['run6'] >= CUTS[3]
P4['stronger_vdrop_big'] = K.cluster_t(o, 'vdrop_big', 'gross')
P4['stronger_10pct_era'] = K.cluster_t(o, 'lim10', 'gross')
NB, _ = K.build({'bear': False}, U, OOS); NB['above'] = [not K.bear_before(K.IDX[s]) for s in NB['s_i']]
P4['stronger_above_200ma(no_bear_switch)'] = K.cluster_t(NB, 'above', 'gross')
P4['weaker_second'] = K.cluster_t(o, 'second', 'gross')
P4['weaker_run6_Q5'] = K.cluster_t(o, 'q5', 'gross')
dd = K.sub(ALL, '2024-01-01', DSC[1]).copy(); dd['new'] = dd['date'] >= K.CHANGE
P4['desc_new_regime_5day_minus_old_2024_2026'] = K.cluster_t(dd, 'new', 'gross')
P4['vdrop_median_pct_OOS'] = round(float(o['vdrop'].median() * 100), 1)
P4['below_200ma_events_nobear'] = K.S(NB[~NB['above']]); P4['above_200ma_events_nobear'] = K.S(NB[NB['above']])
pred = {'stronger_vdrop_big': +1, 'stronger_10pct_era': +1, 'stronger_above_200ma(no_bear_switch)': +1, 'weaker_second': -1, 'weaker_run6_Q5': -1}
P4['direction_ok'] = {k: (None if P4[k] is None else bool(np.sign(P4[k]['diff_pct']) == s)) for k, s in pred.items()}
R['P4'] = P4

# ======================= P5 時序 =======================
P5 = {}
for nm, (a, b) in (('OOS', OOS), ('SEL', SEL), ('DESC_2024_2026', DSC)):
    Pp = K.path_split(K.sub(ALL, a, b))
    P5[nm] = {'early': K.S(Pp, 'early'), 'late': K.S(Pp, 'late'), 'late_minus_early': K.S(Pp, 'late_m_early')}
lm = P5['OOS']['late_minus_early']
P5['verdict_OOS'] = '成立' if lm.get('mean_pct', -9) > 0 and (lm.get('t_month') or -9) >= 2 else ('部分成立' if lm.get('mean_pct', -9) > 0 else '不成立')
R['P5'] = P5

# ======================= Holm =======================
tests = {'P1': (s1.get('t_month'), +1), 'P2a': ((P2['P2a_periodic_minus_prepay_OOS_gross'] or {}).get('t_cluster_month'), +1),
         'P2b': ((P2['P2b_20min_minus_5min_gross'] or {}).get('t_cluster_month'), +1), 'P3': (p3.get('t_month'), +1),
         'P4_vdrop': ((P4['stronger_vdrop_big'] or {}).get('t_cluster_month'), +1), 'P4_lim10': ((P4['stronger_10pct_era'] or {}).get('t_cluster_month'), +1),
         'P4_above': ((P4['stronger_above_200ma(no_bear_switch)'] or {}).get('t_cluster_month'), +1),
         'P4_second': ((P4['weaker_second'] or {}).get('t_cluster_month'), -1), 'P4_q5': ((P4['weaker_run6_Q5'] or {}).get('t_cluster_month'), -1),
         'P5': (lm.get('t_month'), +1)}
ps = {k: p_one(t, s) for k, (t, s) in tests.items()}
order = sorted([k for k in ps if ps[k] is not None], key=lambda k: ps[k]); m = len(order); holm = {}; run = 0
for i, k in enumerate(order):
    run = max(run, min(1.0, (m - i) * ps[k])); holm[k] = round(run, 4)
R['holm'] = {'n_tests': m, 'p_one_sided': {k: (round(v, 4) if v is not None else None) for k, v in ps.items()}, 'p_holm': holm,
             't_used': {k: t for k, (t, s) in tests.items()}}

# ======================= G1：9 種時點 =======================
G1 = {}; grid_tr = {}
for e in ('C1', 'O2', 'C2'):
    for x in ('CL', 'CL1', 'OA'):
        dfe, _ = K.build({'entry': e, 'exit': x}, U)
        Xe = K.exec_cost_event(dfe, 1e5, cap=CAPP, defer=DEF); Xs = K.exec_cost_event(dfe, 1e5, cap=CAPP, stress=True, defer=DEF)
        dfe['net_x'] = Xe['net_x']; dfe['net_x_stress'] = Xs['net_x']; dfe['net_c20'] = dfe['gross'] - 0.02
        grid_tr[f'{e}x{x}'] = dfe
        G1[f'{e}x{x}'] = {p: {'flat': K.S(K.sub(dfe, *v)), 'exec': K.S(K.sub(dfe, *v), 'net_x'), 'exec_stress': K.S(K.sub(dfe, *v), 'net_x_stress'),
                              'flat2.0': K.S(K.sub(dfe, *v), 'net_c20')}
                          for p, v in (('FULL', FULL), ('OOS', OOS), ('2019+', ('2019-01-01', FULL[1])))}
def q(vals, f): v = [x for x in vals if x is not None]; return round(float(f(v)), 3) if v else None
G1s = {}
for p in ('FULL', 'OOS', '2019+'):
    for c in ('flat', 'exec', 'exec_stress', 'flat2.0'):
        vals = [G1[k][p][c].get('mean_pct') for k in G1]; ts = [G1[k][p][c].get('t_month') for k in G1]
        G1s[f'{p}|{c}'] = {'median_mean_pct': q(vals, np.median), 'p10_mean_pct': q(vals, lambda v: np.percentile(v, 10)),
                           'median_t': q(ts, np.median), 'p10_t': q(ts, lambda v: np.percentile(v, 10))}
R['G1_grid'] = G1; R['G1_summary'] = G1s

# ======================= 組合：衛星（閒置 0050） =======================
NAV = {}; SL = {}
def run_sl(name, trades, **kw):
    nav, st = K.sleeve(trades, t_start=FULL[0], cap=kw.pop('cap', CAPP), min_fill=MINF, defer=DEF, **kw)
    NAV[name] = nav; SL[name] = {'stat': st, 'FULL': K.vs0050(nav), 'OOS': K.vs0050(nav, *OOS), '2019+': K.vs0050(nav, '2019-01-01'),
                                 'SEG_2006_2012': K.vs0050(nav, '2006-04-01', '2012-12-31'), 'SEG_2013_2018': K.vs0050(nav, '2013-01-01', '2018-12-31'),
                                 'SEG_2019_2026': K.vs0050(nav, '2019-01-01', FULL[1]), 'SEL': K.vs0050(nav, *SEL), 'DESC': K.vs0050(nav, *DSC)}
    return nav
BASE = grid_tr['C1xCL']
for tier in (1e6, 3e6, 1e7):
    for mode in ('FIX', 'GROW'):
        run_sl(f'C1xCL|{int(tier/1e4)}萬|{mode}|normal', BASE, tier_2026=tier, mode=mode)
        run_sl(f'C1xCL|{int(tier/1e4)}萬|{mode}|stress', BASE, tier_2026=tier, mode=mode, stress=True)
for k, dfe in grid_tr.items():
    if k != 'C1xCL': run_sl(f'{k}|100萬|FIX|normal', dfe, tier_2026=1e6, mode='FIX')
for kk in (5, 20): run_sl(f'C1xCL|100萬|FIX|normal|K{kk}', BASE, tier_2026=1e6, mode='FIX', K=kk)
run_sl('C1xCL|100萬|FIX|normal|cashbuffer2', BASE, tier_2026=1e6, mode='FIX', buffer_slots=2)
NEI = {}
for nm, cfg in (('thr8', {'thr_off': 0.02}), ('thr9.5', {'thr_off': 0.005}), ('liq500', {'liq': 5e6}), ('liq2000', {'liq': 2e7}),
                ('nobear', {'bear': False}), ('first_only', {'first_only': True}), ('regular_only', {'kinds': ('regular',)})):
    dfn, _ = K.build(cfg, U)
    NEI[nm] = {'OOS_event_net': K.S(K.sub(dfn, *OOS)), 'FULL_event_net': K.S(K.sub(dfn, *FULL))}
    run_sl(f'C1xCL|100萬|FIX|normal|{nm}', dfn, tier_2026=1e6, mode='FIX')
NEI['base'] = {'OOS_event_net': K.S(K.sub(ALL, *OOS)), 'FULL_event_net': K.S(K.sub(ALL, *FULL))}
R['neighborhood_event'] = NEI
R['sleeves'] = SL
FOC = 'C1xCL|100萬|FIX|normal'
fnav = NAV[FOC]
# G2、去最好 3 年、三段
ms = K.monthly(fnav); me = K.monthly(K.e50_nav(fnav.index)); ex = (ms - me).dropna()
yr = ex.groupby(ex.index.year).sum().sort_values(ascending=False)
ex3 = ex[~ex.index.year.isin(yr.index[:3])]
R['G2'] = {'FULL_t': SL[FOC]['FULL'].get('t_vs_0050'), '2019+_t': SL[FOC]['2019+'].get('t_vs_0050'),
           'remove_best3_years': {'years': [int(y) for y in yr.index[:3]], 't': K.tstat(ex3), 'mean_pp': round(float(ex3.mean() * 100), 3)},
           'yearly_excess_sum_pp': {int(k): round(float(v * 100), 2) for k, v in ex.groupby(ex.index.year).sum().items()}}
# 去最大 1 個月／1 年／1 檔（事件層 OOS 與 FULL）
def drop_top(df, key):
    g = df.groupby(key)['net'].sum(); top = g.idxmax(); return df[df[key] != top], str(top)
RB = {}
for nm, (a, b) in (('OOS', OOS), ('FULL', FULL)):
    dfp = K.sub(ALL, a, b).copy(); dfp['ym'] = dfp['date'].dt.to_period('M'); dfp['yy'] = dfp['date'].dt.year
    RB[nm] = {}
    for key in ('ym', 'yy', 'code'):
        x, top = drop_top(dfp, key); RB[nm][f'drop_top_{key}'] = {'dropped': top, 'stats': K.S(x)}
R['robust_drop'] = RB
# 核心＋衛星（只報告）
e5 = K.e50_nav(fnav.index)
for w in (0.7, 0.8):
    mix = (w * K.monthly(e5) + (1 - w) * ms).dropna()
    R.setdefault('core_satellite', {})[f'0050_{int(w*100)}/{int(100-w*100)}'] = {
        'ann': round(float((1 + mix).prod() ** (12 / len(mix)) - 1) * 100, 2), 't_vs_0050': K.tstat(mix - K.monthly(e5).reindex(mix.index))}

# ======================= 主策略比較與 P6 =======================
MV = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
cols = [c for c in MV.columns if c.startswith('B10|J') and c.endswith('|K5T|normal|100萬|FIX')]
mm = pd.concat([K.monthly(MV[c].dropna()) for c in cols], axis=1).mean(axis=1)
mex = (mm - me.reindex(mm.index)).dropna()
com = pd.concat([ex.rename('sat'), mex.rename('main')], axis=1).dropna()
R['P6'] = {'corr_sat_main_monthly_excess': round(float(com.corr().iloc[0, 1]), 3), 'n_months': int(len(com)),
           'corr_2019+': round(float(com[com.index.year >= 2019].corr().iloc[0, 1]), 3),
           'main_V7X_100萬FIX_t_vs_0050': K.tstat(mex), 'main_V7X_2019+_t': K.tstat(mex[mex.index.year >= 2019])}
R['P6']['verdict'] = '成立（< 0.3）' if R['P6']['corr_sat_main_monthly_excess'] < 0.3 else '不成立'
# 同帳戶配對：主策略 80＋處置 20 對 0050 80＋處置 20（月再平衡近似）
A_ = (0.8 * mm + 0.2 * ms).dropna(); B_ = (0.8 * me + 0.2 * ms).reindex(A_.index).dropna(); A_ = A_.reindex(B_.index)
R['pair_main80_vs_0050_80'] = {'ann_A': round(float((1 + A_).prod() ** (12 / len(A_)) - 1) * 100, 2),
                               'ann_B': round(float((1 + B_).prod() ** (12 / len(B_)) - 1) * 100, 2), 't_A_minus_B': K.tstat(A_ - B_),
                               't_2019+': K.tstat((A_ - B_)[A_.index.year >= 2019])}

# ======================= G3：SPA＋DSR =======================
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import fam_common as FC, spa_lib as SPA
FD = pd.read_parquet(FC.DATA / 'family_excess.parquet'); MM_ = pd.read_csv(FC.DATA / 'family_members.csv')
a, b = FC.WINDOWS['W_long']; msk = FC.wmask(FD.index, a, b)
mem = MM_[MM_['F_long'] & MM_['cov_W_long']]['member_id'].tolist()
e50w = FC.win_ret(FC.e50_nav())
mine = {}
for k, nav in NAV.items():
    w_ = FC.win_ret(nav) - e50w
    mine['THIS|' + k] = w_.reindex(FD.index)
MY = pd.DataFrame(mine)
X = pd.concat([FD.loc[msk, mem], MY.loc[msk]], axis=1).fillna(0.0)
names = list(X.columns); foc = names.index('THIS|' + FOC)
Wc = SPA.sb_counts(len(X), 6, 10000, 20261102)
sp = SPA.mcp_tests(X.to_numpy(), Wc, 6, 'pr', focus=foc)
et = SPA.eff_trials(X.to_numpy())
sr_all = X.mean(axis=0) / X.std(axis=0, ddof=1)
xf = X.iloc[:, foc].to_numpy()
R['G3'] = {'family_size': len(names), 'n_this_variants': int(MY.shape[1]), 'n_windows': int(len(X)),
           'SPA_c_focus_p': sp.get('p_SPA_c_focus'), 'SPA_l_focus_p': sp.get('p_SPA_l_focus'), 'SPA_c_family_p': sp.get('p_SPA_c'),
           'focus_rank': sp.get('focus_rank'), 'focus_t_window': sp.get('focus_t'), 'max_t': sp.get('max_t'),
           'p_single_focus': sp.get('p_single_focus'), 'eff_trials': et,
           'DSR_LiJi': SPA.dsr(xf, sr_all.to_numpy(), et['M_eff_LiJi']), 'DSR_Nyholt': SPA.dsr(xf, sr_all.to_numpy(), et['M_eff_Nyholt'])}

# ======================= 描述：2024～2026、新制 =======================
R['DESC'] = {'2024_2026': K.S(K.sub(ALL, *DSC)), 'new_regime_2026-08-10_': K.S(K.sub(ALL, *K.PER['NEWREG_2026-08-10_'])),
             'new_regime_n_by_plan_len': {int(k): int(v) for k, v in K.sub(ALL, *K.PER['NEWREG_2026-08-10_'])['plan_n'].value_counts().items()},
             'PREPAY_2004_2006-03': K.S(K.sub(ALL, *PRE)), 'delisted_in_hold': ALL[ALL['delisted']][['date', 'code', 'raw', 'net']].astype(str).to_dict('records')}

# ======================= 輸出 =======================
if DRY:
    print('DRY OK', len(ALL), list(R.keys())); sys.exit(0)
ALL.drop(columns=[c for c in ('announce', 'start', 'end') if c in ALL.columns]).to_csv(HERE / 'trades_all.csv.gz', index=False)
pd.DataFrame(NAV).to_parquet(HERE / 'navs.parquet')
res = json.loads(K.RES_F.read_text(encoding='utf-8'))
assert 'RESULTS' not in res, 'RESULTS 已存在：樣本外只跑一次'
res['RESULTS'] = R
K.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({'P1': R['P1']['net_flat_0.8_1.2']['OOS_2006-04_2021-09'], 'P1v': P1v, 'G2': R['G2'], 'G3': {k: R['G3'][k] for k in ('SPA_c_focus_p', 'focus_rank')}}, ensure_ascii=False, default=str))

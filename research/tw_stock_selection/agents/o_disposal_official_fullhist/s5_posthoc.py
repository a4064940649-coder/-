#!/usr/bin/env python3
"""事後（PREREG 之後）：(1) FIX 實作錯誤修正後重跑全部 FIX 衛星、G2／G4／P6／SPA；(2) 未事前登記的診斷（分期、未看過子集）。
全部寫進 results.json 的 POSTHOC_FIX 與 POSTHOC_DIAG，不得當作事前檢定。"""
import sys, json, datetime
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import core as K
import core_fix as F

RES = K.check_prereg()
assert 'RESULTS' in RES
PR = RES['PREREG']; CAPP = PR['exec']['cap']; DEF = PR['exec']['defer']; MINF = PR['exec']['min_fill']
U = K.universe()
OOS = K.PER['OOS_2006-04_2021-09']; SEL = K.PER['SEL_2021-10_2023']; DSC = K.PER['DESC_2024_2026-10']; FULL = ('2006-04-01', '2026-10-22')
P = {'time_utc': datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat(timespec='seconds') + 'Z',
     'bug': 'core.sleeve 的 FIX：每槽固定 tier/10，但帳戶權益隨 0050 成長不縮放，處置部位占比由約 15% 萎縮到 1～4%（100 萬 FIX 平均投入 4.1%），'
            '與 V7X 的 FIX（帳戶固定規模）不符。修正：每月初把帳戶縮放回 tier（不計成本）。GROW 不受影響（逐日差 2e−15）。'}
grid = {}
for e in ('C1', 'O2', 'C2'):
    for x in ('CL', 'CL1', 'OA'):
        grid[f'{e}x{x}'], _ = K.build({'entry': e, 'exit': x}, U)
NAV = {}; SL = {}
def run(name, tr, **kw):
    nav, st = F.sleeve_fix(tr, t_start=FULL[0], cap=CAPP, min_fill=MINF, defer=DEF, **kw)
    NAV[name] = nav
    SL[name] = {'stat': st, 'FULL': K.vs0050(nav), 'OOS': K.vs0050(nav, *OOS), '2019+': K.vs0050(nav, '2019-01-01'),
                'SEG_2006_2012': K.vs0050(nav, '2006-04-01', '2012-12-31'), 'SEG_2013_2018': K.vs0050(nav, '2013-01-01', '2018-12-31'),
                'SEG_2019_2026': K.vs0050(nav, '2019-01-01', FULL[1]), 'SEL': K.vs0050(nav, *SEL), 'DESC': K.vs0050(nav, *DSC)}
B = grid['C1xCL']
for tier in (1e6, 3e6, 1e7):
    for mode in ('FIX', 'GROW'):
        for c in ('normal', 'stress'):
            run(f'C1xCL|{int(tier/1e4)}萬|{mode}|{c}', B, tier_2026=tier, mode=mode, stress=(c == 'stress'))
for k, tr in grid.items():
    if k != 'C1xCL':
        run(f'{k}|100萬|FIX|normal', tr, tier_2026=1e6, mode='FIX')
        run(f'{k}|100萬|FIX|stress', tr, tier_2026=1e6, mode='FIX', stress=True)
        run(f'{k}|1000萬|GROW|normal', tr, tier_2026=1e7, mode='GROW')
for kk in (5, 20): run(f'C1xCL|100萬|FIX|normal|K{kk}', B, tier_2026=1e6, mode='FIX', K=kk)
run('C1xCL|100萬|FIX|normal|cashbuffer2', B, tier_2026=1e6, mode='FIX', buffer_slots=2)
for nm, cfg in (('thr8', {'thr_off': 0.02}), ('thr9.5', {'thr_off': 0.005}), ('liq500', {'liq': 5e6}), ('liq2000', {'liq': 2e7}),
                ('nobear', {'bear': False}), ('first_only', {'first_only': True}), ('regular_only', {'kinds': ('regular',)})):
    dfn, _ = K.build(cfg, U); run(f'C1xCL|100萬|FIX|normal|{nm}', dfn, tier_2026=1e6, mode='FIX')
P['sleeves'] = SL
FOC = 'C1xCL|100萬|FIX|normal'
fnav = NAV[FOC]; ms = K.monthly(fnav); me = K.monthly(K.e50_nav(fnav.index)); ex = (ms - me).dropna()
yr = ex.groupby(ex.index.year).sum().sort_values(ascending=False); ex3 = ex[~ex.index.year.isin(yr.index[:3])]
P['G2'] = {'FULL_t': SL[FOC]['FULL'].get('t_vs_0050'), '2019+_t': SL[FOC]['2019+'].get('t_vs_0050'),
           'remove_best3_years': {'years': [int(y) for y in yr.index[:3]], 't': K.tstat(ex3)},
           'yearly_excess_sum_pp': {int(k): round(float(v * 100), 2) for k, v in ex.groupby(ex.index.year).sum().items()}}
# 9 種時點的衛星 t 分布（100 萬 FIX，一般／壓力）
for c in ('normal', 'stress'):
    ts = [SL[f'{k}|100萬|FIX|{c}']['FULL'].get('t_vs_0050') for k in grid]; t19 = [SL[f'{k}|100萬|FIX|{c}']['2019+'].get('t_vs_0050') for k in grid]
    ex_ = [SL[f'{k}|100萬|FIX|{c}']['FULL'].get('ex_month_mean_pp') for k in grid]
    P[f'timing_sleeve_100萬FIX_{c}'] = {'t_full_median': float(np.median(ts)), 't_full_p10': float(np.percentile(ts, 10)),
                                         't_2019_median': float(np.median(t19)), 'ex_pp_median': float(np.median(ex_)),
                                         'ex_pp_p10': float(np.percentile(ex_, 10)), 'by_timing_t': dict(zip(grid, ts))}
ts = [SL[f'{k}|1000萬|GROW|normal' if k != 'C1xCL' else 'C1xCL|1000萬|GROW|normal']['FULL'].get('ex_month_mean_pp') for k in grid]
P['timing_sleeve_1000萬GROW_ex_pp'] = dict(zip(grid, ts))
# P6 與主策略
AG = K.AG
MV = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x' / 'out' / 'navs_oos_R4.parquet')
cols = [c for c in MV.columns if c.startswith('B10|J') and c.endswith('|K5T|normal|100萬|FIX')]
mm = pd.concat([K.monthly(MV[c].dropna()) for c in cols], axis=1).mean(axis=1)
mex = (mm - me.reindex(mm.index)).dropna()
com = pd.concat([ex.rename('sat'), mex.rename('main')], axis=1).dropna()
P['P6'] = {'corr': round(float(com.corr().iloc[0, 1]), 3), 'corr_2019+': round(float(com[com.index.year >= 2019].corr().iloc[0, 1]), 3), 'n': int(len(com))}
A_ = (0.8 * mm + 0.2 * ms).dropna(); B_ = (0.8 * me + 0.2 * ms).reindex(A_.index).dropna(); A_ = A_.reindex(B_.index)
P['pair_main80_vs_0050_80'] = {'t': K.tstat(A_ - B_), 't_2019+': K.tstat((A_ - B_)[A_.index.year >= 2019])}
for w in (0.7, 0.8):
    mix = (w * me + (1 - w) * ms).dropna()
    P.setdefault('core_satellite', {})[f'0050_{int(w*100)}/{int(100-w*100)}'] = {'ann': round(float((1 + mix).prod() ** (12 / len(mix)) - 1) * 100, 2),
        'ann_0050': round(float((1 + me.reindex(mix.index)).prod() ** (12 / len(mix)) - 1) * 100, 2), 't_vs_0050': K.tstat(mix - me.reindex(mix.index))}
# SPA（修正後 FIX 取代原 FIX；本題全部變體）
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import fam_common as FC, spa_lib as SPA
FD = pd.read_parquet(FC.DATA / 'family_excess.parquet'); MM_ = pd.read_csv(FC.DATA / 'family_members.csv')
a, b = FC.WINDOWS['W_long']; msk = FC.wmask(FD.index, a, b)
mem = MM_[MM_['F_long'] & MM_['cov_W_long']]['member_id'].tolist()
e50w = FC.win_ret(FC.e50_nav())
MY = pd.DataFrame({'THIS|' + k: (FC.win_ret(nav) - e50w).reindex(FD.index) for k, nav in NAV.items()})
X = pd.concat([FD.loc[msk, mem], MY.loc[msk]], axis=1).fillna(0.0)
names = list(X.columns); foc = names.index('THIS|' + FOC)
Wc = SPA.sb_counts(len(X), 6, 10000, 20261102)
sp = SPA.mcp_tests(X.to_numpy(), Wc, 6, 'pr', focus=foc)
et = SPA.eff_trials(X.to_numpy()); sr_all = (X.mean(axis=0) / X.std(axis=0, ddof=1)).to_numpy(); xf = X.iloc[:, foc].to_numpy()
best_this = max(range(len(names)), key=lambda i: (X.iloc[:, i].mean() / X.iloc[:, i].std(ddof=1)) if names[i].startswith('THIS|') else -9)
sp_best = SPA.mcp_tests(X.to_numpy(), Wc, 6, 'pr', focus=best_this)
P['G3'] = {'family_size': len(names), 'n_this_variants': int(MY.shape[1]), 'SPA_c_focus_p': sp.get('p_SPA_c_focus'), 'focus_rank': sp.get('focus_rank'),
           'focus_t_window': sp.get('focus_t'), 'max_t': sp.get('max_t'), 'p_single_focus': sp.get('p_single_focus'),
           'best_this_variant': names[best_this], 'best_this_SPA_c_p': sp_best.get('p_SPA_c_focus'), 'best_this_t_window': sp_best.get('focus_t'),
           'eff_trials': et, 'DSR_LiJi': SPA.dsr(xf, sr_all, et['M_eff_LiJi']), 'DSR_LiJi_best_this': SPA.dsr(X.iloc[:, best_this].to_numpy(), sr_all, et['M_eff_LiJi'])}
pd.DataFrame(NAV).to_parquet(HERE / 'navs_fixcorrected.parquet')

# ---------------- 未事前登記的診斷 ----------------
ALL = pd.read_csv(HERE / 'trades_all.csv.gz', dtype={'code': str}, parse_dates=['date', 'exit_date'])
D_ = {}
for nm, (a_, b_) in (('2006-04_2015-12', ('2006-04-01', '2015-12-31')), ('2016_2019', ('2016-01-01', '2019-12-31')),
                     ('2020_2021-09', ('2020-01-01', '2021-09-30')), ('2006-04_2019', ('2006-04-01', '2019-12-31'))):
    x = K.sub(ALL, a_, b_)
    D_[nm] = {'net': K.S(x), 'net_x': K.S(x, 'net_x'), 'net50': K.S(x, 'net50')}
O = K.sub(ALL, *OOS)
D_['OOS_unseen_x_era'] = {'unseen_2006_2019': K.S(K.sub(O[O['overlap_proxy'] == False], '2006-04-01', '2019-12-31')),
                          'unseen_2020_2021': K.S(K.sub(O[O['overlap_proxy'] == False], '2020-01-01', '2021-09-30'))}
D_['OOS_prepay_vs_periodic_within_2009_2012'] = {'prepay': K.S(K.sub(O[O['regime'] == 'prepay_only'], '2009-01-01', '2012-12-31'), 'gross'),
                                                 'periodic': K.S(K.sub(O[O['regime'] == 'periodic_match'], '2009-01-01', '2012-12-31'), 'gross')}
P['DIAG'] = D_
res = json.loads(K.RES_F.read_text(encoding='utf-8'))
res['POSTHOC_FIX'] = P
K.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: P[k] for k in ('G2', 'P6', 'pair_main80_vs_0050_80', 'core_satellite', 'timing_sleeve_100萬FIX_normal', 'timing_sleeve_100萬FIX_stress', 'timing_sleeve_1000萬GROW_ex_pp')}, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in P['G3'].items() if k not in ('eff_trials',)}, ensure_ascii=False, default=str)[:1500])
print(json.dumps(D_, ensure_ascii=False)[:3000])
for k, v in SL.items():
    f = v['FULL']; print(k, f.get('ann'), f.get('mdd'), f.get('ex_month_mean_pp'), f.get('t_vs_0050'), '| 19+', v['2019+'].get('t_vs_0050'), '| seg', v['SEG_2006_2012'].get('ex_month_mean_pp'), v['SEG_2013_2018'].get('ex_month_mean_pp'), v['SEG_2019_2026'].get('ex_month_mean_pp'), 'inv', v['stat']['mean_invested'])

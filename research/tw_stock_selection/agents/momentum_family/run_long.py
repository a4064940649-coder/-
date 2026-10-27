#!/usr/bin/env python3
"""價格動能家族 P1～P6（＋P4e 穩健性）長歷史 2005～2026，M2 成交、30/10 檔、成本 gross/滑價0.1%/壓力。
輸出 results_long.json、long_nav.parquet。事前規則：挑選期（2005-01～2019-04）n30 滑價 0.1% 對等權月窗 t 最高者＝選中的價格腿（P4e 不參賽）。"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlib as M
L = M.L
t_start = time.time()
D = list(M.LDATES)
el_univ, t300, inD = M.universe(D)
S = {'P1_52wHigh': M.at(M.P1, D), 'P2_mom12_1': M.at(M.P2, D), 'P3_mom6_1': M.at(M.P3, D), 'P6_volAdj12_1': M.at(M.P6, D)}
el = el_univ & S['P1_52wHigh'].notna() & S['P2_mom12_1'].notna() & S['P3_mom6_1'].notna() & S['P6_volAdj12_1'].notna()
S['P4_resid12_1_taiex'] = M.resid_mom(D, 'taiex')
S['P4e_resid12_1_EW'] = M.resid_mom(D, 'ew')
S['P5_indNeutral12_1'] = M.ind_neutral(S['P2_mom12_1'], el)
ELIG = {k: el for k in S}
el4 = el & S['P4_resid12_1_taiex'].notna(); ELIG['P4_resid12_1_taiex'] = el4
ELIG['P4e_resid12_1_EW'] = el & S['P4e_resid12_1_EW'].notna()
print('signals done', round(time.time() - t_start), flush=True)

W = {'select_2005-01_2019-04': ('2005-01-01', '2019-04-30'), 'valid_2019-05_2026-10': ('2019-05-01', '2026-10-31'),
     'main_2021-03_2026-10': ('2021-03-01', '2026-10-31'), 'full_2005-01_2026-10': ('2005-01-01', '2026-10-31'),
     'valid_survivorOnly_2019-05_2023-12': ('2019-05-01', '2023-12-31'), 'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'),
     'valid_2024-01_2025-12': ('2024-01-01', '2025-12-31'),
     'R_pre_daytrade_2005-01_2013-12': ('2005-01-01', '2013-12-31'), 'R_daytrade_7pct_2014-01_2015-05': ('2014-01-01', '2015-05-31'),
     'R_10pct_preSurv_2015-06_2019-04': ('2015-06-01', '2019-04-30'), 'R_7pct_2005-01_2015-05': ('2005-01-01', '2015-05-31'),
     'R_10pct_2015-06_2026-10': ('2015-06-01', '2026-10-31'), 'P4_common_select_2006-02_2019-04': ('2006-02-01', '2019-04-30')}

ev0, pools = M.select(S['P1_52wHigh'], el, D, 30)
EW, _ = M.sim(pools, 'gross', D)
ev4, pools4 = M.select(S['P4_resid12_1_taiex'], el4, D, 30)
EW4, _ = M.sim(pools4, 'gross', D)
E50 = L.E50_NAV
res = {'_meta': {'sparse_days_dropped_from_v60': [str(x.date()) for x in M.SPARSE], 'dates': [str(D[0].date()), str(D[-1].date())], 'n_rebalances': len(D), 'industry_source_counts': M.IND_SRC,
                 'pool_size_mean_by_year': {int(y): round(float(g.mean()), 1) for y, g in el.sum(axis=1).groupby(el.index.year)},
                 'P4_pool_size_mean': round(float(el4.sum(axis=1)[el4.sum(axis=1) > 0].mean()), 1),
                 'note': 'EW = 共同股池等權（gross、同引擎、d+1 開盤換股）；P4 另報自身股池 EW4。月窗 t = 換股日 d 收盤→下一 d 收盤。2019-05～2023-12 面板只有存活者。'},
       'EW_common': {k: M.stats(EW, a, b, D, E50) for k, (a, b) in W.items()}, 'EW_yearly': M.yearly(EW),
       '0050_yearly': M.yearly(E50[(E50.index >= D[0]) & (E50.index <= D[-1])])}
navs = {'EW': EW, 'EW4': EW4, '0050': E50.reindex(EW.index)}
for name, sc in S.items():
    e = ELIG[name]
    ic = M.ic_series(sc, e, D)
    o = {'IC': {k: {'mean': round(float(ic[(ic.index >= a) & (ic.index <= b)].mean()), 4), 't': L.tstat(ic[(ic.index >= a) & (ic.index <= b)])} for k, (a, b) in W.items()}}
    for n in (30, 10):
        ev, _ = M.select(sc, e, D, n); nb = M.select.nblk
        for cost in ('gross', 0.001, 'stress'):
            nav, tov = M.sim(ev, cost, D)
            key = f'n{n}_{cost}'; navs[f'{name}_{key}'] = nav
            o[key] = {'turnover_value_per_rebalance': round(tov, 3), 'skipped_limit_up': int(nb),
                      'windows': {k: M.stats(nav, a, b, D, E50, EW) for k, (a, b) in W.items()}, 'yearly_pct': M.yearly(nav)}
            if name.startswith('P4'):
                o[key]['windows_vs_ownpool_EW4'] = {k: M.stats(nav, a, b, D, None, EW4) for k, (a, b) in W.items()}
    res[name] = o
    w = o['n30_0.001']['windows']
    print(name, {k: (w[k]['CAGR'], w[k]['t_vs_EW_m'], w[k]['t_vs_0050_m']) for k in list(W)[:3] if w[k]}, round(time.time() - t_start), flush=True)

# ---- 事前選擇規則 ----
cands = [k for k in S if not k.startswith('P4e')]
tsel = {k: res[k]['n30_0.001']['windows']['select_2005-01_2019-04']['t_vs_EW_m'] for k in cands}
best = max(tsel, key=tsel.get)
res['SELECTION'] = {'rule': '挑選期 2005-01～2019-04、n30、滑價 0.1%、對共同股池等權月窗 t 最高', 't_by_signal': tsel, 'chosen': best}
print('SELECTION', tsel, best, flush=True)

# ---- 制度前後差異（Welch t，月超額 vs EW，n30 滑價 0.1%）----
def mex(name):
    nav = navs[f'{name}_n30_0.001']; return M.month_rets(nav, D) - M.month_rets(EW, D)
def welch(x, y):
    x, y = x.dropna(), y.dropna(); return round(float((x.mean() - y.mean()) / np.sqrt(x.var() / len(x) + y.var() / len(y))), 2)
def seg(x, a, b): return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
reg = {}
for name in S:
    x = mex(name)
    reg[name] = {
        'mean_excess_pct_m': {k: round(float(seg(x, *W[k]).mean() * 100), 2) for k in W if k.startswith('R_')},
        'welch_t_10pct_vs_7pct_(2015-06~2026-10 vs 2005-01~2015-05)': welch(seg(x, *W['R_10pct_2015-06_2026-10']), seg(x, *W['R_7pct_2005-01_2015-05'])),
        'welch_t_10pct_preSurv_vs_7pct_(2015-06~2019-04 vs 2005-01~2015-05)': welch(seg(x, *W['R_10pct_preSurv_2015-06_2019-04']), seg(x, *W['R_7pct_2005-01_2015-05'])),
        'welch_t_postDaytrade_vs_pre_(2014-01~2019-04 vs 2005-01~2013-12)': welch(seg(x, '2014-01-01', '2019-04-30'), seg(x, *W['R_pre_daytrade_2005-01_2013-12'])),
    }
res['REGIME'] = reg
# ---- 空頭年 ----
bear = {}
for y in (2008, 2011, 2015, 2018, 2020, 2022):
    a, b = f'{y}-01-01', f'{y}-12-31'
    bear[y] = {'0050_pct': res['0050_yearly'].get(y), 'EW_pct': res['EW_yearly'].get(y)}
    for name in S:
        for n in (30, 10):
            bear[y][f'{name}_n{n}'] = res[name][f'n{n}_0.001']['yearly_pct'].get(y)
    st = {name: M.stats(navs[f'{name}_n30_0.001'], a, b, D, E50, EW) for name in S}
    bear[y]['maxDD_n30'] = {name: st[name]['maxDD'] for name in S}; bear[y]['maxDD_EW'] = M.stats(EW, a, b, D, E50)['maxDD']
    bear[y]['maxDD_0050'] = st['P1_52wHigh']['0050_maxDD']
res['BEAR_YEARS'] = bear
# 訊號間月超額相關（n30 滑價 0.1%，全期）
ex = pd.DataFrame({k: mex(k) for k in S}).dropna()
res['excess_corr_full'] = ex.corr().round(2).to_dict()
(HERE / 'results_long.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'long_nav.parquet')
# 存訊號（供 run_combo 使用）
pd.to_pickle({'S': S, 'el': el}, HERE / '_signals_cache.pkl')
print('done', round(time.time() - t_start))

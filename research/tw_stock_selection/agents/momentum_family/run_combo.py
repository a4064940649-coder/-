#!/usr/bin/env python3
"""2021-03～2026-10：價格腿 × 月營收 3 個月年增率（50/50 百分位平均，與 bt_final_rule 相同），M2 成交，和原主策略比較（配對 t）。
股池與日曆完全沿用 exec_rebuild/exlib（t300_d & yoy3 有值 & 非處置 & 價格腿有值）。輸出 results_combo.json。"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlib as M
L = M.L
D = list(L.DATES)
c = pd.read_pickle(HERE / '_signals_cache.pkl'); S = {k: v.reindex(L.DATES) for k, v in c['S'].items()}
sel = json.loads((HERE / 'results_long.json').read_text(encoding='utf-8'))['SELECTION']['chosen']
S['P1_52wHigh'] = L.s52_d                     # 與 exlib 完全相同的 52 週高點比
base = L.t300_d & L.yoy3.notna() & ~L.inD_d
S['P5_indNeutral12_1'] = M.ind_neutral(S['P2_mom12_1'], base & S['P2_mom12_1'].notna())
Y = L.yoy3
W = {'full_2021-03_2026-10': ('2021-03-01', '2026-10-31'), 'select_2021-03_2023-12': ('2021-03-01', '2023-12-31'),
     'valid_2024-01_2026-10': ('2024-01-01', '2026-10-31'), 'valid_ex2026_2024-01_2025-12': ('2024-01-01', '2025-12-31')}
def combo(p, wp=0.5):
    el = base & p.notna()
    return wp * p.where(el).rank(axis=1, pct=True) + (1 - wp) * Y.where(el).rank(axis=1, pct=True), el
specs = {'MAIN_P1xREV': combo(S['P1_52wHigh'])}
specs['ONLY_P1_52wHigh'] = (S['P1_52wHigh'].where(base & S['P1_52wHigh'].notna()), base & S['P1_52wHigh'].notna())
specs['ONLY_REV_yoy3'] = (Y.where(base & S['P1_52wHigh'].notna()), base & S['P1_52wHigh'].notna())
for k in ['P2_mom12_1', 'P3_mom6_1', 'P4_resid12_1_taiex', 'P4e_resid12_1_EW', 'P5_indNeutral12_1', 'P6_volAdj12_1']:
    specs[f'COMBO_{k}xREV'] = combo(S[k])
    specs[f'ONLY_{k}'] = (S[k].where(base & S[k].notna()), base & S[k].notna())
ev, pools = M.select(*specs['MAIN_P1xREV'], D, 30)
EW, _ = M.sim(pools, 'gross', D)
E50 = L.E50_NAV
res = {'_meta': {'selected_price_leg_from_long_history': sel,
                 'note': 'EW = 主策略股池（t300＆yoy3 有值＆非處置＆d+1 可交易）等權 gross。paired_vs_MAIN = 月窗（d 收盤→下一 d 收盤）報酬差的 t；daily 為日報酬差 t。'}}
navs = {}
for name, (sc, el) in specs.items():
    o = {}
    for n in (30, 10):
        e, _ = M.select(sc, el, D, n)
        for cost in (0.001, 'stress'):
            nav, tov = M.sim(e, cost, D); navs[(name, n, cost)] = nav
            o[f'n{n}_{cost}'] = {'turnover_value_per_rebalance': round(tov, 3),
                                 'windows': {k: M.stats(nav, a, b, D, E50, EW) for k, (a, b) in W.items()}, 'yearly_pct': M.yearly(nav)}
    res[name] = o
for name in specs:
    for n in (30, 10):
        for cost in (0.001, 'stress'):
            x = M.month_rets(navs[(name, n, cost)], D) - M.month_rets(navs[('MAIN_P1xREV', n, cost)], D)
            xd = navs[(name, n, cost)].pct_change() - navs[('MAIN_P1xREV', n, cost)].pct_change()
            res[name][f'n{n}_{cost}']['paired_vs_MAIN'] = {k: {'diff_pct_m': round(float(x[(x.index >= a) & (x.index <= b)].mean() * 100), 2),
                                                               't_m': L.tstat(x[(x.index >= a) & (x.index <= b)]),
                                                               't_daily': L.tstat(xd[(xd.index >= a) & (xd.index <= b)])} for k, (a, b) in W.items()}
# 價格腿 vs 營收腿 的貢獻：主策略、只用 P1、只用營收 的月超額（對 EW）相關
ex = pd.DataFrame({k: M.month_rets(navs[(k, 30, 0.001)], D) - M.month_rets(EW, D) for k in ['MAIN_P1xREV', 'ONLY_P1_52wHigh', 'ONLY_REV_yoy3']}).dropna()
ex = ex[ex.index >= '2021-03-01']
res['DECOMP_n30_slip0.001'] = {'corr_excess_vs_EW': ex.corr().round(2).to_dict(),
                               'mean_excess_vs_EW_pct_m': (ex.mean() * 100).round(2).to_dict(),
                               'reg_MAIN_on_P1only_and_REVonly': dict(zip(['alpha_pct_m', 'b_P1only', 'b_REVonly'],
                                   [round(float(v), 3) for v in np.linalg.lstsq(np.c_[np.ones(len(ex)), ex[['ONLY_P1_52wHigh', 'ONLY_REV_yoy3']].to_numpy()], ex['MAIN_P1xREV'].to_numpy(), rcond=None)[0] * np.array([100, 1, 1])]))}
# 組合層級 50/50（各自 30 檔、每月再平衡）vs 訊號層級 50/50（主策略）：交集效果
mb = 0.5 * M.month_rets(navs[('ONLY_P1_52wHigh', 30, 0.001)], D) + 0.5 * M.month_rets(navs[('ONLY_REV_yoy3', 30, 0.001)], D)
mm_ = M.month_rets(navs[('MAIN_P1xREV', 30, 0.001)], D); ew_ = M.month_rets(EW, D)
k_ = mb.index >= '2021-03-01'
res['DECOMP_n30_slip0.001']['portfolio_blend_5050_P1only_REVonly'] = {'mean_excess_vs_EW_pct_m': round(float((mb - ew_)[k_].mean() * 100), 2),
    't_vs_EW_m': L.tstat((mb - ew_)[k_]), 'MAIN_minus_blend_pct_m': round(float((mm_ - mb)[k_].mean() * 100), 2), 't_MAIN_minus_blend_m': L.tstat((mm_ - mb)[k_])}
# 主策略選股與 P1-only／REV-only 的重疊
ovl = {}
for k2 in ('ONLY_P1_52wHigh', 'ONLY_REV_yoy3'):
    e1, _ = M.select(*specs['MAIN_P1xREV'], D, 30); e2, _ = M.select(*specs[k2], D, 30)
    ovl[k2] = round(float(np.mean([len(set(e1[t][1]) & set(e2[t][1])) / 30 for t in e1 if t in e2])), 3)
res['DECOMP_n30_slip0.001']['avg_overlap_of_MAIN_picks_with'] = ovl
# 選中的價格腿合成
chosen = 'MAIN_P1xREV' if sel == 'P1_52wHigh' else f'COMBO_{sel}xREV'
res['PRE_REGISTERED_TEST'] = {'chosen_price_leg': sel, 'combo_key': chosen,
                              'result_n30_slip0.001': res[chosen]['n30_0.001']['paired_vs_MAIN'],
                              'note': '選中 P1 → 合成即為原主策略，配對差恆為 0；其他價格腿的合成只作參考（非事前選中）。' if chosen == 'MAIN_P1xREV' else ''}
(HERE / 'results_combo.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
for name in specs:
    w = res[name]['n30_0.001']['windows']['full_2021-03_2026-10']; p = res[name]['n30_0.001']['paired_vs_MAIN']['full_2021-03_2026-10']
    v = res[name]['n30_0.001']['windows']['valid_ex2026_2024-01_2025-12']
    print(f"{name:32s} CAGR {w['CAGR']:.3f} DD {w['maxDD']:.3f} t50 {w['t_vs_0050_m']} tEW {w['t_vs_EW_m']} | 24-25 t50 {v['t_vs_0050_m']} | paired {p}")
print(res['DECOMP_n30_slip0.001'])

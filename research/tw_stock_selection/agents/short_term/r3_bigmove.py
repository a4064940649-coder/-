#!/usr/bin/env python3
"""規則 3：前一日大跌／大漲，隔日開盤買、收盤賣（現股當沖 0.23%；收盤跌停→隔日開盤賣 0.5%）。

事前固定：
  3a 前一日（t−1）報酬 ≤ −7%（flag 0；用原始收盤／前收，等同含息報酬）
  3b 前一日報酬 ≥ +7% 且 t−1 收盤未鎖漲停（close 不在漲停價，或 close < high）
  t 日開盤漲停者買不到（剔除）。流動性用 t−1 以前資料。
  注意：2015-06 前漲跌幅只有 7%，3b 在 S0 幾乎沒有事件、3a 等同「跌停或接近跌停」。
  事前附帶（S0 對照，全部報告）：3c 前一日 ≤ −5%、3d 前一日 ≥ +5% 且未鎖漲停。
  另報告：同條件但持有到 t 日收盤後再到 t+1 收盤（非當沖 0.5%）——只作描述，不另做挑選。
敏感度（同規則 2）：收盤跌停假設可在收盤賣出（naive）。
"""
import sys, time
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/short_term')
import numpy as np, pandas as pd
import stlib as L

t0 = time.time()
D = L.load()
idr, elig, F, R = D['id'], D['elig'], D['flag'], D['ret']
cc_prev = D['cc_raw'].shift(1).where(F.shift(1) == 0)
lock_prev = D['lock_up_close'].shift(1).fillna(False).astype(bool)
olu = D['open_limit_up'].fillna(False).astype(bool)
gdt, cdt = L.daytrade_exit(D)
res = {'description': __doc__}
rules = {
    'prev_down_7': cc_prev <= -0.07,
    'prev_up_7_not_locked': (cc_prev >= 0.07) & ~lock_prev,
    'prev_down_5': cc_prev <= -0.05,
    'prev_up_5_not_locked': (cc_prev >= 0.05) & ~lock_prev,
}
summary = {}
r_next = R.shift(-1).where(F.shift(-1).isin([0, 1, 2]) & (R.shift(-1).abs() <= D['lim'].shift(-1).values[:, None] + 0.03))
g2 = (1 + idr) * (1 + r_next) - 1
for name, cond in rules.items():
    mask = elig & cond & ~olu & D['pc'].notna()
    r, g, c, n = L.run_rule(D, gdt, cdt, mask, stress=('cost_0.38pct_pre2017tax', L.COST_DT_PRE2017))
    res[f'{name}__daytrade'] = r; summary[f'{name}__daytrade'] = {w: L.brief(r[w]) for w in L.WINDOWS}
    r_nv, _, _, _ = L.run_rule(D, idr, L.COST_DT, mask)
    res[f'{name}__daytrade_naive_close_exit'] = r_nv; summary[f'{name}__daytrade_naive_close_exit'] = {w: L.brief(r_nv[w]) for w in L.WINDOWS}
    gross, cc, nn = L.cohort(g2, L.COST_RT, mask)
    gd1, _, _ = L.cohort(idr, 0.0, mask & g2.notna()); gd2, _, _ = L.cohort(r_next, 0.0, mask & g2.notna())
    CAL = D['CAL']
    daily = 0.5 * (gd1.reindex(CAL) - L.COST_RT).fillna(0) + 0.5 * gd2.reindex(CAL).shift(1).fillna(0)
    rr = L.evaluate_all(D, gross, L.COST_RT, nn, daily, need_next=True, stress=('cost_1pct', L.COST_RT_STRESS))
    res[f'{name}__hold_to_next_close_descriptive'] = rr; summary[f'{name}__hold_to_next_close_descriptive'] = {w: L.brief(rr[w]) for w in L.WINDOWS}

res['runtime_sec'] = round(time.time() - t0, 1)
L.save('results_r3.json', res)
print(f"done {res['runtime_sec']}s")
for k, v in summary.items():
    print('==', k); print(pd.DataFrame(v).T.to_string())

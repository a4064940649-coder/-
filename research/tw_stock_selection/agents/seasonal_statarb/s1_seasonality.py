#!/usr/bin/env python3
"""檢定 1：個股同月季節性（Heston & Sadka 2008 類型）。

事前固定規則：
  每月最後交易日 t 換股、持有到下一月底（目標月份 m = t 的下一個月）。
  訊號：
    S1a_K5   ：過去 5 年「同一月份 m」的平均月報酬（至少 4 筆）
    S1b_K10  ：過去 10 年同月平均（至少 8 筆）
    S1c_K10rel：過去 10 年同月平均 − 過去 120 個月全部月份平均（至少 8 筆同月、80 筆全部），去掉長期報酬/動能成分
  可投資：t−1 日 60 日均值 ≥ 1,000 萬、股價 ≥ 10、流動性前 300 名；t 收盤鎖漲停者不買。
  組合：訊號最高 30 檔等權（t 收盤成交），成本 0.5%（壓力 1%）× 周轉；另做前 300 名內五分位 Q5−Q1。
  2024-01 資料缺口：2024 年 1、2 月的月報酬設為缺值，不進訊號。
  挑選期 2010-01～2019-04（選 IS「對同池等權 t」最高的訊號）、驗證期 2019-05～2026-10；三個版本全部報告。
"""
import json
import numpy as np, pandas as pd
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *

adj, C, H, V = load('adj', 'close_raw', 'high', 'value')
me = month_ends(adj.index)
R = period_returns(adj, me)                       # index = t, t→t_next
B = bench_returns(e50_series(), me)
# 日曆月報酬表（以結束月份標記）
nxt = dict(zip(me[:-1], me[1:]))
MR = R.copy(); MR.index = pd.PeriodIndex([nxt[t] for t in R.index], freq='M')
MR.loc[[p for p in MR.index if p in (pd.Period('2024-02', 'M'),)]] = np.nan   # 跨 1+2 月，剔除
full = pd.period_range(MR.index.min(), MR.index.max(), freq='M')
MR = MR.reindex(full)

def same_month(K, need):
    lags = [MR.shift(12 * k) for k in range(1, K + 1)]   # 對「目標月」p 而言的 p-12k
    st = np.stack([l.values for l in lags]); cnt = np.isfinite(st).sum(0)
    m = np.nanmean(np.where(np.isfinite(st), st, np.nan), axis=0) if True else None
    out = pd.DataFrame(np.where(cnt >= need, m, np.nan), index=MR.index, columns=MR.columns)
    return out

import warnings; warnings.filterwarnings('ignore')
sig_raw = {'S1a_K5': same_month(5, 4), 'S1b_K10': same_month(10, 8)}
allm = MR.shift(1).rolling(120, min_periods=80).mean()   # 目標月前 120 個月（不含目標月本身）
sig_raw['S1c_K10rel'] = same_month(10, 8) - allm

# 目標月 p → 換股日 t（p 的前一個月月底）
tmap = {pd.Period(nxt[t], 'M'): t for t in R.index}
def to_t(S):
    S = S.loc[[p for p in S.index if p in tmap]]; S.index = pd.DatetimeIndex([tmap[p] for p in S.index]); return S.reindex(R.index)

lock = locked_limit_up(C, H).reindex(me)
el, v60m = eligibility(me, C, V, lag=1)
el &= ~lock.fillna(False)
U300 = top_liq(el, v60m, 300)

res = {}
periods = {'IS_2010-01_2019-04': ('2010-01-01', '2019-04-30'), 'OOS_2019-05_2026-10': (OOS_START, OOS_END)}
for name, S in sig_raw.items():
    sig = to_t(S)
    rr = run_sort(sig, R, U300, nq=5, top_n=30, rt_cost=0.005, min_names=100)
    for pn, (a, b) in periods.items():
        o = {'turnover': round(rr['turnover'], 3), 'n_names': rr['n_names']}
        o['net_0.5pct'] = eval_monthly(rr['best_net'], me, B, rr['universe'], a, b)
        o['net_1pct'] = eval_monthly(rr['best_net_1pct'], me, B, rr['universe'], a, b)
        o['gross'] = ann(cut(rr['best_gross'], a, b), me)
        sp = cut((rr['Q5'] - rr['Q1']).dropna(), a, b)
        o['Q5_Q1_pct_m'] = round(float(sp.mean() * 100), 3); o['t_Q5_Q1'] = tstat(sp)
        o['quintile_CAGR'] = {f'Q{k}': ann(cut(rr[f'Q{k}'], a, b), me).get('CAGR') for k in range(1, 6)}
        res[f'{name}|{pn}'] = o
        print(f"{name:12s} {pn:20s} to={o['turnover']:.2f} gross {o['gross'].get('CAGR')} | net.5 {fmt(o['net_0.5pct'])} | net1 CAGR {o['net_1pct']['strat'].get('CAGR')} | Q5-Q1 {o['Q5_Q1_pct_m']}%/m t={o['t_Q5_Q1']}")

is_key = max(sig_raw, key=lambda n: res[f'{n}|IS_2010-01_2019-04']['net_0.5pct']['t_vs_EW'])
res['IS_selected'] = is_key
print('IS 選中：', is_key)
print('OOS 逐年：', res[f'{is_key}|OOS_2019-05_2026-10']['net_0.5pct']['yearly'])
(OUTDIR / 'results_s1.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')

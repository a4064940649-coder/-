#!/usr/bin/env python3
"""步驟 3：各期年化／t／TE／IR／回撤，配對差，0050 混合，逐年。"""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_capacity_tsmc_gap/replicate_o1')
import rcore as R
from rstat import *
NAV = pd.read_parquet(R.HERE / 'navs_tsmc.parquet')
out = {}
for col in ['n10|w0|normal', 'n10|w10|normal', 'n10|w20|normal', 'n10|w30|normal', 'n10|w0|stress', 'n10|w30|stress', 'n10|w10|stress', 'n10|w20|stress']:
    out[col] = {p: seg_stats(NAV[col], p) for p in PER}
out['0050'] = {p: round(R.cagr(B, PER[p][0], PER[p][1]) * 100, 2) for p in PER}
out['2330'] = {p: round(R.cagr(T, PER[p][0], PER[p][1]) * 100, 2) for p in PER}
# t vs EW
pe = R.win_returns(NAV['EW'], S); p0 = R.win_returns(NAV['n10|w0|normal'], S)
out['w0_t_vs_EW_ALL'] = round(R.tstat((p0 - pe)), 3)
# 配對差 w30 - w0
p30 = R.win_returns(NAV['n10|w30|normal'], S)
for p in ('ALL', 'VAL', 'SEL', 'HOLD'):
    ma, mb = PER[p][2], PER[p][3]; m = (p0.index >= ma) & (p0.index <= mb)
    d = (p30 - p0)[m]
    out[f'pair_w30_w0_{p}'] = {'pp': round(d.mean() * 100, 3), 't': round(R.tstat(d), 3)}
# 對照：0050 30% + 主策略 70%
# (a) 每個換股日開盤重設、0050 無成本；以日頻構造
nav0 = NAV['n10|w0|normal'].dropna()
ex_days = sorted(set(R.IDX.get_loc(d) + 1 for d in S if d < R.EVAL_END))
r_s = nav0.pct_change(); r_b = B.reindex(nav0.index).pct_change()
# 日頻混合：換股日（開盤成交）重設權重 —— 近似：在換股日的前一收盤重設（開盤與收盤差小）
mix = {}
for variant in ('sigclose', 'execday'):
    val = [1.0]; wS, wB = 0.7, 0.3
    idx = nav0.index
    resets = set(S) if variant == 'sigclose' else set(R.IDX[i] for i in ex_days)
    for k in range(1, len(idx)):
        d = idx[k]
        wS *= (1 + r_s.iloc[k]); wB *= (1 + r_b.iloc[k])
        v = wS + wB; val.append(v)
        if d in resets: wS, wB = 0.7 * v, 0.3 * v
    mix[variant] = pd.Series(val, index=idx)
for k, mv in mix.items():
    out[f'mix_{k}'] = {p: seg_stats(mv, p) for p in ('SEL', 'VAL', 'HOLD', 'ALL')}
# 逐年贏 0050
yr = {}
for col in ['n10|w0|normal', 'n10|w30|normal']:
    s = NAV[col].dropna(); r = s.pct_change().dropna(); rb = B.reindex(s.index).pct_change().dropna()
    ys = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1); yb = rb.groupby(rb.index.year).apply(lambda g: (1 + g).prod() - 1)
    yr[col] = int((ys > yb).sum()); yr['nyears'] = int(len(ys))
for k, mv in mix.items():
    r = mv.pct_change().dropna(); rb = B.reindex(mv.index).pct_change().dropna()
    ys = r.groupby(r.index.year).apply(lambda g: (1 + g).prod() - 1); yb = rb.groupby(rb.index.year).apply(lambda g: (1 + g).prod() - 1)
    yr[f'mix_{k}'] = int((ys > yb).sum())
out['years_beat_0050'] = yr
json.dump(out, open(R.HERE / 'res_stats.json', 'w'), ensure_ascii=False, indent=1)
for k, v in out.items(): print(k, v)

#!/usr/bin/env python3
"""三大法人籌碼：投信／外資 20 日買超占成交量比，上市股 2021-11～2026-10。

法人買賣超在收盤後公布 → 訊號只用到 t−1 日，t 日收盤成交（嚴格不偷看）。
單位：inst_5y 為「張」（千股），× 1000 換成股數；成交量 = 成交值 / 收盤（股）。
事前固定的檢定（全部報告）：
  F1 月底換股，投信 20 日買超／20 日成交量 五分位
  F2 月底換股，外資 20 日買超／20 日成交量 五分位
  F3 月底換股，流動性前 300 名中投信比最高 30 檔
  F4 月營收換股日（≥ 11 日），R5（52 週高點 + 營收 yoy3）再加投信比，三者百分位平均最高 30 檔；對照 R5
  F5 避開：R5 入選股中剔除「外資 20 日賣超前 1/5」者，檢查是否改善
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import month_ends, period_returns, bench_returns, run_sort, summarize, ann, tstat, yearly

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
x = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/inst_5y.csv.gz', dtype={'code': str}, parse_dates=['date'])
x = x[x['code'].isin(adj.columns)]
idx = adj.index[(adj.index >= x['date'].min()) & (adj.index <= x['date'].max())]
TR = x.pivot_table(index='date', columns='code', values='trust', aggfunc='sum').reindex(idx).reindex(columns=adj.columns) * 1000
FO = x.pivot_table(index='date', columns='code', values='foreign', aggfunc='sum').reindex(idx).reindex(columns=adj.columns) * 1000
VOLSH = (V / C).reindex(idx)
has = TR.notna() | FO.notna()
TR = TR.where(VOLSH.notna()).fillna(0).where(VOLSH.notna()); FO = FO.where(VOLSH.notna()).fillna(0).where(VOLSH.notna())
vol20 = VOLSH.rolling(20, min_periods=15).sum()
TRR = (TR.rolling(20, min_periods=15).sum() / vol20).shift(1)      # t−1 為止
FOR = (FO.rolling(20, min_periods=15).sum() / vol20).shift(1)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
V60 = V.rolling(60).mean(); S52 = adj / adj.rolling(252).max()
res = {}
# 月底換股
me = [d for d in month_ends(adj.index) if d >= pd.Timestamp('2021-11-01')]
R = period_returns(adj, me); B = bench_returns(e50, me)
v60 = V60.reindex(me); px = C.reindex(me); base = (v60 >= 1e7) & (px >= 10)
top300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
trr, forr = TRR.reindex(me), FOR.reindex(me)
res['F1_trust_quintiles'] = summarize(run_sort(trr, R, base & trr.notna()), me, B, 'F1 trust 20d net buy / volume, quintiles')
res['F2_foreign_quintiles'] = summarize(run_sort(forr, R, base & forr.notna()), me, B, 'F2 foreign 20d net buy / volume, quintiles')
res['F3_top30_trust_of_top300'] = summarize(run_sort(trr, R, top300 & trr.notna(), top_n=30), me, B, 'F3 top30 trust ratio in top300')
# 月營收換股日
rev = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
rev = rev[rev['code'].isin(adj.columns)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M')); YOY3 = YOY.rolling(3, min_periods=3).mean()
cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
ii = adj.index[adj.index >= pd.Timestamp('2021-11-01')]; s = pd.Series(ii, index=ii)
ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
keep = [d for d in ds if (pd.Period(d, 'M') - 1) <= last_full]; dates = keep + [d for d in ds if d > keep[-1]][:1]
R2 = period_returns(adj, dates); B2 = bench_returns(e50, dates)
yoy3 = pd.DataFrame(YOY3.reindex([pd.Period(d, 'M') - 1 for d in dates]).to_numpy(), index=pd.DatetimeIndex(dates), columns=YOY3.columns).reindex(columns=adj.columns)
v60b = V60.reindex(dates); pxb = C.reindex(dates); baseb = (v60b >= 1e7) & (pxb >= 10)
t300 = baseb & (v60b.where(baseb).rank(axis=1, ascending=False) <= 300)
s52 = S52.reindex(dates); trb = TRR.reindex(dates); fob = FOR.reindex(dates)
el = t300 & yoy3.notna() & s52.notna() & trb.notna()
pr = lambda M: M.where(el).rank(axis=1, pct=True)
r5 = (pr(s52) + pr(yoy3)) / 2
f4 = (pr(s52) + pr(yoy3) + pr(trb)) / 3
res['R5_same_window'] = summarize(run_sort(r5, R2, el, top_n=30), dates, B2, 'R5 (52w + yoy3) top30, 2021-11+ window')
res['F4_R5_plus_trust'] = summarize(run_sort(f4, R2, el, top_n=30), dates, B2, 'F4 52w + yoy3 + trust ratio top30')
fo_bad = fob.where(el).rank(axis=1, pct=True) <= 0.2
res['F5_R5_excl_foreign_bottom20'] = summarize(run_sort(r5.where(~fo_bad), R2, el & ~fo_bad, top_n=30), dates, B2, 'F5 R5 excluding foreign-selling bottom 20%')
(OUT / 'bt_flows.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    g, n, u, b = v['best_gross'], v['best_net'], v['universe'], v['bench']
    print(f"{k:32s} n={v['n_names']:5.1f} to={v['turnover']:.2f} | gross {g.get('CAGR')} net {n.get('CAGR')} MDD {n.get('maxDD')} | EW {u.get('CAGR')} 0050 {b.get('CAGR')} | Q5-Q1 {v['Q5_minus_Q1_pct_per_period']}%/m t={v['t_Q5_Q1']} | net-EW {v['net_minus_universe_pct_per_period']} t={v['t_net_vs_universe']} | net-0050 {v['net_minus_bench_pct_per_period']} t={v['t_net_vs_bench']} | yrs {v['years_beat_bench']}")

#!/usr/bin/env python3
"""補測其他規則化方法（上市股；樣本內 2005-01～2019-04，樣本外 2019-05～2026-10，月底換股，扣 0.5% 來回成本）：
  M1 價格動能 12-1（過去 12 個月扣除最近 1 個月報酬）
  M2 低波動（60 日日報酬標準差越低越好）
  M3 避開樂透股（過去 21 日最大單日漲幅越低越好，MAX 效應）
  M4 短期反轉（過去 1 個月報酬越低越好）
擇時開關（事前固定）：換股日加權指數收盤 < 200 日均線 → 全部改持現金（報酬 0）。
  T1 套在 52 週高點 30 檔（2005～2026，含 2008、2011、2015、2018、2022 空頭）
  T2 套在最終規則（52 週高點 × 月營收，2021-03～2026-10）
輸出 output/bt_more_methods.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import month_ends, period_returns, bench_returns, run_sort, summarize, ann, yearly, tstat

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet'); R1 = pd.read_parquet(P / 'ret.parquet')
me = month_ends(adj.index)
R = period_returns(adj, me)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']; B = bench_returns(e50, me)
tx = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close']
V60 = V.rolling(60).mean()
A = adj.ffill(limit=5)
sig = {
    'M1_momentum_12_1': (A.shift(21) / A.shift(252) - 1).reindex(me),
    'M2_low_vol_60d': (-R1.rolling(60, min_periods=50).std()).reindex(me),
    'M3_low_MAX_21d': (-R1.rolling(21, min_periods=15).max()).reindex(me),
    'M4_reversal_1m': (-(A / A.shift(21) - 1)).reindex(me),
}
base = (V60.reindex(me) >= 1e7) & (C.reindex(me) >= 10)
top300 = base & (V60.reindex(me).where(base).rank(axis=1, ascending=False) <= 300)
periods = {'IS': ('2005-01-01', '2019-04-30'), 'OOS': ('2019-05-01', '2026-10-31')}
res = {}
for k, s in sig.items():
    q = run_sort(s, R, base & s.notna())
    t = run_sort(s, R, top300 & s.notna(), top_n=30)
    for pn, (a, b) in periods.items():
        res[f'{k}|quintile|{pn}'] = summarize(q, me, B, k, a, b)
        res[f'{k}|top30|{pn}'] = summarize(t, me, B, k, a, b)
# 擇時開關
ma200 = tx.rolling(200).mean()
def timing_flag(dates):
    x = (tx.reindex(pd.DatetimeIndex(dates), method='ffill') >= ma200.reindex(pd.DatetimeIndex(dates), method='ffill'))
    return x
s52 = (adj / adj.rolling(252).max()).reindex(me)
t52 = run_sort(s52, R, top300 & s52.notna(), top_n=30)
on = timing_flag(me).reindex(t52['best_net'].index).fillna(True)
net = t52['best_net']; netT = net.where(on, 0.0) - (on != on.shift(1)).astype(float).fillna(0) * 0.005 * 0.5
bb = B.reindex(net.index)
def prof(x, start=None):
    x = x[x.index >= pd.Timestamp(start)] if start else x
    a = ann(x, me); return {'CAGR': a['CAGR'], 'maxDD': a['maxDD'], 'vol': a['vol'], 'yearly': yearly(x, me)}
res['T1_52w_top30_no_timing'] = prof(net); res['T1_52w_top30_with_timing'] = prof(netT); res['T1_0050'] = prof(bb)
res['T1_share_months_in_cash'] = round(float((~on).mean()), 3)
import bt_final_rule as fr
n5, u5, tov5, _ = fr.run(fr.yoy3)
on5 = timing_flag(fr.dates).reindex(n5.index).fillna(True)
n5T = n5.where(on5, 0.0) - (on5 != on5.shift(1)).astype(float).fillna(0) * 0.0025
b5 = fr.B.reindex(n5.index)
a = lambda x: ann(x, fr.dates)
res['T2_final_no_timing'] = {'CAGR': a(n5)['CAGR'], 'maxDD': a(n5)['maxDD'], 'yearly': yearly(n5, fr.dates)}
res['T2_final_with_timing'] = {'CAGR': a(n5T)['CAGR'], 'maxDD': a(n5T)['maxDD'], 'yearly': yearly(n5T, fr.dates)}
res['T2_0050'] = {'CAGR': a(b5)['CAGR'], 'maxDD': a(b5)['maxDD']}
res['T2_share_periods_in_cash'] = round(float((~on5).mean()), 3)
(OUT / 'bt_more_methods.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if '|' in k:
        n = v['best_net']; u = v['universe']; b = v['bench']
        print(f"{k:34s} net {n.get('CAGR')} MDD {n.get('maxDD')} | EW {u.get('CAGR')} 0050 {b.get('CAGR')} | Q5-Q1 {v['Q5_minus_Q1_pct_per_period']}%/m t={v['t_Q5_Q1']} | vsEW t={v['t_net_vs_universe']} | vs0050 t={v['t_net_vs_bench']}")
    else:
        print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items()})

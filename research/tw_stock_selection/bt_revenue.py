#!/usr/bin/env python3
"""月營收動能（台灣獨有的月營收揭露）：上市股 2021-01～2026-10。

事前固定的檢定（全部報告，不挑選）：
  換股日 = 每月第一個 ≥ 11 日的交易日（法定 10 日前公告；穩健性：≥ 15 日）；用「上個月」營收；當日收盤進場，持有到下一換股日。
  可投資：60 日平均成交值 ≥ NT$1,000 萬、收盤 ≥ 10 元、有營收資料。
  R1 五分位：單月營收年增率 yoy
  R2 五分位：近 3 個月年增率平均 yoy3
  R3 實務版：流動性前 300 名中 yoy3 最高 30 檔
  R4 事件組合：流動性前 300 名中「單月營收創近 12 個月新高 且 yoy > 0」全部等權
  R5 組合：流動性前 300 名中，52 週高點與 yoy3 的百分位平均最高 30 檔
  R6 實務版但只在流動性前 100 名（大型股）中選 30 檔
基準：0050 含息、可投資全體等權、流動性前 300 名等權。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import period_returns, bench_returns, run_sort, summarize, ann, yearly, tstat

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
rev = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz', dtype={'code': str})
rev = rev[rev['code'].isin(adj.columns)]
rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
REV = rev.pivot_table(index='p', columns='code', values='revenue', aggfunc='last').sort_index()
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
full = pd.period_range(REV.index.min(), REV.index.max(), freq='M'); REV = REV.reindex(full); YOY = YOY.reindex(full)
YOY3 = YOY.rolling(3, min_periods=3).mean()
HIGH12 = REV > REV.shift(1).rolling(12, min_periods=12).max()
cnt = REV.notna().sum(axis=1)
last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()      # 最後一個資料完整的營收月份（2026-09 只有部分公司）
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
S52 = adj / adj.rolling(252).max(); V60 = V.rolling(60).mean()

def rebalance_dates(day):
    idx = adj.index[adj.index >= pd.Timestamp('2021-01-01')]
    s = pd.Series(idx, index=idx)
    ds = [g[g.dt.day >= day].iloc[0] for _, g in s.groupby([idx.year, idx.month]) if (g.dt.day >= day).any()]
    # 只保留「上個月營收」在資料內且完整的換股日，最後再加一個結束日
    keep = [d for d in ds if (pd.Period(d, 'M') - 1) <= last_full and (pd.Period(d, 'M') - 1) >= YOY.index.min()]
    nxt = [d for d in ds if d > keep[-1]]
    return keep + nxt[:1]

def build(day):
    dates = rebalance_dates(day)
    R = period_returns(adj, dates)
    m = [pd.Period(d, 'M') - 1 for d in dates]
    def sig(M): return pd.DataFrame(M.reindex(m).to_numpy(), index=pd.DatetimeIndex(dates), columns=M.columns).reindex(columns=adj.columns)
    yoy, yoy3, hi = sig(YOY), sig(YOY3), sig(HIGH12.astype(float))
    v60 = V60.reindex(dates); px = C.reindex(dates); s52 = S52.reindex(dates)
    base = (v60 >= 1e7) & (px >= 10)
    rk = v60.where(base).rank(axis=1, ascending=False)
    top300 = base & (rk <= 300); top100 = base & (rk <= 100)
    B = bench_returns(e50, dates)
    res = {}
    res['R1_yoy_quintiles'] = summarize(run_sort(yoy, R, base & yoy.notna()), dates, B, 'R1 yoy quintiles (all eligible)')
    res['R2_yoy3_quintiles'] = summarize(run_sort(yoy3, R, base & yoy3.notna()), dates, B, 'R2 yoy3 quintiles (all eligible)')
    res['R3_top30_yoy3_of_top300'] = summarize(run_sort(yoy3, R, top300 & yoy3.notna(), top_n=30), dates, B, 'R3 top30 by yoy3 in top300 liquidity')
    ev = (hi == 1) & (yoy > 0)
    # R4：符合事件者全部等權（用 run_sort 的 top_n=None 取前 1/5 不適用，直接計算）
    port, rest, uni, to, prev = {}, {}, {}, [], None
    for t in R.index:
        ok = top300.loc[t] & R.loc[t].notna() & yoy.loc[t].notna() & hi.loc[t].notna()
        if ok.sum() < 50: continue
        pick = ok & ev.loc[t]
        if pick.sum() < 5: continue
        port[t] = R.loc[t][pick].mean(); rest[t] = R.loc[t][ok & ~ev.loc[t]].mean(); uni[t] = R.loc[t][ok].mean()
        p = set(pick[pick].index)
        if prev is not None: to.append(len(p - prev) / len(p))
        prev = p
    port, rest, uni = pd.Series(port), pd.Series(rest), pd.Series(uni); tov = float(np.mean(to))
    net = port - tov * 0.005; b = B.reindex(net.index)
    res['R4_high12_and_yoy_pos_in_top300'] = {'label': 'R4 12-month revenue high & yoy>0, EW all qualifying in top300', 'avg_names': None,
        'turnover': round(tov, 3), 'gross': ann(port, dates), 'net_0.5pct': ann(net, dates), 'rest_of_top300': ann(rest, dates), 'top300_EW': ann(uni, dates), 'bench': ann(b, dates),
        'event_minus_rest_pct_per_period': round(float((port - rest).mean() * 100), 3), 't_event_minus_rest': tstat(port - rest),
        'net_minus_top300EW_pct': round(float((net - uni).mean() * 100), 3), 't_net_vs_top300EW': tstat(net - uni),
        'net_minus_bench_pct': round(float((net - b).mean() * 100), 3), 't_net_vs_bench': tstat(net - b), 'yearly_net_pct': yearly(net, dates), 'yearly_bench_pct': yearly(b, dates)}
    n_ev = [int((top300.loc[t] & ev.loc[t] & R.loc[t].notna()).sum()) for t in port.index]; res['R4_high12_and_yoy_pos_in_top300']['avg_names'] = round(float(np.mean(n_ev)), 1)
    pr = lambda M, el: M.where(el).rank(axis=1, pct=True)
    combo = (pr(s52, top300 & yoy3.notna() & s52.notna()) + pr(yoy3, top300 & yoy3.notna() & s52.notna())) / 2
    res['R5_top30_combo_52w_yoy3_in_top300'] = summarize(run_sort(combo, R, top300 & combo.notna(), top_n=30), dates, B, 'R5 top30 by avg pct-rank(52w high, yoy3) in top300')
    res['R5b_top30_52w_only_same_dates'] = summarize(run_sort(s52, R, top300 & s52.notna(), top_n=30), dates, B, 'R5b top30 by 52w high only, same dates (control)')
    res['R6_top30_yoy3_of_top100'] = summarize(run_sort(yoy3, R, top100 & yoy3.notna(), top_n=30), dates, B, 'R6 top30 by yoy3 in top100 liquidity')
    res['_info'] = {'rebalance_day_min': day, 'first': str(dates[0].date()), 'last_entry': str(dates[-2].date()), 'end': str(dates[-1].date()), 'n_periods': len(dates) - 1, 'last_full_revenue_month': str(last_full)}
    return res

allres = {'day11': build(11), 'day15': build(15)}
(OUT / 'bt_revenue.json').write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding='utf-8')
for dk, res in allres.items():
    print('=====', dk, res['_info'])
    for k, v in res.items():
        if k.startswith('_'): continue
        if k.startswith('R4'):
            print(f"{k:40s} names={v['avg_names']} to={v['turnover']} | gross {v['gross'].get('CAGR')} net {v['net_0.5pct'].get('CAGR')} MDD {v['net_0.5pct'].get('maxDD')} | rest {v['rest_of_top300'].get('CAGR')} top300EW {v['top300_EW'].get('CAGR')} 0050 {v['bench'].get('CAGR')} | ev-rest {v['event_minus_rest_pct_per_period']}%/m t={v['t_event_minus_rest']} | net-EW t={v['t_net_vs_top300EW']} | net-0050 {v['net_minus_bench_pct']} t={v['t_net_vs_bench']}")
            continue
        g, n, u, b = v['best_gross'], v['best_net'], v['universe'], v['bench']
        print(f"{k:40s} n={v['n_names']:5.1f} to={v['turnover']:.2f} | gross {g.get('CAGR')} net {n.get('CAGR')} net1 {v['best_net_1pct'].get('CAGR')} MDD {n.get('maxDD')} | EW {u.get('CAGR')} 0050 {b.get('CAGR')} | Q5-Q1 {v['Q5_minus_Q1_pct_per_period']}%/m t={v['t_Q5_Q1']} | net-EW {v['net_minus_universe_pct_per_period']} t={v['t_net_vs_universe']} | net-0050 {v['net_minus_bench_pct_per_period']} t={v['t_net_vs_bench']} | yrs {v['years_beat_bench']}")
print('R3 yearly', allres['day11']['R3_top30_yoy3_of_top300']['yearly_net_pct'], '\n0050', allres['day11']['R3_top30_yoy3_of_top300']['yearly_bench_pct'])
print('R5 yearly', allres['day11']['R5_top30_combo_52w_yoy3_in_top300']['yearly_net_pct'])

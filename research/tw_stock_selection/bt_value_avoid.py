#!/usr/bin/env python3
"""價值因子（本益比／股價淨值比／殖利率）與「避開處置股」濾網，上市股 2021-09～2026-10。

換股日 = 月營收換股日（每月第一個 ≥ 11 日的交易日）；估值取該日之前最近一次月初快照（證交所 BWIBBU，無偷看）。
事前固定的檢定：
  V1 殖利率五分位   V2 淨值股價比（1/PB）五分位   V3 盈餘殖利率（1/PE，虧損股排除）五分位
  V4 R5 入選前先剔除 PB 最貴 1/5      V5 52 週高點 + yoy3 + 1/PB 三者百分位平均前 30 檔
  D1 R5 剔除進場日正在處置期間的股票（實務上處置股需預收款項、分盤撮合）
  D2 處置股 vs 非處置股的下一期報酬（事件檢查）
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from btlib import period_returns, bench_returns, run_sort, summarize, ann, tstat

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); V = pd.read_parquet(P / 'value.parquet')
pe = pd.read_csv(X / 'pe_5y.csv.gz', dtype={'code': str}, parse_dates=['date']); pe = pe[pe['code'].isin(adj.columns)]
rev = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str}); rev = rev[rev['code'].isin(adj.columns)]; rev['p'] = pd.PeriodIndex(rev['ym'], freq='M')
YOY = rev.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
YOY = YOY.reindex(pd.period_range(YOY.index.min(), YOY.index.max(), freq='M')); YOY3 = YOY.rolling(3, min_periods=3).mean()
cnt = YOY.notna().sum(axis=1); last_full = cnt[cnt >= 0.8 * cnt.median()].index.max()
ii = adj.index[adj.index >= pd.Timestamp('2021-09-02')]; s = pd.Series(ii, index=ii)
ds = [g[g.dt.day >= 11].iloc[0] for _, g in s.groupby([ii.year, ii.month]) if (g.dt.day >= 11).any()]
keep = [d for d in ds if (pd.Period(d, 'M') - 1) <= last_full]; dates = keep + [d for d in ds if d > keep[-1]][:1]
R = period_returns(adj, dates)
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']; B = bench_returns(e50, dates)
def snap(col):
    W = pe.pivot_table(index='date', columns='code', values=col, aggfunc='last').sort_index()
    return W.reindex(pd.DatetimeIndex(dates), method='ffill').reindex(columns=adj.columns)
PEm, PBm, DYm = snap('pe'), snap('pb'), snap('yield')
EP = (1 / PEm).where(PEm > 0); BP = (1 / PBm).where(PBm > 0); DY = DYm
yoy3 = pd.DataFrame(YOY3.reindex([pd.Period(d, 'M') - 1 for d in dates]).to_numpy(), index=pd.DatetimeIndex(dates), columns=YOY3.columns).reindex(columns=adj.columns)
v60 = V.rolling(60).mean().reindex(dates); px = C.reindex(dates); base = (v60 >= 1e7) & (px >= 10)
t300 = base & (v60.where(base).rank(axis=1, ascending=False) <= 300)
s52 = (adj / adj.rolling(252).max()).reindex(dates)
res = {}
res['V1_dividend_yield_quintiles'] = summarize(run_sort(DY, R, base & DY.notna()), dates, B, 'V1 dividend yield quintiles')
res['V2_book_to_price_quintiles'] = summarize(run_sort(BP, R, base & BP.notna()), dates, B, 'V2 1/PB quintiles')
res['V3_earnings_yield_quintiles'] = summarize(run_sort(EP, R, base & EP.notna()), dates, B, 'V3 1/PE quintiles (profitable only)')
el = t300 & yoy3.notna() & s52.notna()
pr = lambda M, E: M.where(E).rank(axis=1, pct=True)
r5 = (pr(s52, el) + pr(yoy3, el)) / 2
res['R5_same_window'] = summarize(run_sort(r5, R, el, top_n=30), dates, B, 'R5 control, same window')
expensive = pr(BP, el & BP.notna()) <= 0.2
res['V4_R5_excl_most_expensive_PB'] = summarize(run_sort(r5.where(~expensive.fillna(False)), R, el & ~expensive.fillna(False), top_n=30), dates, B, 'V4 R5 excluding most expensive PB quintile')
el5 = el & BP.notna()
v5 = (pr(s52, el5) + pr(yoy3, el5) + pr(BP, el5)) / 3
res['V5_52w_yoy3_BP'] = summarize(run_sort(v5, R, el5, top_n=30), dates, B, 'V5 52w + yoy3 + 1/PB top30')
dsp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['start', 'end'])
inD = pd.DataFrame(False, index=pd.DatetimeIndex(dates), columns=adj.columns)
for _, r in dsp.iterrows():
    if r['code'] in inD.columns and pd.notna(r['start']) and pd.notna(r['end']):
        inD.loc[(inD.index >= r['start']) & (inD.index <= r['end']), r['code']] = True
res['D1_R5_excl_disposal'] = summarize(run_sort(r5.where(~inD), R, el & ~inD, top_n=30), dates, B, 'D1 R5 excluding stocks under disposal at entry')
ev, ot = [], []
for t in R.index:
    ok = base.loc[t] & R.loc[t].notna()
    a = R.loc[t][ok & inD.loc[t]]; b = R.loc[t][ok & ~inD.loc[t]]
    if len(a): ev.append(a.mean() - b.mean())
r5_in_disposal = float(np.mean([inD.loc[t][list(r5.loc[t][el.loc[t]].sort_values(ascending=False).index[:30])].mean() for t in R.index]))
res['D2_disposal_event'] = {'periods_with_disposal_stocks': len(ev), 'disposal_minus_others_next_period_pct': round(float(np.mean(ev) * 100), 3), 't': tstat(ev),
                            'share_of_R5_picks_under_disposal_at_entry': round(r5_in_disposal, 4)}
(OUT / 'bt_value_avoid.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if k.startswith('D2'): print(k, v); continue
    g, n, u, b = v['best_gross'], v['best_net'], v['universe'], v['bench']
    print(f"{k:32s} n={v['n_names']:5.1f} to={v['turnover']:.2f} | gross {g.get('CAGR')} net {n.get('CAGR')} MDD {n.get('maxDD')} | EW {u.get('CAGR')} 0050 {b.get('CAGR')} | Q5-Q1 {v['Q5_minus_Q1_pct_per_period']}%/m t={v['t_Q5_Q1']} | net-EW {v['net_minus_universe_pct_per_period']} t={v['t_net_vs_universe']} | net-0050 {v['net_minus_bench_pct_per_period']} t={v['t_net_vs_bench']} | yrs {v['years_beat_bench']}")

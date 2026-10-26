#!/usr/bin/env python3
"""處置股策略（制度面）：處置第一天收盤買進、處置最後一天收盤賣出；同時持有所有處置中的上市股，等權。
處置公告在盤後，第一天收盤才進場（無偷看）；處置期間由公告即知（結束日事先已知）。
沒有處置股時：(a) 持有現金 (b) 持有 0050。成本：每檔進出來回 0.5%／1%／1.5%。
另做：每年報酬、子樣本（2021–2023／2024–2026）、與最終選股規則合併的配置。
輸出 output/bt_disposal_strategy.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras'); OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); R1 = pd.read_parquet(P / 'ret.parquet'); V = pd.read_parquet(P / 'value.parquet')
idx = adj.index
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].reindex(idx).ffill().pct_change()
dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end']); dp = dp[dp['code'].isin(adj.columns)]
hold = pd.DataFrame(False, index=idx, columns=adj.columns); entries = pd.DataFrame(0.0, index=idx, columns=adj.columns)
lens = []
for _, r in dp.iterrows():
    i0 = idx.searchsorted(r['start']); i1 = idx.searchsorted(r['end'], 'right') - 1
    if i0 >= len(idx) or i1 <= i0: continue
    if not np.isfinite(adj[r['code']].iat[i0]): continue
    hold.iloc[i0 + 1:i1 + 1, hold.columns.get_loc(r['code'])] = True      # 報酬落在第一天之後到最後一天
    entries.iat[i0, entries.columns.get_loc(r['code'])] += 1; lens.append(i1 - i0)
start, end = pd.Timestamp('2021-10-01'), dp['end'].max()
win = (idx >= start) & (idx <= end)
n = hold.sum(axis=1)
port_gross = R1.where(hold).mean(axis=1)
# 每筆進出成本按「進場當天的權重」攤：成本 = 來回成本 × 1/n_next（新進檔的權重）
w_new = entries.sum(axis=1).shift(1).fillna(0) / n.replace(0, np.nan)
def run(rt, idle):
    r = (port_gross - (w_new.fillna(0) * rt)).where(n > 0)
    r = r.fillna(e50 if idle == '0050' else 0.0)[win].fillna(0)
    eq = (1 + r).cumprod(); yrs = (idx[win][-1] - idx[win][0]).days / 365.25
    yearly = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}
    return r, {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eq / eq.cummax() - 1).min()), 4),
               'vol': round(float(r.std() * np.sqrt(252)), 4), 'yearly_pct': yearly}
res = {'events': int(len(lens)), 'median_holding_days': float(np.median(lens)), 'share_days_with_positions': round(float((n[win] > 0).mean()), 3),
       'avg_names_when_invested': round(float(n[win][n[win] > 0].mean()), 1), 'window': [str(idx[win][0].date()), str(idx[win][-1].date())]}
b = e50[win].fillna(0); eqb = (1 + b).cumprod(); yrs = (idx[win][-1] - idx[win][0]).days / 365.25
res['0050'] = {'CAGR': round(float(eqb.iloc[-1] ** (1 / yrs) - 1), 4), 'maxDD': round(float((eqb / eqb.cummax() - 1).min()), 4),
               'yearly_pct': {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in b.groupby(b.index.year)}}
series = {}
for rt in (0.005, 0.01, 0.015):
    for idle in ('cash', '0050'):
        s, st = run(rt, idle); res[f'disposal_rt{rt}_idle_{idle}'] = st; series[(rt, idle)] = s
# 投入日的超額（對 0050）t 值
inv = (n > 0) & win
ex = (port_gross - w_new.fillna(0) * 0.005 - e50)[inv].dropna()
res['excess_vs_0050_on_invested_days'] = {'mean_bp_per_day': round(float(ex.mean() * 1e4), 2), 't': round(float(ex.mean() / (ex.std() / np.sqrt(len(ex)))), 2), 'days': int(len(ex))}
for a, bnd in [('2021-10-01', '2023-12-31'), ('2024-01-01', '2026-12-31')]:
    m = (ex.index >= a) & (ex.index <= bnd); e = ex[m]
    res[f'excess_sub_{a[:4]}_{bnd[:4]}'] = {'mean_bp_per_day': round(float(e.mean() * 1e4), 2), 't': round(float(e.mean() / (e.std() / np.sqrt(len(e)))), 2)}
(OUT / 'bt_disposal_strategy.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
pd.concat({f'{k[0]}_{k[1]}': v for k, v in series.items()}, axis=1).to_csv(OUT / 'disposal_daily_returns.csv')
for k, v in res.items(): print(k, v)

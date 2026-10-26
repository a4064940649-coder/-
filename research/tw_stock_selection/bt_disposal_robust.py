#!/usr/bin/env python3
"""處置股策略穩健性＋最終配置。
  變形：進場延後 1 天（第二天收盤）／提前 2 天出場／只持有前 5 個交易日（模擬處置期縮為 5 天）／只在第一次處置（同股 60 日內未曾處置）
  2026-08-10 之後的事件單獨列出（樣本很少）。
配置（每月換股日再平衡，2021-03～2026-10）：0050／選股規則／處置股策略的組合。
輸出 output/bt_disposal_robust.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
S = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'); P = S / 'panel'
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras'); OUT = Path(__file__).resolve().parent / 'output'
adj = pd.read_parquet(P / 'adj.parquet'); R1 = pd.read_parquet(P / 'ret.parquet'); idx = adj.index
e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].reindex(idx).ffill().pct_change()
dp = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end']); dp = dp[dp['code'].isin(adj.columns)].sort_values('start')
win = (idx >= pd.Timestamp('2021-10-01')) & (idx <= dp['end'].max())
def strategy(entry_lag=0, exit_early=0, max_days=None, first_only=False, rt=0.005, idle='cash', events=None):
    ev = dp if events is None else events
    hold = pd.DataFrame(False, index=idx, columns=adj.columns); ent = pd.Series(0.0, index=idx); last = {}
    rets = []
    for _, r in ev.iterrows():
        c = r['code']
        if first_only and c in last and (r['start'] - last[c]).days < 90: last[c] = r['start']; continue
        last[c] = r['start']
        i0 = idx.searchsorted(r['start']) + entry_lag; i1 = idx.searchsorted(r['end'], 'right') - 1 - exit_early
        if max_days: i1 = min(i1, i0 + max_days)
        if i0 >= len(idx) or i1 <= i0 or not np.isfinite(adj[c].iat[i0]) or not np.isfinite(adj[c].iat[i1]): continue
        hold.iloc[i0 + 1:i1 + 1, hold.columns.get_loc(c)] = True; ent.iat[i0] += 1; rets.append(adj[c].iat[i1] / adj[c].iat[i0] - 1)
    n = hold.sum(axis=1); g = R1.where(hold).mean(axis=1)
    cost = (ent.shift(1).fillna(0) / n.replace(0, np.nan)).fillna(0) * rt
    r = (g - cost).where(n > 0)
    r = r.fillna(e50 if idle == '0050' else 0.0)[win].fillna(0)
    eq = (1 + r).cumprod(); yrs = (r.index[-1] - r.index[0]).days / 365.25
    return r, {'events': len(rets), 'mean_event_pct': round(float(np.mean(rets) * 100), 2), 'median_event_pct': round(float(np.median(rets) * 100), 2),
               'hit_rate': round(float(np.mean(np.array(rets) > 0)), 3), 'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4),
               'maxDD': round(float((eq / eq.cummax() - 1).min()), 4)}
res = {}
base_r, res['base'] = strategy()
_, res['entry_lag_1day'] = strategy(entry_lag=1)
_, res['exit_2days_early'] = strategy(exit_early=2)
_, res['hold_first_5_days_only'] = strategy(max_days=5)
_, res['first_disposal_only_90d'] = strategy(first_only=True)
_, res['lag1_and_cost_1pct'] = strategy(entry_lag=1, rt=0.01)
late = dp[dp['start'] >= pd.Timestamp('2026-08-10')]
rl = [adj[r['code']].iat[idx.searchsorted(r['end'], 'right') - 1] / adj[r['code']].iat[idx.searchsorted(r['start'])] - 1 for _, r in late.iterrows()
      if idx.searchsorted(r['start']) < len(idx) and np.isfinite(adj[r['code']].iat[idx.searchsorted(r['start'])])]
rl = [x for x in rl if np.isfinite(x)]
res['events_after_2026-08-10'] = {'n': len(rl), 'mean_pct': round(float(np.mean(rl) * 100), 2) if rl else None, 'median_pct': round(float(np.median(rl) * 100), 2) if rl else None,
                                  'median_days': float(np.median([(idx.searchsorted(r['end'], 'right') - 1) - idx.searchsorted(r['start']) for _, r in late.iterrows()])) if len(late) else None}
# 配置
import bt_final_rule as fr
net, _, _, _ = fr.run(fr.yoy3)
dates = fr.dates
def period(series):
    eq = (1 + series.reindex(idx).fillna(0)).cumprod()
    v = eq.reindex(pd.DatetimeIndex(dates), method='ffill'); return (v.shift(-1) / v - 1).iloc[:-1]
disp_m = period(base_r); b_m = fr.B
common = net.index.intersection(disp_m.index)
from btlib import ann, yearly
mix = {}
for w0, w1, w2 in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0.5, 0.4, 0.1), (0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.6, 0.3, 0.1)]:
    r = w0 * b_m.reindex(common) + w1 * net.reindex(common) + w2 * disp_m.reindex(common)
    a = ann(r, dates); mix[f'0050 {int(w0*100)}% / rule {int(w1*100)}% / disposal {int(w2*100)}%'] = {'CAGR': a['CAGR'], 'maxDD': a['maxDD'], 'vol': a['vol'], 'yearly': yearly(r, dates)}
res['allocations_2021-10_2026-10'] = mix
(OUT / 'bt_disposal_robust.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in res.items():
    if k.startswith('alloc'):
        for kk, vv in v.items(): print(' ', kk, {x: vv[x] for x in ['CAGR', 'maxDD', 'vol']}, vv['yearly'])
    else: print(k, v)

# 配置對 0050 的每月超額 t 值
res['allocation_vs_0050_t'] = {}
for w0, w1, w2 in [(0.5, 0.5, 0), (0.5, 0.4, 0.1), (0.4, 0.4, 0.2)]:
    r = w0 * b_m.reindex(common) + w1 * net.reindex(common) + w2 * disp_m.reindex(common); ex = r - b_m.reindex(common)
    res['allocation_vs_0050_t'][f'{int(w0*100)}/{int(w1*100)}/{int(w2*100)}'] = {'mean_excess_pct_m': round(float(ex.mean() * 100), 2), 't': round(float(ex.mean() / (ex.std() / np.sqrt(len(ex)))), 2),
        'hit_rate_months': round(float((ex > 0).mean()), 3), 'tracking_error': round(float(ex.std() * np.sqrt(12)), 3)}
ex = disp_m.reindex(common) - b_m.reindex(common)
res['allocation_vs_0050_t']['disposal_alone'] = {'mean_excess_pct_m': round(float(ex.mean() * 100), 2), 't': round(float(ex.mean() / (ex.std() / np.sqrt(len(ex)))), 2)}
(OUT / 'bt_disposal_robust.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['allocation_vs_0050_t'])

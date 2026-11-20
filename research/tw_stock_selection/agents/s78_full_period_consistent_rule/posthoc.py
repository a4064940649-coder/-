#!/usr/bin/env python3
"""S78 事後描述（不在 PREREG 內、不影響判定；只讀 run.py 的輸出檔，不重跑引擎）：
(1) 關鍵格子的分塊資訊比率（超額 ÷ 追蹤誤差）——分辨「超額變大」是因為選股變準，還是只是把底倉拿掉、曝險放大；
(2) 全期最佳格的 W70 孿生格 vs 現行格的配對差；(3) 各維度邊際；(4) 位移診斷摘要；(5) 平均持股檔數與閒置比例。
python3 -I -B posthoc.py → results_posthoc.json"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
EX = pd.read_csv(HERE / 'grid_monthly_excess.csv.gz', index_col=0); EX.index = pd.PeriodIndex(EX.index, freq='M')
G = pd.read_csv(HERE / 'grid.csv').set_index('id'); D = json.load(open(HERE / 'results.json'))
W = {'B1 2008-2012': ('2008-01', '2012-12'), 'B2 2013-2016': ('2013-01', '2016-12'), 'B3 2017-2021': ('2017-01', '2021-12'), 'B4 2022-2026': ('2022-01', '2026-10'),
     '前段 2008-2015': ('2008-01', '2015-12'), '後段 2016-2026': ('2016-01', '2026-10'), '全期': ('2008-01', '2026-10')}
m = lambda a, b: (EX.index >= pd.Period(a, 'M')) & (EX.index <= pd.Period(b, 'M'))


def nw_t(x, lag=6):
    x = np.asarray(x, float); T = len(x); mu = x.mean(); e = x - mu; lr = (e * e).sum() / T
    for i in range(1, lag + 1):
        lr += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / T
    return float(mu / math.sqrt(lr / T))


def prof(x):
    return {k: dict(ex=float(x[m(a, b)].mean() * 1200), te=float(x[m(a, b)].std(ddof=1) * math.sqrt(12) * 100), IR=float(x[m(a, b)].mean() / x[m(a, b)].std(ddof=1) * math.sqrt(12)),
                    t_nw6=nw_t(x[m(a, b)].to_numpy())) for k, (a, b) in W.items()}


cur = 'STD24|MA60|T10M10|X20|NONE|W70'; fp = D['full_sample_pick']['id']; tw = fp.replace('W100', 'W70'); cur100 = cur.replace('W70', 'W100')
keys = {'現行': cur, '現行但不留底倉': cur100, '全期最佳': fp, '全期最佳的 W70 孿生': tw}
out = dict(profiles={k: dict(id=v, cagr=float(G.at[v, 'cagr']), maxdd=float(G.at[v, 'maxdd']), avg_npos=float(G.at[v, 'avg_npos']), avg_idle_frac=float(G.at[v, 'avg_idle_frac']),
                             n_closed=int(G.at[v, 'n_closed']), **prof(EX[v])) for k, v in keys.items()})
pair = lambda a, b: {k: dict(diff=float((EX[a] - EX[b])[m(x, y)].mean() * 1200), t_nw6=nw_t((EX[a] - EX[b])[m(x, y)].to_numpy())) for k, (x, y) in W.items()}
out['pairs'] = {'W70 孿生 − 現行': pair(tw, cur), '全期最佳 − 現行但不留底倉': pair(fp, cur100), '現行但不留底倉 − 現行': pair(cur100, cur)}
ir = EX.apply(lambda c: pd.Series({k: c[m(a, b)].mean() / c[m(a, b)].std(ddof=1) * math.sqrt(12) for k, (a, b) in W.items()}))
out['grid_IR'] = dict(median={k: float(ir.loc[k].median()) for k in W}, max={k: float(ir.loc[k].max()) for k in W}, current={k: float(ir.at[k, cur]) for k in W},
                      current_rank_full_IR=int((ir.loc['全期'] > ir.at['全期', cur]).sum() + 1), n_front_IR_ge_0p5=int((ir.loc['前段 2008-2015'] >= 0.5).sum()),
                      n_back_IR_ge_0p5=int((ir.loc['後段 2016-2026'] >= 0.5).sum()), best_front_IR_id=str(ir.loc['前段 2008-2015'].idxmax()),
                      best_front_IR_profile={k: float(ir.at[k, ir.loc['前段 2008-2015'].idxmax()]) for k in W})
out['weight_pairs'] = {}
g = G.reset_index(); a = g[g.weight == 'W70'].set_index(g[g.weight == 'W70']['id'].str.replace('|W70', '', regex=False)); b = g[g.weight == 'W100'].set_index(g[g.weight == 'W100']['id'].str.replace('|W100', '', regex=False))
for c in ('ex_B1', 'ex_B2', 'ex_B3', 'ex_B4', 'front_ex', 'back_ex', 'full_ex', 'maxdd'):
    out['weight_pairs'][c] = dict(mean_W100_minus_W70=float((b[c] - a[c]).mean()), share_W100_higher=float((b[c] > a[c]).mean()))
out['idle_by_entry'] = {k: dict(avg_npos=float(v['avg_npos'].mean()), avg_idle_frac=float(v['avg_idle_frac'].mean()), mean_targets=float(v['mean_targets'].mean())) for k, v in G.groupby('entry')}
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
r = lambda x: round(x, 2) if isinstance(x, float) else x
for k, v in out['profiles'].items():
    print(k, v['id'], r(v['cagr'] * 100), r(v['maxdd'] * 100), r(v['avg_npos']), r(v['avg_idle_frac']), v['n_closed'])
    for w in W:
        print('   ', w, {q: r(z) for q, z in v[w].items()})
for k, v in out['pairs'].items():
    print(k, {w: (r(z['diff']), r(z['t_nw6'])) for w, z in v.items()})
print(json.dumps({k: out[k] for k in ('grid_IR', 'weight_pairs', 'idle_by_entry')}, ensure_ascii=False, default=r))
for dim, v in D['marginals'].items():
    for lv, z in v.items():
        print(dim, lv, z['n'], [r(x) for x in z['block_ex_mean'].values()], 'front', r(z['front_ex_mean']), 'back', r(z['back_ex_mean']), 'min', r(z['min_block_mean']))
for c, v in D['shift_diagnostics'].items():
    print(c)
    for s, z in v.items():
        print('   ', s, r(z['cagr'] * 100), [r(x) for x in z['block_ex'].values()], 'min', r(z['min_block_ex']), 'front', r(z['front_ex']), 'back', r(z['back_ex']), 'mdd', r(z['maxdd'] * 100))

#!/usr/bin/env python3
"""事後診斷（POSTHOC，不參與判定）：B 序列、B 分組的下月報酬、減碼月份的「錯過漲幅／避開跌幅」。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'HOLDOUT' in R
NS = pd.read_parquet(HERE / 'navs_selval.parquet'); NH = pd.read_parquet(HERE / 'navs_hold.parquet')
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
B = pd.Series({pd.Timestamp(k): v[0] for k, v in R['B_series'].items()}).sort_index()
PER = {'SEL': ('2006-04-01', '2013-12-31'), 'VAL': ('2014-01-01', '2020-12-31'), 'HOLD': ('2021-03-01', '2026-10-31')}
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float('nan')
out = {}
# 1. B 分布
out['B_dist'] = {p: {q: float(B[(B.index >= a) & (B.index <= b)].quantile(q)) for q in (0.05, 0.2, 0.5, 0.8)} for p, (a, b) in PER.items()}
out['B_lowest'] = {str(d.date()): int(v) for d, v in B.nsmallest(25).items()}
# 2. B 分組 → 原版下一個月窗報酬
rows = []
for blk, NV, plans_key in (('long', NS, 'EXPOSURE_PLANS_long'), ('r5y', NH, 'EXPOSURE_PLANS_r5y')):
    for u in ('ALL', 'EXFIN'):
        nav = NV[f'{blk}|{u}|BASE|cash|0.001'].dropna()
        sd = pd.DatetimeIndex(sorted(pd.Timestamp(k) for k in R[plans_key][u]['BASE']))
        sd = sd[sd.isin(nav.index)]
        r = (nav.reindex(sd).shift(-1) / nav.reindex(sd) - 1).iloc[:-1]
        e = E50.reindex(E50.index.union(nav.index)).ffill().reindex(sd); r5 = (e.shift(-1) / e - 1).iloc[:-1]
        for d in r.index:
            per = next((p for p, (a, b) in PER.items() if pd.Timestamp(a) <= d <= pd.Timestamp(b)), None)
            if per: rows.append(dict(u=u, per=per, d=d, B=int(B[d]), r=float(r[d]), r50=float(r5[d])))
df = pd.DataFrame(rows)
bins = [-1, 9, 19, 29, 49, 1e9]; labels = ['0-9', '10-19', '20-29', '30-49', '50+']
df['bin'] = pd.cut(df['B'], bins, labels=labels)
grp = {}
for (u, per), g in df.groupby(['u', 'per']):
    for lb, h in g.groupby('bin', observed=True):
        grp[f'{u}|{per}|{lb}'] = dict(n=len(h), base_mean=round(h['r'].mean() * 100, 2), e50_mean=round(h['r50'].mean() * 100, 2),
                                    base_neg_share=round(float((h['r'] < 0).mean()), 2), base_worst=round(h['r'].min() * 100, 1))
out['next_month_by_B'] = grp
# 全期合併（SEL+VAL+HOLD）
for u in ('ALL', 'EXFIN'):
    g = df[df.u == u]
    out[f'corr_B_vs_next_{u}'] = {'spearman_base': float(g['B'].corr(g['r'], method='spearman')), 'spearman_0050': float(g['B'].corr(g['r50'], method='spearman')),
                                  'spearman_base_minus_0050': float(g['B'].corr(g['r'] - g['r50'], method='spearman'))}
    out[f'by_bin_all_periods_{u}'] = {str(lb): dict(n=len(h), base_mean=round(h['r'].mean() * 100, 2), t=round(tstat(h['r']), 2), e50_mean=round(h['r50'].mean() * 100, 2))
                                       for lb, h in g.groupby('bin', observed=True)}
# 3. 減碼月份：錯過漲幅／避開跌幅（≈(1−e)×原版月報酬）
miss = {}
for u in ('ALL', 'EXFIN'):
    g = df[df.u == u].set_index('d')
    for rule in ('D1_K10', 'D1_K20', 'D1_K30', 'D2', 'D3'):
        ex = {}
        for pk in ('EXPOSURE_PLANS_long', 'EXPOSURE_PLANS_r5y'):
            ex.update({pd.Timestamp(k): v for k, v in R[pk][u][rule].items()})
        for per in PER:
            h = g[g.per == per]
            cut = pd.Series({d: 1 - ex.get(d, 1.0) for d in h.index})
            c = cut * h['r']
            lst = [(str(d.date()), round(cut[d], 2), round(h.loc[d, 'r'] * 100, 1)) for d in h.index if cut[d] > 1e-9]
            miss[f'{u}|{rule}|{per}'] = dict(n_cut=len(lst), missed_gain=round(float(c[c > 0].sum()) * 100, 1), avoided_loss=round(float(-c[c < 0].sum()) * 100, 1),
                                           net=round(float(-c.sum()) * 100, 1), months=lst if per != 'SEL' or len(lst) < 60 else lst[:60])
out['cut_months'] = miss
R['POSTHOC_diagnostics'] = out
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'cut_months'}, ensure_ascii=False, indent=0)[:6000])
for k, v in miss.items():
    if k.startswith('ALL') or 'D2' in k or 'K30' in k:
        print(k, {x: y for x, y in v.items() if x != 'months'})
for k in ('ALL|D1_K30|VAL', 'ALL|D2|VAL', 'ALL|D3|VAL', 'ALL|D1_K30|HOLD', 'ALL|D2|HOLD', 'ALL|D3|HOLD'):
    print(k, miss[k]['months'])

"""Gate checkpoints: compare frozen PIT shares with independent truth snapshots (top 100 mcap)."""
import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/replicate_o8')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rdata import *

cl, op, rt, va = load_panel()
clf = cl.ffill()
sh = load_shares().reindex(index=cl.index).ffill()
src = pd.read_parquet(O8 + '/data/shares_src.parquet').reindex(index=cl.index).ffill()
RAW = O8 + '/raw'
out = {}


def top100(d):
    mc = (sh.loc[d] * clf.loc[d]).dropna()
    mc = mc[[c for c in mc.index if not c.startswith('91')]]
    return mc.sort_values(ascending=False).index[:100]


def compare(name, d, truth):
    d = cl.index[cl.index.searchsorted(pd.Timestamp(d))]
    t = top100(d)
    est = sh.loc[d, t]
    tr = truth.reindex(t)
    df = pd.DataFrame({'est': est, 'tr': tr}).dropna()
    df['err'] = (df.est / df.tr - 1).abs()
    exact = df[df.err < 1e-9].index.tolist()
    df2 = df[df.err >= 1e-9]
    r = dict(date=str(d.date()), n=len(df), n_exact=len(exact),
             med_all=float(df.err.median()), p90_all=float(df.err.quantile(.9)),
             med_ex=float(df2.err.median()), p90_ex=float(df2.err.quantile(.9)),
             sp_all=float(spearmanr(df.est, df.tr)[0]), sp_ex=float(spearmanr(df2.est, df2.tr)[0]),
             worst=df.err.sort_values().tail(5).round(4).to_dict())
    out[name] = r
    print(name, json.dumps(r, ensure_ascii=False))


for f, d in (('StockList_20150522.csv', '2015-05-22'), ('StockList_20170706.csv', '2017-07-06')):
    w = pd.read_csv(RAW + '/weihautin/update_csv/' + f, dtype={'代號': str})
    truth = w.set_index('代號')['股本(億)'].astype(float) * 1e8 / 10
    compare('weihautin_' + d, d, truth)
for d in ('2021-07-02', '2021-09-24'):
    t = pd.read_csv(RAW + f'/tdcc/total_{d}.csv', dtype={'code': str})
    t = t[t.lvl == 17].set_index('code')['shares'].astype(float)
    compare('tdcc_' + d, d, t)

# coverage: top-100 stock-months by source code (month-end)
me = cl.index.to_series().groupby(cl.index.to_period('M')).last()
me = me[(me.index >= pd.Period('2014-01')) & (me.index <= pd.Period('2026-10'))]
cnt = {}
miss = 0
for p, d in me.items():
    t = top100(d)
    s = src.loc[d, t]
    miss += int(s.isna().sum())
    for v in s.dropna().values:
        cnt[int(v)] = cnt.get(int(v), 0) + 1
out['coverage_top100_month'] = {'by_src_code': cnt, 'missing': miss, 'total': int(len(me) * 100)}
print('coverage', cnt, miss)

# identical share fraction 2019-06 vs 2021-10 (all stocks with both)
a = sh.loc['2019-06'].iloc[-1]; b = sh.loc['2021-10'].iloc[-1]
both = a.notna() & b.notna()
out['same_2019_06_vs_2021_10'] = float((a[both] == b[both]).mean())
print('same', out['same_2019_06_vs_2021_10'])

# delisted stocks that ever were top-120 2013-2026
last = cl.apply(lambda s: s.last_valid_index())
dead = last[last < pd.Timestamp('2026-10-01')].index
ever = set()
for d in me.values:
    mc = (sh.loc[d] * clf.loc[d]).dropna().sort_values(ascending=False)
    ever |= set(mc.index[:120])
dd = sorted(set(dead) & ever)
out['delisted_top120'] = {'n': len(dd), 'codes': dd,
                          'with_shares': int(sum(sh[c].notna().any() for c in dd))}
print('delisted top120', len(dd), dd)
json.dump(out, open(OUT + '/gate_check.json', 'w'), ensure_ascii=False, indent=1)

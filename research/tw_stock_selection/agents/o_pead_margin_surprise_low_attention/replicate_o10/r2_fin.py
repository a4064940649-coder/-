"""單季 EPS／營收（Q1 公告值；Q2、Q3 公告單季、缺值用累計差；Q4＝全年−前三季累計），SUE、REVSUE。"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
f = pd.read_parquet(D / 'fin_raw.parquet')
f['y'] = f.period.str[:4].astype(int); f['q'] = f.period.str[-1].astype(int)
f['qi'] = f.y * 4 + f.q - 1
f = f.sort_values(['stock_id', 'qi'])


def single(f, col):
    qv = f[col + '_q'].astype(float); ytd = f[col + '_ytd'].astype(float)
    key = f.set_index(['stock_id', 'qi'])[col + '_ytd']
    prev = pd.Series(list(zip(f.stock_id, f.qi - 1))).map(key.to_dict()).values.astype(float)
    # 前一季須同一年
    prev = np.where(f.q.values == 1, np.nan, prev)
    diff = ytd.values - prev
    out = np.where(f.q.values == 1, np.where(np.isfinite(qv.values), qv.values, ytd.values),
                   np.where(f.q.values == 4, diff, np.where(np.isfinite(qv.values), qv.values, diff)))
    return out


f['eps'] = single(f, 'eps')
f['rev'] = single(f, 'rev')
for c in ['gp', 'opi']:
    f[c] = single(f, c)
g = f[['stock_id', 'period', 'qi', 'y', 'q', 'industry', 'eps', 'rev', 'gp', 'opi']].copy()


def sue(g, col):
    out = np.full(len(g), np.nan)
    for sid, x in g.groupby('stock_id'):
        s = pd.Series(x[col].values, index=x.qi.values)
        s = s.reindex(np.arange(s.index.min(), s.index.max() + 1))
        dq = s - s.shift(4)
        sd = dq.shift(1).rolling(8, min_periods=6).std(ddof=1)
        v = (dq / sd).reindex(x.qi.values).values
        out[x.index.map(lambda i: g.index.get_loc(i))] = v
    return out


g = g.reset_index(drop=True)
g['sue'] = sue(g, 'eps')
g['revsue'] = sue(g, 'rev')
g.loc[~np.isfinite(g.sue), 'sue'] = np.nan
g.loc[~np.isfinite(g.revsue), 'revsue'] = np.nan
g.to_parquet(D / 'fin.parquet', index=False)
print(g.shape, g.sue.notna().sum(), g.revsue.notna().sum())
print(g[g.sue.notna()].groupby('period').size().head(8))

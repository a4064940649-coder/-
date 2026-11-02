"""寬表面板：上市＋上櫃 4 碼與 0050。含息報酬＝close/(close−change)−1。"""
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'

d = pd.read_parquet(D / 'daily.parquet')
d = d[d.market.isin(['twse', 'tpex'])]
d = d[(d.stock_id.str.len() == 4)]
d = d.drop_duplicates(['date', 'stock_id'], keep='last')
dates = sorted(d.date.unique())
codes = sorted(d.stock_id.unique())
di = {x: i for i, x in enumerate(dates)}; ci = {x: i for i, x in enumerate(codes)}
T, N = len(dates), len(codes)
r = d.date.map(di).values; c = d.stock_id.map(ci).values
out = {}
for col in ['open', 'high', 'low', 'close', 'amount', 'change']:
    a = np.full((T, N), np.nan); a[r, c] = d[col].values.astype(float); out[col] = a
mk = np.zeros((T, N), np.int8); mk[r, c] = np.where(d.market.values == 'twse', 1, 2); out['mkt'] = mk
lim = np.zeros((T, N), np.int8); lim[r, c] = np.select([d.limit.values == 'up', d.limit.values == 'down'], [1, -1], 0); out['limflag'] = lim
cl = out['close']; ch = out['change']
o = out['open']; o[o <= 0] = np.nan
cl[cl <= 0] = np.nan
prev = cl - ch
ret = cl / prev - 1
# 沒有交易的日子 ret = 0（價格不動）；異常值保護
ret[~np.isfinite(ret)] = np.nan
out['ret'] = ret
out['ref'] = prev
np.savez_compressed(D / 'panel.npz', dates=np.array(dates), codes=np.array(codes), **out)
print(T, N, np.nanmax(ret), np.nanmin(ret), np.mean(np.abs(ret) > 0.11))

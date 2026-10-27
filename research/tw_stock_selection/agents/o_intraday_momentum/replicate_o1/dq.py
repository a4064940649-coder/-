import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
IDX = R.IDX
ci, co = R.daily_split()
# 全池日均（每日橫斷面平均，再按年平均），只用 GOOD 日
gi = R.GI
dci = pd.Series(np.nanmean(ci[gi], axis=1), index=IDX[gi]); dco = pd.Series(np.nanmean(co[gi], axis=1), index=IDX[gi])
y = pd.DataFrame({'ci_bp': dci.groupby(dci.index.year).mean() * 1e4, 'co_bp': dco.groupby(dco.index.year).mean() * 1e4}).round(1)
print(y.T.to_string())
# 挑選期 252 日累積盤中的橫斷面標準差（股池＝P1 股池）
I, mI = R.roll_sum(ci, 252, 0); O_, mO = R.roll_sum(co, 252, 0)
exec(open(R.HERE / 'p1.py').read().split('SPECS =')[0].split("days = [d")[0])
S52, V60 = R.V.rolling_signals('gapfix')
import types
days = [d for d in R.K.rebalance_days('2005-01-01') if d <= pd.Timestamp('2019-04-30')]
src = open(R.HERE / 'p1.py').read(); fdef = src[src.index('def pool'):src.index('SPECS')]
exec(fdef.split('JOINT')[0])
sds = []
for d in days:
    t = IDX.get_loc(d); el = pool(t); x = np.where(el & mI & mO, I[t], np.nan)
    if np.isfinite(x).sum() > 50: sds.append(np.nanstd(x, ddof=1))
out = {'yearly_bp': y.to_dict(), 'xs_sd_CI252_sel_median': round(float(np.median(sds)), 3), 'xs_sd_CI252_sel_mean': round(float(np.mean(sds)), 3)}
print(out['xs_sd_CI252_sel_median'], out['xs_sd_CI252_sel_mean'])
json.dump(out, open(R.HERE / 'dq.json', 'w'), ensure_ascii=False, indent=1, default=str)

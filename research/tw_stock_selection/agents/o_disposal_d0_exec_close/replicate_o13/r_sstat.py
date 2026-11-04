import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle, numpy as np, pandas as pd
from r_sleeve import *
out = pickle.load(open('sleeve_main.pkl', 'rb'))
EX = pd.DataFrame({k: v[2] for k, v in out.items()})
RM = pd.DataFrame({k: v[0].resample('ME').last().pct_change() for k, v in out.items()})
for a, b in [('2006-04-01', '2012-12-31'), ('2013-01-01', '2018-12-31'), ('2019-01-01', '2026-10-22')]:
    print('seg', a, np.median([per(EX[c], a, b).mean() * 100 for c in EX]))
medser = EX.median(axis=1)
print('median-series t FULL', tstat(medser), 'VAL', tstat(per(medser, '2024-01-01', '2026-10-22')))
# annualized of median monthly return
nav = out[('C1', 'CL')][0]
rm = pd.DataFrame({k: v[0].resample('ME').last() for k, v in out.items()}).pct_change()
rm.iloc[0] = pd.DataFrame({k: v[0].resample('ME').last() for k, v in out.items()}).iloc[0] - 1
mr = rm.median(axis=1); cum = (1 + mr).cumprod()
yrs = len(mr) / 12; print('ann full', cum.iloc[-1] ** (1 / yrs) - 1, 'mdd', (cum / cum.cummax() - 1).min())
b = E50S.resample('ME').last().pct_change().reindex(mr.index); b.iloc[0] = E50S[E50S.index <= mr.index[0]].iloc[-1] / E50S[E50S.index < '2006-04-01'].iloc[-1] - 1
cb = (1 + b).cumprod(); print('0050 ann', cb.iloc[-1] ** (1 / yrs) - 1, 'mdd', (cb / cb.cummax() - 1).min())
v = per(mr, '2024-01-01', '2026-10-22'); print('VAL ann', (1 + v).prod() ** (12 / len(v)) - 1)
# C1xCL daily
s = nav; yrs = (s.index[-1] - s.index[0]).days / 365.25
print('C1xCL daily ann', (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1, 'mdd', (s / s.cummax() - 1).min())
# best-3-years removal
yr = medser.groupby(medser.index.year).sum(); best = yr.nlargest(3).index.tolist(); print('best3', best, 't wo', tstat(medser[~medser.index.year.isin(best)]))

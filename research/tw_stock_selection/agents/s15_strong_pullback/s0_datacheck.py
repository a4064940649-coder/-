#!/usr/bin/env python3
"""S15 第 0 步（寫 PREREG 之前；不算任何報酬）：訊號密度、各條件通過率、和 SURGE 名單重疊、0050 分割停牌日檢查。
python3 -I -B s0_datacheck.py → data_check.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s15lib
from s15lib import s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s15lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
del INNOV
RT = s2lib.rev_tables(P)
F = s15lib.pb_features(P, U)
sig0 = F['sig0']
out = {}
t_from = P.t0 - 1
rows = np.arange(P.T) >= t_from
yr = pd.DatetimeIndex(P.IDX).year
# 條件通過率（股池內、2015-07 起的股日比例）
den = U[rows].sum()
cond = {}
acc = U.copy()
for k in ('strong', 'near', 'pb1', 'vd1', 'turn'):
    acc = acc & F[k]
    cond[k] = dict(alone=float((U & F[k])[rows].sum() / den), cumulative=float(acc[rows].sum() / den), cumulative_n=int(acc[rows].sum()))
out['condition_pass_rate（股池內股日比例，依序累加）'] = cond
n_day = sig0.sum(axis=1)
s = pd.Series(n_day[rows], index=P.IDX[rows])
out['signals_total_since_2015_07'] = int(s.sum())
out['signals_per_year'] = {int(y): int(v) for y, v in s.groupby(s.index.year).sum().items()}
out['days_with_signal_share'] = float((s > 0).mean())
out['signals_per_day_dist'] = {k: float(v) for k, v in s.describe(percentiles=[0.5, 0.9, 0.99]).items()}
out['days_ge_10_signals'] = int((s >= 10).sum())
# 同一檔 20 個交易日內重複訊號（只算第一次的「獨立事件」）
first = np.zeros_like(sig0)
last = np.full(P.N, -10**6)
for t in range(P.T):
    j = np.flatnonzero(sig0[t])
    nf = j[t - last[j] >= 20]
    first[t, nf] = True
    last[j] = t
fs = first[rows].sum()
out['independent_events_20d'] = int(fs)
out['independent_events_per_year'] = {int(y): int(v) for y, v in pd.Series(first[rows].sum(axis=1), index=P.IDX[rows]).groupby(pd.DatetimeIndex(P.IDX[rows]).year).sum().items()}
out['distinct_stocks'] = int(sig0[rows].any(axis=0).sum())
# 和 SURGE 名單重疊：訊號日 k 時最近一次 SURGE 換股（訊號列 ≤ k）的買進名單、L_MA 前 20 名
SE = s15lib.surge_signal(P, U, RT, shift=0)
srows = np.array([x['k'] for x in SE['info']])
buy = {x['k']: set(x['targets']) for x in SE['info']}
in_buy = in_top20 = n_ev = 0
for t in np.flatnonzero(first.any(axis=1)):
    if t < t_from:
        continue
    i = np.searchsorted(srows, t, side='right') - 1
    for j in np.flatnonzero(first[t]):
        n_ev += 1
        if i < 0:
            continue
        kr = srows[i]
        if P.COLS[j] in buy[kr]:
            in_buy += 1
        if not SE['X'][kr, j]:
            in_top20 += 1
out['overlap_with_surge（獨立事件）'] = dict(n=n_ev, in_last_surge_buy_list=in_buy / n_ev, in_last_LMA_top20=in_top20 / n_ev)
# 0050 分割停牌日（2025-06-11～17）：E50O 應等於前一日含息收盤（用前一日收盤），E50 沿用
d = pd.DatetimeIndex(P.IDX)
m = (d >= '2025-06-09') & (d <= '2025-06-19')
out['e50_split_days'] = [dict(date=str(d[t].date()), E50=round(float(P.E50[t]), 4), E50O=round(float(P.E50O[t]), 4), E50O_raw=(None if not np.isfinite(P.E50O_raw[t]) else float(P.E50O_raw[t])),
                              E50O_eq_prev_close=bool(abs(P.E50O[t] - P.E50[t - 1]) < 1e-9)) for t in np.flatnonzero(m)]
# RS 用到的 0050 含息序列：分割前後日報酬
i0 = np.flatnonzero(d == pd.Timestamp('2025-06-18'))
if len(i0):
    t = int(i0[0]); out['e50_ret_2025_06_18'] = float(P.E50[t] / P.E50[t - 1] - 1)
out['note'] = '只算訊號、密度、條件通過率、名單重疊、0050 停牌日價格；沒有算任何持有期報酬或本題格子的淨值。'
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

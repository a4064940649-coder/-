#!/usr/bin/env python3
"""抽樣核對事件條件（直接讀原始 parquet 重算），並做診斷：訊號日收盤起的毛前瞻異常（不可成交，僅判斷形態本身有無資訊）。"""
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
sys.path.insert(0, str(HERE.parents[1])); from btlib import tstat  # noqa: E402
P = R.P
adj = pd.read_parquet(P / 'adj.parquet'); C = pd.read_parquet(P / 'close_raw.parquet'); O = pd.read_parquet(P / 'open.parquet'); V = pd.read_parquet(P / 'value.parquet')
good = adj.notna().sum(axis=1) >= 300; adj, C, O, V = adj[good], C[good], O[good], V[good]
df = pd.read_csv(Path(P).parent / 'w4_engulf_CONF_events.csv.gz', dtype={'code': str}, parse_dates=['sig', 'entry'])
rng = np.random.default_rng(0); bad = 0
for _, r in df.sample(15, random_state=1).iterrows():
    c = r['code']; i = adj.index.get_loc(r['sig'])
    a = adj[c]; cr = C[c]; o = O[c]; vol = V[c] / cr
    ao = o * a / cr
    b0 = (o.iloc[i - 1] - cr.iloc[i - 1]) / o.iloc[i - 1]; b1 = (cr.iloc[i] - o.iloc[i]) / o.iloc[i]
    ok = [b0 >= 0.015, b1 >= 1.5 * b0, ao.iloc[i] <= a.iloc[i - 1] + 1e-9, a.iloc[i] >= ao.iloc[i - 1] - 1e-9,
          a.iloc[i - 1] / a.iloc[i - 11] - 1 <= -0.05, a.iloc[i] < a.iloc[i - 19:i + 1].mean(),
          vol.iloc[i] >= 1.5 * vol.iloc[i - 5:i].mean(), a.iloc[i + 1] > a.iloc[i], adj.index[i + 2] == r['entry']]
    bad += not all(ok)
    print(c, r['sig'].date(), 'O/C t-1', o.iloc[i - 1], cr.iloc[i - 1], 'O/C t', o.iloc[i], cr.iloc[i], 'C t+1', cr.iloc[i + 1], 'vol/vma5p', round(vol.iloc[i] / vol.iloc[i - 5:i].mean(), 2),
          'ret10', round(a.iloc[i - 1] / a.iloc[i - 11] - 1, 3), 'hold', r['hold'], r['why'], 'raw%', round(r['raw'] * 100, 2), 'OK' if all(ok) else ok)
print('sample failures', bad)
# 診斷：各規格訊號日 t 收盤起 k 日（收對收）毛異常（減同池等權收對收），t 按訊號月份分群
ewc = R.ew_cc.to_numpy(); LC = np.concatenate([[0.0], np.cumsum(np.log1p(ewc))])
ACn = R.ACn
for key, side, conf, vol, lag in R.SPECS:
    S = R.signals(side, conf, vol); ts, js = np.where(S)
    for base_name, b in [('from_t_close', 0), ('from_t+1_close', 1)]:
        out = []
        for k in (1, 5, 10):
            m = (ts + b + k) < R.T
            t0 = ts[m] + b; j = js[m]
            raw = ACn[t0 + k, j] / ACn[t0, j] - 1
            ew = np.expm1(LC[t0 + k + 1] - LC[t0 + 1])
            d = pd.DataFrame({'d': R.IDX[ts[m]], 'ab': raw - ew}).dropna()
            for wn, (a_, b_) in [('S', ('2005-01-01', '2019-04-30')), ('V', ('2019-05-01', '2026-10-31'))]:
                x = d[(d['d'] >= a_) & (d['d'] <= b_)]
                out.append(f"{wn}{k}d {x['ab'].mean()*100:+.2f}% t{tstat(x.groupby(x['d'].dt.to_period('M'))['ab'].mean())}")
        print(key, base_name, ' | '.join(out))

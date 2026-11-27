#!/usr/bin/env python3
"""S104 登記前檢查：引擎跑得動、筆數與持股檔數（位移 0、各格），並做會計自檢（零成本、零借券費時放空套筒＝−1×做多同名單近似）。
只輸出筆數、平均持股檔數、自檢誤差；不輸出任何報酬。用法：python3 -I -B step0_counts.py → step0.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s104lib as L, s104cells as C
t0 = time.time(); rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
d = L.load_cache(); L.set_idx(d)
out = dict(TA=str(d.IDX[d.TA].date()), TB=str(d.IDX[d.TB].date()), TE=str(d.IDX[d.TE].date()), K1={}, K2={}, K3={})
def cnt(nav, tr, a, b):
    x = tr[(tr['ent_t'] > a) & (tr['ent_t'] <= b)]
    return dict(n_trades=int(len(x)), avg_hold_days=float((x['exit_t'] - x['ent_t']).mean()) if len(x) else None)
for c in C.K1_CELLS:
    nav, tr, info = C.k1_run(d, c, 0)
    out['K1'][c] = dict(train=cnt(nav, tr, d.TA, d.TB), valid=cnt(nav, tr, d.TB, d.TE), **info)
# 會計自檢：做多 vs 放空 同一名單、零成本、零借券費 → 單一部位報酬應互為相反數
c = 'K1|rebal|all|none'
nS, tS, _ = C.k1_run(d, c, 0, borrow=0.0, cost='zero')
nLc, tLc, _ = C.k1_run(d, c, 0, borrow=0.0, cost='zero', side=+1)
mm = tS.merge(tLc, on=['j', 'ent_t', 'exit_t'], suffixes=('_s', '_l'))
out['K1_selfcheck'] = dict(n_short=int(len(tS)), n_long=int(len(tLc)), n_matched=int(len(mm)), max_abs_ret_sum=float((mm['ret_s'] + mm['ret_l']).abs().max()))
pre = C.K2Pre(d)
for c in C.K2_CELLS:
    nL, tL, nS2, tS2 = C.k2_run(d, pre, c, 0)
    out['K2'][c] = dict(long_train=cnt(nL, tL, d.TA, d.TB), short_train=cnt(nS2, tS2, d.TA, d.TB), long_valid=cnt(nL, tL, d.TB, d.TE), short_valid=cnt(nS2, tS2, d.TB, d.TE))
tr = pd.read_parquet(L.TMP / 'off_trades.parquet'); tr0 = tr[tr['shift'] == 0]
for c, cf in C.K3_CELLS.items():
    nW, tW = C.k3_sleeve(d, tr0, cf['tenor'], 1.35, 0.03, d.TA, d.TE)
    out['K3'][c] = dict(train=cnt(nW, tW, d.TA, d.TB), valid=cnt(nW, tW, d.TB, d.TE), by_kind=tW['why'].value_counts().to_dict())
out['meta'] = dict(runtime_s=round(time.time() - t0), maxrss_mb=rss())
(HERE / 'step0.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=0)[:6000])

#!/usr/bin/env python3
"""第 2 步之二：處置腿變體（挑選期 2021-10～2023-12 只看這段）：d1 上市＋上櫃全部、d2 上市＋上櫃只做第一次處置；對照 d0 只上市（SOP 現行）。
事前準則：處置腿與 0050 的月超額 t 較高者；兩者差 < 0.3 時取規則較簡單的 d1。結果併入 iterations.json 的 DISPOSAL。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import otlib as L, dlib as DL
D = L.D
a, b = pd.Timestamp('2021-10-01'), pd.Timestamp('2023-12-29')
V = {'d0_TWSE_all': dict(markets=('TWSE',), first_only=False), 'd1_ALL_all': dict(markets=('TWSE', 'TPEX'), first_only=False),
     'd2_ALL_first': dict(markets=('TWSE', 'TPEX'), first_only=True)}
E50 = pd.Series(D.E50, index=D.IDX)
res = {}
for k, v in V.items():
    nav, tr, sk = DL.sleeve(DL.events(**v), a, b)
    m = nav.resample('ME').last(); r = m.pct_change().dropna(); rb = E50.reindex(nav.index).resample('ME').last().pct_change().dropna()
    x = (r - rb).dropna()
    res[k] = dict(ann=float((nav.iloc[-1]) ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1) * 100,
                  ann_0050=float((E50[nav.index[-1]] / E50[nav.index[0]]) ** (365.25 / (nav.index[-1] - nav.index[0]).days) - 1) * 100,
                  t_month=L.tstat(x), mean_x_pp=float(x.mean() * 100), mdd=L.mdd(nav), n_trades=int(len(tr)),
                  trade_mean_pct=float(tr.ret.mean() * 100) if len(tr) else None, tpex_trades=int((tr.mkt == 'TPEX').sum()) if len(tr) else 0,
                  tpex_trade_mean_pct=float(tr[tr.mkt == 'TPEX'].ret.mean() * 100) if len(tr) and (tr.mkt == 'TPEX').any() else None, skipped=sk)
    print(k, res[k], flush=True)
c = ['d1_ALL_all', 'd2_ALL_first']
ch = max(c, key=lambda k: res[k]['t_month'])
if abs(res['d1_ALL_all']['t_month'] - res['d2_ALL_first']['t_month']) < 0.3: ch = 'd1_ALL_all'
doc = json.loads((HERE / 'iterations.json').read_text())
doc['DISPOSAL'] = dict(rule=__doc__, results=res, chosen=ch)
(HERE / 'iterations.json').write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str))
print('chosen', ch)

"""步驟 7：事後診斷（不改規格）：2019 年後 E 日異常與成交倍數、生效前 10 日、20 日均成交值中位數的變化。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as L

x = L.event_frame()
X = x[x.explorable].copy()
pre = X[X.win == 'select']; post = X[X.win.isin(['validate', 'holdout'])]
out = {}
for nm, g in (('select', pre), ('oos', post)):
    o = {}
    for c in ['MSCI_STD_DEL', 'MSCI_SC_DEL', 'MSCI_STD_ADD', 'MSCI_SC_ADD', 'TW50_DEL', 'TW50_ADD']:
        h = g[g.cls == c]
        if len(h): o[c] = {'n': int(len(h)), 'abE': round(h.abE.mean() * 100, 2), 'vr_median': round(h.vr.median(), 1),
                           'pre10_sum': round(h.sp10.mean() * 100, 2), 'pre10_bh': round(h.pre10.mean() * 100, 2)}
    dl = g[g.is_del]
    o['all_del'] = {'n': int(len(dl)), 'pre10_sum': round(dl.sp10.mean() * 100, 2), 'pre10_bh': round(dl.pre10.mean() * 100, 2),
                    'adv20_median_e8': round(dl.adv20.median() / 1e8, 2)}
    dm = dl[dl.family == 'MSCI']
    o['msci_del'] = {'n': int(len(dm)), 'pre10_sum': round(dm.sp10.mean() * 100, 2), 'adv20_median_e8': round(dm.adv20.median() / 1e8, 2)}
    out[nm] = o
# 可交易刪除事件（交易用篩選）的 adv20 中位數
T = x[x.tradable & x.is_del]
out['adv20_median_tradable_e8'] = {'select': round(T[T.win == 'select'].adv20.median() / 1e8, 2),
                                   'oos': round(T[T.win.isin(['validate', 'holdout'])].adv20.median() / 1e8, 2)}
L.dump(out, 'posthoc.json')
import pprint; pprint.pprint(out, width=160)

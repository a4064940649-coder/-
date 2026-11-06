# 診斷：我的 V7 FIX100 一般成本 JMED（2019-05 起）與協定殘差逐月差
import sys, os, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib as L
import r4_hist as H
N7 = H.res['V7_FIX100']['_N']
j7 = N7.median(axis=1)
ms = [m for m in j7.index if '2019-05' <= m <= '2026-09']
post = j7.loc[ms]
d = (post - post.mean()).values - L.RESID
for m, v, dd, nj in zip(ms, post.values, d, N7.loc[ms].notna().sum(axis=1)):
    if abs(dd) > 0.05: print(m, round(v, 3), round(dd, 3), nj)
# 嘗試：去掉不完整月後
ok = N7.loc[ms].notna().sum(axis=1) == 13
print('n complete', ok.sum())

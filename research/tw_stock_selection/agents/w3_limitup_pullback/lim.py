#!/usr/bin/env python3
"""資料載入、漲停價（升降單位）判斷。規則見 results.json PREREG。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'exec_rebuild')); sys.path.insert(0, str(HERE.parents[1]))
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')

names = ['adj', 'close_raw', 'open', 'high', 'value', 'flag', 'src', 'ret']
D = {k: pd.read_parquet(P / f'{k}.parquet') for k in names}
good = D['adj'].notna().sum(axis=1) >= 300
BAD = [str(x.date()) for x in D['adj'].index[~good]]
cols = [c for c in D['adj'].columns if c.isdigit() and len(c) == 4 and not c.startswith('91')]
for k in D: D[k] = D[k].loc[good, cols]
IDX = D['adj'].index; T = len(IDX); N = len(cols)
F = D['flag']; bad5 = (F == 5)
ADJ = D['adj'].mask(bad5); CR = D['close_raw'].mask(bad5); OP = D['open'].mask(bad5); HI = D['high'].mask(bad5)
V = D['value']; RET = D['ret'].mask(bad5)
FLAGOK = F.isin([0, 1, 2])


def tick_of(p, old):
    # 2005-03-01 前舊制：<5:0.01、5–15:0.05、15–50:0.1、50–150:0.5、150–1000:1、≥1000:5（以收盤＝漲停價精確比對驗證，見 results.json prereg_amendments）
    new = np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    oldt = np.select([p < 5, p < 15, p < 50, p < 150, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
    return np.where(old, oldt, new)


lim = np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)[:, None]
with np.errstate(invalid='ignore', divide='ignore'):
    REF = (CR / (1 + RET)).to_numpy()
    x = REF * (1 + lim)
    tk = tick_of(np.nan_to_num(x, nan=0.0), (IDX < pd.Timestamp('2005-03-01'))[:, None])
    LIMUP = np.floor(x / tk + 1e-6) * tk
    crn = CR.to_numpy(); opn = OP.to_numpy(); hin = HI.to_numpy()
    CLOSE_LU = (crn >= LIMUP - 0.5 * tk) & (crn >= hin * 0.999) & FLAGOK.to_numpy()
    OPEN_LU = (opn >= LIMUP - 0.5 * tk)
    # 粗判（stlib 口徑）供核對
    CC = (CR / CR.shift(1) - 1).to_numpy()
    ROUGH_LU = (CC >= lim - 0.006) & (crn >= hin * 0.999) & (F == 0).to_numpy()

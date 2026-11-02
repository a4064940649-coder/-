#!/usr/bin/env python3
"""P2 中間版本：car_i（shares_pit 只用在 shares_est 有值格；membership_pit）、car_ii（＋4763／2317 修正）。o7 面板。"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents'); O7 = AG / 'o_capmatched_selection'
HERE = Path(__file__).resolve().parent
src = HERE / 'car_o7'
C = pd.read_parquet(O7 / 'raw/panel/close_raw.parquet'); IDX, COLS = C.index, C.columns
SE = pd.read_parquet(O7 / 'data/shares_est.parquet').reindex(index=IDX, columns=COLS)
SP = pd.read_parquet(AG / 'o_0050_tilt_pit/data/shares_pit.parquet').reindex(index=IDX, columns=COLS)
Si = SP.where(SE.notna())
Sii = Si.copy(); ix = Sii.index
m1 = (ix >= '2019-09-04') & (ix <= '2023-07-26'); Sii.loc[m1, '4763'] /= 12.65
m2 = (ix >= '2023-07-27') & (ix <= '2025-06-29'); Sii.loc[m2, '4763'] /= 10.0
v = Sii.at[pd.Timestamp('2018-06-29'), '2317']; m3 = (ix > '2018-06-29') & (ix <= '2018-10-25'); Sii.loc[m3, '2317'] = v
for name, S in [('car_i', Si), ('car_ii', Sii)]:
    d = HERE / name
    if d.exists(): shutil.rmtree(d)
    (d / 'raw').mkdir(parents=True); (d / 'data').mkdir()
    for f in ['cmlib.py', 'cmstrat.py']: shutil.copy2(src / f, d / f)
    for sub in ['agents', 'bench', 'screener', 'navs', 'exec', 'panel']: os.symlink(O7 / 'raw' / sub, d / 'raw' / sub)
    for f in ['skeleton.json', 'float_phi.parquet']: os.symlink(O7 / 'data' / f, d / 'data' / f)
    S.to_parquet(d / 'data/shares_est.parquet')
    shutil.copy2(AG / 'o_0050_tilt_pit/data/membership_pit.json', d / 'data/membership_pre2015.json')
print('ok')

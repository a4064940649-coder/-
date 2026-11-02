#!/usr/bin/env python3
"""敏感度：car_a2＝(a) 但補值格的 ret 缺值不補（照來源 ret 原樣）。只為找出與研究員 (a) 的小差異來源。"""
import os, shutil
from pathlib import Path
import numpy as np, pandas as pd
AG = Path('/home/user/-/research/tw_stock_selection/agents'); O7 = AG / 'o_capmatched_selection'
HERE = Path(__file__).resolve().parent
P2 = HERE / 'panel_a2'
if P2.exists(): shutil.rmtree(P2)
P2.mkdir()
for f in (HERE / 'panel_a').iterdir():
    if f.name != 'ret.parquet': shutil.copy2(f, P2 / f.name)
R = pd.read_parquet(HERE / 'panel_a/ret.parquet'); R0 = pd.read_parquet(O7 / 'raw/panel/ret.parquet')
DP = pd.read_parquet(AG / 'o_infra_v3_rebuild/delist_prices.parquet'); DP['date'] = pd.to_datetime(DP['date'])
for r in DP.itertuples():
    if not np.isfinite(r.ret): R.at[r.date, r.code] = np.nan
R.to_parquet(P2 / 'ret.parquet')
d = HERE / 'car_a2'
if d.exists(): shutil.rmtree(d)
shutil.copytree(HERE / 'car_a', d, symlinks=True)
os.remove(d / 'raw/panel'); os.symlink(P2, d / 'raw/panel')
print('ok')

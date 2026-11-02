#!/usr/bin/env python3
"""複製者 o9：建立三個「載體」資料夾（o7 cmlib/cmstrat 原檔複製、逐位元核對雜湊），只換輸入。
  car_o7：o7 原輸入
  car_a ：版本 (a)＝shares_pit 全格＋4763/2317 修正；delist_prices 補面板缺值格；V7 上市為否→股數 NaN；2015-09 前名單＝membership_pit
  car_b ：(a) 的股數整體落後 63 個面板交易日（上市遮罩不落後）
python3 -I -B r0_build.py"""
import hashlib, json, os, shutil
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
O7 = AG / 'o_capmatched_selection'
HERE = Path(__file__).resolve().parent
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
EXP = {'cmlib.py': 'da4dba39e3b8d15ed27608c515dc71172c35301a18527e50224f023e578db19c',
       'cmstrat.py': '58d022090aaf0ddc0db351969e0050434aef288874fc2d10641c1a5fde26c8a0'}
INP = {AG / 'o_0050_tilt_pit/data/shares_pit.parquet': '901801d80a30a629f68dc6d578fbe29c5118554e702849872c4a9ce4795bc603',
       AG / 'o_0050_tilt_pit/data/membership_pit.json': 'b9fa9b60d3d737675536b13603e0c8b54bbae85e23ee4ff77a75b8b4b1de4ff6',
       AG / 'o_infra_v3_rebuild/delist_prices.parquet': 'bc4526b4424eaf9b0d13dcc378ce1ccf54a3f3a35e57d414101ea587aa6abf04',
       AG / 'o_v7_listing_exec_engine/data/listed_twse.parquet': '0faa1a996fb5d8e18c8767dd7cda219d41bb81c852bca9e528ebfcf622eacf87',
       O7 / 'data/shares_est.parquet': 'c60a9209b992288819c06d0f4dd65edb4df381ff15708cb9477cb4a5dfcd1b61'}
for f, h in list(EXP.items()): assert sha(O7 / f) == h, f
for f, h in INP.items(): assert sha(f) == h, f

def mkcar(name):
    d = HERE / name
    if d.exists(): shutil.rmtree(d)
    (d / 'raw').mkdir(parents=True); (d / 'data').mkdir()
    for f in EXP: shutil.copy2(O7 / f, d / f); assert sha(d / f) == EXP[f]
    for sub in ['agents', 'bench', 'screener', 'navs', 'exec']:
        os.symlink(O7 / 'raw' / sub, d / 'raw' / sub)
    for f in ['skeleton.json', 'float_phi.parquet']:
        os.symlink(O7 / 'data' / f, d / 'data' / f)
    return d

# ---- o7 ----
d = mkcar('car_o7')
os.symlink(O7 / 'raw' / 'panel', d / 'raw' / 'panel')
os.symlink(O7 / 'data' / 'shares_est.parquet', d / 'data' / 'shares_est.parquet')
os.symlink(O7 / 'data' / 'membership_pre2015.json', d / 'data' / 'membership_pre2015.json')

# ---- panel with delisted fill ----
P = {n: pd.read_parquet(O7 / 'raw/panel' / f'{n}.parquet') for n in ['close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src', 'adj']}
IDX, COLS = P['close_raw'].index, P['close_raw'].columns
DP = pd.read_parquet(AG / 'o_infra_v3_rebuild/delist_prices.parquet')
DP['date'] = pd.to_datetime(DP['date'])
DP = DP[DP.date.isin(IDX) & DP.code.isin(COLS)]
nfill = 0
for code, g in DP.groupby('code'):
    g = g.set_index('date').sort_index()
    miss = P['close_raw'].loc[g.index, code].isna()
    g = g[miss.to_numpy()]
    nfill += len(g)
    for src_c, dst in [('close', 'close_raw'), ('open', 'open'), ('high', 'high'), ('value', 'value')]:
        P[dst].loc[g.index, code] = g[src_c].to_numpy(float)
    # 含息報酬：用來源 ret；缺值則用前一收盤（面板＋補值合併）算
    cc = P['close_raw'][code].ffill().shift(1)
    r = g['ret'].to_numpy(float)
    alt = (g['close'] / cc.loc[g.index] - 1).to_numpy(float)
    r = np.where(np.isfinite(r), r, alt)
    P['ret'].loc[g.index, code] = r
    P['flag'].loc[g.index, code] = 0.0
    # adj：前值接續
    a = P['adj'][code].copy()
    for dd, rr in zip(g.index, r):
        k = IDX.get_loc(dd); prev = a.iloc[:k].dropna()
        a.iloc[k] = (prev.iloc[-1] if len(prev) else 1.0) * (1 + (rr if np.isfinite(rr) else 0.0))
    P['adj'][code] = a
print('delist fill cells', nfill)
PAN = HERE / 'panel_a'
if PAN.exists(): shutil.rmtree(PAN)
PAN.mkdir()
for n, df in P.items(): df.to_parquet(PAN / f'{n}.parquet')

# ---- shares (a) ----
S = pd.read_parquet(AG / 'o_0050_tilt_pit/data/shares_pit.parquet').reindex(index=IDX, columns=COLS)
ix = S.index
m1 = (ix >= '2019-09-04') & (ix <= '2023-07-26'); S.loc[m1, '4763'] = S.loc[m1, '4763'] / 12.65
m2 = (ix >= '2023-07-27') & (ix <= '2025-06-29'); S.loc[m2, '4763'] = S.loc[m2, '4763'] / 10.0
v0629 = S.at[pd.Timestamp('2018-06-29'), '2317']
m3 = (ix > '2018-06-29') & (ix <= '2018-10-25'); S.loc[m3, '2317'] = v0629
LST = pd.read_parquet(AG / 'o_v7_listing_exec_engine/data/listed_twse.parquet').reindex(columns=COLS)
LST = LST.reindex(IDX).fillna(True).astype(bool)   # 面板有、V7 無的日期/代碼 → 不遮罩
Sa = S.where(LST.to_numpy())
Sb = S.shift(63).where(LST.to_numpy())
print('2317 2018-06-29', v0629, 'post', S.at[pd.Timestamp('2018-10-26'), '2317'])
print('listed False cells in panel range', int((~LST.to_numpy()).sum()))

for name, SS in [('car_a', Sa), ('car_b', Sb)]:
    d = mkcar(name)
    os.symlink(PAN, d / 'raw' / 'panel')
    SS.to_parquet(d / 'data' / 'shares_est.parquet')
    shutil.copy2(AG / 'o_0050_tilt_pit/data/membership_pit.json', d / 'data' / 'membership_pre2015.json')
print(json.dumps({k: sha(HERE / k / 'data/shares_est.parquet') for k in ['car_a', 'car_b']}, indent=1))

#!/usr/bin/env python3
"""S2a：觸發強度特徵（面板近似「注意交易資訊」條件），逐月輸出。不產生任何模型預測。
因變數 Y＝universe 普通股且 in_panel 的處置列數（以 start 月份；另存 announce 月份）。
python3 -I -B s2_features.py → data/monthly_features.csv"""
import json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
adj = pd.read_parquet(P / 'adj.parquet'); cr = pd.read_parquet(P / 'close_raw.parquet'); val = pd.read_parquet(P / 'value.parquet')
keep = cr.notna().sum(axis=1) >= 300
adj, cr, val = adj[keep], cr[keep], val[keep]
cols = [c for c in adj.columns if len(c) == 4 and not c.startswith('0')]
adj, cr, val = adj[cols], cr[cols], val[cols]
TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date']).set_index('date')['close'].reindex(adj.index).ffill()
IDX = adj.index
r6 = adj / adj.shift(6) - 1
m6 = TX / TX.shift(6) - 1
vol = val / cr
vr = vol / vol.shift(1).rolling(60, min_periods=40).mean()
ret = adj.pct_change(fill_method=None)
post15 = pd.Series(IDX >= pd.Timestamp('2015-06-01'), index=IDX)
th_reg = pd.Series(np.where(post15, 0.32, 0.25), index=IDX)
def price_flag(th):
    T = np.broadcast_to(th.to_numpy()[:, None], r6.shape)
    d = r6.sub(m6, axis=0)
    up = (r6.to_numpy() > T) & (d.to_numpy() > 0.20)
    dn = (r6.to_numpy() < -T) & (d.to_numpy() < -0.20)
    return pd.DataFrame(up | dn, index=IDX, columns=cols), pd.DataFrame(up, index=IDX, columns=cols)
PF_reg, PFup_reg = price_flag(th_reg)
PF_32, _ = price_flag(pd.Series(0.32, index=IDX))
VF = (vr >= 5) & (r6.abs() > 0.10)
F_reg = PF_reg | VF
F_32 = PF_32 | VF
R14 = pd.Timestamp('2014-04-01')   # 「連續三次」條款與第一次處置 5→10 日（公開沿革；擬合樣本內）
def episodes(F, regime_aware):
    Fi = F.astype(int)
    r3 = Fi.rolling(3).sum() == 3; r5 = Fi.rolling(5).sum() == 5; r10 = Fi.rolling(10).sum() >= 6
    if regime_aware:
        pre = pd.Series(IDX < R14, index=IDX)
        trig = (r5 & pre.to_numpy()[:, None]) | (r3 & ~pre.to_numpy()[:, None]) | r10
    else:
        trig = r3 | r10
    trig = pd.DataFrame(trig, index=IDX, columns=cols)
    prev = trig.astype(int).shift(1).rolling(10, min_periods=1).sum().fillna(0) > 0
    new = trig & ~prev
    # 近似處置起日＝觸發日的下一交易日
    return new.shift(1, fill_value=False)
E_reg = episodes(F_reg, True); E_32 = episodes(F_32, True); E_reg_na = episodes(F_reg, False)
M = IDX.to_period('M')
feat = pd.DataFrame({
    'n_stocks': cr.notna().sum(axis=1).groupby(M).mean(),
    'n_tdays': pd.Series(1, index=IDX).groupby(M).sum(),
    'FD_reg': F_reg.sum(axis=1).groupby(M).sum(),
    'FD_32': F_32.sum(axis=1).groupby(M).sum(),
    'PDup_reg': PFup_reg.sum(axis=1).groupby(M).sum(),
    'VD': VF.sum(axis=1).groupby(M).sum(),
    'K_reg': E_reg.sum(axis=1).groupby(M).sum(),
    'K_32': E_32.sum(axis=1).groupby(M).sum(),
    'K_reg_na': E_reg_na.sum(axis=1).groupby(M).sum(),
    'mkt_absret6_mean': m6.abs().groupby(M).mean(),
})
U = pd.read_csv(HERE / 'data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
U = U[U.code.isin(cols)]
feat['Y_start'] = U.groupby(U.start.dt.to_period('M')).size().reindex(feat.index).fillna(0).astype(int)
feat['Y_announce'] = U.groupby(U.announce.dt.to_period('M')).size().reindex(feat.index).fillna(0).astype(int)
feat.index.name = 'month'
feat.to_csv(HERE / 'data/monthly_features.csv')
print(feat.shape, feat.index.min(), feat.index.max())

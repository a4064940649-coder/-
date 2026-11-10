# -*- coding: utf-8 -*-
"""replicate_o19 資料層（獨立複製員自寫；只依凍結規格文字與 BRIEF 資料）。
python3 -I -B rdata.py  → 在 scratchpad/rep_o19/rdata.npz 產生快取。
v60mode: 'strict'（F1：只有非 GOOD 日沿用前值）或 'lim3'（G1 原引擎讀法：ffill(limit=3)，只作診斷）。
"""
import os, sys, json, math
import numpy as np, pandas as pd

AG = '/home/user/-/research/tw_stock_selection/agents'
V7 = AG + '/o_v7_listing_exec_engine/data'
SCR = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o19'
EXEMPT = [pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')]
EX_LO, EX_HI = pd.Timestamp('2024-02-15'), pd.Timestamp('2024-03-15')


def load_panel():
    f = {k: pd.read_parquet(f'{V7}/{k}.parquet') for k in ['close_raw', 'open', 'value', 'ret', 'flag']}
    idx = f['close_raw'].index
    cols = list(f['close_raw'].columns)
    for k in f:
        f[k] = f[k].reindex(index=idx, columns=cols).astype(float)
    # P：delist_prices 只補缺值格
    p = pd.read_parquet(AG + '/o_infra_v3_rebuild/delist_prices.parquet')
    p['code'] = p['code'].astype(str)
    p['date'] = pd.to_datetime(p['date'])
    nfill = 0
    for src, dst in [('close', 'close_raw'), ('open', 'open'), ('value', 'value'), ('ret', 'ret')]:
        w = p.pivot_table(index='date', columns='code', values=src, aggfunc='last')
        w = w.reindex(index=idx, columns=cols)
        m = f[dst].isna() & w.notna()
        nfill += int(m.values.sum())
        f[dst] = f[dst].where(~m, w)
    return idx, cols, f, nfill


def rolling_good(X, good, win, minp, exempt_mask, mode):
    """X: (T,N) 全部日；只在 GOOD 列上滾動。mode='mean_nonan'：窗內非豁免缺值 → NaN。"""
    gi = np.where(good)[0]
    Xg = X[gi]
    exg = exempt_mask[gi]
    T, N = X.shape
    out_g = np.full(Xg.shape, np.nan)
    if mode == 'mean_v60':
        isn = np.isnan(Xg)
        bad = isn & (~exg[:, None])           # 非豁免日缺值
        val = np.where(isn, 0.0, Xg)
        cnt = (~isn).astype(float)
        cs_v = np.vstack([np.zeros((1, N)), np.cumsum(val, 0)])
        cs_c = np.vstack([np.zeros((1, N)), np.cumsum(cnt, 0)])
        cs_b = np.vstack([np.zeros((1, N)), np.cumsum(bad.astype(float), 0)])
        for i in range(win - 1, len(gi)):
            s = cs_v[i + 1] - cs_v[i + 1 - win]
            c = cs_c[i + 1] - cs_c[i + 1 - win]
            b = cs_b[i + 1] - cs_b[i + 1 - win]
            with np.errstate(invalid='ignore', divide='ignore'):
                m = s / c
            m[(b > 0) | (c == 0)] = np.nan
            out_g[i] = m
    elif mode == 'std':
        Sg = pd.DataFrame(Xg).rolling(win, min_periods=minp).std(ddof=1).values
        out_g = Sg
    return gi, out_g


def build(v60mode='strict'):
    idx, cols, f, nfill = load_panel()
    C = f['close_raw'].values
    O = f['open'].values
    V = f['value'].values
    R = f['ret'].values
    FL = f['flag'].values
    T, N = C.shape
    TR = ~np.isnan(C)
    G = np.where(TR, np.where(np.isnan(R), 1.0, 1.0 + R), 1.0)
    CF = pd.DataFrame(C).ffill().values
    PC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR, G * PC / C, 1.0)
    ADJ[~np.isfinite(ADJ)] = 1.0
    GOOD = TR.sum(1) >= 300
    dts = pd.DatetimeIndex(idx)
    exm = np.zeros(T, bool)
    for e in EXEMPT:
        exm |= (dts == e)
    exm |= (dts >= EX_LO) & (dts <= EX_HI)
    gi, v60g = rolling_good(V, GOOD, 60, 60, exm, 'mean_v60')
    V60 = np.full((T, N), np.nan)
    V60[gi] = v60g
    if v60mode == 'strict':
        # 非 GOOD 日沿用前一個 GOOD 日
        last = None
        for t in range(T):
            if GOOD[t]:
                last = t
            elif last is not None:
                V60[t] = V60[last]
    elif v60mode == 'lim3':
        last = None
        for t in range(T):
            if GOOD[t]:
                last = t
            elif last is not None:
                V60[t] = V60[last]
        V60 = pd.DataFrame(V60).ffill(limit=3).values
    gi2, sg = rolling_good(R, GOOD, 60, 40, exm, 'std')
    SIG = np.full((T, N), np.nan)
    SIG[gi2] = sg
    SIG = pd.DataFrame(SIG).ffill().values
    # vrank：V60≥1e7 且當日收盤（非補值）≥10，依 V60 由大到小；同值依欄位序
    VR = np.full((T, N), np.nan)
    for t in range(T):
        ok = (V60[t] >= 1e7) & (C[t] >= 10)
        ii = np.where(ok)[0]
        if len(ii) == 0:
            continue
        order = ii[np.lexsort((ii, -V60[t, ii]))]
        VR[t, order] = np.arange(1, len(order) + 1)
    # 0050
    e = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj']
    E = e.reindex(dts.union(e.index)).ffill().reindex(dts).values
    return dict(dates=dts.values, codes=np.array(cols), C=C, O=O, V=V, R=R, FL=FL, TR=TR, G=G, CF=CF, PC=PC,
                ADJ=ADJ, GOOD=GOOD, V60=V60, SIG=SIG, VR=VR, E=E, nfill=np.array(nfill))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'strict'
    os.makedirs(SCR, exist_ok=True)
    d = build(mode)
    np.savez(f'{SCR}/rdata_{mode}.npz', **d)
    print('saved', mode, 'nfill', int(d['nfill']), 'GOOD', int(d['GOOD'].sum()), 'T', len(d['dates']))

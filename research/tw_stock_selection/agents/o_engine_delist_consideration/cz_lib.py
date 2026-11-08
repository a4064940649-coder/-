#!/usr/bin/env python3
"""o_engine_delist_consideration 共用小工具（普查用；不 import 任何研究中心引擎）。
只讀：scratchpad/panel（ia_core 紙上面板）、o_v7_listing_exec_engine/data（V7 面板）、
o_infra_v3_rebuild/delist_prices.parquet（2019-05-17～2023 下市股價格修補）、delist_official.csv（終止上市官方名單）。
「可成交」定義與 ia_core.simulate 相同：當日有收盤且有開盤。"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
PANEL = SCR / 'panel'
V7P = AG / 'o_v7_listing_exec_engine' / 'data'
O3 = AG / 'o_infra_v3_rebuild'
WORK = SCR / 'o17_delist'
WORK.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load():
    Cia = pd.read_parquet(PANEL / 'close_raw.parquet'); Oia = pd.read_parquet(PANEL / 'open.parquet')
    C7 = pd.read_parquet(V7P / 'close_raw.parquet'); O7 = pd.read_parquet(V7P / 'open.parquet')
    R7 = pd.read_parquet(V7P / 'ret.parquet')
    IDX = C7.index.union(Cia.index)
    COLS = list(C7.columns)
    Cia = Cia.reindex(index=IDX, columns=COLS); Oia = Oia.reindex(index=IDX, columns=COLS)
    C7 = C7.reindex(index=IDX, columns=COLS); O7 = O7.reindex(index=IDX, columns=COLS); R7 = R7.reindex(index=IDX, columns=COLS)
    TRia = (np.isfinite(Cia.to_numpy()) & np.isfinite(Oia.to_numpy()))
    Cu = C7.to_numpy().copy(); Ou = O7.to_numpy().copy(); Ru = R7.to_numpy().copy()
    # 面板缺值格以 ia 面板補（兩者應相同；保守取聯集）
    m = ~np.isfinite(Cu) & np.isfinite(Cia.to_numpy()); Cu[m] = Cia.to_numpy()[m]; Ou[m] = Oia.to_numpy()[m]
    P = pd.read_parquet(O3 / 'delist_prices.parquet')
    CI = {c: j for j, c in enumerate(COLS)}
    ii = IDX.get_indexer(pd.DatetimeIndex(P['date'])); jj = np.array([CI.get(c, -1) for c in P['code']])
    ok = (ii >= 0) & (jj >= 0); ii, jj, P = ii[ok], jj[ok], P[ok]
    nw = ~np.isfinite(Cu[ii, jj]); ii, jj, P = ii[nw], jj[nw], P[nw]
    Cu[ii, jj] = P['close'].to_numpy(); Ou[ii, jj] = P['open'].to_numpy(); Ru[ii, jj] = P['ret'].to_numpy()
    TRu = np.isfinite(Cu) & np.isfinite(Ou)
    return IDX, COLS, CI, TRia, TRu, Cu, Ru, int(len(ii))


IDX, COLS, CI, TRIA, TRU, CU, RU, NPATCH = _load()
N = len(COLS)
DATES = IDX.values
DL = pd.read_csv(O3 / 'delist_official.csv', dtype=str)
DL['date'] = pd.to_datetime(DL['date'])
DL4 = DL[DL['four'] == 'True']
DLD = {}
for c, d in zip(DL4.code, DL4.date):
    DLD.setdefault(c, []).append(d)


def last_trade_before(j, t, TR):
    """證券 j 在面板索引 t（含）之後的「同一證券」最後可成交日索引：考慮代碼重用（終止上市日後同代碼為新證券）。"""
    c = COLS[j]
    w = np.flatnonzero(TR[:, j])
    if len(w) == 0: return -1
    lim = None
    for dd in sorted(DLD.get(c, [])):
        if IDX[t] <= dd:
            lim = dd; break
    if lim is not None:
        w = w[IDX[w] <= lim]
        if len(w) == 0: return -1
    return int(w[-1])


def next_idx(d):
    """日期 d 之後（不含）第一個面板交易日索引。"""
    return int(np.searchsorted(DATES, np.datetime64(pd.Timestamp(d)), side='right'))


def idx_of(d):
    return int(np.searchsorted(DATES, np.datetime64(pd.Timestamp(d))))

#!/usr/bin/env python3
"""複製員 o8：資料層（自寫）。只依凍結規格文字：V7 面板＋P 下市價補缺值、TR/G/CF/PC/ADJ、GOOD 日、V60、σ60、vrank、dec、
處置名單（rejudge in_panel ＋ disposal_5y）、X 規則（併購）、0050、CPI。結果快取在 cache/d8.npz。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
O7 = AG / 'o_v7_listing_exec_engine'
PNL = O7 / 'data'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
CACHE = HERE / 'cache'; CACHE.mkdir(exist_ok=True)


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def build():
    C = pd.read_parquet(PNL / 'close_raw.parquet')
    IDX, COLS = C.index, C.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
    ld = lambda k: pd.read_parquet(PNL / f'{k}.parquet').reindex(index=IDX, columns=COLS).to_numpy(float).copy()
    Cn = C.to_numpy(float).copy(); On = ld('open'); Vn = ld('value'); Rn = ld('ret'); Fn = ld('flag')
    # P：下市價只補面板缺值格
    P = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
    ii = IDX.get_indexer(pd.DatetimeIndex(P['date'])); jj = np.array([CI.get(c, -1) for c in P['code']])
    m = (ii >= 0) & (jj >= 0); ii, jj, P = ii[m], jj[m], P[m]
    m = ~np.isfinite(Cn[ii, jj]); ii, jj, P = ii[m], jj[m], P[m]
    Cn[ii, jj] = P['close'].to_numpy(); On[ii, jj] = P['open'].to_numpy(); Vn[ii, jj] = P['value'].to_numpy()
    Rn[ii, jj] = P['ret'].to_numpy(); Fn[ii, jj] = 0
    TR = np.isfinite(Cn)
    G = np.where(TR, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
    CF = pd.DataFrame(Cn).ffill().to_numpy()
    PC = np.vstack([np.full((1, N), np.nan), CF[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR & np.isfinite(PC), G * PC / Cn, 1.0)
    ADJ[~np.isfinite(ADJ)] = 1.0
    GOOD = TR.sum(1) >= 300
    # 滾動：只用 GOOD 日
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    Vg = Vn[g]
    miss = pd.DataFrame((np.isnan(Vg) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    v60 = pd.DataFrame(Vg).rolling(60, min_periods=1).mean().to_numpy()
    V60 = np.full((len(IDX), N), np.nan); V60[g] = np.where(miss, v60, np.nan)
    sg = pd.DataFrame(Rn[g]).rolling(60, min_periods=40).std().to_numpy()
    SIG = np.full((len(IDX), N), np.nan); SIG[g] = sg
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    # vrank（t 當日的值；用時取 t−1）
    base = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(Cn) & (Cn >= 10)
    key = np.where(base, V60, -np.inf)
    o = np.argsort(-key, axis=1, kind='stable')
    VR = np.full((len(IDX), N), np.nan)
    rk = np.empty_like(o); r_ = np.arange(1, N + 1)
    for t in range(len(IDX)):
        rk[t, o[t]] = r_
    VR = np.where(base, rk, np.nan)
    # 處置名單
    A_ = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    A_ = A_[A_['in_panel'] == True]
    B_ = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    ka = set(zip(A_.code, A_.start))
    B_ = B_[[(c, s) not in ka for c, s in zip(B_.code, B_.start)]]
    D = pd.concat([A_[['code', 'start', 'end']], B_[['code', 'start', 'end']]])
    D = D[D.code.isin(CI)]
    DISP = np.zeros((len(IDX), N), bool)
    for c, s, e in zip(D.code, D.start, D.end):
        a = IDX.searchsorted(s); b = IDX.searchsorted(e, side='right')
        DISP[a:b, CI[c]] = True
    # 0050
    E = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    E50 = E.reindex(IDX.union(E.index)).ffill().reindex(IDX).to_numpy(float)
    np.savez(CACHE / 'd8.npz', Cn=Cn, On=On, Vn=Vn, Rn=Rn, Fn=Fn, TR=TR, G=G, CF=CF, PC=PC, ADJ=ADJ, GOOD=GOOD,
             V60=V60, SIG=SIG, VR=VR, DISP=DISP, E50=E50, IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str))


def xrules(IDX, COLS, TR, Cn):
    """X 規則（依 V4 規格文字自寫）：最後交易日、four 旗標的停牌恢復、merger_consideration。
    回傳 j -> (f, rel_idx, until_idx, type, delist_idx)"""
    CI = {c: j for j, c in enumerate(COLS)}
    DL = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
    d4 = DL[DL['four'] == 'True']
    lt = np.where(TR.any(0), len(IDX) - 1 - np.argmax(TR[::-1], 0), -1)
    until = {}
    for c, dd in zip(d4.code, pd.to_datetime(d4.date)):
        j = CI.get(c)
        if j is None or lt[j] < 0 or IDX[lt[j]] <= dd: continue
        w = np.where(TR[:, j])[0]; a = w[IDX[w] <= dd]; b = w[IDX[w] > dd]
        if len(a) == 0 or len(b) == 0: continue
        lt[j] = a[-1]; until[j] = IDX[b[0]]
    M = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype={'code': str, 'successor': str})
    for c in ('delist_date', 'pay_date'): M[c] = pd.to_datetime(M[c], errors='coerce')
    M = M.drop_duplicates('code').set_index('code'); out = {}
    for j, c in enumerate(COLS):
        if lt[j] < 0 or IDX[lt[j]] > IDX[-1] - pd.Timedelta(days=30): continue
        ltd = IDX[lt[j]]; lc = Cn[lt[j], j]; typ = None; didx = None
        if c in M.index:
            r = M.loc[c]; typ = r['type']
            dd = r['delist_date'] if pd.notna(r['delist_date']) else ltd + pd.Timedelta(days=14)
            if pd.notna(r['delist_date']): didx = int(np.searchsorted(IDX, r['delist_date']))
            if r['type'] == 'cash' and pd.notna(r['cash']):
                f = float(r['cash']) / lc; rel = r['pay_date'] if pd.notna(r['pay_date']) else dd + pd.Timedelta(days=14)
            elif r['type'] == 'failure': f, rel = 0.0, ltd
            else: f, rel = 1.0, dd
        else: f, rel = 1.0, ltd + pd.Timedelta(days=14)
        out[j] = (f, int(np.searchsorted(IDX, rel)), int(np.searchsorted(IDX, until[j])) if j in until else 10 ** 9, typ, didx)
    return out


_D = None
def load():
    global _D
    if _D is None:
        if not (CACHE / 'd8.npz').exists(): build()
        z = np.load(CACHE / 'd8.npz', allow_pickle=False)
        D = {k: z[k] for k in z.files}
        D['IDX'] = pd.DatetimeIndex(D['IDX']); D['COLS'] = list(D['COLS'])
        D['CI'] = {c: j for j, c in enumerate(D['COLS'])}
        D['XR'] = xrules(D['IDX'], D['COLS'], D['TR'], D['Cn'])
        D['TPOS'] = {d: i for i, d in enumerate(D['IDX'])}
        _D = D
    return _D


if __name__ == '__main__':
    build(); D = load()
    print(len(D['IDX']), len(D['COLS']), int(D['GOOD'].sum()), len(D['XR']), int(D['DISP'].sum()))

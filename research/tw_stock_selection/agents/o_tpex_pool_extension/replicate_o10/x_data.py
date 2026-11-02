#!/usr/bin/env python3
"""replicate_o10 的 x_data 替身（獨立寫；介面對齊 V7X x_exec.py 需要的名稱）。
合併面板＝r0_panel.py 產生的 merged.npz；上市部分沿用 V7X 的下市價補值、X 出清、併購現金、處置名單規則；
上櫃下櫃股出清值＝最後收盤 × OTPX_DELIST_F（預設 1）。上櫃股的十分位用「上市等價名次」。"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
O7 = AG / 'o_v7_listing_exec_engine'
PNL = O7 / 'data'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o10')
SPEC7 = json.loads((PNL / 'model_spec7.json').read_text())
CPI = json.loads((O7 / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']
CACHE = SCR / 'xd_cache.npz'
ODD_SWITCH = pd.Timestamp('2020-10-26')
FEEMIX_SWITCH = pd.Timestamp('2015-01-01')
LIM_SWITCH = pd.Timestamp('2015-06-01')
DELIST_F = float(os.environ.get('OTPX_DELIST_F', '1.0'))


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def _build():
    Z = np.load(SCR / 'merged.npz')
    IDX = pd.DatetimeIndex(Z['IDX']); COLS = list(Z['COLS']); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
    Cn, On, Vn, Rn, Fn = (Z[k].astype(float).copy() for k in ('C', 'O', 'V', 'R', 'F'))
    TPX = Z['TPX']; tponly = set(c for j, c in enumerate(COLS) if TPX[:, j].all())
    P = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
    ii = IDX.get_indexer(pd.DatetimeIndex(P['date'])); jj = np.array([CI.get(c, -1) if c not in tponly else -1 for c in P['code']])
    ok = (ii >= 0) & (jj >= 0); ii, jj, P = ii[ok], jj[ok], P[ok]
    nw = ~np.isfinite(Cn[ii, jj]); ii, jj, P = ii[nw], jj[nw], P[nw]
    Cn[ii, jj] = P['close'].to_numpy(); On[ii, jj] = P['open'].to_numpy(); Vn[ii, jj] = P['value'].to_numpy()
    Rn[ii, jj] = P['ret'].to_numpy(); Fn[ii, jj] = 0
    TR = np.isfinite(Cn)
    G = np.where(TR, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
    CF = pd.DataFrame(Cn).ffill().to_numpy()
    PC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR, G * PC / Cn, 1.0)
    ADJ = np.where(np.isfinite(ADJ), ADJ, 1.0)
    GOOD = (TR & ~TPX).sum(1) >= 300
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    low90 = set(Z['LOW90']) | {'2025-02'}
    exm = pd.Index([str(p) for p in dg.to_period('M')]).isin(low90)
    EXM = ex[:, None] | (exm[:, None] & TPX[g])        # 豁免日（上櫃格多了覆蓋 <90% 的月份）
    Vv = Vn[g]
    mV = pd.DataFrame((np.isnan(Vv) & ~EXM).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    V60g = np.where(mV, v60, np.nan)
    s60 = pd.DataFrame(Rn[g]).rolling(60, min_periods=40).std().ffill().to_numpy()
    V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan)
    V60[g] = V60g; SIG[g] = s60
    V60 = pd.DataFrame(V60).ffill(limit=3).to_numpy()
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    # 52 週高點比：含息累積指數 ÷ 252 個 GOOD 日最高（至少 60 筆）
    AC = np.where(TR, np.cumprod(G, axis=0), np.nan)
    Ag = AC[g]
    mx = pd.DataFrame(Ag).rolling(252, min_periods=60).max().to_numpy()
    S52 = np.full((len(IDX), N), np.nan); S52[g] = Ag / mx
    # 上市等價名次：上市股依 V60 排名；上櫃股＝上市合格股中 V60 比它大的檔數＋1
    el = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(CF) & (CF >= 10)
    VR = np.full((len(IDX), N), np.nan)
    for t in range(len(IDX)):
        wl = np.where(el[t] & ~TPX[t])[0]
        if len(wl):
            o = wl[np.argsort(-V60[t, wl], kind='stable')]
            VR[t, o] = np.arange(1, len(o) + 1)
            wt = np.where(el[t] & TPX[t])[0]
            if len(wt):
                sv = np.sort(V60[t, wl])
                VR[t, wt] = len(wl) - np.searchsorted(sv, V60[t, wt], side='right') + 1
    sh = lambda A: np.vstack([np.full((1, N), np.nan), A[:-1]])
    V60p, SIGp, VRp = sh(V60), sh(SIG), sh(VR)
    with np.errstate(invalid='ignore', divide='ignore'):
        REF = np.where(Fn == 1, PC, Cn / G)
    E50r = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    E50 = E50r.reindex(IDX.union(E50r.index)).ffill().reindex(IDX).to_numpy()
    return dict(IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str), C=Cn, O=On, V=Vn, R=Rn, F=Fn,
             G=G, CF=CF, PC=PC, ADJ=ADJ, V60p=V60p, SIGp=SIGp, VRp=VRp, REF=REF, E50=E50, GOOD=GOOD, V60=V60, S52=S52, TPX=TPX,
             DELIST_TP=Z['DELIST_TP'])


_Z = _build()   # 磁碟空間不足，不寫快取；每次在記憶體重建
IDX = pd.DatetimeIndex(_Z['IDX']); COLS = list(_Z['COLS']); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
C, O, V, R, F, G, CF, PC, ADJ = (_Z[k] for k in ('C', 'O', 'V', 'R', 'F', 'G', 'CF', 'PC', 'ADJ'))
V60p, SIGp, VRp, REF, E50, GOOD, V60, S52, TPX = (_Z[k] for k in ('V60p', 'SIGp', 'VRp', 'REF', 'E50', 'GOOD', 'V60', 'S52', 'TPX'))
DELIST_TP = set(_Z['DELIST_TP'])
TR = np.isfinite(C)
TPOS = {d: i for i, d in enumerate(IDX)}
DATES = IDX.values


def _disp():
    A = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    A = A[A['in_panel'] == True]
    B = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    B = B[B.code.isin(set(COLS))]
    ka = set(zip(A.code, A.start))
    B = B[[(c, s) not in ka for c, s in zip(B.code, B.start)]]
    Dd = pd.concat([A[['code', 'start', 'end']], B[['code', 'start', 'end']]])
    Dd = Dd[Dd.code.isin(CI)].dropna()
    M = np.zeros((len(IDX), N), bool)
    for c, s, e in zip(Dd.code, Dd.start, Dd.end):
        a, b = np.searchsorted(DATES, np.datetime64(s)), np.searchsorted(DATES, np.datetime64(e), side='right')
        M[a:b, CI[c]] = True
    return M, len(Dd)
DISP, N_DISP_EVENTS = _disp()


def _xrules():
    DL = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'delist_official.csv', dtype=str)
    d4 = DL[DL['four'] == 'True']
    lt = np.where(TR.any(0), len(IDX) - 1 - np.argmax(TR[::-1], 0), -1)
    until = {}
    for c, dd in zip(d4.code, pd.to_datetime(d4.date)):
        j = CI.get(c)
        if j is None or c in DELIST_TP or lt[j] < 0 or IDX[lt[j]] <= dd: continue
        w = np.where(TR[:, j])[0]; a = w[IDX[w] <= dd]; b = w[IDX[w] > dd]
        if len(a) == 0 or len(b) == 0: continue
        lt[j] = a[-1]; until[j] = IDX[b[0]]
    E = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype={'code': str, 'successor': str})
    for c in ('delist_date', 'pay_date'): E[c] = pd.to_datetime(E[c], errors='coerce')
    E = E.set_index('code'); out = {}; mcash = {}
    for j, c in enumerate(COLS):
        if lt[j] < 0 or IDX[lt[j]] > IDX[-1] - pd.Timedelta(days=30): continue
        ltd = IDX[lt[j]]; lc = C[lt[j], j]
        uu = int(np.searchsorted(IDX, until[j])) if j in until else 10 ** 9
        if c in DELIST_TP:
            out[j] = (DELIST_F, int(np.searchsorted(IDX, ltd + pd.Timedelta(days=14))), uu); continue
        if c in E.index:
            r = E.loc[c]; dd = r['delist_date'] if pd.notna(r['delist_date']) else ltd + pd.Timedelta(days=14)
            if r['type'] == 'cash' and pd.notna(r['cash']):
                f, rel = float(r['cash']) / lc, (r['pay_date'] if pd.notna(r['pay_date']) else dd + pd.Timedelta(days=14))
            elif r['type'] == 'failure': f, rel = 0.0, ltd
            else: f, rel = 1.0, dd
            if r['type'] == 'cash':
                fc = float(r['cash']) / lc if pd.notna(r['cash']) else 1.0
                di = int(np.searchsorted(IDX, dd))
                mcash[j] = (fc, di, min(di + 20, len(IDX) - 1), uu)
        else: f, rel = 1.0, ltd + pd.Timedelta(days=14)
        out[j] = (f, int(np.searchsorted(IDX, rel)), uu)
    return out, mcash, lt
XR, MCASH, LASTT = _xrules()


def cpi_scale(year):
    y = str(min(max(int(year), 2006), 2026))
    return CPI[y] / CPI['2026']

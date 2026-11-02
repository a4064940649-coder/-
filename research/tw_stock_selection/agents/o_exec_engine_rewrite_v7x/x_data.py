#!/usr/bin/env python3
"""V7X 資料層（獨立重寫；不 import 研究中心任何引擎程式，也沒有讀 ecore7.py／run_exec7*.py／gen_model7.py）。
只讀：V7 凍結面板 data/*.parquet、P 下市價、X 併購對價、終止上市名單、V4 官方處置名單（成交日口徑）、
model_spec7.json、cpi_tw.json、0050 含息。第一次執行把衍生陣列存成 data/x_cache.npz。"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
O7 = AG / 'o_v7_listing_exec_engine'
PNL = O7 / 'data'
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SPEC7 = json.loads((PNL / 'model_spec7.json').read_text())
CPI = json.loads((O7 / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']
CACHE = HERE / 'data' / 'x_cache.npz'
ODD_SWITCH = pd.Timestamp('2020-10-26')
FEEMIX_SWITCH = pd.Timestamp('2015-01-01')
LIM_SWITCH = pd.Timestamp('2015-06-01')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def _build():
    C = pd.read_parquet(PNL / 'close_raw.parquet')
    IDX, COLS = C.index, C.columns
    N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
    ld = lambda k: pd.read_parquet(PNL / f'{k}.parquet').reindex(index=IDX, columns=COLS).to_numpy().astype(float).copy()
    Cn = C.to_numpy().astype(float).copy(); On = ld('open'); Vn = ld('value'); Rn = ld('ret'); Fn = ld('flag')
    # P：下市價補值（只補面板缺值格）
    P = pd.read_parquet(AG / 'o_infra_v3_rebuild' / 'delist_prices.parquet')
    ii = IDX.get_indexer(pd.DatetimeIndex(P['date'])); jj = np.array([CI.get(c, -1) for c in P['code']])
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
    GOOD = TR.sum(1) >= 300
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    Vv = Vn[g]
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    V60g = np.where(mV, v60, np.nan)
    s60 = pd.DataFrame(Rn[g]).rolling(60, min_periods=40).std().ffill().to_numpy()
    V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan)
    V60[g] = V60g; SIG[g] = s60
    V60 = pd.DataFrame(V60).ffill(limit=3).to_numpy()   # 非 GOOD 日沿用前值（只有零星幾天）
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    # vrank：V60≥1e7 且收盤≥10 的上市股，依 V60 由大到小
    el = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(CF) & (CF >= 10)
    VR = np.full((len(IDX), N), np.nan)
    for t in range(len(IDX)):
        w = np.where(el[t])[0]
        if len(w):
            o = w[np.argsort(-V60[t, w], kind='stable')]
            VR[t, o] = np.arange(1, len(o) + 1)
    # t−1 版本
    sh = lambda A: np.vstack([np.full((1, N), np.nan), A[:-1]])
    V60p, SIGp, VRp = sh(V60), sh(SIG), sh(VR)
    with np.errstate(invalid='ignore', divide='ignore'):
        REF = np.where(Fn == 1, PC, Cn / G)    # 漲跌停參考價（與零衝擊漲停遞補口徑相同）
    E50r = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    E50 = E50r.reindex(IDX.union(E50r.index)).ffill().reindex(IDX).to_numpy()
    np.savez_compressed(CACHE, IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str), C=Cn, O=On, V=Vn, R=Rn, F=Fn,
                        G=G, CF=CF, PC=PC, ADJ=ADJ, V60p=V60p, SIGp=SIGp, VRp=VRp, REF=REF, E50=E50, GOOD=GOOD, V60=V60)


if not CACHE.exists():
    _build()
_Z = np.load(CACHE)
IDX = pd.DatetimeIndex(_Z['IDX']); COLS = list(_Z['COLS']); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
C, O, V, R, F, G, CF, PC, ADJ = (_Z[k] for k in ('C', 'O', 'V', 'R', 'F', 'G', 'CF', 'PC', 'ADJ'))
V60p, SIGp, VRp, REF, E50, GOOD, V60 = (_Z[k] for k in ('V60p', 'SIGp', 'VRp', 'REF', 'E50', 'GOOD', 'V60'))
TR = np.isfinite(C)
TPOS = {d: i for i, d in enumerate(IDX)}
DATES = IDX.values

# ---------- 處置（V4 凍結官方名單：rejudge in_panel ＋ disposal_5y 補缺）----------
def _disp():
    A = pd.read_csv(AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    A = A[A['in_panel'] == True]
    B = pd.read_csv(EXT / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    B = B[B.code.isin(set(COLS))]
    ka = set(zip(A.code, A.start))
    B = B[[(c, s) not in ka for c, s in zip(B.code, B.start)]]
    D = pd.concat([A[['code', 'start', 'end']], B[['code', 'start', 'end']]])
    D = D[D.code.isin(CI)].dropna()
    M = np.zeros((len(IDX), N), bool)
    for c, s, e in zip(D.code, D.start, D.end):
        a, b = np.searchsorted(DATES, np.datetime64(s)), np.searchsorted(DATES, np.datetime64(e), side='right')
        M[a:b, CI[c]] = True
    return M, len(D)
DISP, N_DISP_EVENTS = _disp()

# ---------- X 出清規則與現金對價併購 ----------
def _xrules():
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
    E = pd.read_csv(AG / 'o_infra_v3_rebuild' / 'merger_consideration.csv', dtype={'code': str, 'successor': str})
    for c in ('delist_date', 'pay_date'): E[c] = pd.to_datetime(E[c], errors='coerce')
    E = E.set_index('code'); out = {}; mcash = {}
    for j, c in enumerate(COLS):
        if lt[j] < 0 or IDX[lt[j]] > IDX[-1] - pd.Timedelta(days=30): continue
        ltd = IDX[lt[j]]; lc = C[lt[j], j]
        uu = int(np.searchsorted(IDX, until[j])) if j in until else 10 ** 9
        if c in E.index:
            r = E.loc[c]; dd = r['delist_date'] if pd.notna(r['delist_date']) else ltd + pd.Timedelta(days=14)
            if r['type'] == 'cash' and pd.notna(r['cash']):
                f, rel = float(r['cash']) / lc, (r['pay_date'] if pd.notna(r['pay_date']) else dd + pd.Timedelta(days=14))
            elif r['type'] == 'failure': f, rel = 0.0, ltd
            else: f, rel = 1.0, dd
            if r['type'] == 'cash':
                fc = float(r['cash']) / lc if pd.notna(r['cash']) else 1.0
                di = int(np.searchsorted(IDX, dd))
                mcash[j] = (fc, di, min(di + 20, len(IDX) - 1), uu)   # 下市日、撥款日＝下市日＋20 個交易日
        else: f, rel = 1.0, ltd + pd.Timedelta(days=14)
        out[j] = (f, int(np.searchsorted(IDX, rel)), uu)
    return out, mcash
XR, MCASH = _xrules()

def cpi_scale(year):
    y = str(min(max(int(year), 2006), 2026))
    return CPI[y] / CPI['2026']

def plans(strat='AB_V7_B10', js=None, seeds=None, official=False):
    """回傳 {(j, seed): [(exec_idx, picks_idx_array, signal_idx), ...]}；official=True 用 plans_ab7（j='OFF'）。"""
    if official:
        d = pd.read_csv(PNL / 'plans_ab7.csv', dtype={'picks': str}); d['j'] = 'OFF'
    else:
        d = pd.read_csv(PNL / 'plans_j7.csv', dtype={'picks': str})
    d = d[d.strat == strat]
    if js is not None: d = d[d.j.isin(js)]
    if seeds is not None: d = d[d.seed.isin(seeds)]
    out = {}
    for (j, s), g in d.groupby(['j', 'seed'], sort=False):
        L = []
        for sig, e, pk in zip(g.signal, g.exec_idx, g.picks):
            L.append((int(e), np.array([CI[c] for c in pk.split(',')], int), TPOS[pd.Timestamp(sig)]))
        out[(j, int(s))] = sorted(L, key=lambda x: x[0])
    return out

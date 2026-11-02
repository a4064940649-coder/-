#!/usr/bin/env python3
"""V7X 引擎的資料層替身（上市＋上櫃合併面板）。介面與 o_exec_engine_rewrite_v7x/x_data.py 相同，
讓 x_exec.py（只 import、不改）直接在合併面板上模擬。上市欄位的衍生量口徑與 V7X 逐條相同；上櫃欄位另加：
  - 上櫃整體缺漏日（上櫃有值檔數 < 21 日中位數 85%）視為豁免日（V60／52 週窗內缺值不擋）；
  - 流動性名次 VR：上市＝上市合格股內名次（同 V7X）；上櫃＝「放進上市名次表會排第幾」（上市等價名次），
    使開盤量占比、零股上限等十分位參數沿用 V7X 校準值；
  - 上櫃下櫃股（非轉上市、最後交易日早於面板終點 30 天）出清值＝最後收盤×OTPX_DELIST_F（環境變數，預設 1.0；敏感度 0.5／0.0）。
衍生陣列快取於 scratchpad/o_tpex_ext/x_cache.npz（面板 sha 不同會重建）。"""
import json, hashlib, os
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
O7 = AG / 'o_v7_listing_exec_engine'
PNL7 = O7 / 'data'
PNL = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o_tpex_ext/panel')
EXT = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SPEC7 = json.loads((PNL7 / 'model_spec7.json').read_text())
CPI = json.loads((O7 / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']
CACHE = PNL.parent / 'x_cache.npz'
ODD_SWITCH = pd.Timestamp('2020-10-26')
FEEMIX_SWITCH = pd.Timestamp('2015-01-01')
LIM_SWITCH = pd.Timestamp('2015-06-01')
DELIST_F = float(os.environ.get('OTPX_DELIST_F', '1.0'))
EXEMPT_TW = ['2017-08-03', '2018-07-12']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def _roll_ok(isnan_g, exempt_g, w):
    """GOOD 日序列上：窗內缺值只出現在豁免日 → True。"""
    bad = (isnan_g & ~exempt_g).astype(np.int16)
    return pd.DataFrame(bad).rolling(w, min_periods=w).sum().to_numpy() == 0


def _build(key):
    C = pd.read_parquet(PNL / 'close_raw.parquet')
    IDX, COLS = C.index, C.columns
    N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
    ld = lambda k: pd.read_parquet(PNL / f'{k}.parquet').reindex(index=IDX, columns=COLS).to_numpy().astype(float).copy()
    Cn = C.to_numpy().astype(float).copy(); On = ld('open'); Vn = ld('value'); Rn = ld('ret'); Fn = ld('flag'); Hn = ld('high')
    MK = pd.read_parquet(PNL / 'mkt_tpex.parquet').reindex(index=IDX, columns=COLS).fillna(False).to_numpy().astype(bool)
    tp_any = MK.any(0)
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
    # GOOD：上市有值檔數 ≥ 300（與 V7X 相同；上櫃欄位不影響）
    TW = ~tp_any
    TWc = TR & ~MK
    GOOD = TWc.sum(1) >= 300
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex_tw = dg.isin(pd.DatetimeIndex(EXEMPT_TW)) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    # 上櫃整體缺漏日
    tpc = (TR & MK).sum(1).astype(float)
    med = pd.Series(tpc).rolling(21, min_periods=5, center=True).median().to_numpy()
    tp_gap = (tpc < 0.85 * med) | (tpc < 100)
    gm = json.loads((HERE / 'data' / 'gap_map.json').read_text())
    lowm = {k for k, v in gm.items() if v.get('coverage') is not None and v['coverage'] < 0.90}
    tp_gap |= np.array([d.strftime('%Y-%m') in lowm for d in IDX])      # 覆蓋 < 90% 的月份整月豁免
    tp_gap &= IDX >= '2007-04-23'
    ex_tp = ex_tw | tp_gap[g]
    EX = np.where(MK[g], ex_tp[:, None], ex_tw[:, None])     # 每格豁免（依該格市場別）
    Vv = Vn[g]
    okV = _roll_ok(np.isnan(Vv), EX, 60)
    cntV = pd.DataFrame(np.isfinite(Vv).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy()
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    V60g = np.where(okV & (cntV >= 20), v60, np.nan)
    s60 = pd.DataFrame(Rn[g]).rolling(60, min_periods=40).std().ffill().to_numpy()
    # 52 週（252 個 GOOD 日）高點比：含息累積指數，窗內缺值只能在豁免日，且至少 60 筆
    AI = np.cumprod(G, axis=0); AIg = np.where(TR[g], AI[g], np.nan)
    ok52 = _roll_ok(np.isnan(AIg), EX, 252)
    cnt52 = pd.DataFrame(np.isfinite(AIg).astype(np.int16)).rolling(252, min_periods=252).sum().to_numpy()
    mx = pd.DataFrame(AIg).rolling(252, min_periods=1).max().to_numpy()
    S52g = np.where(ok52 & (cnt52 >= 60) & np.isfinite(AIg), AIg / mx, np.nan)
    V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan); S52 = np.full((len(IDX), N), np.nan)
    V60[g] = V60g; SIG[g] = s60; S52[g] = S52g
    V60 = pd.DataFrame(V60).ffill(limit=3).to_numpy()
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    # VR：上市＝上市合格股名次；上櫃＝上市等價名次
    el = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(CF) & (CF >= 10)
    VR = np.full((len(IDX), N), np.nan)
    for t in range(len(IDX)):
        w = np.where(el[t] & ~MK[t])[0]
        if len(w):
            o = w[np.argsort(-V60[t, w], kind='stable')]
            VR[t, o] = np.arange(1, len(o) + 1)
            sv = np.sort(V60[t, w])
            wt = np.where(el[t] & MK[t])[0]
            if len(wt):
                VR[t, wt] = 1 + (len(sv) - np.searchsorted(sv, V60[t, wt], side='right'))
    sh = lambda A: np.vstack([np.full((1, N), np.nan), A[:-1]])
    V60p, SIGp, VRp = sh(V60), sh(SIG), sh(VR)
    with np.errstate(invalid='ignore', divide='ignore'):
        REF = np.where(Fn == 1, PC, Cn / G)
    E50r = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
    E50 = E50r.reindex(IDX.union(E50r.index)).ffill().reindex(IDX).to_numpy()
    np.savez(CACHE, KEY=np.array(key), IDX=IDX.values.astype('datetime64[ns]'), COLS=np.array(COLS, dtype=str), C=Cn, O=On, V=Vn, R=Rn, F=Fn, H=Hn,
             G=G, CF=CF, PC=PC, ADJ=ADJ, V60p=V60p, SIGp=SIGp, VRp=VRp, REF=REF, E50=E50, GOOD=GOOD, V60=V60, MK=MK, S52=S52, AI=AI, TPGAP=tp_gap)


_KEY = sha(PNL / 'close_raw.parquet')[:16] + sha(PNL / 'value.parquet')[:16] + sha(PNL / 'ret.parquet')[:16] + 'v5'
if not CACHE.exists() or str(np.load(CACHE)['KEY']) != _KEY:
    _build(_KEY)
_Z = np.load(CACHE, mmap_mode='r')
IDX = pd.DatetimeIndex(_Z['IDX']); COLS = list(_Z['COLS']); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
C, O, V, R, F, G, CF, PC, ADJ, H = (np.asarray(_Z[k]) for k in ('C', 'O', 'V', 'R', 'F', 'G', 'CF', 'PC', 'ADJ', 'H'))
V60p, SIGp, VRp, REF, E50, GOOD, V60, MK, S52, AI, TPGAP = (np.asarray(_Z[k]) for k in ('V60p', 'SIGp', 'VRp', 'REF', 'E50', 'GOOD', 'V60', 'MK', 'S52', 'AI', 'TPGAP'))
TR = np.isfinite(C)
TPOS = {d: i for i, d in enumerate(IDX)}
DATES = IDX.values
TPEX_EVER = MK.any(0)


# ---------- 處置（同 V7X：V4 凍結官方名單＋disposal_5y；disposal_5y 已含上櫃 2021-09 起）----------
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
                mcash[j] = (fc, di, min(di + 20, len(IDX) - 1), uu)
        else:
            f, rel = 1.0, ltd + pd.Timedelta(days=14)
            if MK[lt[j], j]: f = DELIST_F            # 上櫃下櫃股敏感度
        out[j] = (f, int(np.searchsorted(IDX, rel)), uu)
    return out, mcash
XR, MCASH = _xrules()


def cpi_scale(year):
    y = str(min(max(int(year), 2006), 2026))
    return CPI[y] / CPI['2026']

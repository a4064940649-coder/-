"""前瞻引擎：由截斷資料建立 x_data 替身，逐字載入 V7X x_exec.py。"""
import sys, json, types
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_forward_infra_live/replicate_o9')
import numpy as np, pandas as pd
import rlib as L

V7X = L.ROOT / 'agents/o_exec_engine_rewrite_v7x'
SPEC7 = json.loads((L.ROOT / 'agents/o_v7_listing_exec_engine/data/model_spec7.json').read_text())


def build_D(asof):
    M = L.mp_data(asof); d = M['data']
    IDX = d['cal']; COLS = list(d['C'].columns); N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
    C = d['C'].to_numpy(float).copy(); O = d['O'].reindex_like(d['C']).to_numpy(float).copy()
    V = d['V'].reindex_like(d['C']).to_numpy(float).copy(); R = d['R'].reindex_like(d['C']).to_numpy(float).copy()
    TR = np.isfinite(C)
    F = (TR & ~np.isfinite(R)).astype(float)
    G = np.where(TR, 1.0 + np.nan_to_num(R, nan=0.0), 1.0)
    CF = pd.DataFrame(C).ffill().to_numpy()
    PC = pd.DataFrame(C).ffill().shift(1).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR, G * PC / C, 1.0)
    ADJ = np.where(np.isfinite(ADJ), ADJ, 1.0)
    GOOD = TR.sum(1) >= 300
    g = np.where(GOOD)[0]; dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    Vv = V[g]
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    V60 = np.full((len(IDX), N), np.nan); SIG = np.full((len(IDX), N), np.nan)
    V60[g] = np.where(mV, v60, np.nan)
    SIG[g] = pd.DataFrame(R[g]).rolling(60, min_periods=40).std().ffill().to_numpy()
    V60 = pd.DataFrame(V60).ffill(limit=3).to_numpy(); SIG = pd.DataFrame(SIG).ffill().to_numpy()
    el = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(CF) & (CF >= 10)
    VR = np.full((len(IDX), N), np.nan)
    for t in range(len(IDX)):
        w = np.where(el[t])[0]
        if len(w):
            o = w[np.argsort(-V60[t, w], kind='stable')]; VR[t, o] = np.arange(1, len(o) + 1)
    sh = lambda A: np.vstack([np.full((1, N), np.nan), A[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        REF = np.where(F == 1, PC, C / G)
    e = L.E50[L.E50.index <= pd.Timestamp(asof)]
    E50 = e.reindex(IDX.union(e.index)).ffill().reindex(IDX).to_numpy()
    DISP = np.zeros((len(IDX), N), bool); DATES = IDX.values
    ds = M['dsp']
    for c, s, en in zip(ds.code, ds.start, ds.end):
        if c in CI:
            a, b = np.searchsorted(DATES, np.datetime64(s)), np.searchsorted(DATES, np.datetime64(en), side='right')
            DISP[a:b, CI[c]] = True
    m = types.ModuleType('x_data')
    m.__dict__.update(SPEC7=SPEC7, IDX=IDX, COLS=COLS, N=N, CI=CI, C=C, O=O, V=V, R=R, F=F, G=G, CF=CF, PC=PC, ADJ=ADJ,
                      V60p=sh(V60), SIGp=sh(SIG), VRp=sh(VR), REF=REF, E50=E50, GOOD=GOOD, V60=V60, TR=TR,
                      TPOS={dd: i for i, dd in enumerate(IDX)}, DATES=DATES, DISP=DISP, XR={}, MCASH={},
                      ODD_SWITCH=pd.Timestamp('2020-10-26'), FEEMIX_SWITCH=pd.Timestamp('2015-01-01'),
                      LIM_SWITCH=pd.Timestamp('2015-06-01'), tick=L.MP.tick, cpi_scale=lambda y: 1.0)
    return m, M


def load_engine(Dm):
    sys.modules['x_data'] = Dm
    return L.load_mod('x_exec_fwd', V7X / 'x_exec.py')


BASE_CFG = dict(tier=1e6, mode='GROW', model='K5T', cost='normal', f1=True, f2=True, f3=0.5, f4=True,
                rank_right=True, stress_odd_extra=True)

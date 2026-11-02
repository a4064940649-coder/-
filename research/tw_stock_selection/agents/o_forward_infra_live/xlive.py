#!/usr/bin/env python3
"""V7X 可執行引擎（x_exec.py，原檔只讀、逐字載入）接到前瞻資料（截斷到 asof 的 history.csv.gz）。

作法：用與 x_data.py 相同的屬性名稱建一個資料模組（C、O、V、R、F、G、CF、PC、ADJ、V60p、SIGp、VRp、REF、E50、DISP…），
放進 sys.modules['x_data'] 後載入 x_exec.py，引擎程式一行不改；成本參數一律來自 model_spec7.json。
與 x_data.py 的差異（前瞻資料的必然差異，A3 以 2026-09／10 兩窗量化）：
- 成交值 V＝成交股數×收盤（history.csv.gz 沒有成交金額；研究面板拼接段用面板 value）。
- 旗標 F：history 沒有公司行動旗標；收盤有值但日報酬缺值的格視為 F＝1（漲跌停參考價用前收）。
- 處置名單：截斷資料夾的 disposal_5y（disposal_5y ∪ warnings_hist），處置期間＝start ≤ t ≤ end。
- 下市出清（XR）與現金對價併購（MCASH）：前瞻期間沒有已知事件，設為空；持股停牌時依引擎原規則凍結、以前收計值。
- cpi_scale 一律 1（2026 幣值帳戶；GROW 起始現金＝級距）。
"""
import importlib.util, json, sys, types, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
V7X = AG / 'o_exec_engine_rewrite_v7x'
XEXEC = V7X / 'x_exec.py'
SPEC7_PATH = AG / 'o_v7_listing_exec_engine' / 'data' / 'model_spec7.json'
CFG_V7X = dict(tier=1_000_000, mode='GROW', model='K5T', cost='normal', f1=True, f2=True, f3=0.5, f4=True,
               rank_right=True, stress_odd_extra=True, disp_lim=False)


def tick(p):
    p = np.asarray(p, float)
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def build_module(data, dsp, e50, start=None):
    """data：main_picks_today.load_prices 的回傳（已截斷到 asof）；dsp：處置名單；e50：0050 含息 Series。"""
    m = types.ModuleType('x_data_live')
    C = data['C']; idx = C.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    cols = list(C.columns)
    g = lambda k: data[k].reindex(index=idx, columns=cols).to_numpy(float).copy()
    Cn, On, Vn, Rn = g('C'), g('O'), g('V'), g('R')
    N = len(cols)
    TR = np.isfinite(Cn)
    Fn = np.where(TR & ~np.isfinite(Rn), 1.0, 0.0)
    G = np.where(TR, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
    CF = pd.DataFrame(Cn).ffill().to_numpy()
    PC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ADJ = np.where(TR, G * PC / Cn, 1.0)
    ADJ = np.where(np.isfinite(ADJ), ADJ, 1.0)
    GOOD = TR.sum(1) >= 300
    gi = np.where(GOOD)[0]; dg = idx[gi]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    Vv = Vn[gi]
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy() == 0
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    V60g = np.where(mV, v60, np.nan)
    s60 = pd.DataFrame(Rn[gi]).rolling(60, min_periods=40).std().ffill().to_numpy()
    V60 = np.full((len(idx), N), np.nan); SIG = np.full((len(idx), N), np.nan)
    V60[gi] = V60g; SIG[gi] = s60
    V60 = pd.DataFrame(V60).ffill(limit=3).to_numpy()
    SIG = pd.DataFrame(SIG).ffill().to_numpy()
    el = np.isfinite(V60) & (V60 >= 1e7) & np.isfinite(CF) & (CF >= 10)
    VR = np.full((len(idx), N), np.nan)
    for t in range(len(idx)):
        w = np.where(el[t])[0]
        if len(w):
            o = w[np.argsort(-V60[t, w], kind='stable')]
            VR[t, o] = np.arange(1, len(o) + 1)
    sh = lambda A: np.vstack([np.full((1, N), np.nan), A[:-1]])
    with np.errstate(invalid='ignore', divide='ignore'):
        REF = np.where(Fn == 1, PC, Cn / G)
    E50 = e50.reindex(idx.union(e50.index)).ffill().reindex(idx).to_numpy(float)
    CI = {c: j for j, c in enumerate(cols)}
    DISP = np.zeros((len(idx), N), bool)
    DATES = idx.values
    for c, s, e in zip(dsp['code'], pd.to_datetime(dsp['start']), pd.to_datetime(dsp['end'])):
        j = CI.get(c)
        if j is None or pd.isna(s) or pd.isna(e): continue
        a, b = np.searchsorted(DATES, np.datetime64(s)), np.searchsorted(DATES, np.datetime64(e), side='right')
        DISP[a:b, j] = True
    m.__dict__.update(
        SPEC7=json.loads(SPEC7_PATH.read_text()), ODD_SWITCH=pd.Timestamp('2020-10-26'), FEEMIX_SWITCH=pd.Timestamp('2015-01-01'),
        LIM_SWITCH=pd.Timestamp('2015-06-01'), tick=tick, IDX=idx, COLS=cols, N=N, CI=CI, C=Cn, O=On, V=Vn, R=Rn, F=Fn, G=G, CF=CF, PC=PC,
        ADJ=ADJ, V60p=sh(V60), SIGp=sh(SIG), VRp=sh(VR), REF=REF, E50=E50, GOOD=GOOD, V60=V60, TR=TR,
        TPOS={d: i for i, d in enumerate(idx)}, DATES=DATES, DISP=DISP, N_DISP_EVENTS=len(dsp), XR={}, MCASH={},
        cpi_scale=lambda y: 1.0)
    return m


def load_engine(dmod, name='x_exec_live'):
    """以 dmod 當 x_data 逐字載入 x_exec.py（不改原檔）。"""
    old = sys.modules.get('x_data')
    sys.modules['x_data'] = dmod
    try:
        spec = importlib.util.spec_from_file_location(name, XEXEC)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        if old is None: sys.modules.pop('x_data', None)
        else: sys.modules['x_data'] = old
    return mod


def logging_sim(XE):
    """在 x_exec.Sim 外面包一層，只記錄每筆成交（不改任何成交或成本計算）。"""
    D = XE.D

    class LSim(XE.Sim):
        def __init__(self, pl, cfg):
            super().__init__(pl, cfg); self.fills = []

        def _log(self, t, j, side, n, px, kind):
            o = D.O[t, j]; tk = float(tick(o)) if np.isfinite(o) else np.nan
            dec = int(XE.DEC[t, j]); v60 = D.V60p[t, j]
            oa_pred = XE.OASH[t, j] * v60 if np.isfinite(v60) else np.nan
            oa_act = XE.OASH[t, j] * D.V[t, j] if np.isfinite(D.V[t, j]) else np.nan
            q = n * D.PC[t, j]
            self.fills.append(dict(t=t, date=str(D.IDX[t].date()), code=D.COLS[j], side='buy' if side > 0 else 'sell', kind=kind,
                                   is_odd=int(kind == 'odd'), shares=int(n), fill_px=float(px), open_px=float(o), prev_close=float(D.PC[t, j]),
                                   close_px=float(D.C[t, j]), ticks_from_open=(side * (px - o) / tk) if np.isfinite(tk) and tk > 0 else np.nan,
                                   sigma60_prev=float(D.SIGp[t, j]), V60_prev=float(v60), dec=dec, oa_share_dec=float(XE.OASH[t, j]),
                                   OA_pred=float(oa_pred), share_of_open_vol=float(q / oa_pred) if np.isfinite(oa_pred) and oa_pred > 0 else np.nan,
                                   share_of_open_vol_actual=float(q / oa_act) if np.isfinite(oa_act) and oa_act > 0 else np.nan,
                                   in_disposal=int(bool(D.DISP[t, j]))))

        def fill_whole(self, t, j, L, side):
            n = super().fill_whole(t, j, L, side)
            if n:
                im = self.imp(t, j, n // 1000)
                self._log(t, j, side, n, D.O[t, j] * (1 + side * im), 'whole')
            return n

        def fill_odd(self, t, j, q, side, cashlim=None):
            n = super().fill_odd(t, j, q, side, cashlim)
            if n:
                self._log(t, j, side, n, self.oddpx(t, j) * (1 + side * self.oddc(t, j, side)), 'odd')
            return n

        def fill_disp_sell(self, t, j, n):
            got = super().fill_disp_sell(t, j, n)
            if got:
                self._log(t, j, -1, got, D.C[t, j] - XE.TKC[t, j], 'disp_close')
            return got
    return LSim


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

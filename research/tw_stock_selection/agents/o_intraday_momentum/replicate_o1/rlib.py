#!/usr/bin/env python3
"""replicate_o1 共用：獨立複製員自寫。
只讀 import main_v2/v2_core（中心正式版主策略名單定義：股池、分數、同分、跳動點漲停），
成交引擎、盤中/隔夜拆分、統計全部自寫（沒讀 o_intraday_momentum 研究員的程式）。"""
import sys, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
MV2 = HERE.parent.parent / 'main_v2'
sys.path.insert(0, str(MV2))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
Cn, On, Rn, Fn = K.Cn, K.On, K.Rn, K.Fn
TR = K.TRADED
G = K.GROW
SRCn = K.SRC.reindex(index=IDX, columns=COLS).to_numpy()
Vn = K.V.to_numpy()
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253')

FEE = 0.001425 * 0.6; TAX = 0.003
COST = {'normal': (FEE + 0.001, FEE + TAX + 0.001), 'stress': (0.005, 0.005), 'zero': (0.0, 0.0)}
SEL_END = pd.Timestamp('2019-05-13')     # 2019-05 換股日
END = V.END1012

_cache = {}
def main_strategy():
    if 'S' not in _cache:
        YL, Y5, _ = V.load_revenue()
        S = V.V2('gapfix', YL, Y5, exfin=True)
        _cache['S'] = S
    return _cache['S']

def plan(n=10):
    k = f'pl{n}'
    if k not in _cache: _cache[k] = main_strategy().plan(n, 'official')
    return _cache[k]

# ---------------- 跳動點漲停（自寫） ----------------
def _tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
PREVC = pd.DataFrame(Cn).ffill().shift(1).to_numpy()
def limup(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = Cn[t] / (1 + Rn[t]); ref = np.where(Fn[t] == 1, PREVC[t], ref)
    raw = ref * (1 + lim); tk = _tick(raw)
    return np.floor(raw / tk + 1e-9) * tk
def close_limit_up(t):
    return np.isfinite(Cn[t]) & (Cn[t] >= limup(t) - 1e-6)

# ---------------- 成交引擎（自寫） ----------------
def sim(pl, t0, t1, cost='normal', buy_px='open', cont='rebal', lu_next=False, skip_t=None):
    """pl=[(exec_t, picks, ...)]。換股日：持股前收→開盤用 G×O/C；賣出一律開盤。
    buy_px='open'：在開盤一次解等權目標（含成本，二分法）。
    buy_px='close'：新進股改在收盤買；賣出與續抱的處理見 cont：
       'rebal'＝續抱照原版在開盤調到目標 T，新進股拿原目標金額 T(1+b)，現金開盤→收盤不動，收盤買入
       'hold' ＝續抱不動，賣出淨現金平均分給新進股，收盤買
    lu_next：收盤漲停的新進股改在隔一日開盤買。skip_t：這些換股日一律開盤成交。"""
    b, s = COST[cost]
    reb = {p[0]: np.asarray(p[1], int) for p in pl}
    val = np.zeros(N); cash = 1.0; pend = None
    nav = np.empty(t1 - t0 + 1)
    held = np.zeros(N, bool)
    for t in range(t0, t1 + 1):
        if pend is not None:     # 前一日收盤漲停的新進股：今日開盤買
            amt, js = pend; pend = None
            okj = js[TR[t, js] & np.isfinite(On[t, js])]
            val = val * np.where(TR[t], G[t], 1.0)
            if len(okj):
                val[okj] += amt / len(okj) / (1 + b) * np.where(True, Cn[t, okj] / On[t, okj], 1)
                cash -= amt
            nav[t - t0] = val.sum() + cash
            continue
        if t in reb:
            pk = reb[t]
            tr = TR[t] & np.isfinite(On[t])
            h = np.where(tr, val * G[t] * On[t] / np.where(tr, Cn[t], 1.0), val)
            frz = (h > 0) & ~tr
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            def need(T):
                tg = np.zeros(N); tg[pk] = T
                return T * k + b * np.clip(tg - htr, 0, None).sum() + s * np.clip(htr - tg, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if need(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            mode = 'open' if (buy_px == 'open' or (skip_t is not None and t in skip_t)) else buy_px
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            inpk = np.zeros(N, bool); inpk[pk] = True
            if mode == 'open':
                nv = np.where(frz, h, 0.0); nv[pk] += T
                val = np.where(frz, nv, nv * oc); cash = 0.0
            else:
                new = inpk & ~(htr > 0)
                cont_m = inpk & (htr > 0)
                exit_m = (htr > 0) & ~inpk
                if cont == 'rebal':
                    nv = np.where(frz, h, 0.0)
                    nv[cont_m] = T
                    cash_c = Vt - (T * cont_m.sum() + b * np.clip(T - htr[cont_m], 0, None).sum() + s * np.clip(htr[cont_m] - T, 0, None).sum()
                                   + s * htr[exit_m].sum())
                    val = np.where(frz, nv, nv * oc)
                else:
                    nv = np.where(frz | cont_m, h, 0.0)
                    cash_c = cash + (1 - s) * htr[exit_m].sum()
                    val = np.where(frz, nv, nv * oc)
                js = np.where(new)[0]
                if lu_next and len(js):
                    lu = close_limit_up(t)[js]
                    jn, jl = js[~lu], js[lu]
                else:
                    jn, jl = js, js[:0]
                if len(js):
                    per = cash_c / len(js)
                    val[jn] += per / (1 + b)
                    cash = cash_c - per * len(jn)
                    if len(jl): pend = (per * len(jl), jl)
                else:
                    cash = cash_c
            held = inpk
        else:
            if t > t0: val = val * G[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

def ew_nav(pl, t0, t1):
    return sim([(p[0], p[2]) for p in pl], t0, t1, cost='zero')

# ---------------- 統計（自寫） ----------------
E50 = K.E50
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x))))

def win_rets(nav, sd, a, b):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    x = nav.reindex(D); r = (x.shift(-1) / x - 1).iloc[:-1]
    return r[(r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b))]

def seg(nav, sd, a, b, ew=None, wa=None, wb=None):
    """日頻年化/回撤：a 之後第一筆報酬到 b；訊號窗 t：窗起點在 [wa,wb]（預設＝[a,b]）。"""
    wa = a if wa is None else wa; wb = b if wb is None else wb
    r = nav.pct_change()
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m]; i0 = int(np.argmax(m)); st = r.index[i0 - 1]
    yrs = (rr.index[-1] - st).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2), 'maxDD': round(float((eq / eq.cummax() - 1).min()) * 100, 2)}
    pr = win_rets(nav, sd, wa, wb)
    br = win_rets(E50.reindex(nav.index), sd, wa, wb).reindex(pr.index)
    o['t_0050'] = round(tstat(pr - br), 2); o['n'] = int(pr.notna().sum())
    if ew is not None:
        er = win_rets(ew, sd, wa, wb).reindex(pr.index)
        o['t_EW'] = round(tstat(pr - er), 2)
    o['yearly'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    return o

def paired(nav1, nav0, sd, a, b):
    d = (win_rets(nav1, sd, a, b) - win_rets(nav0, sd, a, b)).dropna()
    return {'mean_pct': round(float(d.mean()) * 100, 3), 't': round(tstat(d), 2), 'n': int(len(d))}

# ---------------- 日拆分（自寫，照凍結規格） ----------------
def daily_split():
    if 'split' in _cache: return _cache['split']
    with np.errstate(invalid='ignore', divide='ignore'):
        ci = np.log(Cn / On)
        tot = np.log1p(Rn)
        co = tot - ci
    L = np.where((IDX < pd.Timestamp('2015-06-01'))[:, None], 0.07, 0.10)
    bad = np.isin(Fn, [3, 4, 5]) | ~np.isfinite(On) | ~np.isfinite(Cn) | ((On == Cn) & (np.nan_to_num(Vn) == 0))
    ci = np.where(bad, np.nan, ci); co = np.where(bad, np.nan, co)
    co = np.where(Fn == 1, np.nan, co)
    xi = np.abs(ci) > np.log((1 + L) / (1 - L)) + 0.005
    ci = np.where(xi, np.nan, ci); co = np.where(xi, np.nan, co)
    xo = (np.abs(co) > np.log(1 + L) + 0.02) & (Fn != 2)
    co = np.where(xo, np.nan, co)
    _cache['split'] = (ci, co)
    return ci, co

GI = np.where(K.GOOD)[0]
def roll_sum(X, look, skip, minfrac=0.8):
    """在 GOOD 日上：窗＝最近 look 個 GOOD 日中去掉最近 skip 個（look-skip 天）。回傳全日曆陣列（只 GOOD 日有值）。"""
    A = pd.DataFrame(X[GI])
    w = look - skip
    s = A.rolling(w, min_periods=1).sum().shift(skip)
    c = A.notna().rolling(w, min_periods=1).sum().shift(skip)
    full = pd.Series(np.ones(len(GI))).rolling(w, min_periods=w).sum().shift(skip).to_numpy()[:, None]
    okm = (c.to_numpy() >= minfrac * w) & np.isfinite(full)
    out = np.where(okm, s.to_numpy(), np.nan)
    R = np.full((len(IDX), N), np.nan); R[GI] = out
    M = np.zeros((len(IDX), N), bool); M[GI] = okm
    return R, M

def src3_frac(look, skip):
    x = pd.Series(((IDX[GI] >= pd.Timestamp('2024-02-15')) & (IDX[GI] <= pd.Timestamp('2025-09-23'))).astype(float))
    w = look - skip
    f = (x.rolling(w, min_periods=w).sum().shift(skip) / w).to_numpy()
    R = np.full(len(IDX), np.nan); R[GI] = f
    return R

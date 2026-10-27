#!/usr/bin/env python3
"""o_intraday_momentum 共用函式庫：盤中／隔夜拆分、IMOM／OMOM 訊號、主策略（main_v2 v2_core，只讀 import）的變體、統計。

挑選期鎖：lock_selection() 會在本行程內把 2019-05 換股日之後的所有價格／報酬／成交值／0050 設成缺值，
任何誤用驗證期資料的計算都會變成 NaN 或觸發斷言。凍結前的程式（explore.py、iterate.py、p4_sel.py）一律先呼叫它。
樣本外程式（oos.py）必須先通過 assert_prereg()。
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V          # 只讀 import（會再 import independent_audit/ia_core）
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N

# ---------------- 期間 ----------------
REB_ALL = K.rebalance_days('2004-01-01')
SEL_FIRST = pd.Timestamp('2005-01-01')
SEL_LAST_SIG = pd.Timestamp('2019-04-30')          # 挑選期最後一個訊號月 2019-04
CUT_DAY = [d for d in REB_ALL if d > SEL_LAST_SIG][0]   # 2019-05 換股日（挑選期最後一期的評價終點）
CUT = IDX.get_loc(CUT_DAY)
VAL_FIRST = pd.Timestamp('2019-05-01')
END1012 = V.END1012
SRC3 = (pd.Timestamp('2024-02-15'), pd.Timestamp('2025-09-23'))

LOCKED = {'on': False}

def lock_selection():
    """把 CUT_DAY 之後的資料全部抹成缺值（只在本行程記憶體內，不動任何檔案）。"""
    nan = np.nan
    for nm in ('Cn', 'On', 'Rn', 'Hn', 'PREVC'):
        a = getattr(K, nm).astype(float, copy=True); a[CUT + 1:] = nan; setattr(K, nm, a)
    g = K.GROW.copy(); g[CUT + 1:] = nan; K.GROW = g
    tr = K.TRADED.copy(); tr[CUT + 1:] = False; K.TRADED = tr
    for nm in ('MYADJ', 'V', 'ADJ', 'C', 'O', 'RET'):
        df = getattr(K, nm).copy(); df.iloc[CUT + 1:] = nan; setattr(K, nm, df)
    e = K.E50.copy(); e.iloc[CUT + 1:] = nan; K.E50 = e
    V._RS.clear()
    LOCKED['on'] = True

def assert_locked_ok(t1):
    if LOCKED['on']:
        assert t1 <= CUT, f'挑選期鎖：評價終點 {IDX[t1].date()} 超過 {CUT_DAY.date()}'

# ---------------- 事前登記 ----------------
def prereg_path(): return HERE / 'results.json'

def assert_prereg():
    r = json.loads(prereg_path().read_text(encoding='utf-8'))
    pr = r.get('PREREG'); sh = r.get('PREREG_sha256')
    assert pr is not None and sh, '尚未凍結：results.json 沒有 PREREG'
    calc = hashlib.sha256(json.dumps(pr, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    assert calc == sh, 'PREREG 內容與 sha256 不符'
    return pr, sh, r.get('PREREG_time')

# ---------------- 盤中／隔夜拆分 ----------------
def limits():
    return np.where(IDX < pd.Timestamp('2015-06-01'), 0.07, 0.10)

def daily_split():
    """回傳 (ID, ON, info)：日對數盤中報酬 log(close/open)、日對數隔夜報酬 log(1+ret) − log(close/open)。
    缺值：flag 3/4/5、open 或 close 缺、open==close 且成交值 0；flag 1（除權息、ret 被設 0）隔夜設缺值（盤中保留）；
    極端值：|ID| > log((1+L)/(1−L)) + 0.005 → 兩者都缺；|ON| > log(1+L) + 0.02 且不是 flag 2 → 隔夜缺。"""
    C, O, R, F = K.Cn, K.On, K.Rn, K.Fn
    Vv = K.V.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        ok = np.isfinite(C) & np.isfinite(O) & (O > 0) & (C > 0) & ~np.isin(F, [3, 4, 5])
        ok &= ~((O == C) & (np.nan_to_num(Vv, nan=0.0) == 0))
        L = limits()[:, None]
        idr = np.where(ok, np.log(C / O), np.nan)
        tot = np.where(ok & (F != 1) & np.isfinite(R) & (R > -1), np.log1p(R), np.nan)
        onr = tot - idr
        bad_id = np.abs(idr) > np.log((1 + L) / (1 - L)) + 0.005
        bad_on = (np.abs(onr) > np.log(1 + L) + 0.02) & (F != 2)
    info = {'n_ok': int(ok.sum()), 'n_bad_id': int(np.nansum(bad_id)), 'n_bad_on': int(np.nansum(bad_on)),
            'n_flag1_on_nan': int((ok & (F == 1)).sum())}
    idr = np.where(bad_id, np.nan, idr); onr = np.where(bad_id | bad_on, np.nan, onr)
    return idr, onr, info

class Split:
    """GOOD 日（有值檔數 ≥ 300）上的累積和，供任意回看窗取用。"""
    def __init__(self):
        self.ID, self.ON, self.info = daily_split()
        self.g = np.where(K.GOOD)[0]
        self.pos = {t: i for i, t in enumerate(self.g)}
        I = self.ID[self.g]; O_ = self.ON[self.g]
        z = np.zeros((1, N))
        self.cI = np.vstack([z, np.cumsum(np.nan_to_num(I), axis=0)])
        self.nI = np.vstack([z, np.cumsum(np.isfinite(I), axis=0)])
        self.cO = np.vstack([z, np.cumsum(np.nan_to_num(O_), axis=0)])
        self.nO = np.vstack([z, np.cumsum(np.isfinite(O_), axis=0)])
        gd = IDX[self.g]
        s3 = ((gd >= SRC3[0]) & (gd <= SRC3[1])).astype(float)
        self.cS3 = np.concatenate([[0.0], np.cumsum(s3)])

    def window(self, t, L, S):
        """訊號日 t（面板列號）、回看 L 個 GOOD 日、跳過最近 S 個 GOOD 日。回傳 (CI, CO, ok, src3_frac)。"""
        p = self.pos[t]
        hi = p - S + 1; lo = hi - L          # 累積和索引：rows [lo, hi) of g
        if lo < 0: return np.full(N, np.nan), np.full(N, np.nan), np.zeros(N, bool), np.nan
        CI = self.cI[hi] - self.cI[lo]; CO = self.cO[hi] - self.cO[lo]
        ok = ((self.nI[hi] - self.nI[lo]) >= 0.8 * L) & ((self.nO[hi] - self.nO[lo]) >= 0.8 * L)
        f3 = (self.cS3[hi] - self.cS3[lo]) / L
        return np.where(ok, CI, np.nan), np.where(ok, CO, np.nan), ok, float(f3)

def form_signal(CI, CO, form):
    """X 越大＝越「盤中型」。cum＝累積盤中；share＝盤中占比（總報酬 ≤ 0 → NaN，結合時給中位百分位）；diff＝盤中−隔夜。"""
    if form == 'cum': return CI.copy()
    if form == 'diff': return CI - CO
    if form == 'share':
        T = CI + CO
        with np.errstate(invalid='ignore', divide='ignore'):
            s = np.clip(CI / T, -1.0, 2.0)
        return np.where(T > 0, s, np.nan)
    raise ValueError(form)

# ---------------- 股池（不要求營收，P1 用）----------------
def base_pool(t, S52, V60, exfin=True):
    e = t + 1
    v60 = V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, int(base.sum()))]] = True
    top &= base
    rank = np.full(N, np.nan); rank[order[:int(base.sum())]] = np.arange(1, int(base.sum()) + 1)
    el = top & np.isfinite(S52[t]) & ~K.disposal_mask(IDX[t])
    if exfin: el &= ~V.FINMASK
    if e < len(IDX): el &= np.isfinite(K.On[e]) & K.TRADED[e]
    return el, rank

# ---------------- 可成交持有期報酬（隔日開盤 → 下一換股隔日開盤）----------------
def hold_returns(e, e2):
    """e 開盤買、e2 開盤賣（e2 沒有開盤成交→用到 e2 的收盤值近似，即凍結近似）。毛報酬，不含成本。"""
    r = K.Cn[e] / K.On[e]
    if e2 > e:
        r = r * np.prod(K.GROW[e + 1:e2 + 1], axis=0)
        tr = K.TRADED[e2] & np.isfinite(K.On[e2])
        with np.errstate(invalid='ignore', divide='ignore'):
            r = np.where(tr, r * K.On[e2] / np.where(tr, K.Cn[e2], 1.0), r)
    return r - 1.0

def split_future(e, e2, sp):
    """持有期 (e, e2] 的盤中、隔夜對數報酬合計（e 當日只算盤中；e2 當日只算隔夜）——機制分析用。"""
    I = np.nansum(sp.ID[e:e2], axis=0)
    O_ = np.nansum(sp.ON[e + 1:e2 + 1], axis=0)
    return I, O_

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def nw_t(x, lag=6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 10: return float('nan')
    m = x.mean(); u = x - m
    v = (u @ u) / n
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); v += 2 * w * (u[l:] @ u[:-l]) / n
    return float(m / math.sqrt(v / n)) if v > 0 else float('nan')

def sig_window_returns(nav, sig_days):
    """訊號日收盤 → 下一訊號日收盤的報酬（index＝起點訊號日）。"""
    D = pd.DatetimeIndex([d for d in sig_days if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def paired(navA, navB, sig_days, a, b, mask_days=None):
    ra = sig_window_returns(navA, sig_days); rb = sig_window_returns(navB, sig_days).reindex(ra.index)
    d = (ra - rb)
    d = d[(d.index >= pd.Timestamp(a)) & (d.index <= pd.Timestamp(b))]
    if mask_days is not None:
        d = d[~d.index.isin(pd.DatetimeIndex(mask_days))]
    d = d.dropna()
    return {'n': int(len(d)), 'mean_pct': float(d.mean() * 100) if len(d) else float('nan'), 't': tstat(d.to_numpy()),
            't_nw6': nw_t(d.to_numpy()), 'win_rate': float((d > 0).mean()) if len(d) else float('nan')}, d

def jdump(o):
    def f(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (pd.Timestamp,)): return str(x.date())
        if isinstance(x, float) and not math.isfinite(x): return None
        return str(x)
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o): return None
        if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    return json.dumps(clean(o), ensure_ascii=False, indent=1, default=f)

def r4(x):
    if isinstance(x, dict): return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [r4(v) for v in x]
    if isinstance(x, (float, np.floating)): return round(float(x), 4) if np.isfinite(x) else None
    return x

# ---------------- 成交時點變體的日頻引擎（改寫自 ia_core.simulate，只改換股日的成交價）----------------
def simulate_exec(plan, buy, sell, t0, t1, buy_px='open', sell_px='open', apply_days=None):
    """buy_px／sell_px ∈ {'open','close','twap'}：換股日新增部位與賣出部位的成交價。續抱部位不受影響。
    close 成交：賣出部位當天開盤→收盤的報酬由原持有人承擔；買進部位當天不吃開盤→收盤報酬。
    twap：(開盤＋收盤)/2。停牌凍結的處理同 ia_core（凍結者不能賣）。"""
    assert_locked_ok(t1)
    G = K.GROW
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            Ct = np.where(tr, K.Cn[t], 1.0); Ot = np.where(tr, K.On[t], 1.0)
            gap = np.where(tr, G[t] * Ot / Ct, 1.0)
            h = val * gap                       # 開盤時的市值
            frz = (h > 0) & ~tr
            htr = np.where(frz, 0.0, h)
            oc = np.where(tr, Ct / Ot, 1.0)
            use = apply_days is None or t in apply_days
            fs = {'open': np.ones(N), 'close': oc, 'twap': 0.5 * (1 + oc)}[sell_px if use else 'open']   # 賣出價 / 開盤價
            fb = {'open': np.ones(N), 'close': oc, 'twap': 0.5 * (1 + oc)}[buy_px if use else 'open']    # 買進價 / 開盤價
            k = len(pk)
            # 以開盤市值計的目標 T：賣出部位以 fs 價格變現，買進以 fb 價格取得
            def resid(T):
                tgt = np.zeros(N); tgt[pk] = T
                sold = np.clip(htr - tgt, 0, None); bought = np.clip(tgt - htr, 0, None)
                cash_in = (sold * fs).sum() * (1 - sell) + cash
                cash_out = (bought * fb).sum() * (1 + buy)
                return cash_in - cash_out
            lo, hi = 0.0, (htr.sum() + cash) * 2 / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if resid(mid) < 0: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            keep = np.minimum(htr, tgt); bought = np.clip(tgt - htr, 0, None)
            # 收盤市值：續抱部分吃 oc；買進部分以 fb 價買進的股數＝bought（開盤市值單位）→ 收盤市值 bought×oc（股數 bought/O，價值 bought/O×C）
            # 注意：bought 是「以開盤價計」的部位大小，實際花費 bought×fb
            val = np.where(frz, h, keep * oc + bought * oc)
            cash = resid(T)
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

#!/usr/bin/env python3
"""o_prepriced_growth 共用：搶在價格之前的成長股（第二條腿）。

只讀 import main_v2/v2_core.py（它再只讀 import independent_audit/ia_core.py）：
  股池、52 週比（gapfix 滾動窗）、營收 3 月年增率、排除金融、處置排除、跳動點開盤漲停遞補、
  日頻引擎、成本、訊號窗 t、AB（只用已公告營收）全部沿用主策略 v2，不另調。
本檔自己的部分：
  * 營收加速度 ACC＝yoy3(m) − yoy3(m−3)；營收 12 月新高比 NH＝最近一月營收 ÷ 前 11 個月最大值
  * 第二腿選股：52 週比在股池內的百分位落在事前固定區間 [lo, hi]，區間內依營收分數排序取 n 檔
  * 加權版日頻引擎（合成：主策略 n10 與第二腿 n10 各半、每月再平衡；重疊持股權重相加、成本只算淨交易）
"""
import sys, math, json, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
MV2 = HERE.parent / 'main_v2'
sys.path.insert(0, str(MV2))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
SWITCH = V.SWITCH
END1012 = V.END1012
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253'), 'value.parquet 不是修正版'

# ---------------- 期間 ----------------
SEL = ('2006-01-01', '2013-12-31')
VAL = ('2014-01-01', '2020-12-31')
HOLD = ('2021-01-01', '2026-10-12')
OOS = ('2014-01-01', '2026-10-12')
ALL = ('2006-01-01', '2026-10-12')
SEL_END = pd.Timestamp('2013-12-31')

# ---------------- 產業 ----------------
PREFIX_IND = {'11': '水泥工業', '12': '食品工業', '13': '塑膠工業', '14': '紡織纖維', '15': '電機機械', '16': '電器電纜',
              '17': '化學工業', '18': '玻璃陶瓷', '19': '造紙工業', '20': '鋼鐵工業', '21': '橡膠工業', '22': '汽車工業',
              '25': '建材營造業', '26': '航運業', '27': '觀光餐旅', '28': '金融保險業', '29': '貿易百貨業', '99': '其他業'}
ELEC_SET = {'電子零組件業', '半導體業', '光電業', '電腦及週邊設備業', '其他電子業', '通信網路業', '資訊服務業', '電子通路業', '數位雲端', '電子工業'}
def _ind(c):
    x = V.SL['industry'].get(c) if c in V.SL.index else None
    if isinstance(x, str): return x
    return PREFIX_IND.get(c[:2], '（未知）')
INDUSTRY = np.array([_ind(c) for c in COLS])
ELEC = np.array([i in ELEC_SET for i in INDUSTRY])

# ---------------- 營收 ----------------
_REV = {}
def revenue():
    if _REV: return _REV
    YL, Y5, Y5f = V.load_revenue()
    # 合併年增率：Y5 優先、缺的月份用長期營收（只用在 d ≥ 2021-03 的「落後 3 個月」那一段）
    YM = YL.reindex(index=YL.index.union(Y5.index), columns=YL.columns.union(Y5.columns))
    Y5r = Y5.reindex(index=YM.index, columns=YM.columns)
    YM = Y5r.where(Y5r.notna(), YM)
    # 營收水準：長期營收；d ≥ 2021-03 時 rev_5y 優先（同為千元）
    d = pd.read_csv(K.LONG, dtype={'code': str, 'market': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d['pri'] = (d['market'] != 'sii').astype(int)
    d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
    LL = d.pivot(index='p', columns='code', values='revenue_k_twd')
    LL = LL.reindex(pd.period_range(LL.index.min(), LL.index.max(), freq='M'))
    r5 = pd.read_csv(K.X / 'rev_5y.csv.gz', dtype={'code': str}); r5['p'] = pd.PeriodIndex(r5['ym'], freq='M')
    L5 = r5.drop_duplicates(['p', 'code'], keep='last').pivot(index='p', columns='code', values='revenue')
    LM = LL.reindex(index=LL.index.union(L5.index), columns=LL.columns.union(L5.columns))
    L5r = L5.reindex(index=LM.index, columns=LM.columns)
    LM = L5r.where(L5r.notna(), LM)
    def nh_tab(LV):
        LV = LV.reindex(columns=COLS)
        prev = LV.shift(1).rolling(11, min_periods=11).max()
        return (LV / prev).where((prev > 0) & (LV > 0))      # 月 p 的值＝p 月營收 ÷ p−1..p−11 月最大值
    _REV.update(YL=YL, Y5=Y5, YM=YM, NH_L=nh_tab(LL), NH_M=nh_tab(LM))
    return _REV

def _row(T, p):
    if p not in T.index: return np.full(N, np.nan)
    return T.reindex(columns=COLS).loc[p].to_numpy(dtype=float)

def yoy3_merged(d, back):
    R = revenue()
    Y = R['YM'] if d >= SWITCH else R['YL']
    return K.yoy3_for_month(Y, pd.Period(d, 'M') - back)

def nh_at(d, back):
    """訊號月 m：最近一月 m−1−back 的 12 月新高比。"""
    R = revenue()
    T = R['NH_M'] if d >= SWITCH else R['NH_L']
    return _row(T, pd.Period(d, 'M') - 1 - back)

# ---------------- 第二腿 ----------------
REVS = ('ACC', 'NH', 'AVG')
BANDS = {'B1': (0.30, 0.80), 'B2': (0.40, 0.85), 'B3': (0.50, 0.85), 'B4': (0.50, 0.90)}
NS = (10, 30)
def vname(rev, band, n): return f'{rev}|{band}|n{n}'
ALL_VERSIONS = [vname(r, b, n) for r in REVS for b in BANDS for n in NS]

def pct_in(el, x):
    return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()

class Leg:
    """S＝V.V2(gapfix, exfin=True[, unavail])。leg_cs(d) 回傳股池、52 週比百分位、營收訊號。"""
    def __init__(self, S):
        self.S = S; self._c = {}
    def leg_cs(self, d):
        if d in self._c: return self._c[d]
        cs = self.S.cross_section(d)
        el = cs['el']
        y3 = cs['y3']
        lag = yoy3_merged(d, 3); nh = nh_at(d, 0)
        if self.S.unavail is not None:
            U = self.S.unavail(d)
            if U is not None and U.any():
                lag = np.where(U, yoy3_merged(d, 4), lag); nh = np.where(U, nh_at(d, 1), nh)
        acc = y3 - lag
        p52 = pct_in(el, cs['s52'])
        pacc = pct_in(el & np.isfinite(acc), acc); pnh = pct_in(el & np.isfinite(nh), nh)
        avg = 0.5 * pacc + 0.5 * pnh
        o = dict(cs=cs, el=el, p52=p52, acc=acc, nh=nh, avg=avg, pacc=pacc, pnh=pnh, y3=y3, py3=pct_in(el, y3))
        self._c[d] = o
        return o
    def candidates(self, d, rev, band):
        o = self.leg_cs(d)
        lo, hi = BANDS[band]
        sc = {'ACC': o['acc'], 'NH': o['nh'], 'AVG': o['avg']}[rev]
        ok = o['el'] & (o['p52'] >= lo) & (o['p52'] <= hi) & np.isfinite(sc)
        s = np.round(np.where(ok, sc, -np.inf), 10)
        col = np.arange(N)
        order = np.lexsort((col, -np.nan_to_num(o['y3'], nan=-1e18), -s))
        return order[:int(ok.sum())], ok
    def plan(self, rev, band, n, days=None):
        out = []; held = set(); self.nblk = 0
        days = self.S.sig_days if days is None else days
        for d in days:
            o = self.leg_cs(d)
            if o['el'].sum() < 50: continue
            cand, ok = self.candidates(d, rev, band)
            if len(cand) == 0: continue
            e = o['cs']['e']; gu = K.gap_up_open(e)
            pick = []
            for j in cand:
                if gu[j] and j not in held: self.nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            out.append((e, np.array(pick, int), np.where(o['el'])[0], d))
            held = set(pick)
        return out

def leg_start(L):
    """第一個 ACC 與 NH 都有值（股池內覆蓋 ≥ 50%）的訊號日；所有版本共用起點。"""
    for d in L.S.sig_days:
        o = L.leg_cs(d)
        if np.isfinite(o['acc'][o['el']]).mean() >= 0.5 and np.isfinite(o['nh'][o['el']]).mean() >= 0.5:
            return d
    raise RuntimeError

# ---------------- 加權引擎（同 ia_core.simulate 的凍結／成本規則，目標部位依權重） ----------------
def simulate_w(plan, buy, sell, t0, t1, G=None):
    """plan＝[(exec_t, idx ndarray, w ndarray（和為 1）)]。"""
    G = K.GROW if G is None else G
    reb = {p[0]: (p[1], p[2]) for p in plan}
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in reb:
            pk, w = reb[t]
            wf = np.zeros(N); np.add.at(wf, pk, w)
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
            def cost_of(X):
                tgt = X * wf
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            tgt = lo * wf
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan}

def eq_plan(pl):
    return [(p[0], p[1], np.full(len(p[1]), 1.0 / len(p[1]))) for p in pl if len(p[1])]

def combo_plan(pl_a, pl_b, wa=0.5):
    """各半合成：同一成交日 a 的持股權重 wa/len、b 的 (1−wa)/len，重疊相加。只取兩者都有的成交日。"""
    B = {p[0]: p for p in pl_b}
    out = []
    for p in pl_a:
        if p[0] not in B: continue
        q = B[p[0]]
        idx = np.concatenate([p[1], q[1]]); w = np.concatenate([np.full(len(p[1]), wa / len(p[1])), np.full(len(q[1]), (1 - wa) / len(q[1]))])
        out.append((p[0], idx, w))
    return out

# ---------------- 統計 ----------------
def window_returns(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def wsel(s, a, b):
    return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]

def seg(nav, ew, a, b, sd, yearly=False):
    return V.seg_stats(nav, ew, a, b, sd, yearly=yearly, ad=a)

def tstat(x): return K.tstat(x)

def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(np.asarray(a)[m], np.asarray(b)[m])[0, 1]) if m.sum() > 5 else float('nan')

def r4(o):
    if isinstance(o, dict): return {k: r4(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [r4(v) for v in o]
    if isinstance(o, (float, np.floating)): return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)): return int(o)
    return o

# max-t bootstrap 從 combo_search/clib.py 抄（clib import lhlib 會載入另一套面板，這裡只取函式本體）
def stationary_idx(T, B, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T):
        new = rng.random(B) < p
        idx[:, t] = np.where(new, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx

def tvec(X):
    m = X.mean(axis=-2); s = X.std(axis=-2, ddof=1)
    return m / (s / math.sqrt(X.shape[-2]))

def maxt_bootstrap(X, B=5000, mean_block=6, seed=0):
    X = np.asarray(X, float); T, Kk = X.shape
    t_obs = tvec(X); mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
    Xc = X - mu
    rng = np.random.default_rng(seed)
    idx = stationary_idx(T, B, mean_block, rng)
    mx = np.empty(B)
    for b0 in range(0, B, 500):
        Xb = Xc[idx[b0:b0 + 500]]
        tb = Xb.mean(axis=1) / (sd / math.sqrt(T))
        mx[b0:b0 + 500] = tb.max(axis=1)
    t_max = float(np.max(t_obs))
    return {'t_obs_max': round(t_max, 3), 'argmax': int(np.argmax(t_obs)), 'p_maxt': round(float((mx >= t_max).mean()), 4),
            'p_each': [round(float((mx >= t).mean()), 4) for t in t_obs], 'boot_max_t_q95': round(float(np.quantile(mx, 0.95)), 3), 'T': T, 'K': Kk}

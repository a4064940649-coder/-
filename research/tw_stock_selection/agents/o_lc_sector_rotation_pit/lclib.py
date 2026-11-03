#!/usr/bin/env python3
"""o_lc_sector_rotation_pit 共用庫：PIT 股數（2013+ 沿用 o8 shares_pit＋修正；2006～2012 由 2013 錨點以除權因子回推）、
PIT 產業表（證交所產業別，依公開快照逐段生效）、PIT 市值前 100 宇宙、產業四訊號與 COMP、有上限的產業超低配權重、
紙上月頻引擎（成本口徑同 tcore.rates）、統計。

唯讀 import o_0050_tilt_pit 的 tlib／tcore（V7 凍結面板＋o_infra_v3 下市股價、除權息日 exfix 報酬、gapfix 52 週比、V4 已公告營收 y3_ab）。
python3 -I -B 執行，不寫 pycache、不改其他資料夾。
時點：訊號日 d＝營收期限日 D_m 之後第 j 個面板交易日（j＝0..12，tlib.jscan_days），d 收盤算訊號，d+1 開盤成交；
評估窗＝d 收盤 → 下一個訊號日收盤（0050 含息同窗）。
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
O8 = AG / 'o_0050_tilt_pit'
DATA = HERE / 'data'; OUT = HERE / 'out'; RAW = HERE / 'raw'
for p in (DATA, OUT, RAW): p.mkdir(exist_ok=True)
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/olc')
SCR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(O8))
import tlib as L      # noqa: E402
import tcore as TC    # noqa: E402
sys.path.pop(0)
IDX, COLS, N, CI = L.IDX, list(L.COLS), L.N, L.CI
J2330 = CI['2330']

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def canon(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda z: float(z) if isinstance(z, (np.floating, np.integer)) else (bool(z) if isinstance(z, np.bool_) else (z.tolist() if isinstance(z, np.ndarray) else str(z))))
def tstat(x): return L.tstat(x)

# ======================================================================
# 一、股數
# ======================================================================
PIT_START = pd.Timestamp('2013-01-02')
FIX4763 = dict(a0='2019-09-04', a1='2023-07-27', div_a=12.65, b1='2025-06-30', div_b=10.0)   # 同 o_capmatched_pit_close/pcore.py
FIX2317 = dict(pre_until='2018-10-25', eff='2018-10-26', pre_val_date='2018-06-29')

def shares_2013():
    """o8 shares_pit（2013-01 起）＋4763／2317 修正（o9 複製員發現的錯誤）。"""
    S = pd.DataFrame(TC.SH.copy(), index=IDX, columns=COLS)
    ix = S.index
    a = (ix >= pd.Timestamp(FIX4763['a0'])) & (ix < pd.Timestamp(FIX4763['a1']))
    b = (ix >= pd.Timestamp(FIX4763['a1'])) & (ix < pd.Timestamp(FIX4763['b1']))
    S.loc[a, '4763'] = S.loc[a, '4763'] / FIX4763['div_a']
    S.loc[b, '4763'] = S.loc[b, '4763'] / FIX4763['div_b']
    pre_val = float(S.loc[:pd.Timestamp(FIX2317['pre_val_date']), '2317'].dropna().iloc[-1])
    m = (ix > pd.Timestamp(FIX2317['pre_val_date'])) & (ix <= pd.Timestamp(FIX2317['pre_until']))
    S.loc[m, '2317'] = pre_val
    S = S.ffill(limit=10)
    S[ix < PIT_START] = np.nan
    return S.to_numpy()

def listed_mask():
    Lt = pd.read_parquet(AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet').reindex(columns=COLS).astype(float)
    Lt = Lt.reindex(IDX.union(Lt.index)).sort_index().ffill().reindex(IDX)
    return Lt.fillna(1.0).to_numpy() > 0.5

# ---- 除權因子（回推用；只用價格，不用報酬檢定）----
def event_factors():
    """每格除權息／參考價事件的總調整因子 f＝前收÷參考價。
    flag 1（無參考價，面板 ret＝0）：參考價估計＝開盤÷(1＋當日非事件股開盤跳空中位數)（同 tlib exfix 口徑）。
    flag 2／3（有官方參考價）：f＝前收×(1＋ret)÷收盤。"""
    Fn = np.nan_to_num(L.Fn, nan=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        gap = L.On / L.PREVC - 1
        normal = L.TRADED & np.isfinite(L.On) & (Fn == 0) & np.isfinite(gap)
        med = np.array([np.median(gap[t][normal[t]]) if normal[t].sum() > 50 else 0.0 for t in range(len(IDX))])
        f1 = L.PREVC * (1 + med[:, None]) / L.On
        f23 = L.PREVC * (1 + np.nan_to_num(L.Rn, nan=0.0)) / L.Cn
    f = np.full((len(IDX), N), np.nan)
    m1 = (Fn == 1) & L.TRADED & np.isfinite(L.On) & (L.On > 0)
    m23 = np.isin(Fn, (2, 3)) & L.TRADED & np.isfinite(L.Rn)
    f[m1] = f1[m1]; f[m23] = f23[m23]
    return f, Fn

TAU_CASH = 0.08     # f−1 ≤ 8%：視為純現金股利（含估計雜訊）
DELTA = 0.02        # 扣掉估計現金殖利率後仍 > 2% 才算配股
def share_multipliers(f, Fn, t_lo, t_hi):
    """[t_lo, t_hi] 內每個事件的股數乘數 m（事件日起股數 ×m）。回傳 dict j -> list[(t, m)]。規則事前寫死：
       flag 1／2：ŷ＝本股 ±3 年內 f−1∈(0.5%,8%] 事件的中位數，否則當年全市場同類事件中位數；m＝f/(1+ŷ)，m ≤ 1.02 時 m＝1。
       flag 3（參考價重設：減資、現增除權等）：|f−1| > 2% 時 m＝f。m 夾在 [0.2, 3]。"""
    yr = IDX.year.values
    ev = np.argwhere(np.isfinite(f[t_lo:t_hi + 1]) & (np.abs(np.nan_to_num(f[t_lo:t_hi + 1], nan=1.0) - 1) > 0.005))
    ev[:, 0] += t_lo
    cash_like = {}
    mkt = {}
    for t, j in ev:
        x = f[t, j] - 1
        if Fn[t, j] in (1, 2) and 0.005 < x <= TAU_CASH:
            cash_like.setdefault(j, []).append((yr[t], x)); mkt.setdefault(yr[t], []).append(x)
    mmed = {y: float(np.median(v)) for y, v in mkt.items()}
    out = {}
    for t, j in ev:
        fx = f[t, j]
        if Fn[t, j] == 3:
            m = fx if abs(fx - 1) > 0.02 else 1.0
        else:
            own = [x for (y, x) in cash_like.get(j, []) if abs(y - yr[t]) <= 3]
            yh = float(np.median(own)) if own else mmed.get(yr[t], 0.03)
            m = fx / (1 + yh)
            if m <= 1 + DELTA: m = 1.0
        m = float(min(max(m, 0.2), 3.0))
        if m != 1.0: out.setdefault(j, []).append((int(t), m))
    return out

def backcast(S_anchor, t_anchor, t_lo, mult):
    """S_anchor：N 向量（t_anchor[j] 當日股數），t_anchor：N 向量（錨點日索引，<0＝無錨）。回傳 [t_lo, max t_anchor] 的回推股數矩陣（只填錨點日之前）。"""
    T = len(IDX); S = np.full((T, N), np.nan)
    for j in range(N):
        ta = t_anchor[j]
        if ta < 0 or not np.isfinite(S_anchor[j]): continue
        evs = sorted([(t, m) for (t, m) in mult.get(j, []) if t_lo < t <= ta], reverse=True)
        cur = S_anchor[j]; hi = ta
        for t, m in evs:
            S[t:hi + 1, j] = cur          # 事件日 t 起是「事件後」股數
            cur = cur / m; hi = t - 1
        S[t_lo:hi + 1, j] = cur
    return S

def adv120():
    V = pd.DataFrame(np.where(L.TRADED, L.Vn, np.nan))
    return V.rolling(120, min_periods=60).mean().to_numpy()

def impute_mcap(t, mc, alive, ADV):
    """無股數的存活股：市值＝ADV120 × 同期有股數股（ADV 名次 30～200）的市值/ADV 中位數。回傳 (mc_full, imputed_mask)。"""
    a = ADV[t]
    have = np.isfinite(mc) & (mc > 0) & alive & np.isfinite(a) & (a > 0)
    need = alive & ~np.isfinite(mc) & np.isfinite(a) & (a > 0)
    out = mc.copy(); imp = np.zeros(N, bool)
    if need.any() and have.sum() > 50:
        o = np.argsort(-np.where(have, a, -np.inf))
        sel = o[29:200]; sel = sel[have[sel]]
        k = float(np.median(mc[sel] / a[sel]))
        out[need] = a[need] * k; imp[need] = True
    return out, imp

# ======================================================================
# 二、PIT 產業
# ======================================================================
REG_2007 = pd.Timestamp('2007-07-02')     # 電子工業細分 8 類；化學生技醫療分立；新增油電燃氣（證交所 2007-07 產業別調整）
REG_2023 = pd.Timestamp('2023-07-03')     # 新增綠能環保、數位雲端、居家生活、運動休閒；觀光事業更名觀光餐旅
NEW4 = {'綠能環保', '數位雲端', '運動休閒', '居家生活'}
ELEC8 = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業'}
FIN_IND = '金融保險業'

def load_industry_pit():
    """raw/industry_pit.csv：code,start,end,industry,source（start 含、end 不含；end 空＝至今）。回傳 (dates, labels list of np arrays)。"""
    T = pd.read_csv(RAW / 'industry_pit.csv', dtype=str)
    T['industry'] = T['industry'].fillna('未分類')
    T['start'] = pd.to_datetime(T['start']); T['end'] = pd.to_datetime(T['end'].fillna('2099-12-31'))
    bps = sorted(set(T['start']) | set(T['end']) | {REG_2007, REG_2023})
    bps = [b for b in bps if b <= pd.Timestamp('2099-01-01')]
    labs = []
    for b in bps:
        x = np.array(['未分類'] * N, dtype=object)
        sub = T[(T.start <= b) & (T.end > b)]
        for c, ind in zip(sub.code, sub.industry):
            j = CI.get(c)
            if j is not None: x[j] = ind
        labs.append(x)
    return pd.DatetimeIndex(bps), labs

_IND = {}
def ind_at(d):
    """d 當日生效的產業別（N 陣列）。2007-07-02 以前電子 8 類合併為「電子工業」、化學／生技醫療合併、油電燃氣併入其他業（主設定）。"""
    if not _IND:
        b, l = load_industry_pit(); _IND['b'] = b; _IND['l'] = l
    b, l = _IND['b'], _IND['l']
    k = int(np.searchsorted(b.values, np.datetime64(pd.Timestamp(d)), side='right')) - 1
    x = l[max(k, 0)].copy()
    if pd.Timestamp(d) < REG_2007:
        x = np.array([('電子工業' if v in ELEC8 else '化學生技醫療業' if v in ('化學工業', '生技醫療業') else '其他業' if v == '油電燃氣業' else v) for v in x], dtype=object)
    return x

def mapping_key(d):
    if not _IND: ind_at(d)
    b = _IND['b']; k = int(np.searchsorted(b.values, np.datetime64(pd.Timestamp(d)), side='right')) - 1
    return (k, pd.Timestamp(d) < REG_2007)

# ======================================================================
# 三、宇宙與市值
# ======================================================================
DRMASK = np.array([c.startswith('91') or c.startswith('0') for c in COLS])
ADV_POOL_BEFORE = pd.Timestamp('2013-01-02')   # D1 未通過：2013 以前改用 120 日均成交值前 100 當代理池（權重仍用回推市值，標「非 PIT 市值」）
FORMAL_START = '2014-01'                         # 計入 G2／G4 的正式段（o8 股數閘門自 2014 起通過）
class Caps:
    """全期股數：2013 起 o8＋修正；2013 以前回推；無股數者 ADV 插補。2013 以前宇宙＝ADV120 前 100（D1 未過）。"""
    def __init__(self, SH_full=None):
        if SH_full is None:
            SH_full = pd.read_parquet(DATA / 'shares_full.parquet').reindex(index=IDX, columns=COLS).to_numpy().astype(float)
        self.SH = SH_full
        self.LST = listed_mask()
        self.ADV = adv120()
        self.tadv = int(IDX.searchsorted(ADV_POOL_BEFORE))
    def mcap(self, t):
        alive = L.TRADED[max(t - 5, 0):t + 1].any(axis=0) & np.isfinite(L.CFF[t]) & self.LST[t] & ~DRMASK
        mc = np.where(alive, self.SH[t] * L.CFF[t], np.nan)
        mc, imp = impute_mcap(t, mc, alive, self.ADV)
        return mc, imp, alive
    def universe(self, t, n=100):
        mc, imp, alive = self.mcap(t)
        if t < self.tadv:
            el = np.where(alive & np.isfinite(mc) & (mc > 0) & np.isfinite(self.ADV[t]), self.ADV[t], -np.inf)
        else:
            el = np.where(alive & np.isfinite(mc) & (mc > 0), mc, -np.inf)
        U = np.argsort(-el, kind='stable')[:n]; U = U[np.isfinite(el[U])]
        U = U[np.argsort(-mc[U], kind='stable')]
        return U, mc[U], imp[U]

# ======================================================================
# 四、產業指數與訊號
# ======================================================================
GOODI = np.where(L.GOOD)[0]
GPOS = np.full(len(IDX), -1); GPOS[GOODI] = np.arange(len(GOODI))
def to_good(t):
    while not L.GOOD[t]: t -= 1
    return t
def gback(t, k):
    p = GPOS[t]; assert p >= 0
    return GOODI[max(p - k, 0)]
CUMG = TC.CUMG

def official_blocks():
    """產業指數的成分期：每月第一個 ≥ 11 日的面板交易日（2005-01 起）。"""
    return [IDX.get_loc(d) for d in L.official_days('2005-01-01')]

class SectorIndex:
    """每個成分期起點 τ：宇宙（前 100，不含 2330）以 τ 收盤 PIT 市值為權重，期內買進持有（含息 exfix 報酬）；逐期串接。
    產業歸屬用訊號日 d 當時生效的分類（套到全部歷史，只用過去價格）。"""
    def __init__(self, caps):
        self.caps = caps
        bl = official_blocks()
        self.bl = bl
        T = len(IDX)
        W = np.zeros((len(bl), N))
        for b, tau in enumerate(bl):
            U, mcU, _ = caps.universe(tau)
            w = np.zeros(N); w[U] = mcU; w[J2330] = 0.0
            W[b] = w
        self.W = W
        # 每日所屬成分期與期內相對值 Q[t,i]＝TRI_i(t)/TRI_i(τ)
        blk = np.full(T, -1)
        for b, tau in enumerate(bl):
            end = bl[b + 1] if b + 1 < len(bl) else T - 1
            blk[tau + 1:end + 1] = b
        self.blk = blk
        self._cache = {}
    def index(self, key, labels):
        """labels：N 產業陣列。回傳 (names, idx[T, S])，idx 在第一個成分期起點＝1。"""
        if key in self._cache: return self._cache[key]
        names = sorted(set(labels))
        sid = {s: k for k, s in enumerate(names)}
        H = np.zeros((N, len(names))); H[np.arange(N), [sid[v] for v in labels]] = 1.0
        T = len(IDX); out = np.full((T, len(names)), np.nan)
        lvl = np.ones(len(names))
        t0 = self.bl[0]; out[t0] = lvl
        for b, tau in enumerate(self.bl):
            end = self.bl[b + 1] if b + 1 < len(self.bl) else T - 1
            w = self.W[b]
            den = w @ H
            ts = np.arange(tau + 1, end + 1)
            if len(ts) == 0: continue
            Q = np.exp(CUMG[ts + 1] - CUMG[tau + 1])            # (len, N)
            num = (Q * w[None, :]) @ H
            with np.errstate(invalid='ignore', divide='ignore'):
                rel = np.where(den[None, :] > 0, num / np.where(den > 0, den, 1.0)[None, :], 1.0)
            out[ts] = lvl[None, :] * rel
            lvl = out[end].copy()
        self._cache[key] = (names, out)
        return names, out

SECT_MIN = 2
REV_CLIP = 100.0
REV_NEUTRAL = ('2013-01', '2014-02')      # IFRS 斷點：此段錨定月的產業營收訊號設中性
REV_COV = 0.70

def _nh_flags():
    s = np.nan_to_num(TC.S52[GOODI], nan=0.0) >= 0.99999
    r = pd.DataFrame(s.astype(np.int8)).rolling(20, min_periods=1).max().to_numpy() > 0
    out = np.zeros((len(IDX), N), bool); out[GOODI] = r
    return out

class Signals:
    def __init__(self, caps, sidx):
        self.caps = caps; self.sidx = sidx
        self.NH = _nh_flags()
    def at(self, d):
        """訊號日 d：宇宙、B 權重、可輪動產業表（含四訊號、COMP、覆蓋率）。"""
        d = pd.Timestamp(d); t = IDX.get_loc(d); tg = to_good(t)
        U, mcU, impU = self.caps.universe(t)
        labs = ind_at(d); key = mapping_key(d)
        indU = labs[U]
        fin = (indU == FIN_IND) | TC.FINMASK[U]
        names, SI = self.sidx.index(key, labs)
        sid = {s: k for k, s in enumerate(names)}
        y3 = TC.y3_ab(d)[U].astype(float)
        anchor = str(pd.Period(d, 'M'))
        rev_neutral = REV_NEUTRAL[0] <= anchor <= REV_NEUTRAL[1]
        a12, a6, a3, b1 = gback(tg, 252), gback(tg, 126), gback(tg, 63), gback(tg, 21)
        h0 = gback(tg, 251)
        liq = (np.nan_to_num(TC.V60[tg], nan=0) >= 1e7) & (np.nan_to_num(L.CFF[tg], nan=0) >= 10) & self.caps.LST[t] & ~DRMASK & \
              L.TRADED[max(t - 5, 0):t + 1].any(axis=0)
        rows = []
        not2330 = U != J2330
        for s in sorted(set(indU[not2330 & ~fin])):
            m = np.where((indU == s) & not2330 & ~fin)[0]
            if len(m) < SECT_MIN or s in ('未分類',): continue
            k = sid[s]; col = SI[:, k]
            w = mcU[m]
            ok = np.isfinite(y3[m])
            cov = float(w[ok].sum() / w.sum()) if w.sum() > 0 else 0.0
            rev = float((w[ok] * np.clip(y3[m][ok], -REV_CLIP, REV_CLIP)).sum() / w[ok].sum()) if ok.sum() else np.nan
            if cov < REV_COV or rev_neutral: rev = np.nan
            def rr(a): return float(col[b1] / col[a] - 1) if np.isfinite(col[a]) and np.isfinite(col[b1]) and col[a] > 0 else np.nan
            seg = col[h0:tg + 1]; seg = seg[np.isfinite(seg)]
            h52 = float(col[tg] / seg.max()) if len(seg) > 200 and np.isfinite(col[tg]) else np.nan
            allm = np.where((labs == s) & liq)[0]
            nh = float(self.NH[tg, allm].mean()) if len(allm) >= 2 else np.nan
            rows.append(dict(sector=s, n=len(m), B=float(w.sum()), REV3=rev, rev_cov=cov, RS12=rr(a12), RS6=rr(a6), RS3=rr(a3), H52=h52, NH=nh, n_nh=len(allm)))
        ST = pd.DataFrame(rows).set_index('sector') if rows else pd.DataFrame()
        if len(ST):
            rs = ST[['RS12', 'RS6', 'RS3']].rank(pct=True).mean(axis=1, skipna=True)
            ST['RS'] = rs
            R = pd.DataFrame({'REV3': ST['REV3'].rank(pct=True), 'RS': ST['RS'].rank(pct=True), 'H52': ST['H52'].rank(pct=True), 'NH': ST['NH'].rank(pct=True)})
            ST['COMP'] = R.mean(axis=1, skipna=True)
            for c in R.columns: ST['p_' + c] = R[c]
        tot = mcU.sum()
        return dict(d=d, t=t, U=U, mc=mcU, imp=impU, ind=indU, fin=fin, b=mcU / tot, ST=ST, imp_share=float(mcU[impU].sum() / tot))

# ======================================================================
# 五、權重
# ======================================================================
def waterfill(total, caps):
    """把 total 平均分給各產業，各自上限 caps；超過的部分再平分給未滿的。回傳分配。"""
    caps = np.asarray(caps, float); x = np.zeros(len(caps)); rem = total; act = caps > 1e-12
    for _ in range(len(caps) + 1):
        if rem <= 1e-12 or not act.any(): break
        share = rem / act.sum()
        add = np.minimum(share, caps - x); add[~act] = 0
        x += add; rem -= add.sum()
        act = act & (caps - x > 1e-12)
    return x

def sector_targets(cs, sig, k, A, held, buf=2, cap_mult=3.0, rng_scores=None):
    """回傳 (stock_idx, weights, new_held, info)。sig：訊號欄名或 None（用 rng_scores：產業名→分數）。"""
    ST = cs['ST']; U, b, ind = cs['U'], cs['b'], cs['ind']
    w = b.copy()
    if len(ST) == 0: return U, w, set(), dict(over=0.0)
    if rng_scores is not None:
        sc = pd.Series([rng_scores.get(s, np.nan) for s in ST.index], index=ST.index)
    else:
        sc = ST[sig]
    Bs = ST['B'] / cs['mc'].sum()
    valid = sc.dropna()
    order = sorted(valid.index, key=lambda s: (-valid[s], -Bs[s]))
    rank = {s: r + 1 for r, s in enumerate(order)}
    keep = [s for s in order if s in held and rank[s] <= k + buf]
    top = list(keep)
    for s in order:
        if len(top) >= k: break
        if s not in top: top.append(s)
    top = top[:k]
    if len(top) == 0: return U, w, set(), dict(over=0.0)
    capv = np.array([cap_mult * Bs[s] for s in top])
    over = waterfill(A, np.minimum(capv, A))     # 單一產業超配 ≤ min(… ,3×B)；平均分配 A/k，滿了再分
    others = [s for s in ST.index if s not in top]
    Bo = np.array([Bs[s] for s in others]); O = float(over.sum())
    if Bo.sum() <= 0: return U, w, set(top), dict(over=0.0)
    if O > Bo.sum(): over = over * Bo.sum() / O; O = float(Bo.sum())
    under = O * Bo / Bo.sum()
    Wsec = {s: Bs[s] + o for s, o in zip(top, over)}
    Wsec.update({s: Bs[s] - u for s, u in zip(others, under)})
    for s, ws in Wsec.items():
        m = np.where((ind == s) & (U != J2330) & ~cs['fin'])[0]
        w[m] = b[m] / b[m].sum() * ws
    w = np.clip(w, 0, None); w = w / w.sum()
    return U, w, set(top), dict(over=O, top=top, rank=rank)

# ======================================================================
# 六、紙上引擎（tcore.run 加「不換股」選項；成本同 tcore.rates）
# ======================================================================
def run_paper(days, plans, cost='normal', daily=False):
    """plans：{d: (idx, w)} 或 {d: None}（None＝該窗不交易，持股續抱）。回傳 (窗報酬 Series, info)。
    info：turnover、cost、W（每窗開盤後權重）、nav（daily=True 時日淨值）。"""
    buy, sell = TC.rates(cost)
    ti = [IDX.get_loc(d) for d in days]
    h = np.zeros(N); cash = 1.0
    out, tov, cst, W = [], [], [], {}
    navd = {}; base = 1.0
    for k in range(len(ti) - 1):
        t, t2 = ti[k], ti[k + 1]; e = t + 1
        V0 = h.sum() + cash
        if V0 <= 0: V0 = 1e-12
        h = h / V0; cash = cash / V0
        fo = TC.factor_close_to_open(t, e)
        tr = np.isfinite(fo)
        dead = (h > 0) & (L.LAST_TRADE < e)
        cash += (h[dead] * (1 - sell)).sum(); h[dead] = 0.0
        frz = (h > 0) & ~tr
        h = np.where(tr, h * np.nan_to_num(fo, nan=1.0), h)
        Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash
        pl = plans[days[k]]
        if pl is None:
            tgt = htr.copy(); T = Vt - cash
            if cash > 0:     # 現金（下市款）按比例買回
                s0 = htr.sum()
                if s0 > 0:
                    tgt = htr * (1 + cash / s0); c = buy * cash; tgt = tgt * (s0 + cash - c) / (s0 + cash); T = tgt.sum()
            tov.append(float(cash / max(Vt + Vfz, 1e-12))); cst.append(float(Vt - T - 0.0) if cash > 0 else 0.0)
        else:
            idx, w = pl
            wv = np.zeros(N); np.add.at(wv, idx, w)
            wv[frz] = 0.0; wv[~tr] = 0.0
            wv = wv / wv.sum() if wv.sum() > 0 else wv
            T = Vt
            for _ in range(3):
                tg_ = wv * T
                c = buy * np.clip(tg_ - htr, 0, None).sum() + sell * np.clip(htr - tg_, 0, None).sum()
                T = Vt - c
            tgt = wv * T
            tov.append(float(np.clip(htr - tgt, 0, None).sum() / max(Vt + Vfz, 1e-12))); cst.append(float(Vt - T))
        f2 = TC.factor_open_to_close(e, t2)
        newh = tgt * np.nan_to_num(f2, nan=1.0)
        hz = np.where(frz, h * TC.factor_close_to_close(e - 1, t2), 0.0)
        if daily:
            # 日淨值：e 收盤 → t2 收盤
            ts = np.arange(e, t2 + 1)
            fe = np.nan_to_num(TC.factor_open_to_close(e, e), nan=1.0)
            Qr = np.exp(CUMG[ts + 1] - CUMG[e + 1])
            vals = (Qr * (tgt * fe)[None, :]).sum(1) + (Qr * np.where(frz, h, 0.0)[None, :]).sum(1)
            for tt, v in zip(ts, vals): navd[tt] = base * v
        h = newh + np.where(frz, hz, 0.0); cash = 0.0
        r = h.sum() - 1.0
        out.append(r); base *= (1 + r)
        W[days[k]] = tgt / max(T, 1e-12) if T > 0 else tgt
    s = pd.Series(out, index=pd.DatetimeIndex(days[:-1]))
    info = {'turnover': np.array(tov), 'cost': np.array(cst), 'W': W}
    if daily:
        info['nav'] = pd.Series(navd).sort_index().rename(index=lambda i: IDX[i])
    return s, info

def bench_win(days): return TC.bench_win(days)

# ======================================================================
# 七、統計
# ======================================================================
def stats(x):
    x = pd.Series(x).dropna()
    if len(x) < 3: return dict(n=len(x), mean_pp=np.nan, t=np.nan, TE=np.nan, IR=np.nan)
    te = float(x.std(ddof=1) * math.sqrt(12))
    return dict(n=int(len(x)), mean_pp=float(x.mean() * 100), t=tstat(x), TE=te * 100, IR=float(x.mean() * 12 / te) if te > 0 else np.nan)
def nw_t(x, lag=3):
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 6: return np.nan
    e = x - x.mean(); s = (e * e).sum() / n
    for i in range(1, lag + 1): s += 2 * (1 - i / (lag + 1)) * (e[i:] * e[:-i]).sum() / n
    return float(x.mean() / math.sqrt(s / n))
def cagr(r):
    r = pd.Series(r).dropna()
    return float(np.prod(1 + r) ** (12 / len(r)) - 1) if len(r) else np.nan
def mdd_from_returns(r):
    v = np.cumprod(1 + pd.Series(r).dropna().to_numpy()); pk = np.maximum.accumulate(v)
    return float((v / pk - 1).min()) if len(v) else np.nan
def mdd_nav(nav):
    v = pd.Series(nav).dropna().to_numpy(); pk = np.maximum.accumulate(v)
    return float((v / pk - 1).min()) if len(v) else np.nan

def jdays(j, m0, m1, mend):
    """錨定月 m0..m1 的 J_j 訊號日＋終點（mend 月 J_j；超出面板 → 面板最後 GOOD 日）。"""
    out = []
    for p in pd.period_range(m0, m1, freq='M'):
        D = L.deadline(p); k = IDX.searchsorted(D)
        if k + j < len(IDX) - 1:
            d = IDX[k + j]
            out.append(d)
    pe = pd.Period(mend, 'M')
    try:
        D = L.deadline(pe); k = IDX.searchsorted(D) + j
    except IndexError:          # 終點月超出交易日曆（2026-11）→ 面板最後 GOOD 日
        k = len(IDX)
    de = IDX[k] if k < len(IDX) else IDX[np.where(L.GOOD)[0][-1]]
    out = [d for d in out if d < de]
    return out + [de]

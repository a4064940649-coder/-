#!/usr/bin/env python3
"""breadth_derisk 核心（複製自 independent_audit/ia_core.py 後追加；不 import 其他資料夾）。
原說明：independent_audit 核心：從原始面板自己重寫的資料、訊號、日曆、成交、成本與日頻引擎。

不 import 研究中心任何既有程式（btlib/exlib/lhlib/mlib/rlib…）。只讀：
  panel/{adj,close_raw,open,high,value,ret,flag,src}.parquet、tw_revenue.csv、rev_5y、disposal_5y、etf_0050.csv

與 exlib/lhlib 刻意採用「不同寫法」：
  * 報酬：逐日把每個部位的市值乘 (1+ret)，不使用 adj 水準；買進日用 close/open，賣出日用 (1+ret)×open/close。
  * 開盤漲停：用跳動點（tick）算出真正的漲停價，與 open 比較（exlib 用 還原開盤/前還原收 −1 ≥ 上限−0.6%）。
  * 成本：用二分法解「調倉後總值」，不用固定點迭代。
  * 停牌：換股日開盤沒有成交的持股「凍結」（不能賣），等恢復交易；從未恢復＝下市，以最後收盤計（另報歸零情境）。
  * 52 週高點、60 日均值：可選 strict（窗內要全有值，＝既有程式）或 lenient（窗內有值 ≥ 200／≥ 50 筆）。
"""
import math
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')
HERE = Path(__file__).resolve().parent

FEE = 0.001425 * 0.6
TAX = 0.003
def rates(slip=0.001):
    if slip == 'stress': return 0.005, 0.005
    return FEE + slip, FEE + TAX + slip

# ---------------- 原始資料 ----------------
ADJ = pd.read_parquet(P / 'adj.parquet')
C = pd.read_parquet(P / 'close_raw.parquet')
O = pd.read_parquet(P / 'open.parquet')
H = pd.read_parquet(P / 'high.parquet')
V = pd.read_parquet(P / 'value.parquet')
RET = pd.read_parquet(P / 'ret.parquet')
FLAG = pd.read_parquet(P / 'flag.parquet')
SRC = pd.read_parquet(P / 'src.parquet')
IDX = C.index; COLS = C.columns; N = len(COLS); CI = {c: j for j, c in enumerate(COLS)}
Cn = C.to_numpy(); On = O.to_numpy(); Rn = RET.to_numpy(); Fn = FLAG.to_numpy(); Hn = H.to_numpy()
TRADED = np.isfinite(Cn)
# 有收盤就有市值變動；ret 缺值（flag 5 資料錯誤）當作 0（面板 adj 也是這樣處理）
GROW = np.where(TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)

# 我自己的還原收盤（由 ret 連乘），只在有收盤的日子有值
MYADJ = pd.DataFrame(np.where(TRADED, np.cumprod(GROW, axis=0), np.nan), index=IDX, columns=COLS)

# 0050 含息（finlab）— 對齊到面板日曆（前值補）
E50_raw = pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = E50_raw.reindex(IDX.union(E50_raw.index)).ffill().reindex(IDX)

# ---------------- 日曆 ----------------
NOBS = TRADED.sum(axis=1)
GOOD = NOBS >= 300          # 有值檔數 < 300 的日子（2018-01-13）不參與滾動訊號
def rebalance_days(start, end=None):
    """每月第一個「日 ≥ 11」的面板交易日。"""
    out = []
    ii = IDX[IDX >= pd.Timestamp(start)]
    if end is not None: ii = ii[ii <= pd.Timestamp(end)]
    seen = set()
    for d in ii:
        if d.day >= 11 and (d.year, d.month) not in seen:
            seen.add((d.year, d.month)); out.append(d)
    return out

# ---------------- 滾動訊號（只用 GOOD 日） ----------------
_RS = {}
def rolling_signals(mode='strict'):
    if mode not in _RS: _RS[mode] = _rolling_signals(mode)
    return _RS[mode]

_med = pd.Series(NOBS, index=IDX).rolling(21, center=True, min_periods=5).median().to_numpy()
GOOD2 = NOBS >= 0.97 * _med   # 另外剔除 2017-08-03、2018-07-12（各缺約 50 檔＝後來下市、只存在 Asoul 的股票）

GOOD3 = GOOD2 & ~((IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2024-03-15')))   # 再剔除 98 檔缺值的 src 3 缺口月

def _rolling_signals(mode):
    g = np.where(GOOD2 if mode in ('strict_fix', 'strict_fix2') else GOOD)[0]
    A = MYADJ.to_numpy()[g]; Vv = V.to_numpy()[g]
    Ad = pd.DataFrame(A); Vd = pd.DataFrame(Vv)
    if mode == 'strict_fix2':
        # 窗內缺值只容許落在 2024-02-15～03-15（src 3 缺 98 檔的那個月）；其他缺值仍照 strict 排除
        exc = pd.Series(((IDX[g] >= pd.Timestamp('2024-02-15')) & (IDX[g] <= pd.Timestamp('2024-03-15'))).astype(float))
        e252 = exc.rolling(252, min_periods=1).sum().to_numpy()[:, None]; e60 = exc.rolling(60, min_periods=1).sum().to_numpy()[:, None]
        c252 = Ad.notna().rolling(252, min_periods=252).sum().to_numpy(); c60 = Vd.notna().rolling(60, min_periods=60).sum().to_numpy()
        hi = np.where(c252 >= 252 - e252, Ad.rolling(252, min_periods=1).max().to_numpy(), np.nan)
        v60 = np.where(c60 >= 60 - e60, Vd.rolling(60, min_periods=1).mean().to_numpy(), np.nan)
    elif mode in ('strict', 'strict_fix'):
        hi = Ad.rolling(252, min_periods=252).max().to_numpy()
        v60 = Vd.rolling(60, min_periods=60).mean().to_numpy()
    else:
        hi = Ad.rolling(252, min_periods=200).max().to_numpy()
        v60 = Vd.rolling(60, min_periods=50).mean().to_numpy()
    s52 = A / hi
    S52 = np.full((len(IDX), N), np.nan); V60 = np.full((len(IDX), N), np.nan)
    S52[g] = s52; V60[g] = v60
    return S52, V60

# ---------------- 營收 ----------------
def yoy_from_long():
    d = pd.read_csv(LONG, dtype={'code': str, 'market': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    # 同代碼同月若上市/上櫃重複，取 sii
    d['pri'] = (d['market'] != 'sii').astype(int)
    d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
    lv = d.pivot(index='p', columns='code', values='revenue_k_twd')
    lv = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq='M'))
    base = lv.shift(12)
    y = (lv / base - 1.0) * 100.0
    return y.where(base > 0)

def yoy_from_rev5y(use_col=True):
    d = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str})
    d['p'] = pd.PeriodIndex(d['ym'], freq='M')
    d = d.drop_duplicates(['p', 'code'], keep='last')
    if use_col:
        y = d.pivot(index='p', columns='code', values='yoy')
    else:
        lv = d.pivot(index='p', columns='code', values='revenue')
        lv = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq='M'))
        y = ((lv / lv.shift(12) - 1) * 100).where(lv.shift(12) > 0)
    return y.reindex(pd.period_range(y.index.min(), y.index.max(), freq='M'))

def yoy3_for_month(Y, m):
    """訊號月 m：用 m−1、m−2、m−3 三個月年增率的平均（三個都要有）。回傳對齊 COLS 的 ndarray。"""
    ms = [m - 1, m - 2, m - 3]
    if any(x not in Y.index for x in ms): return np.full(N, np.nan)
    Z = Y.reindex(index=ms, columns=COLS).to_numpy()
    out = Z.mean(axis=0)
    out[np.isnan(Z).any(axis=0)] = np.nan
    return out

# ---------------- 處置 ----------------
DSP = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
def disposal_mask(d):
    m = np.zeros(N, bool)
    x = DSP[(DSP['start'] <= d) & (DSP['end'] >= d)]
    for c in x['code']:
        if c in CI: m[CI[c]] = True
    return m

# ---------------- 漲停價（跳動點） ----------------
def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)

def limit_up_price(ref, lim):
    raw = ref * (1 + lim)
    t = tick(raw)
    return np.floor(raw / t + 1e-9) * t

def ref_price(t):
    """day t 的參考價：ret 的基準 = close/(1+ret)。flag 1（除權息日 ret 設 0）無法得知 → 用前一筆收盤（偏高，少判漲停）。"""
    c = Cn[t]; r = Rn[t]
    ref = c / (1 + r)
    prev = PREVC[t]
    ref = np.where(Fn[t] == 1, prev, ref)
    return ref

PREVC = C.ffill().shift(1).to_numpy()   # 前一筆收盤（跨停牌）
def gap_up_open(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    ref = ref_price(t)
    lu = limit_up_price(ref, lim)
    return np.isfinite(On[t]) & np.isfinite(ref) & (On[t] >= lu - 1e-6)


class Strategy:
    def __init__(self, Y, start, end=None, sigmode='strict', w52=0.5, use_disposal=True, rev_min_names=100, last_rev_month=None,
                 tiebreak='stable', fin_exclude=None, unavail=None):
        """tiebreak：分數同分時的排序（stable＝代碼欄位順序、reverse、s52、yoy、rand:<seed>）。
        fin_exclude：要把營收設為空值的代碼集合（A 版）。unavail：callable(d)→bool 陣列，True＝訊號日還沒公告新營收（B 版，改用 m−2..m−4）。"""
        if fin_exclude:
            Y = Y.copy(); Y[[c for c in Y.columns if c in fin_exclude]] = np.nan
        self.Y = Y; self.tiebreak = tiebreak; self.unavail = unavail
        self.S52, self.V60 = rolling_signals(sigmode)
        days = rebalance_days(start)
        cnt = Y.notna().sum(axis=1)
        if last_rev_month is None:
            last_rev_month = cnt[cnt >= 0.8 * cnt.median()].index.max()
        keep = []
        for d in days:
            m = pd.Period(d, 'M')
            y3 = yoy3_for_month(Y, m)
            if (m - 1) <= last_rev_month and np.isfinite(y3).sum() > rev_min_names:
                keep.append(d)
        if end is not None: keep = [d for d in keep if d <= pd.Timestamp(end)]
        # 結束日＝最後一個訊號日之後的下一個換股日（只當作評價終點）
        after = [d for d in rebalance_days(start) if d > keep[-1]]
        if after:
            self.sig_days = keep; self.end_day = after[0]
        else:   # 面板最後一個換股日只當評價終點，不交易
            self.sig_days = keep[:-1]; self.end_day = keep[-1]
        self.w52 = w52; self.use_disposal = use_disposal
        self.info = {}

    def cross_section(self, d):
        t = IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = yoy3_for_month(self.Y, pd.Period(d, 'M'))
        if self.unavail is not None:
            U = self.unavail(d)
            if U is not None and U.any():
                y3old = yoy3_for_month(self.Y, pd.Period(d, 'M') - 1)
                y3 = np.where(U, y3old, y3)
        s52 = self.S52[t]
        el = top & np.isfinite(y3) & np.isfinite(s52)
        if self.use_disposal: el &= ~disposal_mask(d)
        el &= np.isfinite(On[e]) & TRADED[e]
        sc = np.full(N, np.nan)
        def pr(x):
            s = pd.Series(np.where(el, x, np.nan)); return s.rank(pct=True).to_numpy()
        r52 = pr(s52); rrv = pr(y3)
        sc = self.w52 * r52 + (1 - self.w52) * rrv
        sc[~el] = np.nan
        return dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3, v60=v60, top=top)

    def order(self, cs):
        sc = np.round(np.nan_to_num(cs['sc'], nan=-9), 10)
        col = np.arange(N)
        tb = self.tiebreak
        if tb == 'stable': keys = (col, -sc)
        elif tb == 'reverse': keys = (-col, -sc)
        elif tb == 's52': keys = (col, -np.nan_to_num(cs['s52'], nan=-9), -sc)
        elif tb == 'yoy': keys = (col, -np.nan_to_num(cs['y3'], nan=-1e9), -sc)
        elif tb.startswith('rand:'):
            rng = np.random.default_rng(int(tb[5:]) * 100003 + cs['t']); keys = (rng.permutation(N), -sc)
        return np.lexsort(keys)

    def plan(self, n, block=True):
        """回傳 [(exec_t, picks ndarray, pool ndarray, sig_day)]。"""
        out = []; held = set(); nblk = 0
        for d in self.sig_days:
            cs = self.cross_section(d)
            el = cs['el']
            if el.sum() < 50: continue
            e = cs['e']; gu = gap_up_open(e) if block else np.zeros(N, bool)
            cand = self.order(cs)[:int(el.sum())]
            pick = []
            for j in cand:
                if gu[j] and j not in held: nblk += 1; continue
                pick.append(j)
                if len(pick) == n: break
            out.append((e, np.array(pick, int), np.where(el)[0], d))
            held = set(pick)
        self.info[f'blocked_n{n}'] = nblk
        return out


def simulate(plan, buy, sell, t0, t1, freeze=True, delist_zero=False, ret_override=None):
    """日頻引擎。plan=[(exec_t, picks, ...)]。回傳 (nav Series（t0..t1 收盤）, log dict)。
    freeze=True：換股日開盤無成交的持股不能賣、保留到可成交；False：以最後收盤價出清（＝既有程式作法）。
    delist_zero=True：之後再也沒有成交的凍結持股市值歸零（下市最壞情境）。"""
    G = GROW if ret_override is None else ret_override
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; frozen_ev = []
    last_trade = np.full(N, -1)
    # 每檔最後成交日（判斷下市）
    lt = np.where(TRADED.any(axis=0), len(IDX) - 1 - np.argmax(TRADED[::-1], axis=0), -1)
    for t in range(t0, t1 + 1):
        if t in reb and t > t0 - 1:
            pk = reb[t]
            tr = TRADED[t] & np.isfinite(On[t])
            # 持股從前一日收盤到今日開盤：(1+ret)×open/close
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            if not freeze:
                frz[:] = False
            if delist_zero:
                dead = frz & (lt <= t)
                h[dead] = 0.0; frz &= ~dead
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0)
            newval[pk] += T   # 若某檔凍結又被選中（不會發生：選中者隔日開盤必有成交）
            # 開盤 → 收盤
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            # 凍結者今天沒成交 → 市值不變；新部位開盤到收盤
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
            if frz.any(): frozen_ev.append((str(IDX[t].date()), [COLS[j] for j in np.where(frz)[0]], float(Vfz / (Vt + Vfz))))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'frozen': frozen_ev}


# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def stats(nav, a, b, sig_days, bench=None, ew=None):
    """日頻年化（a 之後第一筆報酬起到 b）＋月窗（訊號日收盤→下一訊號日收盤，起點落在 [a,b]）t。"""
    r = nav.pct_change()
    m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m]
    i0 = np.argmax(m)
    start = r.index[i0 - 1] if i0 > 0 else r.index[i0]
    yrs = (rr.index[-1] - start).days / 365.25
    eq = (1 + rr).cumprod()
    o = {'CAGR': float(eq.iloc[-1] ** (1 / yrs) - 1), 'maxDD': float((eq / eq.cummax() - 1).min()), 'years': round(yrs, 2)}
    D = pd.DatetimeIndex([d for d in sig_days if d in nav.index])
    nd = nav.reindex(D); pr = (nd.shift(-1) / nd - 1).iloc[:-1]
    pr = pr[(pr.index >= pd.Timestamp(a)) & (pr.index <= pd.Timestamp(b))]
    o['n_m'] = int(pr.notna().sum())
    if bench is not None:
        bd = bench.reindex(D); bp = (bd.shift(-1) / bd - 1).iloc[:-1].reindex(pr.index)
        o['t_0050'] = tstat(pr - bp); o['ex_0050_m'] = float((pr - bp).mean())
        bb = bench.reindex(nav.index).pct_change()[m]
        o['0050_CAGR'] = float((1 + bb.fillna(0)).prod() ** (1 / yrs) - 1)
    if ew is not None:
        ed = ew.reindex(D); ep = (ed.shift(-1) / ed - 1).iloc[:-1].reindex(pr.index)
        o['t_EW'] = tstat(pr - ep)
    o['yearly'] = {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in rr.groupby(rr.index.year)}
    return o

def short(o):
    return {k: (round(v * 100, 2) if k in ('CAGR', 'maxDD', '0050_CAGR', 'ex_0050_m') else (round(v, 2) if isinstance(v, float) else v)) for k, v in o.items() if k != 'yearly'}


# =====================================================================
# breadth_derisk 追加：金融股、廣度 B、曝險／0050 引擎
# =====================================================================
SL = pd.read_csv(X.parent / 'stock_list.csv', dtype=str)
_FIN_LIST = set(SL.loc[SL['industry'] == '金融保險業', 'code'])
FIN = np.array([(c in _FIN_LIST) or c[:2] in ('28', '58') for c in COLS])
FIN_CODES = set(COLS[FIN])

R50 = E50.pct_change().fillna(0.0).to_numpy()
def rates50(slip=0.001):
    if slip == 'stress': return 0.002, 0.002
    return FEE, FEE + 0.001

def pool_top300(t, V60):
    v60 = V60[t]; px = Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, int(base.sum()))]] = True
    return top & base

def breadth_series(start='2005-01-01', thr=0.95, sigmode='strict'):
    S52, V60 = rolling_signals(sigmode)
    out = {}
    for d in rebalance_days(start):
        t = IDX.get_loc(d); top = pool_top300(t, V60); s = S52[t]
        ok = top & np.isfinite(s)
        out[d] = (int((ok & (s >= thr)).sum()), int(ok.sum()))
    return pd.DataFrame(out, index=['B', 'pool']).T


def simulate_x(plan, buy, sell, t0, t1, alt='cash', b50=0.0, s50=0.0):
    """plan=[(exec_t, picks, exposure)]：picks 等權、合計權重 exposure（0～1），其餘 1−exposure 放現金或 0050。
    與 ia_core.simulate（freeze=True）相同的成交／凍結／成本規則；exposure≡1 時應逐日相同。
    0050：執行日以前一日收盤成交（忽略隔夜跳空），新部位吃當日全天報酬。回傳 (nav, stock_weight 日序列, log)。"""
    reb = {p[0]: (p[1], p[2]) for p in plan}
    val = np.zeros(N); cash = 1.0; h50 = 0.0
    nav = np.empty(t1 - t0 + 1); sw = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in reb:
            pk, ex = reb[t]
            ex = float(min(max(ex, 0.0), 1.0))
            if ex <= 1e-12: pk = np.array([], int)
            tr = TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, GROW[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash + h50
            k = len(pk)
            ks = (k / ex) if k > 0 else 1.0
            def cost_of(T):
                tgt = np.zeros(N)
                if k: tgt[pk] = T
                c = buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
                if alt == '0050':
                    g50 = T * (ks - k); c += b50 * max(g50 - h50, 0) + s50 * max(h50 - g50, 0)
                return c
            lo, hi = 0.0, Vt / ks
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * ks + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N)
            if k: tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0); newval += tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            rest = Vt - T * ks - cost_of(T) + T * (ks - k)   # = T*(ks-k)（＋二分法殘差）
            if alt == '0050':
                h50 = rest * (1 + R50[t]); cash = 0.0
            else:
                cash = rest; h50 = 0.0
        else:
            if t > t0:
                val = val * GROW[t]; h50 = h50 * (1 + R50[t])
        nav[t - t0] = val.sum() + cash + h50
        sw[t - t0] = val.sum() / nav[t - t0]
    return (pd.Series(nav, index=IDX[t0:t1 + 1]), pd.Series(sw, index=IDX[t0:t1 + 1]),
            {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan})

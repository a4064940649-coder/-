#!/usr/bin/env python3
"""獨立複製（o_prepriced_growth / replicate_o1）：自己寫的資料、股池、訊號、成交與日頻引擎。
不 import 研究員程式（olib 等），也不 import ia_core / v2_core；只依凍結規格與 BRIEF 的原始資料。"""
import math
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
E50F = Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv')

C = pd.read_parquet(P / 'close_raw.parquet')
IDX = C.index; COLS = list(C.columns); N = len(COLS)
O = pd.read_parquet(P / 'open.parquet').reindex(index=IDX, columns=COLS)
V = pd.read_parquet(P / 'value.parquet').reindex(index=IDX, columns=COLS)
R = pd.read_parquet(P / 'ret.parquet').reindex(index=IDX, columns=COLS)
F = pd.read_parquet(P / 'flag.parquet').reindex(index=IDX, columns=COLS)
ADJP = pd.read_parquet(P / 'adj.parquet').reindex(index=IDX, columns=COLS)
Cn, On, Rn, Fn, Vn = C.to_numpy(), O.to_numpy(), R.to_numpy(), F.to_numpy(), V.to_numpy()
TR = np.isfinite(Cn)
G = np.where(TR, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)          # 日成長因子
ADJ = np.where(TR, np.cumprod(G, axis=0), np.nan)                   # 由 ret 連乘的含息還原收盤

e = pd.read_csv(E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e.reindex(IDX.union(e.index)).ffill().reindex(IDX)

SL = pd.read_csv(X / 'stock_list.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in COLS if c.startswith('28') or c in ('5876', '5880')}
FINM = np.array([c in FIN for c in COLS])
ELEC = set(SL.index[SL['industry'].fillna('').str.contains('電子|半導體|電腦|光電|通信網路|電子零組件|資訊服務')])

# ---------- 換股日 ----------
def reb_days(start, end):
    out, seen = [], set()
    for d in IDX[(IDX >= pd.Timestamp(start)) & (IDX <= pd.Timestamp(end))]:
        if d.day >= 11 and (d.year, d.month) not in seen:
            seen.add((d.year, d.month)); out.append(d)
    return out

# ---------- 滾動窗 gapfix ----------
def _rolling():
    nobs = TR.sum(axis=1)
    g = np.where(nobs >= 300)[0]
    dg = IDX[g]
    ex = dg.isin(pd.DatetimeIndex(['2017-08-03', '2018-07-12'])) | ((dg >= '2024-02-15') & (dg <= '2024-03-15'))
    A = ADJ[g]; Vv = Vn[g]
    mA = pd.DataFrame((np.isnan(A) & ~ex[:, None]).astype(np.int16)).rolling(252, min_periods=252).sum().to_numpy()
    mV = pd.DataFrame((np.isnan(Vv) & ~ex[:, None]).astype(np.int16)).rolling(60, min_periods=60).sum().to_numpy()
    hi = pd.DataFrame(A).rolling(252, min_periods=1).max().to_numpy()
    v60 = pd.DataFrame(Vv).rolling(60, min_periods=1).mean().to_numpy()
    hi = np.where(mA == 0, hi, np.nan); v60 = np.where(mV == 0, v60, np.nan)
    S = np.full(ADJ.shape, np.nan); W = np.full(ADJ.shape, np.nan)
    with np.errstate(invalid='ignore'):
        S[g] = A / hi
    W[g] = v60
    return S, W
S52, V60 = _rolling()

# ---------- 營收（長期） ----------
def _rev():
    d = pd.read_csv(LONG, dtype={'code': str, 'market': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d['pri'] = (d['market'] != 'sii').astype(int)
    d = d.sort_values(['p', 'code', 'pri']).drop_duplicates(['p', 'code'])
    lv = d.pivot(index='p', columns='code', values='revenue_k_twd')
    lv = lv.reindex(pd.period_range(lv.index.min(), lv.index.max(), freq='M')).reindex(columns=COLS)
    b = lv.shift(12)
    y = ((lv / b - 1) * 100).where(b > 0)
    return lv, y
LV, YOY = _rev()

def _rows(D, ms):
    if any(m not in D.index for m in ms): return np.full((len(ms), N), np.nan)
    return D.loc[ms].to_numpy()

def _rev5():
    d = pd.read_csv(X / 'extras' / 'rev_5y.csv.gz', dtype={'code': str})
    d['p'] = pd.PeriodIndex(d['ym'], freq='M')
    y = d.drop_duplicates(['p', 'code'], keep='last').pivot(index='p', columns='code', values='yoy')
    return y.reindex(pd.period_range(y.index.min(), y.index.max(), freq='M')).reindex(columns=COLS)
Y5 = _rev5()
SWITCH = pd.Timestamp('2021-03-01')

def last_full(Y):
    c = Y.notna().sum(axis=1); return c[c >= 0.8 * c.median()].index.max()

def yoy3(m, Y=None):
    Y = YOY if Y is None else Y
    Z = _rows(Y, [m - 1, m - 2, m - 3]); o = Z.mean(axis=0); o[np.isnan(Z).any(axis=0)] = np.nan; return o

NH_POS = 'maxpos'   # 'all'＝規格原文（前 11 個月都 > 0）；'maxpos'＝前 11 個月最大值 > 0 且最近一月 > 0（與研究員數字逐格一致）
def nh(m, pos=None):
    pos = NH_POS if pos is None else pos
    last = _rows(LV, [m - 1])[0]
    prev = _rows(LV, [m - k for k in range(2, 13)])
    ok = np.isfinite(prev).all(axis=0) & np.isfinite(last)
    ok &= (prev > 0).all(axis=0) if pos == 'all' else ((np.where(np.isfinite(prev), prev, -np.inf).max(axis=0) > 0) & (np.nan_to_num(last, nan=-1) > 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        o = last / np.nanmax(np.where(np.isfinite(prev), prev, -np.inf), axis=0)
    o[~ok] = np.nan
    return o

# ---------- 漲停（跳動點） ----------
PREVC = C.ffill().shift(1).to_numpy()
def _tick(p): return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)
def gap_up(t):
    lim = 0.07 if IDX[t] < pd.Timestamp('2015-06-01') else 0.10
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = Cn[t] / (1 + Rn[t])
    ref = np.where(Fn[t] == 1, PREVC[t], ref)
    raw = ref * (1 + lim); tk = _tick(raw)
    lu = np.floor(raw / tk + 1e-9) * tk
    return np.isfinite(On[t]) & np.isfinite(ref) & (On[t] >= lu - 1e-6)

# ---------- 股池 ----------
DSP = pd.read_csv(X / 'extras' / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
CIX = {c: j for j, c in enumerate(COLS)}
def prank(x, el):
    return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()

def pool(d, exfin=True):
    t = IDX.get_loc(d); e = t + 1
    v = V60[t]; px = Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    order = np.argsort(-np.where(base, v, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    y3 = yoy3(pd.Period(d, 'M'), Y5 if d >= SWITCH else YOY)
    if exfin: y3 = np.where(FINM, np.nan, y3)
    s = S52[t]
    el = top & np.isfinite(y3) & np.isfinite(s) & np.isfinite(On[e]) & TR[e]
    # 處置股（disposal_5y：2021-09 起才有事件；挑選期沒有作用）
    x = DSP[(DSP['start'] <= d) & (DSP['end'] >= d)]
    for c in x['code']:
        if c in CIX: el[CIX[c]] = False
    return dict(t=t, e=e, el=el, y3=y3, s52=s, top=top)

def main_score(cs):
    el = cs['el']; sc = 0.5 * prank(cs['s52'], el) + 0.5 * prank(cs['y3'], el); sc[~el] = np.nan; return sc

def leg2_score(cs, d, lo=0.30, hi=0.80, mode='AVG', nanmode='both'):
    el = cs['el']; m = pd.Period(d, 'M')
    p52 = prank(cs['s52'], el)
    acc = yoy3(m) - yoy3(m - 3); h = nh(m)
    ra, rh = prank(acc, el), prank(h, el)
    if mode == 'AVG':
        sc = 0.5 * ra + 0.5 * rh if nanmode == 'both' else np.nanmean(np.vstack([ra, rh]), axis=0)
    elif mode == 'ACC': sc = ra
    elif mode == 'NH': sc = rh
    band = el & (p52 >= lo - 1e-12) & (p52 <= hi + 1e-12)
    sc = np.where(band, sc, np.nan)
    return sc, dict(p52=p52, acc=acc, nh=h, ra=ra, rh=rh)

def order(sc, y3):
    s = np.round(np.nan_to_num(sc, nan=-9), 10)
    return np.lexsort((np.arange(N), -np.nan_to_num(y3, nan=-1e18), -s))

def make_plan(days, n, scorer, block=True):
    out, held = [], set()
    for d in days:
        cs = pool(d)
        sc = scorer(cs, d)
        ok = np.isfinite(sc)
        cand = order(sc, cs['y3'])[:int(ok.sum())]
        gu = gap_up(cs['e']) if block else np.zeros(N, bool)
        pk = []
        for j in cand:
            if gu[j] and j not in held: continue
            pk.append(j)
            if len(pk) == n: break
        out.append((cs['e'], np.array(pk, int), d))
        held = set(pk)
    return out

# ---------- 成本 ----------
FEE = 0.001425 * 0.6
def rates(kind='normal'):
    if kind == 'stress': return 0.005, 0.005
    if kind == 'zero': return 0.0, 0.0
    return FEE + 0.001, FEE + 0.003 + 0.001

# ---------- 日頻引擎（權重版） ----------
def simulate_w(wplan, buy, sell, t0, t1):
    """wplan：{exec_t: 權重向量（和 1）}。換股日開盤成交，停牌持股凍結。回傳 nav（t0..t1 收盤）、單邊周轉列表。"""
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []
    for t in range(t0, t1 + 1):
        if t in wplan:
            w = wplan[t]
            tr = TR[t] & np.isfinite(On[t])
            h = val * np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            Vf = h[frz].sum(); ht = np.where(frz, 0.0, h); Vt = ht.sum() + cash
            def cost(T): tg = w * T; return buy * np.clip(tg - ht, 0, None).sum() + sell * np.clip(ht - tg, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if mid + cost(mid) > Vt: hi = mid
                else: lo = mid
            tg = w * lo
            tov.append(np.clip(ht - tg, 0, None).sum() / (Vt + Vf))
            nv = np.where(frz, h, 0.0) + tg
            val = np.where(frz, nv, nv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)); cash = 0.0
        elif t > t0:
            val = val * G[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), tov

def to_w(plan):
    out = {}
    for e, pk, d in plan:
        w = np.zeros(N); w[pk] = 1.0 / len(pk); out[e] = w
    return out

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))

def win_ret(s, days):
    v = s.reindex(pd.DatetimeIndex(days)).to_numpy()
    return v[1:] / v[:-1] - 1

def cagr_dd(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1, float((nav / nav.cummax() - 1).min())

def yearly(nav):
    r = nav.pct_change().dropna()
    return {int(y): round(float((1 + g).prod() - 1) * 100, 1) for y, g in r.groupby(r.index.year)}

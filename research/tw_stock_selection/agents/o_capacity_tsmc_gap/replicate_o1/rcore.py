#!/usr/bin/env python3
"""獨立複製（replicate_o1）核心：資料、訊號、選股、日頻引擎、統計。
只依凍結規格與 BRIEF 的原始資料從頭寫；不 import 研究中心任何程式。"""
import math
from pathlib import Path
import numpy as np, pandas as pd

P = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel')
D = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data')
X = D / 'extras'
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
HERE = Path(__file__).resolve().parent
EVAL_END = pd.Timestamp('2026-10-12')

FEE = 0.001425 * 0.6
TAX = 0.003
SLIP = 0.001
COST = {'normal': (FEE + SLIP, FEE + TAX + SLIP), 'stress': (0.005, 0.005)}

# ---------------- 面板 ----------------
adj = pd.read_parquet(P / 'adj.parquet')
C = pd.read_parquet(P / 'close_raw.parquet')
O = pd.read_parquet(P / 'open.parquet')
V = pd.read_parquet(P / 'value.parquet')
IDX = adj.index
COLS = list(adj.columns)
N = len(COLS)
CI = {c: j for j, c in enumerate(COLS)}
for df in (C, O, V):
    assert df.index.equals(IDX) and list(df.columns) == COLS

ADJ = adj.to_numpy()
AC = adj.ffill().to_numpy()                      # 評價用（停牌沿用最後還原收盤）
FAC = (adj / C).to_numpy()
OPEN = O.to_numpy()
AO = OPEN * FAC                                  # 還原開盤
CR = C.to_numpy()
CR_ff = C.ffill().to_numpy()
TRAD = np.isfinite(OPEN) & np.isfinite(ADJ) & np.isfinite(FAC)

e50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date')['adj'].sort_index()
E50 = e50.reindex(IDX, method='ffill')

# ---------------- 滾動窗（gapfix） ----------------
GOOD = adj.notna().sum(axis=1) >= 300
EXEMPT = pd.DatetimeIndex([pd.Timestamp('2017-08-03'), pd.Timestamp('2018-07-12')]).append(
    IDX[(IDX >= pd.Timestamp('2024-02-15')) & (IDX <= pd.Timestamp('2024-03-15'))])


def _roll_gapfix(df, win, how):
    g = df.loc[GOOD]
    ex = g.index.isin(EXEMPT)
    bad = (g.isna() & ~ex[:, None]).astype(float)
    nbad = bad.rolling(win, min_periods=win).sum()
    if how == 'max':
        val = g.rolling(win, min_periods=1).max()
    else:
        val = g.rolling(win, min_periods=1).mean()
    out = val.where(nbad == 0)
    return out.reindex(IDX)


HI252 = _roll_gapfix(adj, 252, 'max')
S52 = adj / HI252
V60 = _roll_gapfix(V, 60, 'mean')

# ---------------- 營收 ----------------

def _yoy_long():
    d = pd.read_csv(LONG, dtype={'code': str, 'market': str})
    d['p'] = pd.PeriodIndex(pd.to_datetime(d['month']), freq='M')
    d['mk'] = (d['market'] != 'sii').astype(int)
    d = d.sort_values(['p', 'mk']).drop_duplicates(['p', 'code'], keep='first')
    d = d[d['code'].isin(COLS)]
    L = d.pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
    L = L.reindex(pd.period_range(L.index.min(), L.index.max(), freq='M'))
    prev = L.shift(12)
    return ((L / prev - 1) * 100).where(prev > 0)


def _yoy_5y():
    r = pd.read_csv(X / 'rev_5y.csv.gz', dtype={'code': str})
    r = r[r['code'].isin(COLS)]
    r['p'] = pd.PeriodIndex(r['ym'], freq='M')
    Y = r.pivot_table(index='p', columns='code', values='yoy', aggfunc='last').sort_index()
    return Y.reindex(pd.period_range(Y.index.min(), Y.index.max(), freq='M'))


YL = _yoy_long(); YL3 = YL.rolling(3, min_periods=3).mean().reindex(columns=COLS)
Y5 = _yoy_5y(); Y53 = Y5.rolling(3, min_periods=3).mean().reindex(columns=COLS)


def yoy3_at(d):
    m = pd.Period(d, 'M') - 1
    T = YL3 if d < pd.Timestamp('2021-03-01') else Y53
    if m not in T.index:
        return np.full(N, np.nan)
    return T.loc[m].to_numpy(dtype=float)

# ---------------- 金融、處置 ----------------
SL = pd.read_csv(D / 'stock_list.csv', dtype={'code': str})
FIN = set(SL.loc[SL['industry'] == '金融保險業', 'code']) | {c for c in COLS if c.startswith('28')} | {'5876', '5880'}
FINMASK = np.array([c in FIN for c in COLS])
DSP = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])


def in_disposal(d):
    m = np.zeros(N, bool)
    r = DSP[(DSP['start'] <= d) & (DSP['end'] >= d)]
    for c in r['code']:
        if c in CI: m[CI[c]] = True
    return m

# ---------------- 跳動點漲停 ----------------

def tick(p):
    return np.select([p < 10, p < 50, p < 100, p < 500, p < 1000], [0.01, 0.05, 0.1, 0.5, 1.0], 5.0)


def limit_up_price(ref, lim):
    raw = ref * (1 + lim)
    t = tick(raw)
    return np.floor(raw / t + 1e-9) * t


def open_limit_up(i):
    """第 i 日開盤是否 ≥ 跳動點漲停價（參考價＝前一有收盤日的原始收盤）。"""
    lim = 0.07 if IDX[i] < pd.Timestamp('2015-06-01') else 0.10
    ref = CR_ff[i - 1]
    lp = limit_up_price(ref, lim)
    with np.errstate(invalid='ignore'):
        return np.isfinite(OPEN[i]) & np.isfinite(lp) & (OPEN[i] >= np.round(lp, 4) - 1e-6)

# ---------------- 訊號日 ----------------

def signal_dates(start='2006-04-01', end=EVAL_END):
    s = pd.Series(IDX, index=IDX)
    s = s[(s >= pd.Timestamp(start)) & (s <= pd.Timestamp(end))]
    out = []
    for _, g in s.groupby([s.index.year, s.index.month]):
        g = g[g.dt.day >= 11]
        if len(g): out.append(g.iloc[0])
    return pd.DatetimeIndex(out)


SIG = signal_dates()
POS = {d: IDX.get_loc(d) for d in SIG}

# ---------------- 橫截面 ----------------

class Cross:
    """訊號日 d 的橫截面：前 300 名、合格池、分數、排序。"""
    def __init__(self, d, exfin=True, rank_lo=1, rank_hi=300):
        i = POS[d]; self.d = d; self.i = i; self.ex = i + 1
        v60 = V60.iloc[i].to_numpy(dtype=float); px = CR[i]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        rk = pd.Series(np.where(base, v60, np.nan)).rank(ascending=False, method='first').to_numpy()
        self.vrank = rk
        top = base & (rk >= rank_lo) & (rk <= rank_hi)
        s52 = S52.iloc[i].to_numpy(dtype=float)
        y3 = yoy3_at(d).copy()
        if exfin: y3[FINMASK] = np.nan
        dis = in_disposal(d)
        trad = TRAD[self.ex] if self.ex < len(IDX) else np.zeros(N, bool)
        el = top & np.isfinite(s52) & np.isfinite(y3) & ~dis & trad
        self.el = el
        r52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
        rrv = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
        sc = 0.5 * r52 + 0.5 * rrv
        self.score = sc; self.y3 = y3; self.s52 = s52
        idx = np.where(el)[0]
        key = sorted(idx, key=lambda j: (-round(sc[j], 10), -y3[j], COLS[j]))
        self.order = np.array(key, dtype=int)


def pick(cr, n, held, gu):
    out = []
    for j in cr.order:
        if gu[j] and j not in held:
            continue
        out.append(j)
        if len(out) == n: break
    return out

# ---------------- 引擎 ----------------

def simulate(events, cost='normal', t0=None, t1=None, release='hold'):
    """events: {exec_idx: {col: weight}}（權重和為 1，對可交易資金）。
    換股日開盤成交；停牌（當日無開盤）的持股凍結，恢復交易首日開盤賣出轉現金，下次換股再投入。"""
    buy, sell = COST[cost]
    if t0 is None: t0 = min(events) - 1
    if t1 is None: t1 = IDX.get_loc(EVAL_END)
    u = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); frozen = np.zeros(N, bool)
    for t in range(t0, t1 + 1):
        if frozen.any() and release == 'cash':
            rel = frozen & TRAD[t]
            if rel.any() and t not in events:
                cash += float((u[rel] * AO[t, rel]).sum()) * (1 - sell)
                u[rel] = 0; frozen &= ~rel
        if t in events:
            wts = events[t]
            held = u > 0
            fz = held & ~TRAD[t]
            h = np.where(held & ~fz, u * np.nan_to_num(AO[t]), 0.0)
            Vt = h.sum() + cash
            tgt = np.zeros(N)
            cols = np.array(list(wts.keys()), dtype=int); w = np.array(list(wts.values()), dtype=float)
            w = w / w.sum()
            Vn = Vt
            for _ in range(60):
                tgt[:] = 0; tgt[cols] = w * Vn
                c = buy * np.clip(tgt - h, 0, None).sum() + sell * np.clip(h - tgt, 0, None).sum()
                Vn = Vt - c
            unew = np.zeros(N)
            unew[cols] = tgt[cols] / AO[t, cols]
            unew[fz] = u[fz]
            u = unew; cash = 0.0; frozen = fz
        nav[t - t0] = float(np.nansum(u * AC[t])) + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

# ---------------- 統計 ----------------

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def cagr(nav, a, b):
    """a（含）起算：以 a 之前最後一日為基期。"""
    nav = nav.dropna()
    s = nav[nav.index < pd.Timestamp(a)]
    base_d = s.index[-1] if len(s) else nav.index[0]
    e = nav[nav.index <= pd.Timestamp(b)]
    end_d = e.index[-1]
    yrs = (end_d - base_d).days / 365.25
    return float((nav[end_d] / nav[base_d]) ** (1 / yrs) - 1)


def maxdd(nav, a, b):
    nav = nav.dropna()
    s = nav[nav.index < pd.Timestamp(a)]
    base_d = s.index[-1] if len(s) else nav.index[0]
    x = nav[(nav.index >= base_d) & (nav.index <= pd.Timestamp(b))]
    return float((x / x.cummax() - 1).min())


def win_returns(nav, dates):
    nd = nav.reindex(dates)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

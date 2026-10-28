#!/usr/bin/env python3
"""replicate_o5 核心：獨立複製（只 import 共用引擎 v4lib；不讀 o_rev_event_trigger 研究員程式）。"""
import sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_v4_revenue_patch_official'))
import numpy as np, pandas as pd
import v4lib as L
sys.path.pop(0)
K, V, R = L.K, L.V, L.R
IDX, COLS, N = K.IDX, K.COLS, K.N

XR = L.setup_px()
YFX, Y5F = L.load_rfix()
DMT = L.disp_mask_fn('trade')

TX = pd.DatetimeIndex(sorted(pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']))
EXT = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}

def deadline(m):
    """錨定月 m（Period）→ 營收 m−1 的法定期限 D_m。"""
    if str(m) in EXT: return EXT[str(m)]
    return TX[TX >= pd.Timestamp(m.year, m.month, 10)][0]

def deadline15(m):
    return TX[TX >= pd.Timestamp(m.year, m.month, 15)][0]

def next_panel(d):
    i = IDX.searchsorted(d, side='right'); return IDX[i]

def official():
    S = L.V4(YFX, Y5F, DMT)
    return S

OFF = official()
OFF_DAYS = list(OFF.sig_days)          # 245 個（不含終點 2026-10-12）
OFF_MONTHS = [pd.Period(d, 'M') for d in OFF_DAYS] + [pd.Period(OFF.end_day, 'M')]

def j_day(m, j):
    """j=+1：D 之後第 1 個交易日；j=0：D 當天；j=k：D 之後第 k 個交易日（taiex 日曆）。"""
    D = deadline(m)
    i = TX.get_loc(D) + j
    return TX[i] if i < len(TX) else pd.Timestamp('2100-01-01')

def build_days(j):
    """回傳 (sig_days 不含終點, end_day, 錨定月對照, 跳過月)。"""
    days, skipped, amap = [], [], {}
    for m in OFF_MONTHS:
        d = j_day(m, j)
        if d not in IDX:
            skipped.append((str(m), str(d.date()), 'not in panel')); continue
        t = IDX.get_loc(d)
        if t + 1 >= len(IDX) or not (K.GOOD[t] and K.GOOD[t + 1]):
            skipped.append((str(m), str(d.date()), 'not GOOD')); continue
        days.append(d); amap[d] = m
    return days[:-1], days[-1], amap, skipped

PCOL = V.PCOL; IS15 = V.IS15

def make_unavail(amap, seed, mode='day', q=None, tag=7):
    """day 模式（自寫）：訊號日 < D 全未公告；= D 依 p 抽樣；2020-02 錨定月 D<訊號日≤2020-02-14 也抽 p；
    2026-01..08 營收：INS15 訊號日 < D15 未公告（= D15 抽 p）。mode='night'：= D 時以機率 q 未公告（J0N）。
    mode='worst'：= D 時全未公告。mode='none'：全部已公告（非 AB）。"""
    cache = {}
    def U(d):
        if d in cache: return cache[d]
        m = amap.get(d, pd.Period(d, 'M')); D = deadline(m)
        rm = str(m - 1)
        ymd = int(d.strftime('%Y%m%d'))
        rng = np.random.default_rng([seed, ymd, tag])
        u = np.zeros(N, bool)
        if mode == 'none':
            cache[d] = u; return u
        if d < D:
            u[:] = True
        elif d == D:
            if mode == 'worst': u[:] = True
            elif mode == 'night': u = rng.random(N) < q
            else: u = rng.random(N) < PCOL
        elif str(m) == '2020-02' and d <= pd.Timestamp('2020-02-14'):
            u = rng.random(N) < PCOL
        if '2026-01' <= rm <= '2026-08':
            D15 = deadline15(m)
            if d < D15: u |= IS15
            elif d == D15: u |= IS15 & (np.random.default_rng([seed, ymd, tag + 1]).random(N) < PCOL)
        cache[d] = u
        return u
    return U

def mk(days, end, unavail=None):
    S = L.V4(YFX, Y5F, DMT, unavail=unavail)
    S.sig_days = list(days); S.end_day = end; S._cs = {}
    return S

def run(S, costs=('normal',)):
    t0, t1, sd = S.bounds()
    pl = S.plan(10, 'official')
    out = {}
    for c in costs:
        out[c], _ = L.sim_plan(pl, t0, t1, c, XR)
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
    return out, ew, sd, pl

def mean_nav(lst):
    return (1 + pd.concat([x.pct_change() for x in lst], axis=1).mean(axis=1).fillna(0)).cumprod()

# ---------------- 統計 ----------------
E50 = K.E50
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))

def winx(nav, sd, bench=None):
    bench = E50 if bench is None else bench
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    a = nav.reindex(D); b = bench.reindex(D)
    return ((a.shift(-1) / a) - (b.shift(-1) / b)).iloc[:-1]

def cagr(nav, a, b):
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    yrs = (b - a).days / 365.25
    return (nav.loc[b] / nav.loc[a]) ** (1 / yrs) - 1

def mdd(nav, a, b):
    s = nav.loc[pd.Timestamp(a):pd.Timestamp(b)]
    return float((s / s.cummax() - 1).min())

def cal_x(nav, bench=None, a='2006-05', b='2026-09'):
    bench = E50 if bench is None else bench
    me = nav.groupby(nav.index.to_period('M')).last(); be = bench.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    x = me.pct_change() - be.pct_change()
    return x[(x.index >= pd.Period(a, 'M')) & (x.index <= pd.Period(b, 'M'))].dropna()

SEGS = {'ALL': ('2006-04', '2026-10'), 'SEL': ('2006-04', '2018-12'), 'VAL': ('2019-01', '2023-12'), 'HOLD': ('2024-01', '2026-10'),
        'S1': ('2006-04', '2013-12'), 'S2': ('2014-01', '2020-12'), 'S3': ('2021-03', '2026-10')}

def seg_stats(nav, ew, sd, amap, a, b, endd):
    """t：窗依錨定月歸段；年化：段內第一個訊號日收盤 → 段後第一個訊號日收盤（或終點）。"""
    full = list(sd)  # 含終點
    x = winx(nav, full); xe = winx(nav, full, ew)
    am = pd.PeriodIndex([amap[d] for d in x.index], freq='M')
    sel = (am >= pd.Period(a, 'M')) & (am <= pd.Period(b, 'M'))
    xs = x[sel]; xes = xe[sel]
    inside = [d for d in full[:-1] if pd.Period(a, 'M') <= amap[d] <= pd.Period(b, 'M')]
    st = inside[0]; after = [d for d in full if d > inside[-1]]
    en = after[0] if after else endd
    return dict(CAGR=cagr(nav, st, en) * 100, E50=cagr(E50, st, en) * 100, EW=cagr(ew, st, en) * 100, mdd=mdd(nav, st, en) * 100,
                t=tstat(xs), mean_pp=float(xs.mean() * 100), n=int(len(xs)), t_EW=tstat(xes), start=str(st.date()), end=str(en.date()))

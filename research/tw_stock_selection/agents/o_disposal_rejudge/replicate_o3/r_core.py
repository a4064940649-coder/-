#!/usr/bin/env python3
"""獨立複製核心。
基礎資料、滾動訊號與營收：只讀 import main_v2/v2_core（＝凍結規格指定的主策略定義，非研究員程式）。
自己寫：處置狀態、E 版本規則、排名入選、逐檔成本日頻引擎、統計。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
MV2 = HERE.parent.parent / 'main_v2'
sys.path.insert(0, str(MV2))
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
CI = {c: j for j, c in enumerate(COLS)}
Cn, On, TRADED, GROW = K.Cn, K.On, K.TRADED, K.GROW
FEE = 0.001425 * 0.6
BUY, SELL = FEE + 0.001, FEE + 0.003 + 0.001

YL, Y5, _ = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
SIG = S.sig_days; END = S.end_day
T0, T1 = IDX.get_loc(SIG[0]), IDX.get_loc(END)

# ---------- 處置名單 ----------
LIST = pd.read_csv(HERE / 'my_list.csv', dtype={'code': str}, parse_dates=['ann', 'start', 'end'])

def tdays(a, b):
    return int(((IDX >= a) & (IDX <= b)).sum())

def prep(L):
    L = L[L['code'].isin(CI)].copy()
    L['L'] = [tdays(s, e) for s, e in zip(L['start'], L['end'])]
    return L

LST = prep(LIST)

def state(d, e, L=LST):
    """訊號日 d 的處置狀態：回傳 dict j -> (nth, rem, L, start, end)。"""
    x = L[(L['ann'] <= d) & (L['start'] <= d) & (L['end'] >= d)]
    out = {}
    for c, g in x.groupby('code'):
        g = g.sort_values('end')
        end = g['end'].max(); nth = int(g['nth'].max()); Ln = int(g['L'].iloc[-1])
        rem = int(((IDX >= IDX[e]) & (IDX <= end)).sum())
        out[CI[c]] = (nth, rem, Ln, g['start'].min(), end)
    return out

def rule(name):
    """回傳 f(nth, rem, L) -> True＝排除。"""
    if name == 'E0': return lambda n, r, L: False
    if name == 'E1': return lambda n, r, L: True
    if name == 'E2': return lambda n, r, L: n >= 2
    if name.startswith('E3'):
        f = float(name.split('@')[1]) if '@' in name else 0.6
        return lambda n, r, L: r >= math.ceil(f * L)
    if name.startswith('E5'):
        f = float(name.split('@')[1]) if '@' in name else 0.6
        return lambda n, r, L: n >= 2 or r >= math.ceil(f * L)
    raise ValueError(name)

_CS = {}
def base_cs(d):
    if d in _CS: return _CS[d]
    t = IDX.get_loc(d); e = t + 1
    v60 = S.V60[t]; px = Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    order = np.argsort(-np.where(base, v60, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    y3 = S.y3(d); s52 = S.S52[t]
    el0 = top & np.isfinite(y3) & np.isfinite(s52) & np.isfinite(On[e]) & TRADED[e]
    _CS[d] = (t, e, el0, y3, s52)
    return _CS[d]

def plan(ver, n, L=LST, official_mask=False):
    f = rule(ver)
    out = []; held = set()
    for d in SIG:
        t, e, el0, y3, s52 = base_cs(d)
        el = el0.copy()
        st = state(d, e, L)
        if official_mask:
            el &= ~K.disposal_mask(d)
        else:
            for j, (nth, rem, Ln, a, b) in st.items():
                if f(nth, rem, Ln): el[j] = False
        if el.sum() < 50: continue
        pr = lambda x: pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
        sc = 0.5 * pr(s52) + 0.5 * pr(y3); sc[~el] = np.nan
        scr = np.round(np.nan_to_num(sc, nan=-9), 10)
        od = np.lexsort((np.arange(N), -np.nan_to_num(y3, nan=-1e18), -scr))[:int(el.sum())]
        gu = K.gap_up_open(e)
        pick = []
        for j in od:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        out.append(dict(d=d, t=t, e=e, pick=np.array(pick, int), pool=np.where(el)[0], st=st))
        held = set(pick)
    return out

def in_disp_mask(t, L=LST):
    """day t 是否處於處置期間（start ≤ IDX[t] ≤ end）"""
    d = IDX[t]
    x = L[(L['start'] <= d) & (L['end'] >= d)]
    m = np.zeros(N, bool)
    for c in x['code']: m[CI[c]] = True
    return m

def simulate(pl, buy=BUY, sell=SELL, t0=T0, t1=T1, extra=0.0, L=LST):
    """逐日引擎：停牌凍結、二分法解調倉後總值；extra＞0 時處置期間內的交易每邊加 extra。"""
    reb = {p['e']: p['pick'] for p in pl}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, GROW[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            b = np.full(N, buy); s = np.full(N, sell)
            if extra:
                dm = in_disp_mask(t, L); b = b + extra * dm; s = s + extra * dm
            k = len(pk)
            def tot(T):
                tg = np.zeros(N); tg[pk] = T
                return k * T + (b * np.clip(tg - htr, 0, None)).sum() + (s * np.clip(htr - tg, 0, None)).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if tot(mid) > Vt: hi = mid
                else: lo = mid
            nv = np.where(frz, h, 0.0); nv[pk] += lo
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, nv, nv * oc); cash = 0.0
        elif t > t0:
            val = val * GROW[t]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1])

# ---------- 統計 ----------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    sd = x.std(ddof=1)
    return float(x.mean() / (sd / math.sqrt(len(x))))

SIGX = pd.DatetimeIndex(list(SIG) + [END])
def win_ret(nav):
    nd = nav.reindex(SIGX); return (nd.shift(-1) / nd - 1).iloc[:-1]

E50 = K.E50
PER = {'sel': ('2006-04-01', '2021-08-31'), 'val': ('2021-09-01', '2023-12-31'), 'hold': ('2024-01-01', '2026-10-12'),
       'oos': ('2021-09-01', '2026-10-12'), 'all': ('2006-04-01', '2026-10-12')}

def sub(s, p):
    a, b = PER[p]; return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]

def paired(navA, navB, p):
    x = sub(win_ret(navA) - win_ret(navB), p).dropna()
    return round(float(x.mean() * 100), 3), round(tstat(x), 2), len(x)

def cagr(nav, p, bench=None):
    a, b = PER[p]
    r = nav.pct_change(); m = (r.index >= pd.Timestamp(a)) & (r.index <= pd.Timestamp(b)) & r.notna().to_numpy()
    rr = r[m]; i0 = np.argmax(m); st = r.index[i0 - 1] if i0 > 0 else r.index[i0]
    yrs = (rr.index[-1] - st).days / 365.25
    eq = (1 + rr).cumprod()
    o = dict(CAGR=round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 2), maxDD=round(float((eq / eq.cummax() - 1).min()) * 100, 1))
    br = E50.reindex(nav.index).pct_change()[m]
    o['E50'] = round(float((1 + br.fillna(0)).prod() ** (1 / yrs) - 1) * 100, 2)
    ex = sub(win_ret(nav) - win_ret(E50.reindex(nav.index)), p).dropna()
    o['t50'] = round(tstat(ex), 2); o['ex_m'] = round(float(ex.mean() * 100), 3); o['sd_m'] = round(float(ex.std(ddof=1) * 100), 2)
    return o

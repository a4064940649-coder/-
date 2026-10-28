"""獨立複製用的兩腿再平衡引擎與統計（自寫）。"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
NAV = pd.read_parquet(HERE / 'my_navs.parquet')
G2330 = pd.read_parquet(HERE / 'my_2330.parquet')['g2330']
META = json.loads((HERE / 'my_plan.json').read_text(encoding='utf-8'))
SD = pd.DatetimeIndex(pd.to_datetime(META['sd']))
IDX = NAV.index
FEE = 0.001425 * 0.6
def leg_rates(kind, stress=False, slip=None):
    a = 0.003 if stress else 0.0
    if kind == 'main' or kind == 'stock':
        s = 0.001 if slip is None else slip
        return FEE + s + a, FEE + 0.003 + s + a
    s = 0.0005 if slip is None else slip
    return FEE + s + a, FEE + 0.001 + s + a
SEG = {'SEL': ('2006-04-11', '2019-01-11'), 'VAL': ('2019-01-11', '2024-02-15'), 'HOLD': ('2024-02-15', '2026-10-12'),
       'POST': ('2019-01-11', '2026-10-12'), 'ALL': ('2006-04-11', '2026-10-12')}
def growth(s):
    g = (s / s.shift(1)).to_numpy().copy(); g[0] = 1.0
    return np.where(np.isfinite(g), g, 1.0)
GM = {'normal': growth(NAV['normal']), 'stress3': growth(NAV['stress3']), 'nodisc': growth(NAV['nodisc'])}
GF = growth(NAV['e50'])
GS = G2330.reindex(IDX).fillna(1.0).to_numpy().copy(); GS[0] = 1.0
SDPOS = np.array([IDX.get_loc(d) for d in SD])
def run(w, rule, a, b, gm, gf, rb_m, rb_f, months=None, init=None):
    """回傳 (nav Series a..b, 每日 0050 權重, 移轉 list)。rb_m/rb_f=(buy, sell)。rule：M/Q/H/A/NONE/B5/B10/B15/MOM12/A{m}"""
    ia, ib = IDX.get_loc(pd.Timestamp(a)), IDX.get_loc(pd.Timestamp(b))
    sdset = {int(p) for p in SDPOS if ia < p < ib}
    M, F = (1 - w), w
    if init is not None: M, F = init
    nav = np.empty(ib - ia + 1); wt = np.empty(ib - ia + 1); trans = []
    nav[0] = M + F; wt[0] = F / (M + F)
    for t in range(ia + 1, ib + 1):
        M *= gm[t]; F *= gf[t]
        if t in sdset:
            V = M + F; cw = F / V; do = False
            mo = IDX[t].month
            if rule == 'M': do = True
            elif rule == 'Q': do = mo in (1, 4, 7, 10)
            elif rule == 'H': do = mo in (1, 7)
            elif rule == 'A': do = mo == 1
            elif rule.startswith('A') and rule[1:].isdigit(): do = mo == int(rule[1:])
            elif rule.startswith('B'): do = abs(cw - w) > int(rule[1:]) / 100 + 1e-12
            elif rule == 'MOM12':
                if abs(cw - w) > 1e-12:
                    k = int(np.searchsorted(SDPOS, t))
                    j0 = SDPOS[max(0, k - 12)]
                    rm = np.prod(gm[j0 + 1:t + 1]); rf = np.prod(gf[j0 + 1:t + 1])
                    over_f = cw > w
                    rel = (rf / rm - 1) if over_f else (rm / rf - 1)
                    do = not (k >= 1 and rel > 0)
            if do and abs(cw - w) > 1e-15:
                if cw < w:   # main → 0050
                    k_ = 1 - rb_m[1] - rb_f[0]
                    x = (w * V - F) / (k_ + w * (1 - k_))
                    M -= x; F += x * k_
                else:
                    k_ = 1 - rb_f[1] - rb_m[0]
                    x = (F - w * V) / (1 - w * (1 - k_))
                    F -= x; M += x * k_
                trans.append((IDX[t], x / V, (1 - k_) * x / V))
        nav[t - ia] = M + F; wt[t - ia] = F / (M + F)
    return pd.Series(nav, index=IDX[ia:ib + 1]), pd.Series(wt, index=IDX[ia:ib + 1]), trans
def cagr(s, base_prev=False):
    v0 = s.iloc[0]; d0 = s.index[0]
    yrs = (s.index[-1] - d0).days / 365.25
    return (s.iloc[-1] / v0) ** (1 / yrs) * 100 - 100
def mdd(s): return float((s / s.cummax() - 1).min() * 100)
def winret(s):
    d = SD[(SD >= s.index[0]) & (SD <= s.index[-1])]
    x = s.reindex(d)
    return (x.shift(-1) / x - 1).iloc[:-1]
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))
def seg_nav(col, a, b):
    s = NAV[col]; s = s[(s.index >= a) & (s.index <= b)]; return s / s.iloc[0]
def stats(s, bench):
    r = winret(s); rb = winret(bench); d = r - rb
    return dict(CAGR=cagr(s), MDD=mdd(s), t=tstat(d), TE=float(d.std(ddof=1) * math.sqrt(12) * 100), n=len(d))
def paired(s1, s2):
    d = winret(s1) - winret(s2)
    return dict(dCAGR=cagr(s1) - cagr(s2), t=tstat(d), n=len(d))

#!/usr/bin/env python3
"""格點執行：每個 j 一條路徑；同一路徑上一次算好所有格點的權重（各自的緩衝帶狀態），再跑紙上引擎。
格點：訊號 {REV3, RS, H52, NH, COMP} × k {3,5} × A {15%,30%} × 頻率 {M 每月, B2 雙月（奇數錨定月換股）}＝40 組。
參數 P（迭代時才可改）：sect_min（產業合併門檻）、cap_mult（超配上限倍數）、buf（緩衝帶寬度）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lclib as X
L, TC = X.L, X.TC
IDX, N = X.IDX, X.N

SIGS = ['REV3', 'RS', 'H52', 'NH', 'COMP']
KS = [3, 5]; AS = [0.15, 0.30]; FREQS = ['M', 'B2']
def cfg_name(s, k, A, f): return f'{s}|k{k}|A{int(round(A * 100))}|{f}'
GRID = [(s, k, A, f) for s in SIGS for k in KS for A in AS for f in FREQS]
DEFAULT = cfg_name('COMP', 5, 0.30, 'M')
P0 = dict(sect_min=2, cap_mult=3.0, buf=2)

_G = {}
def init():
    if _G: return _G
    caps = X.Caps()
    sidx = X.SectorIndex(caps)
    sig = X.Signals(caps, sidx)
    _G.update(caps=caps, sidx=sidx, sig=sig)
    return _G

_CS = {}
def cs_at(d, P):
    key = (pd.Timestamp(d), P['sect_min'])
    if key in _CS: return _CS[key]
    g = init()
    X.SECT_MIN = P['sect_min']
    cs = g['sig'].at(d); _CS[key] = cs
    return cs

def is_reb(d, f):
    if f == 'M': return True
    return pd.Timestamp(d).month % 2 == 1

def build(j, m0, m1, mend, P=P0, grid=GRID, placebo=None, keep_cs=False):
    """回傳 dict：days、plans{cfg: {d: (idx,w)|None}}、Bplan、secrec（每訊號日：產業表、各格產業權重）。
    placebo：dict(scores=callable(cs, state)->{sector: score}) 用隨機排名取代訊號（只對 grid 內各格）。"""
    days = X.jdays(j, m0, m1, mend)
    plans = {cfg_name(*c): {} for c in grid}; held = {cfg_name(*c): set() for c in grid}
    Bplan = {}; rec = {}
    for i, d in enumerate(days[:-1]):
        cs = cs_at(d, P)
        Bplan[d] = (cs['U'], cs['b'])
        rs = placebo['scores'](cs) if placebo else None
        secw = {}
        for c in grid:
            nm = cfg_name(*c); s, k, A, f = c
            if i > 0 and not is_reb(d, f):
                plans[nm][d] = None; continue
            U, w, h2, info = X.sector_targets(cs, s, k, A, held[nm], buf=P['buf'], cap_mult=P['cap_mult'], rng_scores=rs)
            held[nm] = h2; plans[nm][d] = (U, w)
            secw[nm] = dict(top=info.get('top', []), over=info.get('over', 0.0))
        rec[d] = dict(ST=cs['ST'] if keep_cs else None, secw=secw, imp_share=cs['imp_share'], n_rot=len(cs['ST']),
                      rot_B=float(cs['ST']['B'].sum() / cs['mc'].sum()) if len(cs['ST']) else 0.0,
                      w2330=float(cs['b'][cs['U'] == X.J2330].sum()), wfin=float(cs['b'][cs['fin']].sum()), cs=cs if keep_cs else None)
    return dict(days=days, plans=plans, Bplan=Bplan, rec=rec)

def run_all(B, costs=('normal', 'zero'), daily=False):
    """對 build() 結果跑紙上引擎。回傳 {(cfg, cost): (series, info)}，cfg 含 'B'。"""
    out = {}
    for cost in costs:
        out[('B', cost)] = X.run_paper(B['days'], B['Bplan'], cost, daily=daily)
        for nm, pl in B['plans'].items():
            out[(nm, cost)] = X.run_paper(B['days'], pl, cost, daily=daily)
    return out

def sector_window_returns(cs, t, t2):
    """訊號日 t 收盤 → t2 收盤，各可輪動產業（d 時成員、d 時市值權重）與 B 的含息報酬。"""
    r = TC.factor_close_to_close(t, t2) - 1
    U, b, ind = cs['U'], cs['b'], cs['ind']
    rb = float(np.nansum(b * r[U]))
    out = {}
    for s in cs['ST'].index:
        m = np.where((ind == s) & (U != X.J2330) & ~cs['fin'])[0]
        w = b[m]; out[s] = float(np.nansum(w * r[U[m]]) / w.sum())
    return out, rb

def jstats(series_by_j, bench_by_j):
    """13 個 j 的統計：每 j 的 mean/t/IR/TE，與 j 中位數、p10。"""
    rows = {}
    for j, s in series_by_j.items():
        x = (s - bench_by_j[j]).dropna()
        rows[j] = X.stats(x)
    df = pd.DataFrame(rows).T
    return dict(jmed={c: float(df[c].median()) for c in ('mean_pp', 't', 'IR', 'TE')}, p10={c: float(df[c].quantile(0.10)) for c in ('mean_pp', 't', 'IR')},
                per_j=df.round(4).to_dict('index'))

# ---------------- 安慰劑：隨機產業排名（AR(1) 持續度與實際訊號相同；同一錨定月所有 j 共用同一組隨機分數）----------------
def phi_from_rank_ac(rho):
    return float(min(max(2 * math.sin(math.pi * rho / 6), 0.0), 0.999))

def placebo_scores(months, sectors, phi, rng):
    z = {}; prev = {}
    for m in months:
        cur = {}
        for s in sectors:
            e = rng.standard_normal()
            cur[s] = phi * prev[s] + math.sqrt(1 - phi ** 2) * e if s in prev else e
        z[m] = cur; prev = cur
    return z

_PB = {}
def placebo_prepare(m0, m1, mend, P):
    if _PB.get('key') == (m0, m1, mend, tuple(sorted(P.items()))): return _PB
    paths = {}
    names = set()
    for j in range(13):
        days = X.jdays(j, m0, m1, mend)
        css = [cs_at(d, P) for d in days[:-1]]
        for cs in css: names |= set(cs['ST'].index)
        Bplan = {d: (cs['U'], cs['b']) for d, cs in zip(days[:-1], css)}
        sb = {c: X.run_paper(days, Bplan, c)[0] for c in ('normal', 'zero')}
        paths[j] = (days, css, sb)
    _PB.clear(); _PB.update(key=(m0, m1, mend, tuple(sorted(P.items()))), paths=paths, names=sorted(names),
                            months=[str(p) for p in pd.period_range(m0, m1, freq='M')])
    return _PB

def placebo_draw(args):
    """args：(seed, cfg tuple, phi, m0, m1, mend, P)。回傳 (gross j 中位數, net j 中位數) 的月均超額（pp）。"""
    seed, c, phi, m0, m1, mend, P = args
    pb = placebo_prepare(m0, m1, mend, P)
    rng = np.random.default_rng(seed)
    Z = placebo_scores(pb['months'], pb['names'], phi, rng)
    s, k, A, f = c
    g, n = [], []
    for j, (days, css, sb) in pb['paths'].items():
        held = set(); plan = {}
        for i, (d, cs) in enumerate(zip(days[:-1], css)):
            if i > 0 and not is_reb(d, f): plan[d] = None; continue
            U, w, held, _ = X.sector_targets(cs, None, k, A, held, buf=P['buf'], cap_mult=P['cap_mult'], rng_scores=Z[str(pd.Period(d, 'M'))])
            plan[d] = (U, w)
        rg = X.run_paper(days, plan, 'zero')[0]; rn = X.run_paper(days, plan, 'normal')[0]
        g.append(float((rg - sb['zero']).mean() * 100)); n.append(float((rn - sb['normal']).mean() * 100))
    return float(np.median(g)), float(np.median(n))

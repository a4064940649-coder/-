#!/usr/bin/env python3
"""紙上 ia 口徑的共用載入（只讀 import；不改任何其他資料夾）。
load(patched)：patched=True 時，把 ia_core_dispmask_official.patch 套到 independent_audit/ia_core.py 的「副本」（scratchpad），
以模組名 ia_core 註冊（與 o_v7 v7env 同法），之後 import 的 pclib→v2_core、cz_engine、flib 都用這份；DISP_OFFICIAL 預設 None。
名單：c0_plan()（main_v2 gapfix 正式、排除金融 n10）、uopt_plan()（o17 U_opt＝flib.VM(C0MASK,'opt','SURV')）。
統計：win()／stats_one() 與 integrate_o17/run.py 逐行相同。"""
import sys, types, hashlib, json, math, subprocess, shutil
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SC = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/o18_delistterms')
IA = AG / 'independent_audit' / 'ia_core.py'
PATCH = HERE / 'ia_core_dispmask_official.patch'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
IA_SHA = sha(IA)

def load(patched=False):
    assert 'ia_core' not in sys.modules
    if patched:
        d = SC / 'patchwork'; shutil.rmtree(d, ignore_errors=True); (d / 'independent_audit').mkdir(parents=True)
        shutil.copy(IA, d / 'independent_audit' / 'ia_core.py')
        r = subprocess.run(['patch', '-p1', '-s', '-i', str(PATCH)], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        src = (d / 'independent_audit' / 'ia_core.py').read_text(encoding='utf-8')
        m = types.ModuleType('ia_core'); m.__file__ = str(IA)
        sys.modules['ia_core'] = m
        exec(compile(src, str(IA), 'exec'), m.__dict__)
        assert m.DISP_OFFICIAL is None
    sys.path.insert(0, str(AG / 'independent_audit'))
    sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count')); import pclib as P; sys.path.pop(0)
    sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_engine as CE; sys.path.pop(0)
    sys.path.pop(0)
    assert CE.K is P.K
    return P, CE

def c0_plan(P):
    S = P.V.V2('gapfix', P.YL, P.Y5, exfin=True)
    pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
    return dict(plan=pl, t0=t0, t1=t1, sd=list(sd))

def uopt_plan(P):
    FM = AG / 'o_finmask_delisted_fin_erratum'
    sys.path.insert(0, str(FM)); import flib as FL; sys.path.pop(0)
    assert FL.K is P.K
    C0M = FL.C0MASK.copy()
    S = FL.VM(lambda d: C0M, 'opt', 'SURV')
    pl = S.plan(10, 'official'); t0, t1, sd = S.bounds()
    return dict(plan=pl, t0=t0, t1=t1, sd=list(sd))

def with_mask(P, mode, fn):
    """mode：None／'union'／'replace'（patched ia_core 的 DISP_OFFICIAL）。"""
    K = P.K; old = K.DISP_OFFICIAL
    K.DISP_OFFICIAL = mode
    try: return fn()
    finally: K.DISP_OFFICIAL = old

sys.path.insert(0, str(AG / 'o_forward_sequential_v2')); import fslib as F; sys.path.pop(0)
E50 = F.e50()
END = pd.Timestamp('2026-10-12')
def win(nav, SD):
    SD = pd.DatetimeIndex(SD)
    pts = list(SD[SD < END]) + [END]
    rows = [(pd.Period(a, 'M'), F._at(nav, b) / F._at(nav, a) - 1, F._at(E50, b) / F._at(E50, a) - 1, a.year) for a, b in zip(pts[:-1], pts[1:])]
    W = pd.DataFrame(rows, columns=['per', 'r', 'r50', 'yr']).set_index('per')
    return W[~W.index.duplicated(keep='last')].dropna()
PERS = {'ALL': (None, None), 'SEG1': ('2006-01', '2012-12'), 'SEG2': ('2013-01', '2018-12'), 'SEG3': ('2019-01', None),
        'SEL': (None, '2019-04'), 'VAL': ('2019-05', None), 'HOLD': ('2024-01', None), 'Y1419': ('2014-01', '2019-12'), 'Y14ON': ('2014-01', None)}
def pmask(idx, k):
    a, b = PERS[k]; m = np.ones(len(idx), bool)
    if a: m &= idx >= pd.Period(a, 'M')
    if b: m &= idx <= pd.Period(b, 'M')
    return m
def tt(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * math.sqrt(len(x))) if len(x) > 2 and s > 1e-15 else None
def ann(r): r = np.asarray(r, float); return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
def mdd(r):
    nav = np.cumprod(1 + np.asarray(r, float)); pk_ = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    return float((nav / pk_ - 1).min()) * 100
def stats_one(rv, r50, rM, idx, yrs):
    o = {}
    for k in PERS:
        m = pmask(idx, k); a, b, c = rv[m], r50[m], rM[m]
        o[k] = dict(n=int(m.sum()), ann=ann(a), ann_0050=ann(b), gex=ann(a) - ann(b), mdd=mdd(a), t_vs_0050=tt(a - b),
                    mean_ex_pp=float(np.mean(a - b) * 100), ann_diff_vs_ref=ann(a) - ann(c), t_diff_vs_ref=tt(a - c))
    s = pd.Series(rv - r50).groupby(np.asarray(yrs)).sum()
    b3 = list(s.sort_values(ascending=False).index[:3]); m = ~np.isin(yrs, b3)
    o['ALL']['t_drop_best3'] = tt(rv[m] - r50[m]); o['ALL']['best3'] = [int(x) for x in b3]
    o['yearly_ex_pp'] = {int(y): float(v * 100) for y, v in pd.Series(rv - r50).groupby(np.asarray(yrs)).sum().items()}
    o['yearly_diff_vs_ref_pp'] = {int(y): float(v * 100) for y, v in pd.Series(rv - rM).groupby(np.asarray(yrs)).sum().items()}
    return o

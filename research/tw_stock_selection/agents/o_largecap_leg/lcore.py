#!/usr/bin/env python3
"""o_largecap_leg 核心：只讀 import o_rev_survivorship/osurv_core.py 與 o_capacity_tsmc_gap/ocore.py
（→ main_v2/v2_core.py → independent_audit/ia_core.py）。不重寫成本、成交、跳動點漲停、停牌凍結。
自己實作：大型股池（事前可知）、池內分數與選股、組合權重計畫、0050 袖子混合、拆解與 β 等統計。"""
import sys, json, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_rev_survivorship'))
import osurv_core as OS
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_capacity_tsmc_gap'))
import ocore as O
sys.path.pop(0)
import numpy as np, pandas as pd
V, K = OS.V, OS.K
assert O.K is K and O.V is V
IDX, COLS, N, CI = K.IDX, K.COLS, K.N, K.CI
J2330 = CI['2330']
NORMAL = K.rates(0.001)
STRESS3 = K.rates(0.004)          # 每邊再加 0.3% 滑價
STRESS_OLD = K.rates('stress')    # 研究中心舊壓力：買賣各 0.5%
FOREIGN_MASK = np.array([c in OS.FOREIGN for c in COLS])
FINMASK = V.FINMASK
ELEC_IND = {'半導體業', '電子零組件業', '光電業', '電腦及週邊設備業', '其他電子業', '通信網路業', '資訊服務業', '電子通路業'}
IND = np.array([V.SL['industry'].get(c) if isinstance(V.SL['industry'].get(c), str) else '無' for c in COLS])
ELEC = np.isin(IND, list(ELEC_IND))

def dumps(o): return O.dumps(o)
def sha_obj(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

# ---------------- 主策略（MAIN_R） ----------------
YD, Y5 = OS.load_yl('R_DEL')
SR = OS.V2B('gapfix', YD, Y5, exfin=True)
MAIN_PLAN = SR.plan(10, 'official')
T0, T1, SD = SR.bounds()
SPLIT = [d for d in SD if d.year == 2019][0]          # 2019-01 換股訊號日：SEL 窗起點 < SPLIT
T_SPLIT = IDX.get_loc(SPLIT)
LAST_SEL = [d for d in SD if d < SPLIT][-1]
SIG = [p[3] for p in MAIN_PLAN]
EXEC = [p[0] for p in MAIN_PLAN]

PERIODS = {
    'SEL': ('2006-04-01', str(LAST_SEL.date()), '2006-04-01', str(SPLIT.date())),
    'VAL': (str(SPLIT.date()), '2026-10-11', str((SPLIT + pd.Timedelta(days=1)).date()), '2026-10-12'),
    'VAL_A': (str(SPLIT.date()), '2023-12-31', str((SPLIT + pd.Timedelta(days=1)).date()), '2023-12-31'),
    'HOLD': ('2024-01-01', '2026-10-11', '2024-01-01', '2026-10-12'),
    'S1': ('2006-04-01', '2013-12-31', '2006-04-01', '2013-12-31'),
    'S2': ('2014-01-01', '2018-12-31', '2014-01-01', str(SPLIT.date())),
    'ALL': ('2006-04-01', '2026-10-11', '2006-04-01', '2026-10-12'),
}

# ---------------- 大型股池 ----------------
_LCS = {}
def large_cs(d, P, score='55', excl=(), members=None):
    """回傳 dict(t, e, pool idx（依 60 日均成交值由大到小）, sc, y3, s52, v60)。合格＝v60≥1,000 萬、收盤≥10、非金融、非 KY/DR、
    營收與 52 週比有值、非處置、隔日有開盤成交；members（代碼集合）給定時再限成分。"""
    key = (d, P, score, tuple(sorted(excl)), None if members is None else frozenset(members))
    if key in _LCS: return _LCS[key]
    t = IDX.get_loc(d); e = t + 1
    v60 = SR.V60[t]; px = K.Cn[t]; y3 = SR.y3(d); s52 = SR.S52[t]
    el = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10) & np.isfinite(y3) & np.isfinite(s52)
    el &= ~FINMASK & ~FOREIGN_MASK & ~K.disposal_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
    for c in excl:
        if c in CI: el[CI[c]] = False
    if members is not None:
        mm = np.zeros(N, bool)
        for c in members:
            if c in CI: mm[CI[c]] = True
        el &= mm
    vv = np.where(el, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')[:int(el.sum())]
    pool = order[:P]
    inpool = np.zeros(N, bool); inpool[pool] = True
    def pr(x): return pd.Series(np.where(inpool, x, np.nan)).rank(pct=True).to_numpy()
    if score == '55': sc = 0.5 * pr(s52) + 0.5 * pr(y3)
    elif score == '52': sc = pr(s52)
    elif score == 'rev': sc = pr(y3)
    elif score == 'v60': sc = pr(v60)
    else: raise ValueError(score)
    sc = np.array(sc, dtype=float, copy=True); sc[~inpool] = np.nan
    cs = dict(t=t, e=e, pool=pool, sc=sc, y3=y3, s52=s52, v60=v60, el=inpool)
    _LCS[key] = cs
    return cs

def plan_leg(P, n, score='55', wmode='ew', excl=(), members_fn=None, upto=None):
    """回傳 [(exec_t, picks, weights, sig_day, pool)]；開盤漲停且原本未持有者遞補（同 v2）。"""
    out = []; held = set()
    for e0, d in zip(EXEC, SIG):
        if upto is not None and d >= upto: break
        mem = members_fn(d) if members_fn is not None else None
        cs = large_cs(d, P, score, excl, mem)
        e = cs['e']; assert e == e0
        if len(cs['pool']) == 0:
            out.append((e, np.array([], int), np.array([]), d, cs['pool'])); held = set(); continue
        gu = K.gap_up_open(e)
        cand = V.V2.order(cs, 'official')[:len(cs['pool'])]
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        pick = np.array(pick, int)
        if wmode == 'ew': w = np.full(len(pick), 1.0 / len(pick))
        elif wmode == 'vw': w = cs['v60'][pick] / cs['v60'][pick].sum()
        else: raise ValueError(wmode)
        out.append((e, pick, w, d, cs['pool']))
        held = set(pick.tolist())
    return out

def main_plan(upto=None):
    return [p for p in MAIN_PLAN if upto is None or p[3] < upto]

def wplan_main(pl, w=1.0):
    return [(p[0], np.asarray(p[1], int), np.full(len(p[1]), w / len(p[1]))) for p in pl]

def wplan_mix(mainpl, legpl, wleg):
    out = []
    for a, b in zip(mainpl, legpl):
        assert a[0] == b[0]
        idx = np.concatenate([np.asarray(a[1], int), b[1]])
        w = np.concatenate([np.full(len(a[1]), (1 - wleg) / len(a[1])), wleg * b[2]])
        out.append((a[0], idx, w))
    return out

def wplan_leg(legpl):
    return [(p[0], p[1], p[2]) for p in legpl]

def wplan_pool(legpl, wleg, mainpl=None):
    """池等權（安慰劑）：池內每檔 wleg/|池|；mainpl 給定時和主策略混合。"""
    out = []
    for i, b in enumerate(legpl):
        pool = np.asarray(b[4], int); wp = np.full(len(pool), wleg / len(pool))
        if mainpl is None: out.append((b[0], pool, wp / wleg)); continue
        a = mainpl[i]; assert a[0] == b[0]
        out.append((a[0], np.concatenate([np.asarray(a[1], int), pool]), np.concatenate([np.full(len(a[1]), (1 - wleg) / len(a[1])), wp])))
    return out

def sim(wp, cost, t1, G=None):
    nav, lg = O.simulate_w(wp, *cost, T0, t1, G=G)
    return nav, lg['turnover']

def pool_ew(legpl, t1):
    nav, _ = K.simulate([(p[0], np.asarray(p[4], int)) for p in legpl], 0, 0, T0, t1)
    return nav

def sleeve_mix(navA, wB, t1, cost=NORMAL, B=None):
    """主策略袖子 navA（1−wB）＋0050 袖子 wB；每個換股成交日收盤回到目標權重，移轉金額扣成本。"""
    B = K.E50 if B is None else B
    ix = IDX[T0:t1 + 1]
    ra = navA.reindex(ix).pct_change().fillna(0).to_numpy()
    rb = B.reindex(ix).pct_change().fillna(0).to_numpy()
    ex = set(i - T0 for i in EXEC if T0 <= i <= t1)
    fee_buy = cost[0]; sell_stock = cost[1]; sell_etf = cost[0] + 0.001
    a, b = 1 - wB, wB; out = np.empty(len(ix))
    for i in range(len(ix)):
        if i > 0: a *= 1 + ra[i]; b *= 1 + rb[i]
        if i in ex and i > 0:
            tot = a + b; ta = (1 - wB) * tot
            if a < ta:  # 賣 0050 買股票
                m = ta - a; c = m * (sell_etf + fee_buy)
            else:
                m = a - ta; c = m * (sell_stock + fee_buy)
            tot -= c; a, b = (1 - wB) * tot, wB * tot
        out[i] = a + b
    return pd.Series(out, index=ix)

# ---------------- 統計 ----------------
def wr(nav, sd=None):
    return O.window_returns(nav, SD if sd is None else sd)
E50W = None
def e50w():
    global E50W
    if E50W is None: E50W = wr(K.E50)
    return E50W
TSW = None
def tsw():
    global TSW
    if TSW is None: TSW = wr(O.TSMC_IDX)
    return TSW

def win(x, k):
    a, b = PERIODS[k][:2]
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()

def segstat(nav, ew, k):
    st = O.segstat(nav, ew, SD, *PERIODS[k])
    st = {kk: v for kk, v in st.items() if kk != 'yearly'}
    x = win(wr(nav) - e50w(), k)
    st['sd_ex_0050_m'] = round(float(x.std(ddof=1)) * 100, 3)
    return st

def tstat(x): return K.tstat(x)

def decomp(xc, xm):
    """xc：組合月超額、xm：主策略月超額（同月份）。t＝√n·m/s。"""
    z = pd.concat([xc, xm], axis=1).dropna(); c, m_ = z.iloc[:, 0], z.iloc[:, 1]
    n = len(z); m1, s1, m0, s0 = c.mean(), c.std(ddof=1), m_.mean(), m_.std(ddof=1)
    rt = math.sqrt(n)
    t1_, t0_ = rt * m1 / s1, rt * m0 / s0
    mean_only = rt * (m1 - m0) / s0; sd_only = rt * m0 * (1 / s1 - 1 / s0)
    sh_mean = 0.5 * rt * (m1 - m0) * (1 / s0 + 1 / s1); sh_sd = 0.5 * rt * (m0 + m1) * (1 / s1 - 1 / s0)
    dt = t1_ - t0_
    return {'n': n, 'mean_main_pp': round(m0 * 100, 3), 'mean_combo_pp': round(m1 * 100, 3), 'sd_main_pp': round(s0 * 100, 3), 'sd_combo_pp': round(s1 * 100, 3),
            't_main': round(t0_, 3), 't_combo': round(t1_, 3), 'dt': round(dt, 3), 'dt_mean_only': round(mean_only, 3), 'dt_sd_only': round(sd_only, 3),
            'dt_shapley_mean': round(sh_mean, 3), 'dt_shapley_sd': round(sh_sd, 3),
            'share_from_sd': round(sh_sd / dt, 3) if abs(dt) > 1e-9 else None}

def beta(y, x):
    z = pd.concat([y, x], axis=1).dropna()
    if len(z) < 10: return {'beta': None}
    Y, X = z.iloc[:, 0].to_numpy(), z.iloc[:, 1].to_numpy()
    Xm = np.column_stack([np.ones(len(X)), X]); b = np.linalg.lstsq(Xm, Y, rcond=None)[0]
    e = Y - Xm @ b; s2 = e @ e / (len(Y) - 2); se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    return {'alpha_pp': round(b[0] * 100, 3), 't_alpha': round(b[0] / se[0], 2), 'beta': round(b[1], 3), 't_beta': round(b[1] / se[1], 2),
            'R2': round(1 - (e @ e) / ((Y - Y.mean()) @ (Y - Y.mean())), 3), 'n': int(len(Y))}

def corr(x, y):
    z = pd.concat([x, y], axis=1).dropna()
    return round(float(z.iloc[:, 0].corr(z.iloc[:, 1])), 3)

def mstat(x):
    x = x.dropna()
    return {'mean_pp': round(float(x.mean()) * 100, 3), 'sd_pp': round(float(x.std(ddof=1)) * 100, 3), 't': round(tstat(x), 3), 'n': int(len(x)),
            'win': round(float((x > 0).mean()), 3)}

def leg_period_returns(legpl, t_end, G=None):
    """每期（成交日開盤→下一成交日開盤）每檔毛報酬：[(sig, picks, w, r_picks, pool, r_pool)]，用 v2_core.stock_contrib 的報酬定義。"""
    p1 = [(p[0], p[1], None, p[3]) for p in legpl]
    p2 = [(p[0], np.asarray(p[4], int), None, p[3]) for p in legpl]
    _, per1 = V.stock_contrib(p1, t_end, G=G); _, per2 = V.stock_contrib(p2, t_end, G=G)
    m2 = {d: (pk, r) for d, pk, r in per2}
    wmap = {p[3]: p[2] for p in legpl}
    return [(d, pk, wmap[d], r, m2[d][0], m2[d][1]) for d, pk, r in per1]

def drift_zero_G(t_split, t1):
    """2330 對 0050 的日均漂移在 [T0, t_split] 與 (t_split, t1] 兩段各自歸零。"""
    g = K.GROW[:, J2330]; e = (K.E50.pct_change() + 1).reindex(IDX).to_numpy()
    G2 = K.GROW.copy(); info = {}
    for lab, lo, hi in (('SEL', T0 + 1, t_split), ('VAL', t_split + 1, t1)):
        m = np.zeros(len(IDX), bool); m[lo:hi + 1] = True
        m &= np.isfinite(e) & K.TRADED[:, J2330]
        dr = float(np.mean(g[m] - e[m])); G2[m, J2330] = g[m] - dr
        info[lab] = round(dr * 1e4, 3)
    return G2, info

# ---------------- 輔助股池：臺灣50 倒推成分 ----------------
_TW = json.loads((HERE / 'tw50_members.json').read_text(encoding='utf-8'))
_TW_STEPS = sorted([(pd.Timestamp(s['eff']), set(s['members_before'])) for s in _TW['steps_backward']])
_TW_CUR = set(_TW['current_2026-10-09'])
TW50_START = pd.Timestamp('2015-04-01')
def tw50_members(d):
    if d < TW50_START: return set()
    for eff, mem in _TW_STEPS:
        if d <= eff: return mem
    return _TW_CUR

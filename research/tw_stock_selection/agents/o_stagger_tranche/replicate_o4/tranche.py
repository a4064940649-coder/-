import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
C.REVRULE['mode'] = 'anchor'
OFFD = [C.OFFICIAL[m] for m in C.ANCHORS]
SD = OFFD + [C.END]
T0 = C.POS[OFFD[0]]
C1DAYS = {C.next_good(C.POS[d]) for d in OFFD}
def per(d):
    m = pd.Period(d, 'M')
    return 'SEL' if m <= pd.Period('2018-12','M') else ('VAL' if m <= pd.Period('2023-12','M') else 'HOLD')
PER = {d: per(d) for d in OFFD}

def b1_rank(m, s, U, uid, r1):
    """b1：營收腿百分位凍結在 J+1 批的橫斷面；合格＝自己的 el ∩ J+1 的 el；價格腿在交集內重排。"""
    key = ('b1', m, s, uid)
    if key in C._CS: return C._CS[key]
    r = C.ranking(m, s, U, uid)
    t = r['t']; el = r['el'] & r1['el']
    s52 = C.S52[t]
    pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
    sc = 0.5 * pr52 + 0.5 * r1['prr']; sc[~el] = np.nan
    scr = np.round(np.nan_to_num(sc, nan=-9), 10)
    cand = np.lexsort((np.arange(C.N), -np.nan_to_num(r1['y3'], nan=-1e18), -scr))[:int(el.sum())]
    out = dict(t=t, e=r['e'], el=el, cand=cand, nel=int(el.sum()))
    C._CS[key] = out
    return out

def plans_for(js, a='a0', b='b0', n=10, seed=0):
    """依時間順序產生各批 plan（a1 需要其他批當下持股）。"""
    k = len(js)
    abs_ = [C.AB(seed) for _ in js]
    ev = []
    for m in C.ANCHORS:
        for i, j in enumerate(js):
            s = C.jday(m, j)
            if s is None or s not in C.POS or s > C.END: continue
            ev.append((s, i, m))
    ev.sort(key=lambda x: (x[0], x[1]))
    held = [set() for _ in js]; out = [[] for _ in js]
    r1cache = {}
    for s, i, m in ev:
        U = abs_[i](m, s); uid = abs_[i].uid
        if b == 'b1' and js[i] != js[0]:
            s1 = C.jday(m, js[0])
            if s1 not in C.POS: continue
            U1 = abs_[0](m, s1)
            r1 = C.ranking(m, s1, U1, abs_[0].uid)
            if 'prr' not in r1:
                el1 = r1['el']; y3 = C.y3_for(m, s1)
                if U1 is not None and U1.any(): y3 = np.where(U1, C.y3_for(m, s1, 1), y3)
                r1['prr'] = pd.Series(np.where(el1, y3, np.nan)).rank(pct=True).to_numpy(); r1['y3'] = y3
            r = b1_rank(m, s, U, uid, r1)
        else:
            r = C.ranking(m, s, U, uid)
        if r['nel'] < 50: continue
        gu = C.gap_up(r['e']); pick = []
        others = set().union(*[held[q] for q in range(k) if q != i]) if a == 'a1' else set()
        for jj in r['cand']:
            jj = int(jj)
            if gu[jj] and jj not in held[i]: continue
            if jj in others: continue
            pick.append(jj)
            if len(pick) == n: break
        out[i].append((r['e'], np.array(pick, int), m, s))
        held[i] = set(pick)
    return out

def run(js, a='a0', b='b0', c='c0', cost=None, seed=0):
    pls = plans_for(js, a, b, seed=seed)
    cost = cost or C.NORMAL
    nav, lg = C.tsim(pls, *cost, T0, C.TEND, c1_days=C1DAYS if c == 'c1' else None)
    return nav, lg, pls

def win(nav, bench=None):
    x = C.excess(nav, SD, bench)
    return x

def selstats(x, p='SEL'):
    y = x[[PER[d] == p for d in x.index]] if p != 'ALL' else x
    return y

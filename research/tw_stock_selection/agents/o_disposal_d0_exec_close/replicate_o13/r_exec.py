# Independent replication (o13) - per-trade executable model.
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from r_data import *

ENTRIES = ['O1', 'C1', 'O2']; EXITS = ['CL1', 'CL', 'OA']
ODD_CUT = pd.Timestamp('2020-10-26'); FEE_CUT = pd.Timestamp('2015-01-01')
SH = {'last': {5: .1071, 20: .1342}, 'first': {5: .0301, 20: .0802}}
SH_OTHER = {'last': .1153, 'first': .0393}; SH_NORM = {'last': .1296, 'first': .0437}

def bar_share(ev, i, kind):
    """kind 'last'/'first'; in-period periodic match uses match-minute shares."""
    if ev.si <= i <= ev.ei and ev.regime == 'periodic_match':
        mm = ev.match_min
        if mm in (5.0, 20.0): return SH[kind][int(mm)]
        return SH_OTHER[kind]
    return SH_NORM[kind]

def vadj(ev, i):
    v = V[i, ev.ci]
    if SRC[i, ev.ci] == 3: v = v / (1 - bar_share(ev, i, 'last'))
    return v

def fee(a, d, stress):
    if a <= 0: return 0.0
    fr = 0.001425 * (0.6 if (stress or d < FEE_CUT) else 0.28)
    return max(20.0, fr * a)

def priced(i, c): return np.isfinite(C[i, c]) and C[i, c] > 0

def find_sell_day(c, i0):
    """first day >= i0 priced and close > limit down; None if none (delisted)."""
    for i in range(i0, NT):
        if not GOOD[i]: continue
        if priced(i, c) and C[i, c] > limdn(REF[i, c], IDX[i]) + 1e-9: return i
    return None

LASTP = {}
def last_priced(c, upto):
    if upto == NT - 1:
        if c not in LASTP: LASTP[c] = _last_priced(c, upto)
        return LASTP[c]
    return _last_priced(c, upto)
def _last_priced(c, upto):
    for i in range(min(upto, NT - 1), -1, -1):
        if priced(i, c): return i
    return None

def next_good(i):
    i += 1
    while i < NT and not GOOD[i]: i += 1
    return i

def prev_good(i):
    i -= 1
    while i >= 0 and not GOOD[i]: i -= 1
    return i

def run_trade(ev, ent, ex, slot, stress=False, odd_sp=0.004, kappa=0.5, lim_off=None, budget=None, ret_detail=False):
    """returns dict or None (not filled / invalid). slot in 2026 NT$. budget overrides (current NT$)."""
    c = ev.ci
    # entry bar
    if ent == 'O2': i = next_good(ev.si); kind = 'first'
    else: i = ev.si; kind = 'first' if ent == 'O1' else 'last'
    if i >= NT or not priced(i, c) or not np.isfinite(O[i, c]): return None
    d = IDX[i]
    P = O[i, c] if kind == 'first' else C[i, c]
    lu = limup(REF[i, c], d)
    L = lu if lim_off is None else np.floor(REF[i, c] * (1 + limpct(d) + lim_off) / tick(REF[i, c]) + 1e-9) * tick(REF[i, c])
    if P >= L - 1e-9: return None
    B = budget if budget is not None else slot * cpi_ratio(d.year)
    N = int(np.floor(B / L)); lots = N // 1000; odd = N % 1000
    Qa = vadj(ev, i) * bar_share(ev, i, kind)
    maxl = int(np.floor(0.2 * Qa / (1000 * P))) if Qa > 0 else 0
    lots_wanted = lots; lots = min(lots, maxl)
    sp = odd_sp + (0.004 if stress else 0.0)
    tranches = []  # (shares, entry level, entry price, kind_is_lot)
    cost = 0.0; fair = 0.0
    if lots > 0:
        A = lots * 1000 * P
        r = max(tick(P) / P, 0.001) + kappa * SIGp[i, c] * np.sqrt(A / Qa) + (0.004 if stress else 0)
        cost += A * (1 + r) + fee(A, d, stress); fair += A
        tranches.append((lots * 1000, (LAO if kind == 'first' else LA)[i, c], P, 1))
    if odd > 0:
        if kind == 'first' and d < ODD_CUT: Po = C[i, c]; lvl = LA[i, c]
        else: Po = P; lvl = (LAO if kind == 'first' else LA)[i, c]
        A = odd * Po
        cost += A * (1 + sp) + fee(A, d, stress); fair += A
        tranches.append((odd, lvl, Po, 0))
    if cost <= 0: return None
    if stress and cost < fair * 1.01: cost = fair * 1.01
    ent_lvl_ew = (EWLO if kind == 'first' else EWL)[i]; ent_lvl_50 = (E50LO if kind == 'first' else E50L)[i]
    # exit bar
    if ex == 'CL1': j = prev_good(ev.ei); xk = 'last'
    elif ex == 'CL': j = ev.ei; xk = 'last'
    else: j = next_good(ev.ei); xk = 'first'
    if j < i or (j == i and not (kind == 'first' and xk == 'last')): return None
    if j >= NT:
        if ret_detail: return dict(entry_i=i, cost=cost, tranches=tranches, sells=[], open=True, lots=lots, odd=odd)
        return None
    if xk == 'first':
        if j >= NT or not priced(j, c) or not np.isfinite(O[j, c]) or O[j, c] <= limdn(REF[j, c], IDX[j]) + 1e-9:
            xk = 'last'  # switch to close with R5 rule
    sells = []  # (day, kind, frac_lots, odd flag)
    lot_sh = lots * 1000; odd_sh = odd
    proceeds = 0.0; sfair = 0.0; ew_w = 0.0; e50_w = 0.0; last_day = None
    def val(sh, tr_lvl, tr_P, lvl):  # adjusted value of sh shares
        return sh * tr_P * np.exp(lvl - tr_lvl)
    lot_tr = [t for t in tranches if t[3] == 1]; odd_tr = [t for t in tranches if t[3] == 0]
    cur = j; first = True
    while lot_sh > 0 or odd_sh > 0:
        if first and xk == 'first':
            k = cur; bk = 'first'
        else:
            k = find_sell_day(c, cur)
            bk = 'last'
            if k is None:
                k = last_priced(c, NT - 1)
                if k is None or k < i: return None
        first = False
        dk = IDX[k]
        Px = O[k, c] if bk == 'first' else C[k, c]
        lvl = (LAO if bk == 'first' else LA)[k, c]
        if not np.isfinite(lvl):
            lvl = LA[k, c]; Px = C[k, c]; bk = 'last'
        delisted = (k == last_priced(c, NT - 1)) and k < NT - 1
        Qa = vadj(ev, k) * bar_share(ev, k, bk)
        ew_lvl = (EWLO if bk == 'first' else EWL)[k]; l50 = (E50LO if bk == 'first' else E50L)[k]
        if lot_sh > 0:
            maxl = int(np.floor(0.2 * Qa / (1000 * Px))) if Qa > 0 else 0
            nl = lot_sh // 1000 if delisted else min(lot_sh // 1000, maxl)
            if nl > 0:
                tr = lot_tr[0]
                A = val(nl * 1000, tr[1], tr[2], lvl); Araw = nl * 1000 * Px
                r = max(tick(Px) / Px, 0.001) + kappa * SIGp[k, c] * np.sqrt(min(Araw / Qa, 1e6) if Qa > 0 else 1.0) + (0.004 if stress else 0)
                cst = A * (r + 0.003) + fee(A, dk, stress)
                if stress: cst = max(cst, 0.01 * A)
                proceeds += A - cst; sfair += A
                ew_w += A * (ew_lvl - ent_lvl_ew); e50_w += A * (l50 - ent_lvl_50)
                lot_sh -= nl * 1000
                sells.append((k, A - cst, nl * 1000, 0))
        if odd_sh > 0:
            tr = odd_tr[0]
            if bk == 'first' and dk < ODD_CUT: lv2 = LA[k, c]; ew2 = EWL[k]; l502 = E50L[k]
            else: lv2 = lvl; ew2 = ew_lvl; l502 = l50
            A = val(odd_sh, tr[1], tr[2], lv2)
            cst = A * (sp + 0.003) + fee(A, dk, stress)
            if stress: cst = max(cst, 0.01 * A)
            proceeds += A - cst; sfair += A
            ew_w += A * (ew2 - ent_lvl_ew); e50_w += A * (l502 - ent_lvl_50)
            sells.append((k, A - cst, 0, odd_sh))
            odd_sh = 0
        last_day = k
        cur = k + 1
        if lot_sh > 0 and cur >= NT: return None
    ret = proceeds / cost - 1
    ew = np.exp(ew_w / sfair) - 1; r50 = np.exp(e50_w / sfair) - 1
    out = dict(ret=ret, ew=ew, r50=r50, net_x=ret - ew, net50_x=ret - r50, entry_i=i, exit_i=last_day,
               lots=lots, lots_wanted=lots_wanted, odd=odd, cost=cost, gross=None)
    if ret_detail: out.update(tranches=tranches, sells=sells)
    return out

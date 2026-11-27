#!/usr/bin/env python3
"""S104 評估：每題 20 條路徑，給定窗 (a, b) 只算該窗的數字（訓練步只傳訓練窗）。"""
import numpy as np, pandas as pd
import s104lib as L, s104cells as C


def _flat(nav, t0):
    v = np.array(nav, float); v[:t0] = 1.0
    return v


def _m(nav, d, a, b, offi=None, trades=None):
    return L.metrics(nav, d, a, b, trades=trades, navref=offi)


def off_array(d):
    return L.official_nav(d).bfill().to_numpy(float)   # 2015-07-01 以前補成起始值（不影響訓練窗）


# ---------------------------------------------------------------- K1
def k1_paths(d, cell, a, b, borrow=0.02, keep_nav=False):
    offi = off_array(d)
    P = []; navs = {}
    for si in L.SHIFTS:
        nav, tr, info = C.k1_run(d, cell, si, borrow=borrow)
        nav = _flat(nav, d.TA)
        o = dict(short=_m(nav, d, a, b, offi, tr))
        base = _m(offi, d, a, b)
        o['official'] = base
        for w in C.K1_W:
            cb = L.overlay(offi, nav, w)
            m = _m(cb, d, a, b, offi)
            m['d_sharpe'] = m['sharpe'] - base['sharpe']; m['d_cagr'] = m['cagr'] - base['cagr']; m['d_mdd'] = m['mdd'] - base['mdd']
            o[f'combo_w{w}'] = m
        P.append(o)
        if keep_nav:
            navs[si] = (nav, tr)
    return P, navs


# ---------------------------------------------------------------- K2
def k2_paths(d, pre, cell, a, b, borrow=0.02, keep_nav=False):
    offi = off_array(d)
    P = []; navs = {}
    for off in L.SHIFTS:
        nL, tL, nS, tS = C.k2_run(d, pre, cell, off, borrow=borrow)
        t0 = d.TA + off
        nL, nS = _flat(nL, t0), _flat(nS, t0)
        dual = L.blend(nL, nS, 0.5)
        aa = max(a, t0)
        base = _m(offi, d, aa, b)
        o = dict(long=_m(nL, d, aa, b, offi, tL), short=_m(nS, d, aa, b, offi, tS), dual=_m(dual, d, aa, b, offi), official=base)
        o['dual']['n_trades'] = o['long']['n_trades'] + o['short']['n_trades']
        nw = o['long']['n_trades'] * o['long']['win_rate'] + o['short']['n_trades'] * o['short']['win_rate']
        o['dual']['win_rate'] = float(nw / max(o['dual']['n_trades'], 1))
        cb = L.blend(offi, dual, 0.3)
        m = _m(cb, d, aa, b, offi)
        m['d_sharpe'] = m['sharpe'] - base['sharpe']; m['d_cagr'] = m['cagr'] - base['cagr']; m['d_mdd'] = m['mdd'] - base['mdd']
        o['combo_70off_30dual'] = m
        P.append(o)
        if keep_nav:
            navs[off] = (nL, tL, nS, tS, dual)
    return P, navs


# ---------------------------------------------------------------- K3
def k3_paths(d, cell, a, b, ivm, spread, ON, TR, keep_nav=False):
    c = C.K3_CELLS[cell]
    P = []; navs = {}
    for si in L.SHIFTS:
        offs = ON[str(si)].reindex(d.IDX).to_numpy(float)
        tr = TR[TR['shift'] == si]
        nW, tW = C.k3_sleeve(d, tr, c['tenor'], ivm, spread, d.TA, d.TE)
        nW = _flat(nW, d.TA)
        rebal = set(int(x) for x in tr['ent_t'].unique())
        cb = C.blend_rebal(offs, nW, c['f'], rebal, d.TA, d.TE)
        cb[:d.TA] = 1.0
        base = _m(offs, d, a, b)
        offi0 = off_array(d)
        o = dict(warrant_sleeve=_m(nW, d, a, b, offi0, tW), official_path=base)
        m = _m(cb, d, a, b, offi0)
        m['d_sharpe'] = m['sharpe'] - base['sharpe']; m['d_cagr'] = m['cagr'] - base['cagr']; m['d_mdd'] = m['mdd'] - base['mdd']
        o['combo'] = m
        P.append(o)
        if keep_nav:
            navs[si] = (nW, tW, cb, offs)
    return P, navs


def med(P, *keys):
    v = []
    for p in P:
        x = p
        for k in keys:
            x = x[k]
        v.append(x)
    v = np.array(v, float)
    return float(np.nanmedian(v)), v


def summarize(P):
    """每個子物件每個欄位的中位數＋最小＋最大。"""
    out = {}
    for k in P[0]:
        out[k] = {}
        for f in P[0][k]:
            v = np.array([p[k][f] for p in P], float)
            out[k][f] = dict(median=float(np.nanmedian(v)), min=float(np.nanmin(v)), max=float(np.nanmax(v)))
    return out

#!/usr/bin/env python3
"""Q7 安慰劑：(a) 隨機產業排名＋同一個股規則；(b) 真產業排名＋產業內隨機選股；(c) 兩者皆隨機。
隨機分數用 AR(1)（z ← ρz＋√(1−ρ²)ε），ρ 事前以「周轉對齊」校準；緩衝帶、權重、上限、載體與實際規格相同。
每個模擬回傳每窗報酬（一般成本與零成本）。"""
import numpy as np, pandas as pd
import lc_lib as X, lc_run as Rn

class AR:
    def __init__(self, rng, rho):
        self.rng, self.rho, self.z = rng, rho, {}
    def get(self, keys):
        out = np.empty(len(keys))
        for i, k in enumerate(keys):
            if k in self.z: self.z[k] = self.rho * self.z[k] + np.sqrt(1 - self.rho ** 2) * self.rng.standard_normal()
            else: self.z[k] = self.rng.standard_normal()
            out[i] = self.z[k]
        return out

def make_fns(kind, rng, rho_sec, rho_stk):
    srf = sf = None
    if kind in ('a', 'c'):
        ars = AR(rng, rho_sec); cache = {}
        def srf(cs):
            d = cs['d']
            if d not in cache:
                names = list(cs['ST'].index); z = ars.get(names)
                cache.clear(); cache[d] = [names[i] for i in np.argsort(-z, kind='stable')]
            return cache[d]
    if kind in ('b', 'c'):
        art = AR(rng, rho_stk); cache2 = {}
        def sf(cs):
            d = cs['d']
            if d not in cache2:
                real = cs['score']; z = art.get([X.COLS[u] for u in cs['U']])
                sc = np.where(np.isfinite(real), pd.Series(np.where(np.isfinite(real), z, np.nan)).rank(pct=True).to_numpy(), np.nan)
                cache2.clear(); cache2[d] = sc
            return cache2[d]
    return srf, sf

def sim_weights(sp, days, kind, P, seed, rho_sec, rho_stk):
    """回傳每窗 P×N 權重矩陣列表。"""
    T = len(days) - 1
    W = [np.zeros((P, X.N)) for _ in range(T)]
    for p in range(P):
        rng = np.random.default_rng([seed, p])
        srf, sf = make_fns(kind, rng, rho_sec, rho_stk)
        s = Rn.make(sp, srf=srf, sf=sf)
        for k, d in enumerate(days[:-1]):
            cs = X.cs_cached(d); idx, w, _ = s.step(cs)
            np.add.at(W[k][p], idx, w)
    return W

def run(sp, j, per, kind, P=1000, seed=20261103, rho_sec=0.8, rho_stk=0.8):
    days = X.days_for(j, per)
    W = sim_weights(sp, days, kind, P, seed + j, rho_sec, rho_stk)
    Rn_, TOn = X.vrun(days, W, 'normal'); Rz, _ = X.vrun(days, W, 'zero')
    b = X.bench_win(days)
    return dict(days=days, net=Rn_, gross=Rz, to=TOn, b=b)

def calibrate(sp, per, rhos=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95), js=(0, 6, 12), P=40):
    """ρ 周轉對齊：產業 ρ 對齊產業周轉（只隨機產業），個股 ρ 對齊組合周轉（真產業＋隨機個股）。"""
    real_to = np.mean([Rn.run(sp, j, per)['to'].mean() for j in js])
    real_sec = []
    for j in js:
        info = Rn.run(sp, j, per)['info']; sl = [tuple(i.get('sectors', [])) for i in info]
        real_sec.append(np.mean([len(set(a) - set(b)) / max(len(a), 1) for a, b in zip(sl[1:], sl[:-1])]))
    real_sec = float(np.mean(real_sec))
    res = {'real_turnover': float(real_to), 'real_sector_turnover': real_sec, 'stk': {}, 'sec': {}}
    for r in rhos:
        tos = []
        for j in js:
            days = X.days_for(j, per); W = sim_weights(sp, days, 'b', P, 7 + j, 0.8, r)
            _, TO = X.vrun(days, W, 'normal'); tos.append(TO.mean())
        res['stk'][str(r)] = float(np.mean(tos))
        sect = []
        for j in js:
            days = X.days_for(j, per)
            for p in range(10):
                rng = np.random.default_rng([99, j, p]); srf, _ = make_fns('a', rng, r, 0.8)
                s_ = Rn.make(dict(sp, kind='IND_ONLY'), srf=srf)
                prev = None; ch = []
                for d in days[:-1]:
                    cur = set(s_.top_sectors(X.cs_cached(d)))
                    if prev is not None: ch.append(len(cur - prev) / max(len(cur), 1))
                    prev = cur
                sect.append(np.mean(ch))
        res['sec'][str(r)] = float(np.mean(sect))
    rs = min(rhos, key=lambda r: abs(res['stk'][str(r)] - real_to))
    rc = min(rhos, key=lambda r: abs(res['sec'][str(r)] - real_sec))
    res['rho_stk'] = rs; res['rho_sec'] = rc
    return res

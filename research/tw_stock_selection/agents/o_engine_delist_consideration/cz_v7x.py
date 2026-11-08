#!/usr/bin/env python3
"""V7X 可執行引擎的 consideration 口徑（只讀 import o_exec_engine_rewrite_v7x 的 x_data／x_exec；不修改）。
CSim(pl, cfg, settle=None, cmode='orig')：
  cmode='orig'：與 x_exec.Sim.run 逐行相同（A16：現金型以 0050 計值、下市日＋20 交易日撥款；換股型 ×1 於終止日轉現金；經營惡化 0）。
  cmode='consideration'：settle＝cz_engine.build_settlement(events, bound, TR=D.TR, …) 的結果（同一張表、同一套結算規則）：
    lt＋1 日把持股轉為請求權；cash 入帳前 0%（cash_0050=True 時以 0050 計值）、付款日（終止日＋20 交易日）入帳、不課稅；
    stock 部分在存續股首個可交易日 s 以「股數 × 比例」加入存續股持股，之後由一般換股流程（V7X 成本模型）賣出；
    fixed_sell（特別股、價格缺口）在 s 起的下一個換股日以 V7X 賣出稅費（證交稅＋手續費）出清；writeoff＝0。
    不在事件表的永久停牌部位：沿用原 XR／MCASH 規則（並計數 n_fallback）。"""
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
V7X = HERE.parent / 'o_exec_engine_rewrite_v7x'
sys.path.insert(0, str(V7X))
import x_data as D, x_exec as X
sys.path.pop(0)


class CSim(X.Sim):
    def __init__(self, pl, cfg, settle=None, cmode='orig'):
        super().__init__(pl, cfg)
        self.settle = settle or {}; self.cmode = cmode
        self.acc['n_settled'] = 0; self.acc['n_fallback'] = 0
        self.trig = {}
        for j, rules in self.settle.items():
            for ru in rules: self.trig.setdefault(ru['trig'], []).append((j, ru))
        self.log = []

    def run(self, t1=None):
        c = self.c; t0 = self.t0
        t1 = len(D.IDX) - 1 if t1 is None else t1
        self.S = np.zeros(D.N); self.cash = c['tier'] * D.cpi_scale(2006); self.cash_late = 0.0
        self.pend = {}; mpend = []; claims = []
        unit = 1.0; eq_prev = self.cash
        out = np.empty(t1 - t0 + 1); out[0] = 1.0
        cons = self.cmode == 'consideration'
        for t in range(t0 + 1, t1 + 1):
            r50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
            for m in mpend:
                if not (t in self.reb and m[1] < t): m[0] *= r50
            H = np.flatnonzero(self.S > 0)
            if len(H): self.S[H] *= D.ADJ[t, H]
            if self.cash_late:
                self.cash += self.cash_late; self.cash_late = 0.0
            tr = D.TR[t] & np.isfinite(D.O[t])
            isreb = t in self.reb
            if cons:
                # 觸發
                for j, ru in self.trig.get(t, []):
                    if self.S[j] <= 0: continue
                    sh = self.S[j]; V = sh * D.CF[t - 1, j]
                    self.S[j] = 0.0; self.pend.pop(j, None); self.acc['n_settled'] += 1
                    ev = dict(date=str(D.IDX[t].date()), code=ru['code'], type=ru['type'], value_lastclose=float(V), parts=[])
                    for p in ru['parts']:
                        if p['kind'] == 'cash': claims.append(dict(kind='cash', value=sh * p['per_share'], rel=max(p['rel'], t), c50=ru['cash_0050'], ev=ev))
                        elif p['kind'] == 'fixed_sell': claims.append(dict(kind='fixed_sell', value=sh * p['per_share'], rel=max(p['rel'], t), zero=p.get('zero', False), ev=ev))
                        elif p['kind'] == 'stock': claims.append(dict(kind='stock', value=sh * p['per_share_window'], rel=max(p['rel'], t), succ=p['succ'], q=sh * p['ratio'], ev=ev))
                    self.log.append(ev)
                for cl in claims:
                    if cl['kind'] == 'cash' and cl.get('c50') and t <= cl['rel']: cl['value'] *= r50
                keep = []
                for cl in claims:
                    if t >= cl['rel']:
                        if cl['kind'] == 'cash':
                            self.cash += cl['value']; cl['ev']['parts'].append(('cash', str(D.IDX[t].date()), float(cl['value']))); continue
                        if cl['kind'] == 'stock':
                            self.S[cl['succ']] += cl['q']
                            cl['ev']['parts'].append(('stock', str(D.IDX[t].date()), D.COLS[cl['succ']], float(cl['q'] * D.C[t, cl['succ']]))); continue
                        if cl['kind'] == 'fixed_sell' and isreb:
                            pr = 0.0 if cl.get('zero') else cl['value'] * (1 - X.TAX - self.feerate(t))
                            self.cash += pr; cl['ev']['parts'].append(('writeoff' if cl.get('zero') else 'fixed_sell', str(D.IDX[t].date()), float(pr))); continue
                    keep.append(cl)
                claims = keep
            if isreb:
                pk = self.reb[t]
                self.pend = {}
                H = np.flatnonzero(self.S > 0)
                frz = H[~tr[H]]
                for j in frz:
                    if cons:
                        if j in D.MCASH or j in D.XR:
                            # 只有不在事件表的永久停牌才會走到這裡
                            lt_ok = (j in D.XR and D.XR[j][1] <= t < D.XR[j][2]) or (j in D.MCASH and D.MCASH[j][1] <= t < D.MCASH[j][3])
                            if lt_ok: self.acc['n_fallback'] += 1
                        else:
                            continue
                    if j in D.MCASH:
                        fc, di, pi, uu = D.MCASH[j]
                        if di <= t < uu:
                            mpend.append([self.S[j] * D.CF[t - 1, j] * D.XR[j][0] * r50, pi]); self.S[j] = 0.0
                            self.acc['n_merger_delay'] += 1
                        continue
                    if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                        self.cash += self.S[j] * D.CF[t - 1, j] * D.XR[j][0]; self.S[j] = 0.0
                keep = []
                for m in mpend:
                    if m[1] < t: self.cash += m[0]
                    else: keep.append(m)
                mpend = keep
                H = np.flatnonzero(self.S > 0); trH = H[tr[H]]; frH = H[~tr[H]]
                mval = sum(m[0] for m in mpend) + sum(cl['value'] for cl in claims)
                eq_pre = self.cash + (self.S[trH] * D.O[t, trH]).sum() + (self.S[frH] * D.CF[t - 1, frH]).sum() + mval
                unit *= eq_pre / eq_prev
                flow = 0.0
                if c['mode'] == 'FIX':
                    tgt = c['tier'] * D.cpi_scale(X.YEAR[t])
                    self.trk['flow_ratio'].append((tgt - eq_pre) / eq_pre)
                    if c.get('f1'):
                        flow = tgt - eq_pre; self.cash += flow
                    else:
                        sc = tgt / eq_pre
                        self.S *= sc; self.cash *= sc
                        for m in mpend: m[0] *= sc
                        for cl in claims: cl['value'] *= sc; cl['q'] = cl.get('q', 0) * sc
                        flow = tgt - eq_pre
                fr = self.S[trH] - np.floor(self.S[trH] + 1e-9)
                fr = np.where(fr > 0, fr, 0.0)
                self.cash += (fr * D.O[t, trH]).sum(); self.S[trH] -= fr
                self.rebalance(t, pk, tr)
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + sum(cl['value'] for cl in claims)
                unit *= eq / (eq_pre + flow)
            else:
                if self.pend:
                    self.deferred(t, tr); self.acc['deferred_days'] += 1
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + sum(cl['value'] for cl in claims)
                unit *= eq / eq_prev
            eq_prev = eq
            out[t - t0] = unit
        return pd.Series(out, index=D.IDX[t0:t1 + 1])

#!/usr/bin/env python3
"""V7X 可執行引擎（o_exec_engine_rewrite_v7x/x_exec.py，唯讀 import）加「目標權重」子類。
只覆寫 rebalance 的目標金額：原版每檔 avail/k（等權），這裡每檔 λ·w_i·avail·(1−摩擦)，λ 用同一個二分法讓估計成本 ≤ 現金。
成本（K5T 衝擊、零股價差、手續費最低 20 元、證交稅、處置期賣出、漲跌停遞補、下市／併購出清、FIX／GROW）一律沿用 V7X 原碼，不重寫。
V7X 正式設定 f1=f2=True、f3=0.5、f4=True（o_exec_engine_rewrite_v7x PREREG final_cfg）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
V7X = Path(__file__).resolve().parent.parent / 'o_exec_engine_rewrite_v7x'
sys.path.insert(0, str(V7X))
import x_data as D   # noqa: E402
import x_exec as XE  # noqa: E402
sys.path.pop(0)
SP = D.SPEC7
FINAL = dict(f1=True, f2=True, f3=0.5, f4=True)
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}

class WSim(XE.Sim):
    def __init__(self, pl, cfg, wmap, band=0.0025):
        super().__init__(pl, cfg); self.wmap = wmap; self.band = band
    def rebalance(self, t, pk, tr):
        c = self.c
        wv = self.wmap[t]                       # 與 pk 對齊的權重（和為 1）
        ok = tr[pk]                             # 成交日沒開盤／已停止交易的目標股不買（同紙上引擎 wv[~tr]=0），權重重新正規化
        pk = pk[ok]; wv = wv[ok]; wv = wv / wv.sum()
        H = np.flatnonzero(self.S > 0)
        cur = H[tr[H]]
        U = np.union1d(cur, pk)
        curS = np.floor(self.S[U] + 1e-9)
        ispk = np.isin(U, pk)
        wU = np.zeros(len(U)); pos = {j: i for i, j in enumerate(U)}
        for j, w in zip(pk, wv): wU[pos[j]] += w
        dmask = D.DISP[t, U] & ~ispk
        avail = self.cash + (self.S[cur] * D.O[t, cur]).sum()
        hiv = wU * avail * (1 - SP['frictions']['cash'])
        p = D.PC[t, U]
        def deltas(lam):
            T = lam * hiv
            with np.errstate(invalid='ignore', divide='ignore'):
                Lq = np.floor(T / (1000.0 * p) + 1e-12)
                q = np.floor((T - 1000.0 * Lq * p) / p + 1e-9)
            Lq = np.nan_to_num(Lq); q = np.nan_to_num(q)
            tg = np.where(ispk, 1000.0 * Lq + np.maximum(q, 0), 0.0)
            return tg - curS
        def need(lam):
            dl = deltas(lam); s = 0.0
            for i in range(len(U)):
                d = dl[i]
                if d == 0: continue
                if c.get('f2') and dmask[i] and d < 0: continue
                s += self.cost_est(t, U[i], int(d), dmask[i])
            return s
        lam = 1.0
        if need(1.0) > self.cash:
            lo, h2 = 0.0, 1.0
            for _ in range(30):
                mid = (lo + h2) / 2
                if need(mid) <= self.cash: lo = mid
                else: h2 = mid
            lam = lo
        dl = deltas(lam)
        if self.band > 0:     # 不交易帶（同 o_capmatched_pit_close/s6_exec.py 的 0.25%）：目標仍持有、且調整金額 < band×可用資金的單不下
            tg_pos = (curS + dl) > 0
            small = (np.abs(dl * np.nan_to_num(p)) < self.band * avail) & tg_pos & (curS > 0)
            dl = np.where(small, 0.0, dl)
        order = np.argsort(dl, kind='stable')
        for i in order:
            d = int(dl[i]); j = U[i]
            if d == 0: continue
            side = 1 if d > 0 else -1
            if dmask[i] and side < 0:
                if c.get('disp_lim') and self.lim_block(t, j, -1):
                    self.pend[j] = d; self.acc['n_lim_block'] += 1; continue
                n = self.fill_disp_sell(t, j, -d)
                if n < -d: self.pend[j] = d + n
                continue
            if self.lim_block(t, j, side):
                self.pend[j] = d; self.acc['n_lim_block'] += 1; continue
            if side > 0 and self.cost_est(t, j, d, False) > self.cash + 1e-9:
                self.pend[j] = d; continue
            a = abs(d); L, q = a // 1000, a % 1000
            dq = self.fill_odd(t, j, q, side)
            dL = self.fill_whole(t, j, L, side)
            rem = a - dq - dL
            if rem > 0: self.pend[j] = side * rem

def run_weighted(sched, tier, mode, cost, t1=None, band=0.0025):
    """sched：[(signal_day Timestamp, idx array, w array)]（None 權重＝不換股的月份請直接略過）。回傳 (日淨值 Series, acc)。"""
    pl = []; wmap = {}
    for d, idx, w in sched:
        s = D.TPOS[pd.Timestamp(d)]; e = s + 1
        m = w > 1e-9
        pl.append((e, np.asarray(idx)[m].astype(int), s)); wmap[e] = np.asarray(w)[m] / np.asarray(w)[m].sum()
    cfg = dict(tier=TIERS[tier], mode=mode, model='K5T', cost=cost); cfg.update(FINAL)
    sim = WSim(pl, cfg, wmap, band=band)
    nav = sim.run(t1=t1)
    return nav, sim.acc

def win_returns(nav, days):
    v = nav.reindex(pd.DatetimeIndex(days)).to_numpy()
    return pd.Series(v[1:] / v[:-1] - 1, index=pd.DatetimeIndex(days[:-1]))

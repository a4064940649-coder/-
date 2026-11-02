#!/usr/bin/env python3
"""V7X 可執行引擎（依 model_spec7.json 與 interp_register.json 逐條自寫）。
cfg 欄位：
  tier（2026 幣值元）、mode 'FIX'|'GROW'、model 'K5T'|'TICK'|'K5'、cost 'normal'|'stress'
  修正：f1（FIX 經由現金進出）、f2（處置期 20% 日量上限＋款項隔日可用）、f3（2020 前盤後零股量上限，值＝ratio_pre；0＝關）、f4（參與上限用事前 OA）
  解讀開關（iterations 用）：stress_odd_extra（壓力時零股也加 0.4%）、disp_lim（處置賣出受跌停限制）
"""
import math
import numpy as np, pandas as pd
import x_data as D

SP = D.SPEC7
OA_SHARE = np.array([SP['oa_share_by_dec'][str(i)] for i in range(1, 11)])
Z_SHARE = np.array([SP['zcap']['share_by_dec'][str(i)] for i in range(1, 11)])
Z_DEF = SP['zcap']['share_default']
DR_E = np.array(SP['drift']['edges']); DR_V = np.array(SP['drift']['vals'])
OD_PE = np.array(SP['odd']['price_edges']); OD_RE = SP['odd']['rank_edges'][0]
OD_B = np.array(SP['odd']['buy']); OD_S = np.array(SP['odd']['sell'])
BASE = SP['base']; KAPPA = SP['kappa']; CAP = SP['cap']; EXTRA = SP['stress']['extra_per_side']
TAX = 0.003
I_ODD = int(np.searchsorted(D.DATES, np.datetime64(D.ODD_SWITCH)))
I_FEE = int(np.searchsorted(D.DATES, np.datetime64(D.FEEMIX_SWITCH)))
I_LIM = int(np.searchsorted(D.DATES, np.datetime64(D.LIM_SWITCH)))
YEAR = D.IDX.year.values

# 漲跌停價（每日每檔）
with np.errstate(invalid='ignore', divide='ignore'):
    _lim = np.where(np.arange(len(D.IDX))[:, None] < I_LIM, 0.07, 0.10)
    _up = D.REF * (1 + _lim); _dn = D.REF * (1 - _lim)
    LIMUP = np.floor(_up / D.tick(_up) + 1e-9) * D.tick(_up)
    LIMDN = np.ceil(_dn / D.tick(_dn) - 1e-9) * D.tick(_dn)
del _lim, _up, _dn
DEC = np.where(np.isfinite(D.VRp) & (D.VRp <= 300), np.ceil(np.nan_to_num(D.VRp, nan=300) / 30).clip(1, 10), 10).astype(np.int8)
ZSH = np.where(np.isfinite(D.VRp) & (D.VRp <= 300), Z_SHARE[DEC - 1], Z_DEF)
OASH = OA_SHARE[DEC - 1]
RB = (np.nan_to_num(D.VRp, nan=1e9) > OD_RE).astype(np.int8)
# 預先算好的逐格陣列（加速；數值與逐筆計算相同）
with np.errstate(invalid='ignore', divide='ignore'):
    TKO = D.tick(D.O) / D.O
    TKC = D.tick(D.C)
    _gap = np.where(np.isfinite(D.PC) & (D.PC > 0), D.O / D.PC - 1, 0.0)
    _gap = np.where(np.isfinite(_gap), _gap, 0.0)
    _drift = DR_V[np.searchsorted(DR_E, _gap, side='right')]
    ODDP = np.where(np.arange(len(D.IDX))[:, None] < I_ODD, D.C, D.O * (1 + _drift))
    _pb = np.searchsorted(OD_PE, np.nan_to_num(D.PC, nan=0.0), side='right')
    OCB = OD_B[_pb, RB]
    OCS = np.maximum(OD_S[_pb, RB], 0.0)
    RB1 = (np.nan_to_num(D.VRp, nan=1e9) >= OD_RE).astype(np.int8)   # 名次桶以 searchsorted 'right'（名次 150 歸第 2 桶）
    OCB1 = OD_B[_pb, RB1]
    OCS1 = np.maximum(OD_S[_pb, RB1], 0.0)
del _gap, _drift, _pb


class Sim:
    def __init__(self, pl, cfg):
        self.pl = pl; self.c = cfg
        self.reb = {e: pk for e, pk, sg in pl}
        self.t0 = pl[0][2]
        self.acc = dict(tax=0.0, fee=0.0, impact=0.0, odd_spread=0.0, disp_cost=0.0, flow_cost=0.0, n_orders=0, n_cap_hit=0,
                        n_zcap_hit=0, n_lim_block=0, deferred_days=0, n_disp_sells=0, n_disp_cap_hit=0, n_merger_delay=0)
        self.part = []   # 整股成交 Q/OA_true
        self._lastpart = None
        self.trk = dict(disp_ratio=[], odd_pre_ratio=[], flow_ratio=[], oa_ratio=[])
        st = cfg['cost'] == 'stress'
        self.extra = EXTRA if st else 0.0
        self.fee_b = st or False

    # ---------- 價格與成本 ----------
    def feerate(self, t):
        if self.c['cost'] == 'stress' or t < I_FEE: return 0.001425 * 0.6
        return 0.001425 * 0.28

    def imp(self, t, j, L):
        """整股每邊衝擊率；L＝實際成交張數。"""
        tk = TKO[t, j]
        m = self.c['model']
        if m == 'TICK':
            r = tk
        else:
            r = BASE
            oa = OASH[t, j] * D.V[t, j]; sg = D.SIGp[t, j]
            if L > 0 and np.isfinite(oa) and oa > 0 and np.isfinite(sg):
                q = L * 1000.0 * D.PC[t, j]
                r = r + KAPPA * sg * math.sqrt(q / oa)
            if m == 'K5T': r = max(r, tk)
        return r + self.extra

    def oddpx(self, t, j):
        return ODDP[t, j]

    def oddc(self, t, j, side):
        rr = self.c.get('rank_right')
        if side > 0:
            c = (OCB1 if rr else OCB)[t, j]
            if self.c['cost'] == 'stress': c *= 2
        else:
            c = (OCS1 if rr else OCS)[t, j]
        if self.c['cost'] == 'stress' and self.c.get('stress_odd_extra'): c += EXTRA
        return c

    def oddcap(self, t, j):
        """零股當日上限股數（None＝無上限）。"""
        if t >= I_ODD:
            px = self.oddpx(t, j); v = D.V60p[t, j]
            if not np.isfinite(v): v = 0.0
            return math.floor(SP['zcap']['part_main'] * ZSH[t, j] * v / px)
        r = self.c.get('f3', 0)
        if r:
            px = D.C[t, j]; v = D.V60p[t, j]
            if not np.isfinite(v): v = 0.0
            return math.floor(r * ZSH[t, j] * v / px)
        return None

    def lotcap(self, t, j):
        if self.c.get('f4'):
            oa = OASH[t, j] * D.V60p[t, j]
        else:
            oa = OASH[t, j] * D.V[t, j]
        if not (np.isfinite(oa) and oa > 0): return 1
        return max(1, math.floor(CAP * oa / (1000.0 * D.PC[t, j])))

    def cost_est(self, t, j, delta, disp):
        """單檔估計成本（現金支出為正；賣出為負的淨收入），不考慮上限。"""
        if delta == 0: return 0.0
        side = 1 if delta > 0 else -1; a = abs(delta)
        L, q = int(a // 1000), int(a % 1000)
        fr = self.feerate(t); tot = 0.0
        if disp and side < 0:
            px = D.C[t, j] - TKC[t, j]
            for n in (L * 1000, q):
                if n > 0:
                    v = n * px; tot += -v + max(20.0, fr * v) + TAX * v
            return tot
        if L > 0:
            v = 1000.0 * L * D.O[t, j] * (1 + side * self.imp(t, j, L))
            tot += side * v + max(20.0, fr * v) + (TAX * v if side < 0 else 0.0)
        if q > 0:
            v = q * self.oddpx(t, j) * (1 + side * self.oddc(t, j, side))
            tot += side * v + max(20.0, fr * v) + (TAX * v if side < 0 else 0.0)
        return tot

    # ---------- 實際成交 ----------
    def fill_whole(self, t, j, L, side):
        if L <= 0: return 0
        cap = self.lotcap(t, j)
        if L > cap: L = cap; self.acc['n_cap_hit'] += 1
        im = self.imp(t, j, L)
        oa = OASH[t, j] * D.V[t, j]
        if np.isfinite(oa) and oa > 0:
            self.part.append(1000.0 * L * D.PC[t, j] / oa)
            if np.isfinite(D.V60p[t, j]) and D.V60p[t, j] > 0: self.trk['oa_ratio'].append(D.V[t, j] / D.V60p[t, j])
        o = D.O[t, j]; v = 1000.0 * L * o * (1 + side * im)
        fe = max(20.0, self.feerate(t) * v); tx = TAX * v if side < 0 else 0.0
        self.cash += -side * v - fe - tx
        self.S[j] += side * 1000.0 * L
        self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['impact'] += 1000.0 * L * o * im; self.acc['n_orders'] += 1
        return L * 1000

    def fill_odd(self, t, j, q, side, cashlim=None):
        if q <= 0: return 0
        cap = self.oddcap(t, j)
        if cap is not None and q > cap:
            q = max(cap, 0); self.acc['n_zcap_hit'] += 1
            if q == 0: return 0
        px = self.oddpx(t, j); c = self.oddc(t, j, side)
        if t < I_ODD:
            vv = ZSH[t, j] * D.V60p[t, j]
            if np.isfinite(vv) and vv > 0: self.trk['odd_pre_ratio'].append(q * px / vv)
        v = q * px * (1 + side * c)
        fe = max(20.0, self.feerate(t) * v); tx = TAX * v if side < 0 else 0.0
        if cashlim is not None and side > 0 and v + fe > cashlim: return 0
        self.cash += -side * v - fe - tx
        self.S[j] += side * q
        self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['odd_spread'] += q * px * c; self.acc['n_orders'] += 1
        return q

    def fill_disp_sell(self, t, j, n):
        """處置期賣出：收盤減一跳；f2 時受當日成交值 20% 上限。回傳成交股數。"""
        if n <= 0: return 0
        if self.c.get('f2'):
            capn = math.floor(0.20 * np.nan_to_num(D.V[t, j]) / D.C[t, j])
            if n > capn: n = max(capn, 0); self.acc['n_disp_cap_hit'] += 1
            if n == 0: return 0
        px = D.C[t, j] - TKC[t, j]; fr = self.feerate(t)
        vv = D.V[t, j]
        if np.isfinite(vv) and vv > 0: self.trk['disp_ratio'].append(n * D.C[t, j] / vv)
        L, q = int(n // 1000), int(n % 1000); proceeds = 0.0
        for m in (q, L * 1000):
            if m > 0:
                v = m * px; fe = max(20.0, fr * v); tx = TAX * v
                proceeds += v - fe - tx
                self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['disp_cost'] += m * TKC[t, j]; self.acc['n_orders'] += 1
        if self.c.get('f2'): self.cash_late += proceeds
        else: self.cash += proceeds
        self.S[j] -= n; self.acc['n_disp_sells'] += 1
        return n

    def lim_block(self, t, j, side):
        o = D.O[t, j]
        if side > 0: return bool(np.isfinite(LIMUP[t, j]) and o >= LIMUP[t, j] - 1e-6)
        return bool(np.isfinite(LIMDN[t, j]) and o <= LIMDN[t, j] + 1e-6)

    # ---------- 主迴圈 ----------
    def run(self, t1=None):
        c = self.c; t0 = self.t0
        t1 = len(D.IDX) - 1 if t1 is None else t1
        self.S = np.zeros(D.N); self.cash = c['tier'] * D.cpi_scale(2006); self.cash_late = 0.0
        self.pend = {}; mpend = []; used_m = set()
        unit = 1.0; eq_prev = self.cash
        out = np.empty(t1 - t0 + 1); out[0] = 1.0
        for t in range(t0 + 1, t1 + 1):
            r50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
            for m in mpend:
                if not (t in self.reb and m[1] < t): m[0] *= r50
            H = np.flatnonzero(self.S > 0)
            if len(H): self.S[H] *= D.ADJ[t, H]
            if self.cash_late:
                self.cash += self.cash_late; self.cash_late = 0.0
            tr = D.TR[t] & np.isfinite(D.O[t])
            if t in self.reb:
                pk = self.reb[t]
                self.pend = {}
                H = np.flatnonzero(self.S > 0)
                frz = H[~tr[H]]
                for j in frz:
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
                mval = sum(m[0] for m in mpend)
                eq_pre = self.cash + (self.S[trH] * D.O[t, trH]).sum() + (self.S[frH] * D.CF[t - 1, frH]).sum() + mval
                unit *= eq_pre / eq_prev
                flow = 0.0
                if c['mode'] == 'FIX':
                    tgt = c['tier'] * D.cpi_scale(YEAR[t])
                    self.trk['flow_ratio'].append((tgt - eq_pre) / eq_pre)
                    if c.get('f1'):
                        flow = tgt - eq_pre; self.cash += flow
                    else:
                        sc = tgt / eq_pre
                        self.S *= sc; self.cash *= sc
                        for m in mpend: m[0] *= sc
                        flow = tgt - eq_pre
                # 小數股
                fr = self.S[trH] - np.floor(self.S[trH] + 1e-9)
                fr = np.where(fr > 0, fr, 0.0)
                self.cash += (fr * D.O[t, trH]).sum(); self.S[trH] -= fr
                self.rebalance(t, pk, tr)
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend)
                unit *= eq / (eq_pre + flow)
            else:
                if self.pend:
                    self.deferred(t, tr); self.acc['deferred_days'] += 1
                H = np.flatnonzero(self.S > 0)
                eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend)
                unit *= eq / eq_prev
            eq_prev = eq
            out[t - t0] = unit
        return pd.Series(out, index=D.IDX[t0:t1 + 1])

    def rebalance(self, t, pk, tr):
        c = self.c; k = len(pk)
        H = np.flatnonzero(self.S > 0)
        cur = H[tr[H]]
        U = np.union1d(cur, pk)
        curS = np.floor(self.S[U] + 1e-9)
        ispk = np.isin(U, pk)
        dmask = D.DISP[t, U] & ~ispk
        avail = self.cash + (self.S[cur] * D.O[t, cur]).sum()
        hi = avail / k * (1 - SP['frictions']['cash'])
        p = D.PC[t, U]
        # f2：處置賣出款不能拿來付開盤買單 → 求解時現金不含處置賣出款（以零計）
        def deltas(T):
            L = np.floor(T / (1000.0 * p) + 1e-12)
            q = np.floor((T - 1000.0 * L * p) / p + 1e-9)
            tg = np.where(ispk, 1000.0 * L + np.maximum(q, 0), 0.0)
            return tg - curS
        def need(T):
            dl = deltas(T); s = 0.0
            for i in range(len(U)):
                d = dl[i]
                if d == 0: continue
                if c.get('f2') and dmask[i] and d < 0: continue
                s += self.cost_est(t, U[i], int(d), dmask[i])
            return s
        T = hi
        if need(hi) > self.cash:
            lo, h2 = 0.0, hi
            for _ in range(40):
                mid = (lo + h2) / 2
                if need(mid) <= self.cash: lo = mid
                else: h2 = mid
            T = lo
        dl = deltas(T)
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

    def deferred(self, t, tr):
        items = sorted(self.pend.items(), key=lambda x: x[1])
        for j, d in items:
            if not tr[j]: continue
            side = 1 if d > 0 else -1
            if side < 0:
                n = min(-d, round(self.S[j]))
                if n <= 0: del self.pend[j]; continue
                if D.DISP[t, j] and not (self.c.get('disp_lim') and self.lim_block(t, j, -1)):
                    got = self.fill_disp_sell(t, j, n)
                    rem = -d - got
                    if rem > 0: self.pend[j] = -rem
                    else: del self.pend[j]
                    continue
                if self.lim_block(t, j, -1): self.acc['n_lim_block'] += 1; continue
                L, q = n // 1000, n % 1000
                got = self.fill_odd(t, j, q, -1) + self.fill_whole(t, j, L, -1)
                rem = -d - got
                if rem > 0: self.pend[j] = -rem
                else: del self.pend[j]
            else:
                if self.lim_block(t, j, 1): self.acc['n_lim_block'] += 1; continue
                o = D.O[t, j]
                Lmax = math.floor(self.cash / (o * 1000 * 1.05)) if self.cash > 0 else 0
                L = min(d // 1000, Lmax); q = d % 1000
                got = 0
                if q > 0:
                    lim = self.cash - L * o * 1050
                    got += self.fill_odd(t, j, q, 1, cashlim=lim)
                got += self.fill_whole(t, j, L, 1)
                rem = d - got
                if rem > 0: self.pend[j] = rem
                else: del self.pend[j]


def run_one(pl, cfg):
    s = Sim(pl, cfg); nav = s.run()
    pa = np.array(s.part) if s.part else np.array([np.nan])
    s.acc['part_median'] = float(np.nanmedian(pa)); s.acc['part_p90'] = float(np.nanpercentile(pa, 90)) if np.isfinite(pa).any() else float('nan')
    s.acc['part_gt10'] = float(np.mean(pa > 0.10)) if np.isfinite(pa).any() else float('nan')
    s.acc['n_part'] = int(np.isfinite(pa).sum())
    s.acc['part_mean'] = float(np.nanmean(pa))
    return nav, s.acc


def run_track(pl, cfg, t1=None):
    s = Sim(pl, cfg); nav = s.run(t1=t1)
    return nav, s.acc, s.trk, s.part

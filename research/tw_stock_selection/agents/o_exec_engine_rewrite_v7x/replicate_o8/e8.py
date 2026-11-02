#!/usr/bin/env python3
"""複製員 o8：可執行引擎（只依 V7X 凍結規格文字自寫）。
設定 cfg：model K5T/TICK、cost normal/stress、tier、acct FIX/GROW、F1/F2/F3(zprop)/F4、f2fix（事後修正版：處置款隔日可用）、
zcap_nan（V60 缺值時零股上限：'zero'＝0，'none'＝不設上限）。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import d8

D = d8.load()
IDX = D['IDX']; NT = len(IDX)
Cn, On, Vn, Rn, Fn = D['Cn'], D['On'], D['Vn'], D['Rn'], D['Fn']
TR, CF, PC, ADJ = D['TR'], D['CF'], D['PC'], D['ADJ']
V60, SIG, VR, DISP, E50 = D['V60'], D['SIG'], D['VR'], D['DISP'], D['E50']
XR = D['XR']; TPOS = D['TPOS']; CI = D['CI']
CASHJ = {j: v for j, v in XR.items() if v[3] == 'cash' and v[4] is not None}
SPEC = json.loads((d8.O7 / 'data' / 'model_spec7.json').read_text())
CPI = json.loads((d8.O7 / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text())['index_2006_100']
CPI = {int(k): v for k, v in CPI.items()}
OASH = {int(k): v for k, v in SPEC['oa_share_by_dec'].items()}
ZSH = {int(k): v for k, v in SPEC['zcap']['share_by_dec'].items()}
DR_E = np.array(SPEC['drift']['edges']); DR_V = np.array(SPEC['drift']['vals'])
OD_PE = np.array(SPEC['odd']['price_edges']); OD_RE = np.array(SPEC['odd']['rank_edges'])
OD_B = SPEC['odd']['buy']; OD_S = SPEC['odd']['sell']
T_ZC = TPOS[IDX[IDX >= '2020-10-26'][0]]
T_FEE = TPOS[IDX[IDX >= '2015-01-01'][0]]
T_LIM10 = TPOS[IDX[IDX >= '2015-06-01'][0]]
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
YEAR = IDX.year.to_numpy()


def tk(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0


# 漲跌停（開盤 ≥ 漲停 → 買不到；開盤 ≤ 跌停 → 賣不掉）
def _limits():
    with np.errstate(invalid='ignore', divide='ignore'):
        ref = np.where(Fn == 1, PC, Cn / (1 + Rn))
        lim = np.where(np.arange(NT)[:, None] < T_LIM10, 0.07, 0.10)
        ru = ref * (1 + lim); rd = ref * (1 - lim)
        tu = d8.tick(ru); td = d8.tick(rd)
        lu = np.floor(ru / tu + 1e-9) * tu; ld = np.ceil(rd / td - 1e-9) * td
        bb = np.isfinite(On) & np.isfinite(ref) & (On >= lu - 1e-6)
        sb = np.isfinite(On) & np.isfinite(ref) & (On <= ld + 1e-6)
    return bb, sb
BUYBLK, SELLBLK = _limits()


class Sim:
    def __init__(self, plan, t1, cfg):
        self.plan = plan; self.t1 = t1; self.c = cfg
        self.reb = {e: pk for e, sg, pk in plan}
        self.t0 = plan[0][0] - 1
        st = cfg['cost'] == 'stress'
        self.extra = 0.004 if st else 0.0
        self.oddmul = 2.0 if st else 1.0
        self.feeB_all = st
        self.k5t = cfg['model'] == 'K5T'
        self.tier = TIERS[cfg['tier']]
        self.fix = cfg['acct'] == 'FIX'
        self.F1 = cfg.get('F1', False); self.F2 = cfg.get('F2', False); self.F3 = cfg.get('F3', 0.0); self.F4 = cfg.get('F4', False)
        self.f2fix = cfg.get('f2fix', False); self.zcap_nan = cfg.get('zcap_nan', 'zero')
        self.acc = dict(n_disp=0, n_mdelay=0, n_cap=0, qoa=[], fee=0.0, tax=0.0, odd_spread=0.0, impact=0.0, disp_cost=0.0, n_orders=0, n_lim_block=0, n_zcap_hit=0, deferred_days=0)

    # ---------- 價格與成本 ----------
    def fee(self, t, amt):
        r = 0.001425 * (0.6 if (self.feeB_all or t < T_FEE) else 0.28)
        return max(20.0, r * amt)

    def dec(self, t, j):
        v = VR[t - 1, j]
        if not np.isfinite(v) or v > 300: return 10
        return int(math.ceil(v / 30))

    def oa(self, t, j):
        return OASH[self.dec(t, j)] * (Vn[t, j] if np.isfinite(Vn[t, j]) else 0.0)

    def lotcap(self, t, j):
        if self.F4:
            v = V60[t - 1, j]; oa = OASH[self.dec(t, j)] * v if np.isfinite(v) else 0.0
        else:
            oa = self.oa(t, j)
        pc = PC[t, j]
        return max(1, int(math.floor(0.2 * oa / (1000 * pc)))) if pc > 0 else 1

    def imp(self, t, j, lots):
        o = On[t, j]; ti = tk(o) / o
        if self.k5t and lots > 0:
            oa = self.oa(t, j); q = lots * 1000 * PC[t, j]
            s = SIG[t - 1, j]; s = s if np.isfinite(s) else 0.0
            x = 0.001 + 0.5 * s * math.sqrt(q / oa) if oa > 0 else 1.0
            if oa > 0: self.acc['qoa'].append(q / oa) if self.c.get('track_qoa') else None
            ti = max(x, ti)
        return ti + self.extra

    def oddref(self, t, j):
        if t < T_ZC: return Cn[t, j]
        o = On[t, j]; g = o / PC[t, j] - 1
        return o * (1 + DR_V[int(np.searchsorted(DR_E, g, side='right'))])

    def oddc(self, t, j, ref, buy):
        pb = int(np.searchsorted(OD_PE, ref, side='right'))
        v = VR[t - 1, j]; rb = int(np.searchsorted(OD_RE, v, side='right')) if np.isfinite(v) else 1
        if buy: return OD_B[pb][rb] * self.oddmul + self.extra
        return max(OD_S[pb][rb], 0.0) + self.extra

    def oddcap(self, t, j, ref):
        if t >= T_ZC:
            v = V60[t - 1, j]
            if not np.isfinite(v):
                return 10 ** 12 if self.zcap_nan == 'none' else 0
            return int(math.floor(ZSH[self.dec(t, j)] * v / ref))
        if self.F3 > 0:
            v = V60[t - 1, j]
            if not np.isfinite(v): return 0
            return int(math.floor(self.F3 * ZSH[self.dec(t, j)] * v / Cn[t, j]))
        return 10 ** 12

    def dispprice(self, t, j):
        c = Cn[t, j]; return c - tk(c)

    # ---------- 單檔現金流（不含上限；need 用）----------
    def flow(self, t, j, d):
        """d＝目標股數−持股；回傳淨現金需求（買為正）。"""
        if d == 0: return 0.0
        n = abs(d); L, o = divmod(n, 1000)
        if d < 0:
            if DISP[t, j]:
                p = self.dispprice(t, j); a = n * p
                return -(a - self.fee(t, a) - 0.003 * a)
            out = 0.0
            if o:
                r = self.oddref(t, j); a = o * r * (1 - self.oddc(t, j, r, False)); out -= a - self.fee(t, a) - 0.003 * a
            if L:
                a = L * 1000 * On[t, j] * (1 - self.imp(t, j, L)); out -= a - self.fee(t, a) - 0.003 * a
            return out
        out = 0.0
        if o:
            r = self.oddref(t, j); a = o * r * (1 + self.oddc(t, j, r, True)); out += a + self.fee(t, a)
        if L:
            a = L * 1000 * On[t, j] * (1 + self.imp(t, j, L)); out += a + self.fee(t, a)
        return out

    # ---------- 成交 ----------
    def _bk(self, t, gross, side, spread=0.0, impc=0.0, disp=0.0):
        """記帳並回傳現金變動。gross＝成交金額（已含價差/衝擊）。"""
        f = self.fee(t, gross); A = self.acc
        A['fee'] += f; A['odd_spread'] += spread; A['impact'] += impc; A['disp_cost'] += disp; A['n_orders'] += 1
        if side < 0:
            A['tax'] += 0.003 * gross; return gross - f - 0.003 * gross
        return -(gross + f)

    def sell(self, t, j, o, L):
        """賣 o 零股＋L 張；回傳未成交 (o, L)。"""
        h = self.sh.get(j, 0.0)
        tot = min(o + 1000 * L, int(math.floor(h + 1e-9)))
        if tot <= 0: return 0, 0
        if o + 1000 * L > tot: L, o = divmod(tot, 1000)
        if DISP[t, j]:
            p = self.dispprice(t, j); n = o + 1000 * L
            if self.F2:
                v = Vn[t, j] if np.isfinite(Vn[t, j]) else 0.0
                n = min(n, int(math.floor(0.2 * v / p)))
            if n <= 0: return o, L
            net = self._bk(t, n * p, -1, disp=n * (Cn[t, j] - p))
            if self.F2 and not self.f2fix: self.locked += net
            elif self.F2 and self.f2fix: self.nextday += net
            else: self.cash += net
            self.sh[j] = h - n; self.acc['n_disp'] += 1
            rem = o + 1000 * L - n
            return rem % 1000, rem // 1000
        if SELLBLK[t, j]: self.acc['n_lim_block'] += 1; return o, L
        ro, rL = 0, 0
        if o:
            r = self.oddref(t, j); q = min(o, self.oddcap(t, j, r)); ro = o - q
            if ro: self.acc['n_zcap_hit'] += 1
            if q > 0:
                c = self.oddc(t, j, r, False)
                self.cash += self._bk(t, q * r * (1 - c), -1, spread=q * r * c); h -= q
        if L:
            cap = self.lotcap(t, j); q = min(L, cap); rL = L - q
            if rL: self.acc['n_cap'] += 1
            im = self.imp(t, j, q); g = q * 1000 * On[t, j]
            self.cash += self._bk(t, g * (1 - im), -1, impc=g * im); h -= q * 1000
        self.sh[j] = h
        return ro, rL

    def buy(self, t, j, o, L, deferred=False):
        if BUYBLK[t, j]: self.acc['n_lim_block'] += 1; return o, L
        r = self.oddref(t, j)
        qo = min(o, self.oddcap(t, j, r)) if o else 0
        if qo < o: self.acc['n_zcap_hit'] += 1
        cap = self.lotcap(t, j); qL = min(L, cap) if L else 0
        if qL < L: self.acc['n_cap'] += 1
        c = self.oddc(t, j, r, True) if o else 0.0
        if deferred:
            # 先零股，再整股；整股張數上限＝floor(現金/(開盤×1050))
            if qo:
                po = r * (1 + c)
                qo = min(qo, int(math.floor(max(self.cash, 0.0) / po)))
                while qo > 0 and qo * po + self.fee(t, qo * po) > self.cash: qo -= 1
                if qo > 0:
                    self.cash += self._bk(t, qo * po, 1, spread=qo * r * c); self.sh[j] = self.sh.get(j, 0.0) + qo
            if qL:
                qL = min(qL, int(math.floor(max(self.cash, 0.0) / (On[t, j] * 1050))))
                if qL > 0:
                    im = self.imp(t, j, qL); g = qL * 1000 * On[t, j]
                    if g * (1 + im) + self.fee(t, g * (1 + im)) <= self.cash:
                        self.cash += self._bk(t, g * (1 + im), 1, impc=g * im); self.sh[j] = self.sh.get(j, 0.0) + qL * 1000
                    else: qL = 0
            return o - qo, L - max(qL, 0)
        cost = 0.0
        if qo:
            a = qo * r * (1 + c); cost += a + self.fee(t, a)
        if qL:
            im = self.imp(t, j, qL); a = qL * 1000 * On[t, j] * (1 + im); cost += a + self.fee(t, a)
        if cost > self.cash: return o, L
        if qo: self.cash += self._bk(t, qo * r * (1 + c), 1, spread=qo * r * c)
        if qL:
            g = qL * 1000 * On[t, j]; self.cash += self._bk(t, g * (1 + im), 1, impc=g * im)
        self.sh[j] = self.sh.get(j, 0.0) + qo + 1000 * qL
        return o - qo, L - qL

    # ---------- 換股 ----------
    def rebalance(self, t, pk, eq_prev):
        self.defer = []
        if self.locked: self.cash += self.locked; self.locked = 0.0
        # 凍結股與併購
        for j in list(self.sh):
            h = self.sh[j]
            if h <= 0: del self.sh[j]; continue
            if TR[t, j] and np.isfinite(On[t, j]): continue
            if j in CASHJ and t >= CASHJ[j][4]:
                v = CASHJ[j]; self.pend[j] = (h * CF[t, j] * v[0], t - 1, v[4] + 20); del self.sh[j]; self.acc['n_mdelay'] += 1; continue
            if j in XR and XR[j][1] <= t < XR[j][2]:
                self.cash += h * CF[t, j] * XR[j][0]; del self.sh[j]
        for j in list(self.pend):
            v0, b, ip = self.pend[j]
            if t > ip:
                self.cash += v0 * E50[t - 1] / E50[b]; del self.pend[j]
        # FIX
        base = eq_prev
        if self.fix:
            tgt = self.tier * CPI[int(YEAR[t])] / CPI[2026]
            if self.F1:
                self.cash += tgt - eq_prev
            else:
                sc = tgt / eq_prev
                for j in self.sh: self.sh[j] *= sc
                self.cash *= sc; self.nextday *= sc
                self.pend = {j: (v0 * sc, b, ip) for j, (v0, b, ip) in self.pend.items()}
            base = tgt
        # 小數股 → 現金（開盤價）
        trd = [j for j in self.sh if TR[t, j] and np.isfinite(On[t, j])]
        for j in trd:
            h = self.sh[j]; f = math.floor(h + 1e-9)
            if h - f > 0: self.cash += (h - f) * On[t, j]
            self.sh[j] = float(f)
        k = len(pk); pks = set(int(x) for x in pk)
        excl = set(j for j in trd if DISP[t, j]) if self.F2 else set()
        hi = (sum(self.sh[j] * On[t, j] for j in trd if j not in excl) + self.cash) / k * 0.98
        names = sorted(set(trd) | pks)
        names = [j for j in names if TR[t, j] and np.isfinite(On[t, j])]

        def deltas(T):
            return {j: (int(math.floor(T / On[t, j])) if j in pks else 0) - int(self.sh.get(j, 0.0)) for j in names}

        def need(T):
            s = 0.0
            for j, d in deltas(T).items():
                if self.F2 and d < 0 and DISP[t, j]: continue
                s += self.flow(t, j, d)
            return s
        T = hi
        if need(hi) > self.cash:
            lo, up = 0.0, hi
            for _ in range(40):
                m = 0.5 * (lo + up)
                if need(m) > self.cash: up = m
                else: lo = m
            T = lo
        dl = deltas(T)
        orders = sorted([(d * On[t, j], j, d) for j, d in dl.items() if d != 0])
        for _, j, d in orders:
            L, o = divmod(abs(d), 1000)
            if d < 0:
                ro, rL = self.sell(t, j, o, L)
                if ro or rL: self.defer.append([j, -1, ro, rL])
            else:
                ro, rL = self.buy(t, j, o, L)
                if ro or rL: self.defer.append([j, 1, ro, rL])
        return base

    def do_deferred(self, t):
        self.defer.sort(key=lambda x: x[2] + 1000 * x[3])
        keep = []
        for j, sd, o, L in self.defer:
            if not (TR[t, j] and np.isfinite(On[t, j])): keep.append([j, sd, o, L]); continue
            if sd < 0: ro, rL = self.sell(t, j, o, L)
            else: ro, rL = self.buy(t, j, o, L, deferred=True)
            if ro or rL: keep.append([j, sd, ro, rL])
        self.defer = keep

    def equity(self, t):
        e = self.cash + self.locked + self.nextday
        for j, h in self.sh.items(): e += h * CF[t, j]
        for j, (v0, b, ip) in self.pend.items(): e += v0 * E50[t] / E50[b]
        return e

    def run(self):
        t0, t1 = self.t0, self.t1
        self.sh = {}; self.pend = {}; self.defer = []; self.locked = 0.0; self.nextday = 0.0
        self.cash = self.tier * CPI[2006] / CPI[2026]
        ret = np.zeros(t1 - t0 + 1); eq = self.cash
        for t in range(t0 + 1, t1 + 1):
            for j in self.sh: self.sh[j] *= ADJ[t, j]
            if self.nextday: self.cash += self.nextday; self.nextday = 0.0
            base = eq
            if t in self.reb: base = self.rebalance(t, self.reb[t], eq)
            elif self.defer: self.acc['deferred_days'] += 1; self.do_deferred(t)
            e = self.equity(t)
            ret[t - t0] = e / base - 1
            eq = e
        return pd.Series(ret, index=IDX[t0:t1 + 1])

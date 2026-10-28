#!/usr/bin/env python3
"""V7 可執行引擎（由 o_open_impact_calib/ecore6.py 與 o_exec_capacity_tiers/ecore.py 移植修改；不 import 原檔）。
資料直接取自 V7 面板上的 V4 引擎（v7env → v4lib；setup_px 之後的 K 陣列、X 出清規則、官方處置名單），每個名單（plan）各自模擬，
衝擊、順延、處置期賣出、併購結算延遲都落在該名單自己的日子（o6 教訓 13）。

模擬器 Sim（單一路徑、單一級距）：
  impact：'K5T'（上界：每邊 max(0.1%＋0.5·σ60·√(Q/OA), 一跳/開盤價)）、'K5'（o5 原式，敏感度）、'TICK'（下界：每邊一跳/開盤價）、
          'ENG'（引擎口徑：每邊 0.1%、無參與上限；只做拆解）。
  OA＝share(ceil(前一日 V60 名次/30)) × 當日成交值（MIS 校正占比，名次 > 300 或缺值＝第 10 十分位）；參與上限 20%×OA，超過的張數順延。
  opts：lim（開盤漲停買／跌停賣順延）、odd0、cash（每次換股保留現金比例）、feemix（2015 前手續費 B）、zpart（零股量上限）、
        disp（處置期賣出：收盤減一跳成交、無衝擊與上限）、mdelay（現金對價併購：下市日起以 0050 計值，下市日＋20 交易日撥款後的第一個換股日才再投入）、
        extra（壓力：每邊再加 extra）、odd_stress。
理想模式 ideal()：逐行重寫 O3.simulate（市值記帳、開盤調倉、二分法 60 次），flags 全關時必須與 v4lib.sim_plan 逐日相同（斷言 < 1e-9）；
  可另開 disp／mdelay 做零衝擊拆解。"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
FEE_FULL = 0.001425
FEES = {'A': dict(rate=FEE_FULL * 0.28, min_lot=20.0, min_odd=20.0), 'B': dict(rate=FEE_FULL * 0.6, min_lot=20.0, min_odd=20.0)}
TAX = 0.003
ENG_FEE = FEE_FULL * 0.6
ODD_INTRADAY = pd.Timestamp('2020-10-26')
FEEMIX_CUT = pd.Timestamp('2015-01-01')
_CPI = json.loads((HERE / 'raw' / 'exec_inputs' / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100']
CPI = {int(k): v for k, v in _CPI.items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]

def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

class Data:
    def __init__(self, L, XR):
        K, V = L.K, L.V
        self.IDX = K.IDX; self.CODES = list(K.COLS); self.CI = dict(K.CI); N = len(self.CODES)
        self.O = K.On; self.C = K.Cn; self.G = K.GROW; self.TR = K.TRADED; self.PC = K.PREVC
        self.CF = pd.DataFrame(self.C).ffill().to_numpy()
        self.VAL = K.V.to_numpy()
        _, V60 = V.rolling_signals('gapfix'); self.V60 = np.asarray(V60)
        G = np.asarray(K.GOOD)
        Rr = pd.DataFrame(K.Rn[G], index=self.IDX[G])
        self.SIG = Rr.rolling(60, min_periods=40).std().reindex(self.IDX).ffill().to_numpy()
        base = np.isfinite(self.V60) & (self.V60 >= 1e7) & np.isfinite(self.C) & (self.C >= 10)
        vv = np.where(base, self.V60, -np.inf); order = np.argsort(-vv, axis=1, kind='stable')
        ranks = np.empty_like(order); ar = np.arange(N)
        for t in range(len(self.IDX)): ranks[t, order[t]] = ar
        self.VRANK = np.where(base, ranks + 1.0, np.nan)
        self.XR = {j: (v[0], int(np.searchsorted(self.IDX, pd.Timestamp(v[1]))), int(np.searchsorted(self.IDX, pd.Timestamp(v[3]))) if len(v) > 3 else 10 ** 9, v[2])
                   for j, v in XR.items()}
        E = L._STATE['E']
        self.XDEL = {}
        for j, v in self.XR.items():
            c = self.CODES[j]
            if v[3] == 'cash' and c in E.index and pd.notna(E.loc[c, 'delist_date']):
                td = int(np.searchsorted(self.IDX, pd.Timestamp(E.loc[c, 'delist_date'])))
                self.XDEL[j] = (td, min(td + 20, len(self.IDX) - 1))
        self.E50 = K.E50.ffill().to_numpy()
        # 官方處置名單（V4 凍結版，L.disposal_table）：處置期間 start～end（含）
        D = L.disposal_table()
        self.DISP = np.zeros((len(self.IDX), N), bool)
        for c, a, b in zip(D['code'], D['start'], D['end']):
            j = self.CI.get(c)
            if j is None or pd.isna(a) or pd.isna(b): continue
            ia, ib = np.searchsorted(self.IDX, a), np.searchsorted(self.IDX, b, side='right')
            self.DISP[ia:ib, j] = True
        with np.errstate(invalid='ignore', divide='ignore'):
            adj = np.where(self.TR, self.G * self.PC / self.C, 1.0)
        self.ADJ = np.where(np.isfinite(adj), adj, 1.0)
        self.GAP = np.where(self.TR & np.isfinite(self.O), self.O / self.PC - 1, np.nan)
        ref = np.where(self.TR & np.isfinite(self.G) & (self.G > 0), self.C / self.G, self.PC)
        lim = np.where((self.IDX >= pd.Timestamp('2015-06-01'))[:, None], 0.10, 0.07)
        up_raw = ref * (1 + lim); dn_raw = ref * (1 - lim)
        vt = np.vectorize(tick)
        with np.errstate(invalid='ignore'):
            tu = vt(np.nan_to_num(up_raw, nan=1.0)); td_ = vt(np.nan_to_num(dn_raw, nan=1.0))
            self.LU = np.floor(up_raw / tu + 1e-6) * tu; self.LD = np.ceil(dn_raw / td_ - 1e-6) * td_

class Model:
    def __init__(self, spec, impact):
        self.s = spec; self.kind = impact
        self.share = {int(k): float(v) for k, v in spec['oa_share_by_dec'].items()}
        self.cap = float(spec['cap']); self.base = float(spec['base']); self.kappa = float(spec['kappa'])
        self.dr_edges = np.array(spec['drift']['edges']); self.dr_vals = np.array(spec['drift']['vals'])
        self.odd_pe = np.array(spec['odd']['price_edges']); self.odd_re = np.array(spec['odd']['rank_edges'])
        self.odd_buy = np.array(spec['odd']['buy']); self.odd_sell = np.array(spec['odd']['sell'])
        self.z_share = {int(k): float(v) for k, v in spec['zcap']['share_by_dec'].items()}; self.z_def = float(spec['zcap']['share_default'])
    def dec(self, rank):
        if not np.isfinite(rank): return 10
        return int(min(max(math.ceil(rank / 30), 1), 10))
    def drift(self, gap): return float(self.dr_vals[int(np.searchsorted(self.dr_edges, gap, side='right'))])
    def odd_cost(self, price, rank, side, odd0, stress):
        i = int(np.searchsorted(self.odd_pe, price, side='right')); j = int(np.searchsorted(self.odd_re, rank if np.isfinite(rank) else 1e9, side='right'))
        c = float((self.odd_buy if side > 0 else self.odd_sell)[i, j])
        if stress: c = 2 * c if side > 0 else max(c, 0.0)
        if odd0 and side < 0: c = max(c, 0.0)
        return c
    def zcap_value(self, rank, v60, part):
        d = self.dec(rank)
        return part * self.z_share.get(d, self.z_def) * v60

DEF_OPTS = dict(lim=False, odd0=False, cash=0.0, feemix=False, zpart=None, odd_stress=False, extra=0.0, disp=False, mdelay=False)

class Sim:
    def __init__(self, D, M, opts):
        self.D, self.M = D, M; self.o = dict(DEF_OPTS); self.o.update(opts or {})
    def fee(self, value, odd, t):
        if value <= 0: return 0.0
        fr = FEES['B'] if (self.o['feemix'] and self.D.IDX[t] < FEEMIX_CUT) else self.fr
        return max(fr['min_odd'] if odd else fr['min_lot'], fr['rate'] * value)
    def oa(self, t, j):
        D, M = self.D, self.M
        v = D.VAL[t, j]
        if not np.isfinite(v) or v <= 0: v = D.V60[t - 1, j]
        if not np.isfinite(v): return np.nan
        return M.share[M.dec(D.VRANK[t - 1, j])] * v
    def imp(self, t, j, Q):
        D, M = self.D, self.M
        o = D.O[t, j]; tk = tick(o) / o if np.isfinite(o) and o > 0 else 0.0
        ex = self.o['extra']
        if M.kind == 'TICK': return tk + ex
        if M.kind == 'ENG': return M.base + ex
        sig = D.SIG[t - 1, j]; oa = self.oa(t, j)
        k5 = M.base + (M.kappa * sig * math.sqrt(Q / oa) if (np.isfinite(sig) and np.isfinite(oa) and oa > 0 and Q > 0) else 0.0)
        if M.kind == 'K5': return k5 + ex
        return max(k5, tk) + ex                                   # K5T
    def in_disp(self, t, j, side):
        return self.o['disp'] and side < 0 and self.D.DISP[t, j]
    def disp_px(self, t, j):
        c = self.D.C[t, j]
        if not np.isfinite(c): c = self.D.CF[t, j]
        return max(c - tick(c), 0.01)
    def blocked(self, t, j, side):
        if not self.o['lim']: return False
        if self.in_disp(t, j, side): return False
        o = self.D.O[t, j]
        return bool(o >= self.D.LU[t, j] - 1e-6) if side > 0 else bool(o <= self.D.LD[t, j] + 1e-6)
    def odd_px(self, t, j):
        D = self.D
        if D.IDX[t] < ODD_INTRADAY: return D.C[t, j]
        g = D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0
        return D.O[t, j] * (1.0 + self.M.drift(g))
    def zcap_sh(self, t, j, px):
        part = self.o['zpart']
        if part is None or self.D.IDX[t] < ODD_INTRADAY: return None
        v = self.D.V60[t - 1, j]
        if not np.isfinite(v): return None
        return int(math.floor(self.M.zcap_value(self.D.VRANK[t - 1, j], v, part) / px))
    def book(self, t, **kw):
        eq = self.acc['_eq']
        for k, v in kw.items():
            self.acc[k] = self.acc.get(k, 0.0) + v; self.acc['pct_' + k] = self.acc.get('pct_' + k, 0.0) + v / eq
    def lot_exec(self, t, j, lots, side):
        D = self.D
        if self.in_disp(t, j, side):
            n = lots; px = self.disp_px(t, j); val = n * 1000.0 * px
            f = self.fee(val, False, t); tax = TAX * val
            self.book(t, fee=f, tax=tax, disp_cost=n * 1000.0 * (D.O[t, j] - px) if np.isfinite(D.O[t, j]) else 0.0)
            self.acc['n_disp_sells'] += 1; self.acc['n_orders'] += 1
            return n, val - f - tax
        p_ref = D.PC[t, j]; n = lots
        if self.M.kind != 'ENG':
            oa = self.oa(t, j)
            if np.isfinite(oa):
                mx = max(1, int(math.floor(self.M.cap * oa / (1000.0 * p_ref))))
                if n > mx: n = mx; self.acc['n_cap_hit'] += 1
        Q = n * 1000.0 * p_ref
        im = self.imp(t, j, Q)
        ref = D.O[t, j]; px = ref * (1 + side * im); val = n * 1000.0 * px
        f = self.fee(val, False, t); tax = TAX * val if side < 0 else 0.0
        self.book(t, fee=f, tax=tax, impact=n * 1000.0 * ref * im)
        self.acc['n_orders'] += 1; self.acc['n_lot_orders'] += 1
        return n, ((-val - f) if side > 0 else (val - f - tax))
    def odd_exec(self, t, j, sh, side):
        D = self.D
        if self.in_disp(t, j, side):
            px = self.disp_px(t, j); val = sh * px; f = self.fee(val, True, t); tax = TAX * val
            self.book(t, fee=f, tax=tax); self.acc['n_disp_sells'] += 1; self.acc['n_orders'] += 1
            return val - f - tax
        base = self.odd_px(t, j)
        c = self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], side, self.o['odd0'], self.o['odd_stress']) + self.o['extra']
        px = base * (1 + side * c); val = sh * px
        f = self.fee(val, True, t); tax = TAX * val if side < 0 else 0.0
        self.book(t, fee=f, tax=tax, odd_spread=sh * base * c)
        self.acc['n_orders'] += 1; self.acc['n_odd_orders'] += 1
        return (-val - f) if side > 0 else (val - f - tax)
    def cost(self, t, j, delta):
        key = (j, delta)
        if key in self.memo: return self.memo[key]
        D = self.D
        if delta == 0: return 0.0
        side = 1 if delta > 0 else -1; a = abs(delta)
        lots, odd = a // 1000, a % 1000; tot = 0.0
        if self.in_disp(t, j, side):
            px = self.disp_px(t, j)
            if lots: val = lots * 1000.0 * px; tot += -val + self.fee(val, False, t) + TAX * val
            if odd: val = odd * px; tot += -val + self.fee(val, True, t) + TAX * val
            self.memo[key] = tot; return tot
        if lots:
            Q = lots * 1000.0 * D.PC[t, j]; im = self.imp(t, j, Q)
            val = lots * 1000.0 * D.O[t, j] * (1 + side * im)
            tot += side * val + self.fee(val, False, t) + (TAX * val if side < 0 else 0.0)
        if odd:
            px = self.odd_px(t, j) * (1 + side * (self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], side, self.o['odd0'], self.o['odd_stress']) + self.o['extra']))
            val = odd * px
            tot += side * val + self.fee(val, True, t) + (TAX * val if side < 0 else 0.0)
        self.memo[key] = tot
        return tot
    def do_odd(self, t, j, sh, side, pend):
        if self.in_disp(t, j, side): return self.odd_exec(t, j, sh, side), sh
        px = self.odd_px(t, j); cap = self.zcap_sh(t, j, px)
        n = sh if cap is None else min(sh, max(cap, 0))
        if cap is not None and n < sh: self.acc['n_zcap_hit'] += 1; pend[j] = pend.get(j, 0) + side * (sh - n)
        if n <= 0: return 0.0, 0
        return self.odd_exec(t, j, n, side), n
    def rebalance(self, t, pk, S, tr, frz, cash):
        D = self.D; pka = list(pk); pks = set(pka)
        cur = {j: int(round(S[j])) for j in np.where(tr & ~frz & (S > 0))[0]}
        Vtr = sum(S[j] * D.O[t, j] for j in cur) + cash
        k = len(pka)
        def deltas(T):
            out = {}
            for j in set(cur) | pks:
                if j in pks:
                    p = D.PC[t, j]
                    if not np.isfinite(p) or p <= 0: tgt = 0
                    else:
                        L = int(math.floor(T / (1000.0 * p) + 1e-12)); q = int(math.floor((T - L * 1000.0 * p) / p + 1e-9)); tgt = L * 1000 + max(q, 0)
                else: tgt = 0
                d = tgt - cur.get(j, 0)
                if d != 0: out[j] = d
            return out
        self.memo = {}
        def need(T): return sum(self.cost(t, j, d) for j, d in deltas(T).items())
        hi = Vtr / k * (1 - self.o['cash']); lo = 0.0
        if need(hi) <= cash: lo = hi
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if need(mid) <= cash: lo = mid
                else: hi = mid
        dl = deltas(lo); pend = {}
        for j, d in sorted(dl.items(), key=lambda x: x[1]):
            side = 1 if d > 0 else -1; a = abs(d)
            if self.blocked(t, j, side):
                pend[j] = pend.get(j, 0) + d; self.acc['n_lim_block'] += 1; continue
            if side > 0 and self.cost(t, j, d) > cash + 1e-9:
                pend[j] = pend.get(j, 0) + d; self.acc['deferred_days'] += 1; continue
            lots, odd = a // 1000, a % 1000
            if odd:
                dc, n = self.do_odd(t, j, odd, side, pend); cash += dc; S[j] += side * n
            if lots:
                n, dc = self.lot_exec(t, j, lots, side)
                cash += dc; S[j] += side * n * 1000
                if n < lots: pend[j] = pend.get(j, 0) + side * (lots - n) * 1000
        return S, cash, pend
    def work_pending(self, t, S, cash, pend):
        D = self.D; newp = {}
        for j, d in sorted(pend.items(), key=lambda x: x[1]):
            if not (D.TR[t, j] and np.isfinite(D.O[t, j])): newp[j] = d; continue
            side = 1 if d > 0 else -1
            if self.blocked(t, j, side): newp[j] = d; continue
            a = abs(d)
            if side < 0: a = min(a, int(round(S[j])))
            lots, odd = a // 1000, a % 1000
            self.acc['deferred_days'] += 1
            if side > 0:
                unit = D.O[t, j] * 1000.0 * 1.05
                lots = min(lots, int(cash // unit)) if unit > 0 else 0
                if odd:
                    v_ = odd * self.odd_px(t, j) * (1 + self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], 1, self.o['odd0'], self.o['odd_stress']) + self.o['extra'])
                    if v_ + self.fee(v_, True, t) > cash - lots * unit: odd = 0
            got = 0
            if odd:
                tmp = {}
                dc, n = self.do_odd(t, j, odd, side, tmp); cash += dc; S[j] += side * n; got += n
            if lots:
                n, dc = self.lot_exec(t, j, lots, side)
                cash += dc; S[j] += side * n * 1000; got += n * 1000
            rem = abs(d) - got if side > 0 else a - got
            if rem > 0: newp[j] = side * rem
        return S, cash, newp
    def run(self, plan, t0, t1, tier, version='GROW', fee='A'):
        D = self.D; self.fr = FEES[fee]
        reb = {p[0]: p for p in plan}
        N = len(D.CODES); O, CF, TR, ADJ, E50 = D.O, D.CF, D.TR, D.ADJ, D.E50
        S = np.zeros(N); E0 = tier * cpi_scale(D.IDX[t0]); cash = E0
        nav = np.empty(t1 - t0 + 1); nav[0] = 1.0; prev_eq = E0; unit = 1.0
        pend = {}; rec = []           # rec：[金額（以下市日 0050 計）, 下市日 t_d, 撥款日 t_p]
        self.acc = dict(tax=0.0, fee=0.0, impact=0.0, odd_spread=0.0, disp_cost=0.0, n_orders=0, n_lot_orders=0, n_odd_orders=0, n_cap_hit=0, n_zcap_hit=0,
                        n_lim_block=0, deferred_days=0, n_disp_sells=0, n_merger_delay=0, flow=0.0, _eq=E0)
        for t in range(t0 + 1, t1 + 1):
            S = S * ADJ[t]
            flow = 0.0; eq_pre = None
            if t in reb:
                e, pk, sg = reb[t]
                tr = TR[t] & np.isfinite(O[t])
                pend = {}
                held = S > 0; frz = held & ~tr
                for j in np.where(frz)[0]:
                    if j in D.XR:
                        xr = D.XR[j]
                        if self.o['mdelay'] and j in D.XDEL and D.XDEL[j][0] <= t < xr[2]:
                            td, tp = D.XDEL[j]
                            amt = S[j] * CF[t - 1, j] * xr[0] * E50[td] / E50[t - 1]      # 以下市日的 0050 計價基準
                            rec.append([amt, td, tp]); S[j] = 0.0; frz[j] = False; self.acc['n_merger_delay'] += 1
                        elif xr[1] <= t < xr[2] and not (self.o['mdelay'] and j in D.XDEL):
                            cash += S[j] * CF[t - 1, j] * xr[0]; S[j] = 0.0; frz[j] = False
                if rec:
                    keep = []
                    for r in rec:
                        if t >= r[2]: cash += r[0] * E50[t - 1] / E50[r[1]]
                        else: keep.append(r)
                    rec = keep
                recv = sum(r[0] * E50[t - 1] / E50[r[1]] for r in rec)
                if version == 'FIX':
                    Vfz0 = float((S[frz] * CF[t - 1, frz]).sum()); Vtr0 = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
                    eq_pre = Vtr0 + Vfz0 + cash + recv
                    target = tier * cpi_scale(D.IDX[t]); sc = target / eq_pre
                    S = S * sc; cash = cash * sc; flow = target - eq_pre; self.acc['flow'] += flow
                    for r in rec: r[0] *= sc
                fra = np.where(tr & (S > 0), S - np.floor(S + 1e-9), 0.0); fra = np.clip(fra, 0, None)
                cash += float((fra * np.where(tr, O[t], 0.0)).sum()); S = S - fra
                Vfz = float((S[frz] * CF[t - 1, frz]).sum()); Vtr = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
                self.acc['_eq'] = Vtr + Vfz + cash + sum(r[0] * E50[t - 1] / E50[r[1]] for r in rec)
                S, cash, pend = self.rebalance(t, pk, S, tr, frz, cash)
            elif pend:
                self.acc['_eq'] = prev_eq
                S, cash, pend = self.work_pending(t, S, cash, pend)
            nz = S != 0
            eq = cash + float((S[nz] * CF[t, nz]).sum()) + sum(r[0] * E50[t] / E50[r[1]] for r in rec)
            assert cash > -1e-6 * max(1.0, eq), (t, cash)
            if eq_pre is None: unit *= eq / prev_eq
            else: unit *= (eq_pre / prev_eq) * (eq / (eq_pre + flow))
            nav[t - t0] = unit; prev_eq = eq
        self.acc['end_equity'] = prev_eq; self.acc['start_equity'] = E0
        return pd.Series(nav, index=D.IDX[t0:t1 + 1]), self.acc

def ideal(D, plan, t0, t1, buy, sell, disp=False, mdelay=False):
    """O3.simulate 逐行重寫（市值記帳、開盤調倉、二分法 60 次、X 出清）。disp／mdelay 全關時應與 v4lib.sim_plan 相同。
    disp：處置期間被賣出的持股，賣價改成收盤減一跳（開盤到該價的報酬計入）；mdelay：現金對價併購的出清值改以 0050 計、撥款後才再投入。"""
    reb = {p[0]: p[1] for p in plan}
    N = len(D.CODES); O, C, G, TR, E50 = D.O, D.C, D.G, D.TR, D.E50
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); rec = []
    nd = 0; nm = 0
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = np.array(reb[t]); pks = set(reb[t])
            tr = TR[t] & np.isfinite(O[t])
            gap = np.where(tr, G[t] * O[t] / np.where(tr, C[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in D.XR:
                    xr = D.XR[j]
                    if mdelay and j in D.XDEL and D.XDEL[j][0] <= t < xr[2]:
                        td, tp = D.XDEL[j]; rec.append([h[j] * xr[0] * E50[td] / E50[t - 1], td, tp]); h[j] = 0.0; frz[j] = False; nm += 1
                    elif xr[1] <= t < xr[2] and not (mdelay and j in D.XDEL):
                        cash += h[j] * xr[0]; h[j] = 0.0; frz[j] = False
            if rec:
                keep = []
                for r in rec:
                    if t >= r[2]: cash += r[0] * E50[t - 1] / E50[r[1]]
                    else: keep.append(r)
                rec = keep
            if disp:
                for j in np.where((h > 0) & tr & D.DISP[t])[0]:
                    if j not in pks:
                        c = C[t, j]; h[j] = h[j] * max(c - tick(c), 0.01) / O[t, j]; nd += 1
            htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash; k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            newval = np.where(frz, h, 0.0); newval[pk] += lo
            oc = np.where(tr, C[t] / np.where(tr, O[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash + sum(r[0] * E50[t] / E50[r[1]] for r in rec)
    return pd.Series(nav, index=D.IDX[t0:t1 + 1]), {'n_disp_sells': nd, 'n_merger_delay': nm}

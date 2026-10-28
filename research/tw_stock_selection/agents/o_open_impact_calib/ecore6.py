#!/usr/bin/env python3
"""o_open_impact_calib 容量模擬器（由 o_exec_capacity_tiers/ecore.py 移植；本題資料夾 ecore_o5.py 為原檔副本）。
只讀本題 data/ 與 raw/；執行時不 import 其他資料夾。

和 o5 的差別（全部由 opts 控制；opts 全關＋impact='O5' 時逐日重現 o5 navs_main，見 check_engine.py）：
  impact：'M3V'（主：0.1%＋b·σ·(Q/V60)^δ）、'M2'（0.1%＋Y·σ·√(Q/V60)）、'M1'（0.1%＋κ·σ·√(Q/OA_hat)）、'O5'（o5 原式 0.1%＋0.5σ√(Q/OA_o5)）。
  參與上限 20%：分母為 OA_hat（本題事前開盤量模型；名次缺值時以 300 代入）；'O5' 模式用 o5 的 OA。
  四項摩擦：lim（開盤漲停買／跌停賣不成交、順延次日）、odd0（零股賣成本下限 0）、cash（每次換股保留現金比例）、feemix（2015 前手續費 B）。
  zpart：模型 Z 零股量上限（2020-10-26 起；每檔每日零股成交值 ≤ zpart × 十分位占比 × V60；超過的股數順延到次日）。
  e1：2020-10-26 起，買單（含整股部分）若總值 ≤ 預測日零股成交值（zpart 或 1.0）則全部改在 9:10 零股成交。
  e2：新進股整股 ≥ 2 張時，ceil(L/2) 張在開盤（衝擊按該半計），其餘在 09:30 前以 開盤×exp(−prem_hat(跳空)) 成交（衝擊按該半計、無參與上限）。
  odd_stress：零股買成本 ×2、賣成本 <0 → 0。
"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RAW = HERE / 'data', HERE / 'raw'
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
FEE_FULL = 0.001425
FEES = {'A': dict(rate=FEE_FULL * 0.28, min_lot=20.0, min_odd=20.0),
        'B': dict(rate=FEE_FULL * 0.6, min_lot=20.0, min_odd=20.0),
        'C': dict(rate=FEE_FULL * 0.28, min_lot=20.0, min_odd=1.0)}
TAX = 0.003
ENG_FEE = FEE_FULL * 0.6
ODD_INTRADAY = pd.Timestamp('2020-10-26')
FEEMIX_CUT = pd.Timestamp('2015-01-01')
A_, B_ = '2006-04-01', '2026-10-12'
PER = {'ALL': ('2006-04-01', '2026-10-12'), 'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'),
       'HOLD': ('2024-01-01', '2026-10-12'), 'FROM2008': ('2008-01-01', '2026-10-12'), 'Y19_26': ('2019-01-01', '2026-10-12'),
       'PRE_ODD': ('2006-04-01', '2020-10-25'), 'POST_ODD': ('2020-10-26', '2026-10-12'),
       'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12')}
_CPI = json.loads((RAW / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100']
CPI = {int(k): v for k, v in _CPI.items()}
def cpi_scale(ts): return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]

def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

class Data:
    def __init__(self, load_j=False):
        O = pd.read_parquet(DATA / 'px_open.parquet')
        self.IDX = O.index; self.CODES = list(O.columns); self.CI = {c: j for j, c in enumerate(self.CODES)}
        self.O = O.to_numpy(); self.C = pd.read_parquet(DATA / 'px_close.parquet').to_numpy()
        self.G = pd.read_parquet(DATA / 'px_grow.parquet').to_numpy()
        self.TR = pd.read_parquet(DATA / 'px_traded.parquet').to_numpy().astype(bool)
        self.PC = pd.read_parquet(DATA / 'prevclose.parquet').to_numpy()
        self.CF = pd.DataFrame(self.C, index=self.IDX).ffill().to_numpy()
        F = pd.read_parquet(DATA / 'feat_daily.parquet')
        def w(col): return F.pivot(index='date', columns='code', values=col).reindex(index=self.IDX, columns=self.CODES).to_numpy()
        self.V60, self.SIG, self.VRANK = w('V60'), w('sig60'), w('vrank')
        xr = json.loads((DATA / 'xrules.json').read_text(encoding='utf-8'))
        self.XR = {self.CI[c]: (v[0], int(np.searchsorted(self.IDX, pd.Timestamp(v[1]))),
                                int(np.searchsorted(self.IDX, pd.Timestamp(v[3]))) if v[3] else 10 ** 9) for c, v in xr.items()}
        B = pd.read_parquet(DATA / 'bench.parquet'); self.BENCH = B
        self.E50 = B['E50']
        P4 = pd.read_csv(DATA / 'plan_v4.csv', dtype={'picks': str})
        self.sig_days = [pd.Timestamp(x) for x in P4.signal] + [pd.Timestamp(B_)]
        self.t0 = int(self.IDX.get_loc(pd.Timestamp(P4.signal.iloc[0]))); self.t1 = int(self.IDX.get_loc(pd.Timestamp(B_)))
        self.plans = {}
        for name, f in (('AB', 'plans_ab.csv'), ('B10', 'plans_b10.csv')):
            P = pd.read_csv(DATA / f, dtype={'picks': str})
            for s, g in P.groupby('seed'):
                self.plans[(name, int(s))] = self._mk(g)
        self.plans[('V4', -1)] = self._mk(P4)
        self.jmeta = {}
        if load_j:
            PJ = pd.read_csv(DATA / 'plans_j.csv', dtype={'picks': str})
            meta = json.load(open(DATA / 'jscan_meta.json'))['meta']
            for (st, j, s), g in PJ.groupby(['strat', 'j', 'seed']):
                self.plans[(f'{st}j{j}', int(s))] = self._mk(g)
                end = pd.Timestamp(meta[f'{st}|j{j}']['end'])
                self.jmeta[f'{st}j{j}'] = dict(t0=int(self.IDX.get_loc(pd.Timestamp(g.signal.iloc[0]))), t1=int(self.IDX.get_loc(end)),
                                               sig_days=[pd.Timestamp(x) for x in g.signal] + [end])
        with np.errstate(invalid='ignore', divide='ignore'):
            adj = np.where(self.TR, self.G * self.PC / self.C, 1.0)
        self.ADJ = np.where(np.isfinite(adj), adj, 1.0)
        self.GAP = np.where(self.TR & np.isfinite(self.O), self.O / self.PC - 1, np.nan)
        # 漲跌停價（o5 成本審查 review_exec_o5.limits 同法）
        ref = np.where(self.TR & np.isfinite(self.G) & (self.G > 0), self.C / self.G, self.PC)
        lim = np.where((self.IDX >= pd.Timestamp('2015-06-01'))[:, None], 0.10, 0.07)
        up_raw = ref * (1 + lim); dn_raw = ref * (1 - lim)
        vt = np.vectorize(tick)
        with np.errstate(invalid='ignore'):
            tu = vt(np.nan_to_num(up_raw, nan=1.0)); td = vt(np.nan_to_num(dn_raw, nan=1.0))
            self.LU = np.floor(up_raw / tu + 1e-6) * tu; self.LD = np.ceil(dn_raw / td - 1e-6) * td

    def _mk(self, g):
        return [(int(e), [self.CI[c] for c in pk.split(',')], sg) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]

    def bounds(self, name):
        if name in self.jmeta: m = self.jmeta[name]; return m['t0'], m['t1'], m['sig_days']
        return self.t0, self.t1, self.sig_days

class Model:
    def __init__(self, spec, impact='M3V', q='point', params=None):
        self.s = spec; self.kind = impact
        if impact == 'M3V':
            p = params or spec['impact'][q]; self.delta, self.b = float(p['delta']), float(p['b'])
        elif impact == 'M2': self.Y = float((params or spec['sens']['M2'])['Y'])
        elif impact == 'M1': self.kappa = float((params or spec['sens']['M1'])['kappa'])
        self.base = float(spec['impact']['base'])
        oa = spec['oa']; self.oa_coef = np.array(oa['coef']); self.oa_smear = float(oa['smear']); self.cap = float(oa['cap'])
        o5 = json.loads((DATA / 'model_spec.json').read_text(encoding='utf-8'))
        self.o5 = o5; self.o5_coef = np.array(o5['oa']['coef']); self.o5_vars = o5['oa']['vars']; self.o5_kappa = o5['oa']['kappa']
        self.dr_edges = np.array(spec['drift']['edges']); self.dr_vals = np.array(spec['drift']['vals'])
        self.odd_pe = np.array(spec['odd']['price_edges']); self.odd_re = np.array(spec['odd']['rank_edges'])
        self.odd_buy = np.array(spec['odd']['buy']); self.odd_sell = np.array(spec['odd']['sell'])
        self.z_share = {int(k): float(v) for k, v in spec['zcap']['share_by_dec'].items()}; self.z_def = float(spec['zcap']['share_default'])
        self.e2_edges = np.array(spec['e2']['edges']); self.e2_vals = np.array(spec['e2']['prem'])

    def oa_hat(self, rank, price, sig, cx):
        r = rank if np.isfinite(rank) else 300.0
        x = np.array([1.0, math.log(max(r, 1.0)), math.log(price), math.log(max(sig, 1e-4)), float(cx == 'pick'), float(cx == 'drop')])
        return self.oa_smear * math.exp(float(self.oa_coef @ x))

    def oa_o5(self, rank, price, sig, gap, cx):
        cols = {'pick': float(cx == 'pick'), 'drop': float(cx == 'drop'), 'const': 1.0, 'lnrank': np.log(np.clip(rank, 1, None)), 'lnprice': np.log(price),
                'lnsig': np.log(np.clip(sig, 1e-4, None)), 'absgap': np.abs(gap)}
        return self.o5_kappa * float(np.exp(self.o5_coef @ np.array([cols[v] for v in self.o5_vars], dtype=float)))

    def drift(self, gap):
        return float(self.dr_vals[int(np.searchsorted(self.dr_edges, gap, side='right'))])

    def odd_cost(self, price, rank, side, odd0, stress):
        i = int(np.searchsorted(self.odd_pe, price, side='right')); j = int(np.searchsorted(self.odd_re, rank if np.isfinite(rank) else 1e9, side='right'))
        c = float((self.odd_buy if side > 0 else self.odd_sell)[i, j])
        if stress: c = 2 * c if side > 0 else max(c, 0.0)
        if odd0 and side < 0: c = max(c, 0.0)
        return c

    def e2_prem(self, gap):
        return float(self.e2_vals[int(np.searchsorted(self.e2_edges, gap, side='right'))])

    def zcap_value(self, rank, v60, part):
        d = int(math.ceil(rank / 30)) if np.isfinite(rank) else 10
        d = min(max(d, 1), 10)
        return part * self.z_share.get(d, self.z_def) * v60

DEF_OPTS = dict(lim=False, odd0=False, cash=0.0, feemix=False, zpart=None, e1=False, e2=False, odd_stress=False, base=None)
FRICTIONS = dict(lim=True, odd0=True, cash=0.02, feemix=True)

class Sim:
    """單一路徑（plan）在單一級距的模擬。"""
    def __init__(self, D, M, opts):
        self.D, self.M = D, M; self.o = dict(DEF_OPTS); self.o.update(opts or {})
        self.base = self.o['base'] if self.o['base'] is not None else M.base

    def fee(self, value, odd, t):
        if value <= 0: return 0.0
        fr = FEES['B'] if (self.o['feemix'] and self.D.IDX[t] < FEEMIX_CUT) else self.fr
        return max(fr['min_odd'] if odd else fr['min_lot'], fr['rate'] * value)

    def oa(self, t, j, cx):
        D, M = self.D, self.M
        if M.kind == 'O5':
            g = D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0
            r = M.oa_o5(D.VRANK[t - 1, j], D.PC[t, j], D.SIG[t - 1, j], g, cx); v = D.V60[t - 1, j]
            return r * v if np.isfinite(v) and np.isfinite(r) else np.nan
        v = D.V60[t - 1, j]; s = D.SIG[t - 1, j]
        if not (np.isfinite(v) and np.isfinite(s)): return np.nan
        return M.oa_hat(D.VRANK[t - 1, j], D.PC[t, j], s, cx) * v

    def imp(self, t, j, Q, cx):
        D, M = self.D, self.M
        sig, v = D.SIG[t - 1, j], D.V60[t - 1, j]
        if not (np.isfinite(sig) and Q > 0): return self.base
        if M.kind == 'M3V':
            return self.base + (M.b * sig * (Q / v) ** M.delta if np.isfinite(v) and v > 0 else 0.0)
        if M.kind == 'M2':
            return self.base + (M.Y * sig * math.sqrt(Q / v) if np.isfinite(v) and v > 0 else 0.0)
        oa = self.oa(t, j, cx)
        if not (np.isfinite(oa) and oa > 0): return self.base
        if M.kind == 'M1': return self.base + M.kappa * sig * math.sqrt(Q / oa)
        return self.base + 0.5 * sig * math.sqrt(Q / oa)        # O5

    def blocked(self, t, j, side):
        if not self.o['lim']: return False
        o = self.D.O[t, j]
        return bool(o >= self.D.LU[t, j] - 1e-6) if side > 0 else bool(o <= self.D.LD[t, j] + 1e-6)

    def odd_px(self, t, j):
        D = self.D
        if D.IDX[t] < ODD_INTRADAY: return D.C[t, j]
        g = D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0
        return D.O[t, j] * (1.0 + self.M.drift(g))

    def zcap_sh(self, t, j, px, part=None):
        part = part if part is not None else self.o['zpart']
        if part is None or self.D.IDX[t] < ODD_INTRADAY: return None
        v = self.D.V60[t - 1, j]
        if not np.isfinite(v): return None
        return int(math.floor(self.M.zcap_value(self.D.VRANK[t - 1, j], v, part) / px))

    def book(self, t, **kw):
        eq = self.acc['_eq']
        for k, v in kw.items():
            if k == 'traded': self.acc['_tv'][t] = self.acc['_tv'].get(t, 0.0) + v / eq; continue
            self.acc[k] = self.acc.get(k, 0.0) + v; self.acc['pct_' + k] = self.acc.get('pct_' + k, 0.0) + v / eq

    # ---- 成交 ----
    def lot_exec(self, t, j, lots, side, cx, at_open=True):
        D = self.D; p_ref = D.PC[t, j]
        n = lots; oa = np.nan
        if at_open:
            oa = self.oa(t, j, cx)
            if np.isfinite(oa):
                mx = max(1, int(math.floor(self.M.cap * oa / (1000.0 * p_ref))))
                if n > mx: n = mx; self.acc['n_cap_hit'] += 1
        Q = n * 1000.0 * p_ref
        im = self.imp(t, j, Q, cx)
        if self.rows is not None:
            self.rows.append((t, j, side, n, 1 if at_open else 0, (Q / oa) if np.isfinite(oa) and oa > 0 else np.nan, Q * (im - self.base), Q))
        ref = D.O[t, j] if at_open else D.O[t, j] * math.exp(-self.M.e2_prem(D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0))
        px = ref * (1 + side * im)
        val = n * 1000.0 * px
        f = self.fee(val, False, t); tax = TAX * val if side < 0 else 0.0
        self.book(t, fee=f, tax=tax, impact=n * 1000.0 * ref * (im - self.base), base_slip=n * 1000.0 * ref * self.base,
                  e2_drift=(side * n * 1000.0 * (ref - D.O[t, j]) if not at_open else 0.0), traded=val)
        self.acc['n_orders'] += 1; self.acc['n_lot_orders'] += 1
        return n, ((-val - f) if side > 0 else (val - f - tax))

    def odd_exec(self, t, j, sh, side):
        D = self.D
        base = self.odd_px(t, j)
        c = self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], side, self.o['odd0'], self.o['odd_stress'])
        px = base * (1 + side * c); val = sh * px
        f = self.fee(val, True, t); tax = TAX * val if side < 0 else 0.0
        self.book(t, fee=f, tax=tax, odd_spread=sh * base * c, odd_timing=side * sh * (base - D.O[t, j]), traded=val)
        self.acc['n_orders'] += 1; self.acc['n_odd_orders'] += 1
        return (-val - f) if side > 0 else (val - f - tax)

    def cost(self, t, j, delta, cx, newentry):
        """delta 股的現金需求（買＞0）或淨收入（賣＜0 → 負）；用當日實際價格（已知開盤的理想化，同 o5）；不含參與上限與量上限。"""
        key = (j, delta)
        if key in self.memo: return self.memo[key]
        D = self.D
        if delta == 0: return 0.0
        side = 1 if delta > 0 else -1; a = abs(delta)
        tot = 0.0
        if side > 0 and self.e1_ok(t, j, a):
            px = self.odd_px(t, j) * (1 + self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], 1, self.o['odd0'], self.o['odd_stress']))
            val = a * px; tot = val + self.fee(val, True, t)
            self.memo[key] = tot; return tot
        lots, odd = a // 1000, a % 1000
        if lots:
            for L, op in self.split(lots, side, newentry):
                Q = L * 1000.0 * D.PC[t, j]; im = self.imp(t, j, Q, cx)
                ref = D.O[t, j] if op else D.O[t, j] * math.exp(-self.M.e2_prem(D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0))
                val = L * 1000.0 * ref * (1 + side * im)
                tot += side * val + self.fee(val, False, t) + (TAX * val if side < 0 else 0.0)
        if odd:
            px = self.odd_px(t, j) * (1 + side * self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], side, self.o['odd0'], self.o['odd_stress']))
            val = odd * px
            tot += side * val + self.fee(val, True, t) + (TAX * val if side < 0 else 0.0)
        self.memo[key] = tot
        return tot

    def split(self, lots, side, newentry):
        if self.o['e2'] and side > 0 and newentry and lots >= 2:
            h = int(math.ceil(lots / 2)); return [(h, True), (lots - h, False)]
        return [(lots, True)]

    def e1_ok(self, t, j, a):
        if not self.o['e1'] or self.D.IDX[t] < ODD_INTRADAY: return False
        v = self.D.V60[t - 1, j]
        if not np.isfinite(v): return False
        cap = self.M.zcap_value(self.D.VRANK[t - 1, j], v, self.o['zpart'] if self.o['zpart'] is not None else 1.0)
        return a * self.D.PC[t, j] <= cap

    def do_odd(self, t, j, sh, side, pend):
        """零股（含量上限 Z）；回傳現金變動與實際成交股數。"""
        px = self.odd_px(t, j)
        cap = self.zcap_sh(t, j, px)
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
        def need(T): return sum(self.cost(t, j, d, 'pick' if j in pks else 'drop', j not in cur) for j, d in deltas(T).items())
        hi = Vtr / k * (1 - self.o['cash']); lo = 0.0
        if need(hi) <= cash: lo = hi
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if need(mid) <= cash: lo = mid
                else: hi = mid
        dl = deltas(lo); pend = {}
        for j, d in sorted(dl.items(), key=lambda x: x[1]):
            side = 1 if d > 0 else -1; a = abs(d); cx = 'pick' if j in pks else 'drop'; newentry = j not in cur
            if self.blocked(t, j, side):
                pend[j] = pend.get(j, 0) + d; self.acc['n_lim_block'] += 1; continue
            if side > 0 and self.cost(t, j, d, cx, newentry) > cash + 1e-9:
                pend[j] = pend.get(j, 0) + d; self.acc['deferred_days'] += 1; continue
            if side > 0 and self.e1_ok(t, j, a):
                dc, n = self.do_odd(t, j, a, 1, pend); cash += dc; S[j] += n; self.acc['n_e1'] += 1; continue
            lots, odd = a // 1000, a % 1000
            if odd:
                dc, n = self.do_odd(t, j, odd, side, pend); cash += dc; S[j] += side * n
            for L, op in (self.split(lots, side, newentry) if lots else []):
                n, dc = self.lot_exec(t, j, L, side, cx, at_open=op)
                cash += dc; S[j] += side * n * 1000
                if n < L: pend[j] = pend.get(j, 0) + side * (L - n) * 1000
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
                    v_ = odd * self.odd_px(t, j) * (1 + self.M.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], 1, self.o['odd0'], self.o['odd_stress']))
                    if v_ + self.fee(v_, True, t) > cash - lots * unit: odd = 0
            got = 0
            if odd:
                tmp = {}
                dc, n = self.do_odd(t, j, odd, side, tmp); cash += dc; S[j] += side * n; got += n
            if lots:
                n, dc = self.lot_exec(t, j, lots, side, None, at_open=True)
                cash += dc; S[j] += side * n * 1000; got += n * 1000
            rem = abs(d) - got if side > 0 else a - got
            if rem > 0: newp[j] = side * rem
        return S, cash, newp

    def run(self, plan, t0, t1, tier, version='GROW', fee='A', collect=False):
        D = self.D; self.fr = FEES[fee]
        reb = {p[0]: p for p in plan}
        N = len(D.CODES); O, CF, TR, ADJ = D.O, D.CF, D.TR, D.ADJ
        S = np.zeros(N); E0 = tier * cpi_scale(D.IDX[t0]); cash = E0
        nav = np.empty(t1 - t0 + 1); nav[0] = 1.0; prev_eq = E0; unit = 1.0
        pend = {}
        self.acc = dict(tax=0.0, fee=0.0, impact=0.0, odd_spread=0.0, odd_timing=0.0, base_slip=0.0, e2_drift=0.0, n_orders=0, n_lot_orders=0, n_odd_orders=0,
                        n_cap_hit=0, n_zcap_hit=0, n_lim_block=0, n_e1=0, deferred_days=0, unfilled_at_reb=0.0, flow=0.0, _eq=E0, _tv={}, cash_frac_sum=0.0)
        self.rows = [] if collect else None
        for t in range(t0 + 1, t1 + 1):
            S = S * ADJ[t]
            flow = 0.0; eq_pre = None
            if t in reb:
                e, pk, sg = reb[t]
                tr = TR[t] & np.isfinite(O[t])
                self.acc['unfilled_at_reb'] += sum(abs(v) * CF[t, j] for j, v in pend.items())
                pend = {}
                held = S > 0; frz = held & ~tr
                for j in np.where(frz)[0]:
                    if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                        cash += S[j] * CF[t - 1, j] * D.XR[j][0]; S[j] = 0.0; frz[j] = False
                if version == 'FIX':
                    Vfz0 = float((S[frz] * CF[t - 1, frz]).sum()); Vtr0 = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
                    eq_pre = Vtr0 + Vfz0 + cash
                    target = tier * cpi_scale(D.IDX[t]); sc = target / eq_pre
                    S = S * sc; cash = cash * sc; flow = target - eq_pre; self.acc['flow'] += flow
                fra = np.where(tr & (S > 0), S - np.floor(S + 1e-9), 0.0); fra = np.clip(fra, 0, None)
                cash += float((fra * np.where(tr, O[t], 0.0)).sum()); S = S - fra
                Vfz = float((S[frz] * CF[t - 1, frz]).sum()); Vtr = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
                self.acc['_eq'] = Vtr + Vfz + cash
                S, cash, pend = self.rebalance(t, pk, S, tr, frz, cash)
            elif pend:
                self.acc['_eq'] = prev_eq
                S, cash, pend = self.work_pending(t, S, cash, pend)
            nz = S != 0
            eq = cash + float((S[nz] * CF[t, nz]).sum())
            assert cash > -1e-6 * max(1.0, eq), (t, cash)
            if eq_pre is None: unit *= eq / prev_eq
            else: unit *= (eq_pre / prev_eq) * (eq / (eq_pre + flow))
            self.acc['cash_frac_sum'] += cash / eq if eq > 0 else 0.0
            nav[t - t0] = unit; prev_eq = eq
        self.acc['end_equity'] = prev_eq; self.acc['start_equity'] = E0
        return pd.Series(nav, index=D.IDX[t0:t1 + 1]), self.acc, self.rows

# ---------------- 統計 ----------------
def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    i0 = nav.index.get_loc(s.index[0])
    base = nav.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
    start = nav.index[i0 - 1] if i0 > 0 else s.index[0]
    yrs = (s.index[-1] - start).days / 365.25
    eq = s / base
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def win_ret(nav, sd):
    Dd = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(Dd)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def excess(nav, bench, sd, a=A_, b=B_):
    x = win_ret(nav, sd) - win_ret(bench.reindex(nav.index), sd)
    return x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))].dropna()

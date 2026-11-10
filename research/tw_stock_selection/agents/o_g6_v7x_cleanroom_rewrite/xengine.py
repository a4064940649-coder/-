# V7X 可執行引擎（clean-room 重寫；只依 frozen_spec／interp_register／model_spec7／o5 第六節文字）
import numpy as np, pandas as pd, json, math
import xdata
AG = xdata.AG
M7 = json.load(open(AG + 'o_v7_listing_exec_engine/data/model_spec7.json'))
SHARE = np.array([M7['oa_share_by_dec'][str(k)] for k in range(1, 11)])
ZSH = np.array([M7['zcap']['share_by_dec'][str(k)] for k in range(1, 11)]); ZDEF = M7['zcap']['share_default']
DR_E = np.array(M7['drift']['edges']); DR_V = np.array(M7['drift']['vals'])
OD_PE = np.array(M7['odd']['price_edges']); OD_RE = np.array(M7['odd']['rank_edges'])
OD_B = np.array(M7['odd']['buy']); OD_S = np.array(M7['odd']['sell'])
ODD_START = pd.Timestamp('2020-10-26'); FEE_SWITCH = pd.Timestamp('2015-01-01')

BASE_CFG = dict(tier=1_000_000, cost='normal', merger='cons_main', whole_pre2020=False, odd_pre_ratio=0.5,
                lim=True, tax=True, fee=True, imp=True, caps=True, lots=True, cash98=True, disp=True, timing='open',
                record=False)

class World:
    def __init__(self):
        X = xdata.load(); self.X = X
        self.dates = X['dates']; self.codes = X['codes']; self.ci = {c: j for j, c in enumerate(self.codes)}
        T = len(self.dates)
        self.C, self.O, self.V, self.TR, self.G, self.CF, self.PC, self.ADJ = (X[k] for k in ['C', 'O', 'V', 'TR', 'G', 'CF', 'PC', 'ADJ'])
        self.UP, self.DN, self.E = X['UP'], X['DN'], X['E']
        sh = lambda a: np.vstack([np.full((1, a.shape[1]), np.nan), a[:-1]])
        self.V60p = sh(X['V60']); self.SIGp = sh(X['SIG']); self.VRp = sh(X['VR'])
        vr = self.VRp
        self.DEC = np.where(np.isnan(vr) | (vr > 300), 10, np.ceil(np.nan_to_num(vr, nan=3000) / 30)).astype(int)
        self.DEC = np.clip(self.DEC, 1, 10)
        self.in300 = ~(np.isnan(vr) | (vr > 300))
        self.odd_day = np.array(self.dates >= ODD_START)
        self.fee_pre = np.array(self.dates < FEE_SWITCH)
        # 處置期遮罩（V4 凍結官方名單）
        self.DISP = np.zeros((T, len(self.codes)), dtype=bool)
        du = xdata.disposal_v4()
        dv = self.dates.values
        for r in du.itertuples():
            j = self.ci.get(r.code)
            if j is None: continue
            a = np.searchsorted(dv, np.datetime64(r.start), 'left'); b = np.searchsorted(dv, np.datetime64(r.end), 'right')
            self.DISP[a:b, j] = True
        self._xrules()

    def didx(self, d):  # 第一個 ≥ d 的面板日
        return int(np.searchsorted(self.dates.values, np.datetime64(pd.Timestamp(d)), 'left'))

    def _xrules(self):
        T = len(self.dates)
        lastTR = {}
        for j in range(len(self.codes)):
            w = np.where(self.TR[:, j])[0]
            lastTR[j] = int(w[-1]) if len(w) else -1
        self.lastTR = lastTR
        # orig：merger_consideration＋delist_official（未列者 unknown）
        m = pd.read_csv(open(AG + 'o_infra_v3_rebuild/merger_consideration.csv'), dtype={'code': str, 'successor': str})
        dl = pd.read_csv(open(AG + 'o_infra_v3_rebuild/delist_official.csv'), dtype={'code': str})
        self.XO = {}
        def valid_until(j, rel):
            w = np.where(self.TR[rel:, j])[0]
            return rel + int(w[0]) if len(w) else T + 1
        for r in m.itertuples():
            j = self.ci.get(r.code)
            if j is None: continue
            dd = pd.Timestamp(r.delist_date) if isinstance(r.delist_date, str) else pd.Timestamp(r.last_trade) + pd.Timedelta(days=14)
            rel = self.didx(dd); lt = self.didx(pd.Timestamp(r.last_trade))
            if r.type == 'cash':
                ratio = (r.cash / r.last_close) if (isinstance(r.cash, float) and not math.isnan(r.cash)) else 1.0
                p0 = min(rel, T - 1); pay = p0 + 20
                self.XO[j] = dict(kind='mcash', rel=rel, pay=pay, ratio=ratio, vu=valid_until(j, rel))
            elif r.type == 'failure':
                self.XO[j] = dict(kind='zero', rel=lt + 1, vu=valid_until(j, lt + 1))
            else:
                self.XO[j] = dict(kind='one', rel=rel, vu=valid_until(j, rel))
        for r in dl.itertuples():
            j = self.ci.get(r.code)
            if j is None or j in self.XO: continue
            rel = self.didx(pd.Timestamp(r.date))
            if rel >= T: continue
            self.XO[j] = dict(kind='one', rel=rel, vu=valid_until(j, rel))
        # cons_main：events_v2（凍結後修正 F4）
        e = pd.read_csv(open(AG + 'o_engine_delist_consideration/events_v2.csv'), dtype={'code': str, 'successor': str})
        self.XC = {}
        num = lambda v: (float(v) if isinstance(v, (float, int)) and not math.isnan(v) else None)
        for r in e.itertuples():
            j = self.ci.get(r.code)
            if j is None or not isinstance(r.last_trade, str) or not r.last_trade: continue
            ltU = pd.Timestamp(r.last_trade)
            term = pd.Timestamp(r.terminate_date) if isinstance(r.terminate_date, str) and r.terminate_date else ltU + pd.Timedelta(days=14)
            cut = max(term, ltU)
            w = np.where(self.TR[:, j] & (self.dates <= cut))[0]
            if not len(w): continue
            lt_panel = int(w[-1]); trig = lt_panel + 1
            if trig >= T: continue
            ti = int(np.searchsorted(self.dates.values, np.datetime64(term)))
            pay = min(ti + 20, T - 1)
            lc = float(self.C[lt_panel, j])
            d = dict(type=r.type, trig=trig, pay=pay, last_close=lc, cash=num(r.cash), ratio=num(r.ratio), pref_ratio=num(r.pref_ratio), pref_value=num(r.pref_value), succ=None, s=None)
            if self.dates[lt_panel] < ltU - pd.Timedelta(days=7):
                d['type'] = 'price_gap'
            else:
                sj = self.ci.get(r.successor) if isinstance(r.successor, str) and r.successor else None
                sf = None
                if sj is not None and isinstance(r.succ_first_trade, str) and r.succ_first_trade:
                    s0 = int(np.searchsorted(self.dates.values, np.datetime64(pd.Timestamp(r.succ_first_trade))))
                    ww = np.where(self.TR[s0:, sj] & np.isfinite(self.O[s0:, sj]))[0]
                    sf = s0 + int(ww[0]) if len(ww) else None
                d['succ'] = sj; d['s'] = sf
                if d['type'] == 'swap_ur' and sj is not None and sf is not None:
                    if np.isfinite(self.C[lt_panel, sj]): d['ratio'] = lc / float(self.C[lt_panel, sj])
                    else: d['s'] = None
                if d['type'] in ('swap', 'conversion', 'mixed', 'swap_ur') and (sj is None or d['s'] is None):
                    d['type'] = 'acq_ur'
            self.XC.setdefault(j, []).append(d)

def tick(p):
    return 0.01 if p < 10 else 0.05 if p < 50 else 0.1 if p < 100 else 0.5 if p < 500 else 1.0 if p < 1000 else 5.0

class Sim:
    def __init__(self, W, plan, end_idx, cfg):
        self.W = W; self.cfg = dict(BASE_CFG, **cfg); c = self.cfg
        self.plan = plan; self.end = end_idx
        self.stress = c['cost'] == 'stress'
        self.trades = [] if c['record'] else None

    # ---------- 成本 ----------
    def fee(self, val, t):
        if not self.cfg['fee'] or val <= 0: return 0.0
        rate = 0.001425 * (0.6 if (self.stress or self.W.fee_pre[t]) else 0.28)
        return max(20.0, rate * val)

    def imp(self, t, j, lots):
        W = self.W; o = self.px_open(t, j); c = self.cfg
        if not c['imp']: x = 0.0
        else:
            x = 0.001
            oa = SHARE[W.DEC[t, j] - 1] * W.V[t, j]; sg = W.SIGp[t, j]
            if lots > 0 and np.isfinite(oa) and oa > 0 and np.isfinite(sg):
                x += 0.5 * sg * math.sqrt(lots * 1000 * W.PC[t, j] / oa)
            x = max(x, tick(o) / o)
        if self.stress: x += 0.004
        return x

    def odd_px(self, t, j):
        W = self.W
        if not self.cfg['lots']: return self.px_open(t, j)
        if W.odd_day[t]:
            pc = W.PC[t, j]; o = W.O[t, j]
            gap = o / pc - 1 if (np.isfinite(pc) and pc > 0) else 0.0
            return o * (1 + DR_V[np.searchsorted(DR_E, gap, 'right')])
        return W.C[t, j]

    def odd_c(self, t, j, side):
        W = self.W
        if not self.cfg['lots']:
            return self.imp(t, j, 0) if self.cfg['imp'] else (0.004 if self.stress else 0.0)
        pb = int(np.searchsorted(OD_PE, W.PC[t, j], 'right'))
        vr = W.VRp[t, j]; rb = int(np.searchsorted(OD_RE, vr if np.isfinite(vr) else 1e9, 'right'))
        if side > 0:
            c = OD_B[pb, rb]
            if self.stress: c = 2 * c + 0.004
        else:
            c = max(OD_S[pb, rb], 0.0)
            if self.stress: c += 0.004
        return c

    def odd_cap(self, t, j, px):
        W = self.W
        if not self.cfg['caps']: return 10**12
        v = W.V60p[t, j]
        if not np.isfinite(v) or not (px > 0): return 0
        s = ZSH[W.DEC[t, j] - 1] if W.in300[t, j] else ZDEF
        ratio = 1.0 if W.odd_day[t] else self.cfg['odd_pre_ratio']
        return int(math.floor(ratio * s * v / px))

    def lot_cap(self, t, j):
        W = self.W
        if not self.cfg['caps']: return 10**12
        oap = SHARE[W.DEC[t, j] - 1] * W.V60p[t, j]
        if not np.isfinite(oap): return 1
        return max(1, int(math.floor(0.20 * oap / (1000 * W.PC[t, j]))))

    def is_disp(self, t, j):
        return self.cfg['disp'] and self.W.DISP[t, j]

    def lim_block(self, t, j, side):
        if not self.cfg['lim']: return False
        W = self.W; o = self.px_open(t, j)
        if side > 0: return o >= W.UP[t, j] - 1e-9
        return o <= W.DN[t, j] + 1e-9

    def split(self, t, j, n):
        n = int(round(abs(n)))
        if not self.cfg['lots']: return 0, n
        if self.cfg['whole_pre2020'] and not self.W.odd_day[t]:
            return n // 1000, 0
        return n // 1000, n % 1000

    def leg_cost(self, t, j, side, lots, odd, est=False):
        """回傳 (現金變動 = -cost, 成交明細)；side +1 買 −1 賣；lots 張、odd 股"""
        W = self.W; cash = 0.0; c = self.cfg
        if side < 0 and self.is_disp(t, j):
            px = W.C[t, j] - tick(W.C[t, j]); val = (lots * 1000 + odd) * px
            if not c['lots']: val = (lots * 1000 + odd) * px
            f = self.fee(lots * 1000 * px, t) + self.fee(odd * px, t)
            tx = 0.003 * val if c['tax'] else 0.0
            return val - f - tx, ('disp', px)
        if self.cfg['lots']:
            if odd > 0:
                b = self.odd_px(t, j); cc = self.odd_c(t, j, side); px = b * (1 + side * cc); val = odd * px
                cash += -side * val - self.fee(val, t) - (0.003 * val if (side < 0 and c['tax']) else 0.0)
            if lots > 0:
                x = self.imp(t, j, lots); px = W.O[t, j] * (1 + side * x); val = lots * 1000 * px
                cash += -side * val - self.fee(val, t) - (0.003 * val if (side < 0 and c['tax']) else 0.0)
        else:
            n = lots * 1000 + odd
            if n > 0:
                x = self.imp(t, j, n / 1000.0) if c['imp'] else (0.004 if self.stress else 0.0)
                px = self.px_open(t, j) * (1 + side * x); val = n * px
                cash += -side * val - self.fee(val, t) - (0.003 * val if (side < 0 and c['tax']) else 0.0)
        return cash, None

    # ---------- 主迴圈 ----------
    def run(self):
        W = self.W; c = self.cfg; plan = self.plan
        T = W.C.shape[0]
        exec_map = {}
        for (s, e, picks) in plan:
            if c['timing'] == 'close': exec_map[s] = picks
            else: exec_map[e] = picks
        t0 = plan[0][0]
        self.S = {}; self.cash = c['tier'] * xdata.cpi()[2006] / xdata.cpi()[2026]
        self.cash_next = 0.0; self.p50 = []  # [value, pay_idx]
        self.claims = []  # dict(kind, value, pay/ s ...)
        self.pend = {}
        self.done_x = set()
        self.slot = {}  # 右尾：現行槽 (j)->dict
        nav = np.full(T, np.nan); eq_prev = self.cash; nav[t0] = 1.0; unit = 1.0
        self.turn = np.zeros(T)
        self.win = -1; self.pnl = {}; self.eq_open = {}; self.cf = {}; self.vprev = {}
        rec_on = self.trades is not None
        ekeys = sorted(exec_map)
        for t in range(t0 + 1, self.end + 1):
            # 0050 待撥款計值
            r50 = W.E[t] / W.E[t - 1]
            for p in self.p50: p[0] *= r50
            for j in list(self.S):
                self.S[j] *= W.ADJ[t, j]
            if self.cash_next:
                self.cash += self.cash_next; self.cash_next = 0.0
            # 現金請求權撥款
            keep = []
            for cl in self.claims:
                if cl['kind'] == 'fixed_sell' and t >= cl['rel'] and t in exec_map:
                    rate = 0.001425 * (0.6 if (self.stress or W.fee_pre[t]) else 0.28)
                    self.cash += cl['value'] * (1 - (0.003 if c['tax'] else 0.0) - (rate if c['fee'] else 0.0))
                elif cl['kind'] == 'cash' and t >= cl['pay']:
                    self.cash += cl['value']
                elif cl['kind'] == 'stock' and cl['s'] is not None and t >= cl['s']:
                    sj = cl['succ']; self.S[sj] = self.S.get(sj, 0.0) + cl['shares']
                else: keep.append(cl)
            self.claims = keep
            if c['merger'] == 'cons_main': self.cons_trigger(t)
            if t in exec_map:
                if rec_on:
                    self.win_prev = self.win; self.win = ekeys.index(t); self.newpk = set(exec_map[t])
                    self.eq_open[self.win] = self.equity(t - 1) if t > t0 else self.cash
                self.pend = {}
                self.frozen_x(t)
                for j in list(self.S):
                    if W.TR[t, j] and self.S[j] > 0:
                        fl = math.floor(self.S[j] + 1e-9)
                        fr = self.S[j] - fl
                        if fr > 0 and c['lots']:
                            self.cash += fr * self.px_open(t, j); self.S[j] = fl
                            if rec_on: self.cf[j] = self.cf.get(j, 0.0) + fr * self.px_open(t, j)
                self.rebalance(t, exec_map[t])
            elif self.pend:
                self.deferred(t)
            eq = self.equity(t)
            if rec_on and self.win >= 0:
                isre = t in exec_map
                for j in set(self.S) | set(self.vprev) | set(self.cf):
                    v = self.S.get(j, 0.0) * W.CF[t, j] if self.S.get(j, 0.0) else 0.0
                    p = v - self.vprev.get(j, 0.0) + self.cf.get(j, 0.0)
                    k = self.win if (not isre or j in self.newpk or self.win_prev < 0) else self.win_prev
                    self.pnl[(k, j)] = self.pnl.get((k, j), 0.0) + p
                    self.vprev[j] = v
                self.cf = {}
            unit *= eq / eq_prev if eq_prev > 0 else 1.0
            nav[t] = unit; eq_prev = eq
        return nav

    def px_open(self, t, j):
        return self.W.C[t, j] if self.cfg['timing'] == 'close' else self.W.O[t, j]

    def equity(self, t):
        W = self.W
        v = self.cash + self.cash_next + sum(p[0] for p in self.p50)
        for cl in self.claims: v += cl['value']
        for j, s in self.S.items():
            if s: v += s * W.CF[t, j]
        return v

    def cons_trigger(self, t):
        W = self.W
        for j in list(self.S):
            ds = [x for x in W.XC.get(j, []) if x['trig'] == t]
            if not ds or self.S[j] <= 0: continue
            d = ds[0]
            sh = self.S.pop(j); lc = d['last_close']; ty = d['type']
            if ty == 'cash':
                self.claims.append(dict(kind='cash', value=sh * (d['cash'] if d['cash'] else lc), pay=d['pay']))
            elif ty in ('conversion', 'swap', 'swap_ur') and d['succ'] is not None and d['ratio']:
                self.claims.append(dict(kind='stock', value=sh * lc, shares=sh * d['ratio'], succ=d['succ'], s=d['s']))
            elif ty == 'mixed' and d['succ'] is not None:
                if d['cash']: self.claims.append(dict(kind='cash', value=sh * d['cash'], pay=d['pay']))
                stock_v = lc - (d['cash'] or 0)
                if d['ratio']: self.claims.append(dict(kind='stock', value=sh * max(stock_v, 0), shares=sh * d['ratio'], succ=d['succ'], s=d['s']))
                if d['pref_ratio'] and d['pref_value']:
                    self.claims.append(dict(kind='fixed_sell', value=sh * d['pref_ratio'] * d['pref_value'], rel=d['s']))
            elif ty == 'price_gap':
                self.claims.append(dict(kind='fixed_sell', value=sh * lc, rel=t))
            elif ty == 'distress':
                pass
            else:  # acq_ur、unknown（main 上界）、其他
                self.claims.append(dict(kind='cash', value=sh * lc, pay=d['pay']))

    def frozen_x(self, t):
        W = self.W; c = self.cfg
        for j in list(self.S):
            if W.TR[t, j] or self.S[j] <= 0: continue
            if c['merger'] == 'cons_main' and j in W.XC: continue
            x = W.XO.get(j)
            if x is None or not (x['rel'] <= t < x['vu']): continue
            sh = self.S.pop(j); pv = sh * W.CF[t - 1, j]
            if x['kind'] == 'mcash':
                self.p50.append([pv * x['ratio'] * (W.E[t] / W.E[t - 1]), x['pay']])
            elif x['kind'] == 'one':
                self.cash += pv
            # zero：歸零
        # 0050 待撥款：撥款日之後第一個換股日以前一日收盤值併入
        keep = []
        r50 = W.E[t] / W.E[t - 1]
        for p in self.p50:
            if t > p[1] and p[1] >= 0 and not getattr(p, 'new', False):
                self.cash += p[0] / r50
            else: keep.append(p)
        self.p50 = keep

    def target(self, t, j, T):
        W = self.W; p = W.PC[t, j]
        if self.cfg['timing'] == 'close': p = W.C[t, j]
        if not (p > 0): return 0
        if not self.cfg['lots']:
            po = self.px_open(t, j)
            return T / po if po > 0 else 0
        L = math.floor(T / (1000 * p) + 1e-12)
        if self.cfg['whole_pre2020'] and not W.odd_day[t]: return 1000 * L
        q = math.floor((T - 1000 * L * p) / p + 1e-9)
        return 1000 * L + max(q, 0)

    def est_cost(self, t, j, delta):
        if delta == 0: return 0.0
        W = self.W
        if not (W.TR[t, j] and np.isfinite(self.px_open(t, j))): return 0.0  # 無價可估（順延）
        side = 1 if delta > 0 else -1
        if side < 0 and self.is_disp(t, j): return 0.0  # F2：處置賣出款不計入求解
        lots, odd = self.split(t, j, delta)
        if not self.cfg['lots']: lots, odd = 0, abs(delta)
        cash, _ = self.leg_cost_est(t, j, side, lots, odd, delta)
        return -cash

    def leg_cost_est(self, t, j, side, lots, odd, delta):
        if not self.cfg['lots']:
            W = self.W; n = abs(delta)
            x = self.imp(t, j, n / 1000.0) if self.cfg['imp'] else (0.004 if self.stress else 0.0)
            px = self.px_open(t, j) * (1 + side * x); val = n * px
            return -side * val - self.fee(val, t) - (0.003 * val if (side < 0 and self.cfg['tax']) else 0.0), None
        return self.leg_cost(t, j, side, lots, odd)

    def rebalance(self, t, picks):
        W = self.W; c = self.cfg
        k = len(picks)
        cur = {j: s for j, s in self.S.items() if s > 0 and W.TR[t, j]}
        hi = (sum(s * self.px_open(t, j) for j, s in cur.items()) + self.cash) / k * (0.98 if c['cash98'] else 1.0)
        names = list(dict.fromkeys(list(picks) + list(cur)))
        pk = set(picks)
        def deltas(T):
            out = {}
            for j in names:
                tg = self.target(t, j, T) if j in pk else 0
                out[j] = tg - cur.get(j, 0.0) if (j in cur or j in pk) else 0
            return out
        def need(T):
            return sum(self.est_cost(t, j, d) for j, d in deltas(T).items())
        if need(hi) <= self.cash: Tt = hi
        else:
            lo, h = 0.0, hi
            for _ in range(40):
                mid = (lo + h) / 2
                if need(mid) <= self.cash: lo = mid
                else: h = mid
            Tt = lo
        self.Tt = Tt
        ds = deltas(Tt)
        for j, d in sorted(ds.items(), key=lambda kv: kv[1]):
            if abs(d) < 1e-9: continue
            self.exec_order(t, j, d, first=True)

    def exec_order(self, t, j, d, first):
        """執行一筆（換股日或順延日）委託；未成交部分寫入 self.pend"""
        W = self.W; c = self.cfg; side = 1 if d > 0 else -1
        if c['timing'] == 'close':
            tradable = W.TR[t, j]
        else:
            tradable = W.TR[t, j] and np.isfinite(W.O[t, j])
        disp_sell = side < 0 and self.is_disp(t, j)
        if not tradable or (not disp_sell and self.lim_block(t, j, side)):
            self.pend[j] = self.pend.get(j, 0) + d; return
        if side < 0:
            n = min(abs(d), round(self.S.get(j, 0.0))) if not first else abs(d)
            n = int(round(n)) if c['lots'] else n
            if disp_sell:
                cap = int(math.floor(0.20 * W.V[t, j] / W.C[t, j])) if c['caps'] else 10**12
                ex = min(n, cap)
                lots, odd = (ex // 1000, ex % 1000) if c['lots'] else (0, ex)
                if ex > 0:
                    cash, _ = self.leg_cost(t, j, -1, lots, odd)
                    self.cash_next += cash; self.S[j] = self.S.get(j, 0.0) - ex
                    self.rec(t, j, -ex, cash)
                rem = abs(d) - ex
                if rem > 1e-9: self.pend[j] = self.pend.get(j, 0) - rem
                return
            lots, odd = self.split(t, j, n) if c['lots'] else (0, n)
            if c['lots']:
                oc = self.odd_cap(t, j, self.odd_px(t, j)); odd_x = min(odd, oc)
                lots_x = min(lots, self.lot_cap(t, j))
                if c['whole_pre2020'] and not W.odd_day[t]: odd_x = 0
            else:
                odd_x, lots_x = odd, 0
            cash, _ = self.leg_cost(t, j, -1, lots_x, odd_x)
            ex = lots_x * 1000 + odd_x
            if ex > 0:
                self.cash += cash; self.S[j] = self.S.get(j, 0.0) - ex; self.rec(t, j, -ex, cash)
                if -0.5 < self.S[j] < 0: self.S[j] = 0.0  # round(S) 賣量造成的不足一股
            rem = abs(d) - ex
            if rem > 1e-9: self.pend[j] = self.pend.get(j, 0) - rem
            return
        # 買
        n = d
        if c['lots']:
            lots, odd = self.split(t, j, n)
            if c['whole_pre2020'] and not W.odd_day[t]:
                odd = 0
            oc = self.odd_cap(t, j, self.odd_px(t, j)); odd_x = min(odd, oc)
            lots_x = min(lots, self.lot_cap(t, j))
            if not first:
                o = W.O[t, j]
                lots_x = min(lots_x, int(math.floor(self.cash / (o * 1000 * 1.05))))
                if odd_x > 0:
                    oc_cash, _ = self.leg_cost(t, j, 1, 0, odd_x)
                    if -oc_cash > self.cash - lots_x * o * 1050: odd_x = 0
            cash, _ = self.leg_cost(t, j, 1, lots_x, odd_x)
            if first and self.est_cost(t, j, d) > self.cash + 1e-9:  # 凍結後修正 F3：完整委託 cost(j, delta)
                self.pend[j] = self.pend.get(j, 0) + d; return
            ex = lots_x * 1000 + odd_x
            dropped = (lots * 1000 + odd) - ex
            if c['whole_pre2020'] and not W.odd_day[t]:
                dropped = lots * 1000 - lots_x * 1000
        else:
            cash, _ = self.leg_cost_est(t, j, 1, 0, n, n)
            if -cash > self.cash + 1e-9:
                # 可分割：縮量成交
                scale = self.cash / (-cash) if cash < 0 else 0
                n2 = n * scale * (1 - 1e-12); cash, _ = self.leg_cost_est(t, j, 1, 0, n2, n2); ex = n2; dropped = 0
            else:
                ex = n; dropped = 0
        if ex > 0:
            self.cash += cash; self.S[j] = self.S.get(j, 0.0) + ex; self.rec(t, j, ex, cash)
        rem = d - ex
        if c['lots'] and c['whole_pre2020'] and not W.odd_day[t]:
            rem = dropped
        if rem > 1e-9: self.pend[j] = self.pend.get(j, 0) + rem

    def deferred(self, t):
        items = sorted(self.pend.items(), key=lambda kv: kv[1])
        self.pend = {}
        for j, d in items:
            if abs(d) < 1e-9: continue
            self.exec_order(t, j, d, first=False)

    def rec(self, t, j, n, cash):
        self.turn[t] += abs(cash)
        if self.trades is not None: self.cf[j] = self.cf.get(j, 0.0) + cash
        if self.trades is not None:
            self.trades.append((t, j, n, cash))

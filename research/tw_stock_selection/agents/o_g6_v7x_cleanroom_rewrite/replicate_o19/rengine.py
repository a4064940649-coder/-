# -*- coding: utf-8 -*-
"""replicate_o19 可執行引擎（V7X＝F1＋F2＋F3(0.5)＋F4、K5T、GROW；合併下市 orig＝A16＋o3 X）。
獨立複製員依凍結規格文字自寫；沒有讀任何研究員 .py。
"""
import json, math
import numpy as np, pandas as pd

AG = '/home/user/-/research/tw_stock_selection/agents'
SPEC = json.load(open(AG + '/o_v7_listing_exec_engine/data/model_spec7.json'))
CPI = json.load(open(AG + '/o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json'))['index_2006_100']
OA_SH = {int(k): v for k, v in SPEC['oa_share_by_dec'].items()}
ZC_SH = {int(k): v for k, v in SPEC['zcap']['share_by_dec'].items()}
ZC_DEF = SPEC['zcap']['share_default']
DR_E = np.array(SPEC['drift']['edges']); DR_V = np.array(SPEC['drift']['vals'])
OD_PE = np.array(SPEC['odd']['price_edges']); OD_RE = np.array(SPEC['odd']['rank_edges'])
OD_BUY = np.array(SPEC['odd']['buy'])
ODD_START = np.datetime64('2020-10-26')
FEE_SWITCH = np.datetime64('2015-01-01')
LIM_SWITCH = np.datetime64('2015-06-01')


def tick(p):
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0


def limits(ref, L):
    up = ref * (1 + L); dn = ref * (1 - L)
    tu = tick(up); td = tick(dn)
    return math.floor(up / tu + 1e-9) * tu, math.ceil(dn / td - 1e-9) * td


class Data:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        for k in z.files:
            setattr(self, k, z[k])
        self.T, self.N = self.C.shape
        self.col = {c: i for i, c in enumerate(self.codes)}
        self.dates = self.dates.astype('datetime64[D]')
        self.dpos = {d: i for i, d in enumerate(self.dates)}
        self.E = self.E.astype(float)
        self.ER = np.ones(self.T)
        self.ER[1:] = self.E[1:] / self.E[:-1]
        self._disp()
        self._merger()

    def _disp(self):
        r = pd.read_csv(AG + '/o_disposal_rejudge/disposal_hist_rejudge.csv', dtype={'code': str})
        r = r[r['in_panel'] == True][['code', 'start', 'end']]
        z = pd.read_csv(AG + '/o_v7_listing_exec_engine/raw/screener/extras/disposal_5y.csv.gz', dtype={'code': str})
        z = z[['code', 'start', 'end']]
        have = set(zip(r.code, r.start))
        z = z[[(c, s) not in have for c, s in zip(z.code, z.start)]]
        u = pd.concat([r, z])
        D = np.zeros((self.T, self.N), bool)
        for c, s, e in zip(u.code, u.start, u.end):
            if c not in self.col or not isinstance(s, str) or not isinstance(e, str):
                continue
            s = np.datetime64(s); e = np.datetime64(e)
            m = (self.dates >= s) & (self.dates <= e)
            D[m, self.col[c]] = True
        self.DISP = D

    def _merger(self):
        """X 規則：merger_consideration（型態、對價）；表外的永久停牌以 delist_official 終止日當 unknown。
        代碼重用：規則以「終止日前最後一筆成交 lt_old」為準，新證券第一筆成交後失效。"""
        m = pd.read_csv(AG + '/o_infra_v3_rebuild/merger_consideration.csv', dtype={'code': str})
        off = pd.read_csv(AG + '/o_infra_v3_rebuild/delist_official.csv', dtype={'code': str})
        off = off[off['in_panel'] == True]
        self.TRIDX = {n: np.where(self.TR[:, n])[0] for n in range(self.N)}
        self.MRG = {}   # n -> list of rules
        def add(n, typ, dd, ratio):
            tri = self.TRIDX[n]
            p0 = int(np.searchsorted(self.dates, dd, 'left'))
            before = tri[tri < p0]
            if len(before) == 0:
                return
            lt_old = int(before[-1])
            after = tri[tri > lt_old]
            nxt = int(after[0]) if len(after) else self.T + 10
            pay = min(p0 + 20, self.T - 1)
            self.MRG.setdefault(n, []).append(dict(type=typ, dd=dd, lt=lt_old, nxt=nxt, ratio=ratio, pay=pay))
        seen = set()
        for _, r in m.iterrows():
            c = r['code']
            if c not in self.col:
                continue
            dd = r['delist_date']
            if not isinstance(dd, str):
                dd = str(np.datetime64(r['last_trade']) + np.timedelta64(14, 'D'))
            ratio = r['cash_over_last_close'] if np.isfinite(r['cash_over_last_close']) else 1.0
            add(self.col[c], r['type'], np.datetime64(dd), float(ratio))
            seen.add(c)
        for c, dd in zip(off.code, off.date):
            if c in seen or c not in self.col:
                continue
            add(self.col[c], 'unknown', np.datetime64(dd), 1.0)

    def rule_for(self, n, t):
        for r in self.MRG.get(n, []):
            if r['lt'] < t < r['nxt']:
                return r
        return None


class Cfg:
    def __init__(self, tier=100e4, stress=False, w30=False, var=''):
        self.tier = tier; self.stress = stress; self.w30 = w30
        self.var = set(v for v in var.split(',') if v)


def run_sim(D, plan, cfg, end_idx=None, trace=None, inv=None):
    """plan: list of (signal_idx, exec_idx, [col idx...]). 回傳 (t0, equity array from t0..T-1)."""
    T = D.T
    s0 = plan[0][0]
    cash = cfg.tier * CPI['2006'] / CPI['2026']
    cash_next = 0.0
    S = {}            # n -> shares
    pend = {}         # n -> signed shares remaining (deferred)
    claims = []       # [value_0050, pay_idx]
    eq = np.full(T, np.nan)
    eq[s0] = cash
    rebal = {p[1]: p[2] for p in plan}
    rate_fn = lambda d: 0.001425 * 0.6 if (cfg.stress or d < FEE_SWITCH) else 0.001425 * 0.28
    C, O, V, PC, CF, ADJ, G = D.C, D.O, D.V, D.PC, D.CF, D.ADJ, D.G
    TRm = D.TR

    for t in range(s0 + 1, T):
        d = D.dates[t]
        # 隔日現金
        cash += cash_next; cash_next = 0.0
        # 股數×ADJ
        for n in list(S.keys()):
            S[n] *= ADJ[t, n]
        # 0050 待撥款
        er = D.ER[t]
        rate = rate_fn(d)
        L = 0.07 if d < LIM_SWITCH else 0.10
        vr = D.VR[t - 1]; v60 = D.V60[t - 1]; sig = D.SIG[t - 1]
        odd_post = d >= ODD_START

        def dec_of(n):
            r = vr[n]
            if not np.isfinite(r) or r > 300: return 10
            return int(math.ceil(r / 30.0))

        def tradable(n):
            return TRm[t, n] and np.isfinite(O[t, n]) and O[t, n] > 0

        def lim_hit(n, side):
            if not TRm[t, n]:
                return True
            ref = PC[t, n] if D.FL[t, n] == 1 else C[t, n] / G[t, n]
            up, dn = limits(ref, L)
            if side > 0:
                return O[t, n] >= up - 1e-9
            return O[t, n] <= dn + 1e-9

        def odd_px(n, side):
            """回傳 (成交價, 量上限)。"""
            pc = PC[t, n]
            rk = vr[n] if np.isfinite(vr[n]) else 1e9
            pb = int(np.searchsorted(OD_PE, pc, 'right')); rb = int(np.searchsorted(OD_RE, rk, 'right'))
            c = OD_BUY[pb][rb]
            if odd_post:
                o = O[t, n]
                gap = o / pc - 1 if (np.isfinite(pc) and pc > 0) else 0.0
                b = o * (1 + DR_V[int(np.searchsorted(DR_E, gap, 'right'))])
                part = 1.0
            else:
                b = C[t, n]
                part = 0.5
            zs = ZC_DEF if (not np.isfinite(vr[n]) or vr[n] > 300) else ZC_SH[dec_of(n)]
            cap = math.floor(part * zs * v60[n] / b + 1e-9) if np.isfinite(v60[n]) else 0
            if side > 0:
                cc = c * 2 + 0.004 if cfg.stress else c
                return b * (1 + cc), cap
            cc = 0.0 + (0.004 if cfg.stress else 0.0)
            return b * (1 - cc), cap

        def lot_imp(n, nl):
            o = O[t, n]
            sh = OA_SH[dec_of(n)]
            oa = sh * V[t, n]
            base = 0.001
            if np.isfinite(sig[n]) and np.isfinite(oa) and oa > 0:
                base = 0.001 + 0.5 * sig[n] * math.sqrt(nl * 1000 * PC[t, n] / oa)
            imp = max(base, tick(o) / o)
            if cfg.stress:
                imp += 0.004
            return imp

        def lot_cap(n):
            sh = OA_SH[dec_of(n)]
            if not np.isfinite(v60[n]):
                return 1
            return max(1, math.floor(0.2 * sh * v60[n] / (1000 * PC[t, n]) + 1e-12))

        def fee(val):
            return max(20.0, rate * val) if val > 0 else 0.0

        def cost_est(n, delta):
            """need 用：signed 現金需求（買正、賣負）；整股用全部張數、不套上限；處置賣出 0。"""
            if delta == 0:
                return 0.0
            if delta < 0 and D.DISP[t, n]:
                return 0.0
            q = abs(delta); nl = int(q // 1000); no = q - nl * 1000
            tot = 0.0
            side = 1 if delta > 0 else -1
            if no > 0:
                px, _ = odd_px(n, side)
                val = no * px
                tot += side * val + fee(val) + (0.003 * val if side < 0 else 0.0)
            if nl > 0:
                imp = lot_imp(n, nl)
                px = O[t, n] * (1 + side * imp)
                val = nl * 1000 * px
                tot += side * val + fee(val) + (0.003 * val if side < 0 else 0.0)
            return tot

        def do_disposal_sell(n, q):
            """處置期賣出：C−tick，量 ≤ floor(0.2×value/C)，款項隔日。回傳成交股數。"""
            nonlocal cash_next
            if not TRm[t, n]:
                return 0
            c = C[t, n]
            capq = math.floor(0.2 * V[t, n] / c + 1e-9)
            qq = min(q, capq)
            if qq <= 0:
                return 0
            px = c - tick(c)
            if cfg.stress and 'dispstress' in cfg.var:
                px *= (1 - 0.004)
            nl = int(qq // 1000); no = qq - nl * 1000
            got = 0.0
            for part in (no, nl * 1000):
                if part > 0:
                    val = part * px
                    got += val - fee(val) - 0.003 * val
            cash_next += got
            return qq

        def exec_order(n, delta, rebal_day, cash_avail_fn=None):
            """執行一檔委託；回傳未成交 signed 股數。"""
            nonlocal cash
            if delta == 0:
                return 0
            side = 1 if delta > 0 else -1
            q = abs(delta)
            if side < 0:
                held = S.get(n, 0.0)
                if not rebal_day:
                    q = min(q, int(round(held)))
                if q <= 0:
                    return delta
                if D.DISP[t, n]:
                    done = do_disposal_sell(n, q)
                    S[n] = S.get(n, 0.0) - done
                    if -0.5 < S[n] < 1e-9: S.pop(n)
                    return -(abs(delta) - done)
            if not tradable(n) or lim_hit(n, side):
                return delta
            nl = int(q // 1000); no = int(q - nl * 1000)
            if cfg.w30 and not odd_post:
                no = 0
            if side > 0:
                if rebal_day:
                    # F3：以完整委託 cost 判定
                    if 'f3off' in cfg.var:
                        _, capo = odd_px(n, 1) if no > 0 else (0, 0)
                        qe = min(nl, lot_cap(n)) * 1000 + min(no, capo)
                        if cost_est(n, qe) > cash + 1e-9:
                            return delta
                    elif cost_est(n, delta) > cash + 1e-9:
                        return delta
                else:
                    o = O[t, n]
                    nl = min(nl, int(math.floor(cash / (o * 1000 * 1.05))))
                    nl = max(nl, 0)
            done = 0
            # 零股
            if no > 0:
                px, cap = odd_px(n, side)
                qo = min(no, cap)
                if (not cfg.w30 or odd_post) and qo > 0:
                    val = qo * px
                    if side > 0:
                        c_ = val + fee(val)
                        nlc = min(nl, lot_cap(n)) if 'g4' in cfg.var else nl
                        if (not rebal_day) and c_ > cash - nlc * O[t, n] * 1050:
                            qo = 0
                        else:
                            cash -= c_
                    else:
                        cash += val - fee(val) - 0.003 * val
                    if qo > 0:
                        S[n] = S.get(n, 0.0) + side * qo
                        done += qo
            # 整股
            if nl > 0:
                nl2 = min(nl, lot_cap(n))
                if nl2 > 0:
                    imp = lot_imp(n, nl2)
                    px = O[t, n] * (1 + side * imp)
                    val = nl2 * 1000 * px
                    if side > 0:
                        cash -= val + fee(val)
                    else:
                        cash += val - fee(val) - 0.003 * val
                    S[n] = S.get(n, 0.0) + side * nl2 * 1000
                    done += nl2 * 1000
            if n in S and -0.5 < S[n] < 1e-9:
                S.pop(n)
            return side * (abs(delta) - done)

        if t in rebal:
            picks = rebal[t]
            # 1 待撥款：撥款後的換股日以前一日收盤值併入現金（當日 0050 不計）
            keep = []
            for cl in claims:
                if (t >= cl[1]) if 'paygte' in cfg.var else (t > cl[1]):
                    cash += cl[0]
                else:
                    cl[0] *= er
                    keep.append(cl)
            claims = keep
            # 2 清掉順延
            pend = {}
            # 3 凍結股（X／A16）
            for n in list(S.keys()):
                if TRm[t, n]:
                    continue
                mr = D.rule_for(n, t)
                if mr is None:
                    continue
                val = S[n] * CF[t, n]
                if mr['type'] == 'cash':
                    if d >= mr['dd']:
                        claims.append([val * mr['ratio'] * er, mr['pay']])
                        S.pop(n)
                elif mr['type'] == 'failure':
                    if t > mr['lt']:
                        S.pop(n)
                else:
                    if (d > mr['dd']) if 'swapgt' in cfg.var else (d >= mr['dd']):
                        cash += val
                        S.pop(n)
            # 4 小數股
            for n in list(S.keys()):
                if tradable(n):
                    fr = S[n] - math.floor(S[n] + 1e-9)
                    if fr > 0:
                        cash += fr * O[t, n]
                        S[n] -= fr
                    if S[n] <= 1e-9:
                        S.pop(n)
            # 5 目標
            k = len(picks)
            hi = (sum(S[n] * O[t, n] for n in S if tradable(n)) + cash) / k * 0.98
            pickset = set(picks)
            others = [n for n in S if n not in pickset and tradable(n)]

            def targets(Tm):
                out = {}
                for n in picks:
                    p = PC[t, n]
                    Lt = math.floor(Tm / (1000 * p) + 1e-12)
                    q = math.floor((Tm - 1000 * Lt * p) / p + 1e-9)
                    if cfg.w30 and not odd_post:
                        q = 0
                    tg = 1000 * Lt + max(q, 0)
                    out[n] = tg - int(round(S.get(n, 0.0)))
                for n in others:
                    out[n] = -int(round(S[n]))
                return out

            def need(Tm):
                return sum(cost_est(n, dl) for n, dl in targets(Tm).items())

            if need(hi) <= cash:
                Tm = hi
            else:
                lo, h2 = 0.0, hi
                for _ in range(40):
                    mid = (lo + h2) / 2
                    if need(mid) <= cash:
                        lo = mid
                    else:
                        h2 = mid
                Tm = lo
            dl = targets(Tm)
            for n in picks:
                if n in others:
                    pass
            # 非可交易的非名單持股：目標 0（順延）
            for n in S:
                if n not in pickset and n not in dl:
                    dl[n] = -int(round(S[n]))
            order = sorted([(v, n) for n, v in dl.items() if v != 0], key=lambda x: (x[0], x[1]))
            for v, n in order:
                rem = exec_order(n, v, True)
                if rem != 0:
                    pend[n] = rem
            if trace is not None:
                trace.append((t, Tm, dict(S), cash))
        elif pend:
            for v, n in sorted([(v, n) for n, v in pend.items()], key=lambda x: (x[0], x[1])):
                rem = exec_order(n, v, False)
                if rem == 0:
                    pend.pop(n)
                else:
                    pend[n] = rem
            for cl in claims:
                cl[0] *= er
        else:
            for cl in claims:
                cl[0] *= er
        eqv = cash + cash_next + sum(S[n] * CF[t, n] for n in S) + sum(cl[0] for cl in claims)
        eq[t] = eqv
        if inv is not None:
            inv[t] = sum(S[n] * CF[t, n] for n in S) / eqv if eqv > 0 else np.nan
    return s0, eq

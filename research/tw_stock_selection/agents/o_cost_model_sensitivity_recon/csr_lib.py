#!/usr/bin/env python3
"""o_cost_model_sensitivity_recon 引擎包裝與共用函式（研究員用；複製員不得 import）。

只讀 import 正式 V7X 引擎 o_engine_delist_consideration/cz_v7x.CSim（consideration × cons_main 結算）← o_exec_engine_rewrite_v7x/x_exec.Sim。
設定＝integrate_o19/run_v7x.py 的 R4：orig_cfg＋f1,f2,f3=0.5,f4；GROW；K5T；名單＝o_turnover_fixedcost_diagnosis/data/plans_real_allseeds.csv.gz。

SensSim＝CSim 子類別，加四組旋鈕（全部預設＝正式引擎；斷言預設時淨值與 CSim 逐位相同）：
  mo：零股價差乘數（x_exec OCB/OCS；壓力成本的買進 ×2 照舊乘在後面）
  mc：成交量上限乘數，同時作用在三種參與上限：整股 CAP×OA（lotcap）、零股量上限（2020-10-26 前 f3×ZSH×V60、之後 part_main×ZSH×V60）、處置賣出 20% 日量
  mt：跳動點乘數（K5T 的跳動點下限 TKO 與處置賣出「收盤減一跳」TKC）；0.5＝半個跳動點假設
  sw：Shapley 開關 tax/fee/tick/imp/caps/timing（同 o_turnover_fixedcost_diagnosis/tfc_lib 的定義；timing='close' 需 engine(close=True)）
  refund：None｜'cash'（每筆成本當下退回現金；處置賣出退回 cash_late）｜'prop'（每個有成交的交易日收盤，把當日成本按比例退回：
          全部持股、現金、cash_late、未完成順延單同乘 1＋C_d／E_d；E_d＝當日收盤估值）。兩者都保留成本模型造成的順延、取整、上限與成本預估造成的部位縮小。
  wholelot：2020-10-26 前零股上限＝0（＝integrate_o19 的 WholeLotPre，「30 萬整股」格）
log=True：逐筆成交日誌（每筆 fill 的 t、欄位 j、買賣、股數、W/O/D、參考價 ref（W＝e 開盤、O＝零股基準價 ODDP、D＝收盤）、成交價、
          tk、sq、im、零股價差率、手續費、稅、名目＝股數×參考價、要求量、R/P、跳動點／衝擊／零股／壓力金額）與換股日紀錄。"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/csr')
TFC = AG / 'o_turnover_fixedcost_diagnosis'
TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
# 6 格：(成本, 資金, 模式)
CELLS = [('normal', '30萬', 'std'), ('normal', '100萬', 'std'), ('normal', '300萬', 'std'), ('normal', '1000萬', 'std'),
         ('stress', '100萬', 'std'), ('normal', '30萬', 'wholelot')]
CELLKEY = {c: f'{c[0]}|{c[1]}|{c[2]}' for c in CELLS}
VARS = ['REAL_opt', 'REAL_p90', 'REAL_carry']
GRID = [(mo, mc) for mo in (0.5, 1.0, 2.0) for mc in (0.5, 1.0, 2.0)]
CORNERS = [(0.5, 0.5), (0.5, 2.0), (2.0, 0.5), (2.0, 2.0), (1.0, 1.0)]
ALLON = dict(tax=1, fee=1, tick=1, imp=1, caps=1, timing='open')
FILL_COLS = ['t', 'j', 'side', 'shares', 'kind', 'ref', 'px', 'tk', 'sq', 'im', 'oddc', 'fee', 'tax', 'notional', 'req', 'ctx',
             'tick_amt', 'imp_amt', 'odd_amt', 'stress_amt']
# 期間（日期為收盤點；年化＝起點收盤到終點收盤；成交歸屬＝(起點, 終點]）
SEL_END = pd.Timestamp('2014-12-31')     # 題 1 的挑選期終點；本題探索與迭代只用 ≤ 此日
PER = {'ALL': ('1900-01-01', '2100-01-01'), 'SEL14': ('1900-01-01', '2014-12-31'), 'S0612': ('2006-01-01', '2012-12-31'),
       'S1318': ('2013-01-01', '2018-12-31'), 'S0618': ('1900-01-01', '2018-12-31'), 'Y19on': ('2019-01-01', '2100-01-01'),
       'V1520': ('2015-01-01', '2020-12-31'), 'H2126': ('2021-01-01', '2100-01-01'), 'PREODD': ('1900-01-01', '2020-10-25'),
       'POSTODD': ('2020-10-26', '2100-01-01')}

def sha256f(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha256j(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

_E = {}
def engine(close=False):
    """close=True：只能在 import cz_v7x 之前呼叫（worker initializer）：把 V7X 資料層的開盤價換成收盤價（同 tfc_lib.setup_close_timing）。"""
    if _E:
        assert _E['close'] == close
        return _E
    V7X = AG / 'o_exec_engine_rewrite_v7x'
    if close:
        sys.path.insert(0, str(V7X)); import x_data as D0; sys.path.pop(0)
        assert 'x_exec' not in sys.modules
        D0.O = D0.C.copy()
    sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
    D = CV.D
    sys.path.insert(0, str(CV.V7X)); import x_stats as S; import x_exec as XE; sys.path.pop(0)
    VR = json.loads((CV.V7X / 'results.json').read_text())
    BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
    R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
    dd_ = R2['PREREG']['events_v2']['summary']['d']['d_used']
    E2 = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
    ST = CE.build_settlement(E2, 'main', d=dd_, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
    _E.update(CV=CV, CE=CE, D=D, S=S, XE=XE, BASE=BASE, ST=ST, close=close)
    _E['Sim'] = _make_class(CV, XE, D)
    return _E

def _make_class(CV, X, D):
    BASE_, KAPPA, CAP = X.BASE, X.KAPPA, X.CAP
    PART = X.SP['zcap']['part_main']
    class SensSim(CV.CSim):
        def __init__(self, pl, cfg, settle=None, cmode='consideration', mo=1.0, mc=1.0, mt=1.0, sw=None, refund=None, log=False, wholelot=False):
            super().__init__(pl, cfg, settle=settle, cmode=cmode)
            self.mo, self.mc, self.mt = float(mo), float(mc), float(mt)
            self.sw = dict(ALLON); self.sw.update(sw or {})
            assert refund in (None, 'cash', 'prop')
            self.refund = refund; self.dolog = log; self.wholelot = wholelot
            self.fills = []; self.rebs = []; self._ctx = 'R'; self._dcost = 0.0; self.daycost = {}; self.refunded = 0.0
        # ---------- 成本元件 ----------
        def _fee(self, t, v):
            if not self.sw['fee']: return 0.0
            return max(20.0, self.feerate(t) * v)
        def _tko(self, t, j):
            tk = X.TKO[t, j]
            return tk * self.mt if self.mt != 1.0 else tk
        def _tkc(self, t, j):
            if not self.sw['tick']: return 0.0
            tk = X.TKC[t, j]
            return tk * self.mt if self.mt != 1.0 else tk
        def _sq(self, t, j, L):
            r = BASE_; oa = X.OASH[t, j] * D.V[t, j]; sg = D.SIGp[t, j]
            if L > 0 and np.isfinite(oa) and oa > 0 and np.isfinite(sg):
                q = L * 1000.0 * D.PC[t, j]; r = r + KAPPA * sg * math.sqrt(q / oa)
            return r
        def imp(self, t, j, L):
            assert self.c['model'] == 'K5T'
            tk = self._tko(t, j); sq = self._sq(t, j, L)
            a, b = self.sw['tick'], self.sw['imp']
            r = max(sq, tk) if (a and b) else (tk if a else (sq if b else 0.0))
            return r + self.extra
        def oddpx(self, t, j):
            if self.sw['timing'] == 'close': return D.C[t, j]
            return X.ODDP[t, j]
        def oddc(self, t, j, side):
            rr = self.c.get('rank_right')
            if self.sw['tick']:
                if side > 0:
                    c = (X.OCB1 if rr else X.OCB)[t, j]
                    if self.mo != 1.0: c = c * self.mo
                    if self.c['cost'] == 'stress': c *= 2
                else:
                    c = (X.OCS1 if rr else X.OCS)[t, j]
                    if self.mo != 1.0: c = c * self.mo
            else:
                c = 0.0
            if self.c['cost'] == 'stress' and self.c.get('stress_odd_extra'): c += X.EXTRA
            return c
        def oddcap(self, t, j):
            if not self.sw['caps']: return None
            if self.wholelot and t < X.I_ODD: return 0
            if t >= X.I_ODD:
                px = self.oddpx(t, j); v = D.V60p[t, j]
                if not np.isfinite(v): v = 0.0
                pm = PART * self.mc if self.mc != 1.0 else PART
                return math.floor(pm * X.ZSH[t, j] * v / px)
            r = self.c.get('f3', 0)
            if r:
                px = D.C[t, j]; v = D.V60p[t, j]
                if not np.isfinite(v): v = 0.0
                rm = r * self.mc if self.mc != 1.0 else r
                return math.floor(rm * X.ZSH[t, j] * v / px)
            return None
        def lotcap(self, t, j):
            if not self.sw['caps']: return 10 ** 9
            if self.c.get('f4'): oa = X.OASH[t, j] * D.V60p[t, j]
            else: oa = X.OASH[t, j] * D.V[t, j]
            if not (np.isfinite(oa) and oa > 0): return 1
            cm = CAP * self.mc if self.mc != 1.0 else CAP
            return max(1, math.floor(cm * oa / (1000.0 * D.PC[t, j])))
        def cost_est(self, t, j, delta, disp):
            if delta == 0: return 0.0
            side = 1 if delta > 0 else -1; a = abs(delta)
            L, q = int(a // 1000), int(a % 1000)
            tot = 0.0; TAX = X.TAX
            if disp and side < 0:
                px = D.C[t, j] - self._tkc(t, j)
                for n in (L * 1000, q):
                    if n > 0:
                        v = n * px; tot += -v + self._fee(t, v) + TAX * v
                return tot
            if L > 0:
                v = 1000.0 * L * D.O[t, j] * (1 + side * self.imp(t, j, L))
                tot += side * v + self._fee(t, v) + (TAX * v if side < 0 else 0.0)
            if q > 0:
                v = q * self.oddpx(t, j) * (1 + side * self.oddc(t, j, side))
                tot += side * v + self._fee(t, v) + (TAX * v if side < 0 else 0.0)
            return tot
        def _book(self, cost, late=False):
            self._dcost += cost
            if self.refund == 'cash':
                if late: self.cash_late += cost
                else: self.cash += cost
                self.refunded += cost
        # ---------- 成交（與 x_exec 同式，加旋鈕、日誌、退款） ----------
        def fill_whole(self, t, j, L, side):
            if L <= 0: return 0
            req = L
            cap = self.lotcap(t, j)
            if L > cap: L = cap; self.acc['n_cap_hit'] += 1
            im = self.imp(t, j, L)
            oa = X.OASH[t, j] * D.V[t, j]
            if np.isfinite(oa) and oa > 0:
                self.part.append(1000.0 * L * D.PC[t, j] / oa)
                if np.isfinite(D.V60p[t, j]) and D.V60p[t, j] > 0: self.trk['oa_ratio'].append(D.V[t, j] / D.V60p[t, j])
            o = D.O[t, j]; v = 1000.0 * L * o * (1 + side * im)
            fe = self._fee(t, v); tx = X.TAX * v if side < 0 else 0.0
            self.cash += -side * v - fe - tx
            self.S[j] += side * 1000.0 * L
            self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['impact'] += 1000.0 * L * o * im; self.acc['n_orders'] += 1
            nt = 1000.0 * L * o
            self._book(nt * im + fe + tx)
            if self.dolog:
                tk = self._tko(t, j) if self.sw['tick'] else 0.0
                imx = max(im - self.extra, 0.0)
                tka = nt * min(imx, tk); ima = nt * max(0.0, imx - tk)
                self.fills.append((t, j, side, 1000 * L, 'W', o, o * (1 + side * im), tk, self._sq(t, j, L), im, 0.0, fe, tx, nt, 1000 * req, self._ctx,
                                   tka, ima, 0.0, nt * self.extra))
            return L * 1000
        def fill_odd(self, t, j, q, side, cashlim=None):
            if q <= 0: return 0
            req = q
            cap = self.oddcap(t, j)
            if cap is not None and q > cap:
                q = max(cap, 0); self.acc['n_zcap_hit'] += 1
                if q == 0: return 0
            px = self.oddpx(t, j); c = self.oddc(t, j, side)
            if t < X.I_ODD:
                vv = X.ZSH[t, j] * D.V60p[t, j]
                if np.isfinite(vv) and vv > 0: self.trk['odd_pre_ratio'].append(q * px / vv)
            v = q * px * (1 + side * c)
            fe = self._fee(t, v); tx = X.TAX * v if side < 0 else 0.0
            if cashlim is not None and side > 0 and v + fe > cashlim: return 0
            self.cash += -side * v - fe - tx
            self.S[j] += side * q
            self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['odd_spread'] += q * px * c; self.acc['n_orders'] += 1
            nt = q * px
            self._book(nt * c + fe + tx)
            if self.dolog:
                self.fills.append((t, j, side, q, 'O', px, px * (1 + side * c), self._tko(t, j), np.nan, 0.0, c, fe, tx, nt, req, self._ctx,
                                   0.0, 0.0, nt * c, 0.0))
            return q
        def fill_disp_sell(self, t, j, n):
            if n <= 0: return 0
            req = n
            if self.c.get('f2') and self.sw['caps']:
                cp = 0.20 * self.mc if self.mc != 1.0 else 0.20
                capn = math.floor(cp * np.nan_to_num(D.V[t, j]) / D.C[t, j])
                if n > capn: n = max(capn, 0); self.acc['n_disp_cap_hit'] += 1
                if n == 0: return 0
            tkc = self._tkc(t, j)
            px = D.C[t, j] - tkc
            vv = D.V[t, j]
            if np.isfinite(vv) and vv > 0: self.trk['disp_ratio'].append(n * D.C[t, j] / vv)
            L, q = int(n // 1000), int(n % 1000); proceeds = 0.0; fes = txs = 0.0
            for m in (q, L * 1000):
                if m > 0:
                    v = m * px; fe = self._fee(t, v); tx = X.TAX * v
                    proceeds += v - fe - tx; fes += fe; txs += tx
                    self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['disp_cost'] += m * tkc; self.acc['n_orders'] += 1
            if self.c.get('f2'): self.cash_late += proceeds
            else: self.cash += proceeds
            self.S[j] -= n; self.acc['n_disp_sells'] += 1
            self._book(n * tkc + fes + txs, late=bool(self.c.get('f2')))
            if self.dolog:
                C = D.C[t, j]
                self.fills.append((t, j, -1, n, 'D', C, px, tkc / C, np.nan, tkc / C, 0.0, fes, txs, n * C, req, self._ctx, n * tkc, 0.0, 0.0, 0.0))
            return n
        def _endday(self, t):
            dc = self._dcost
            if dc:
                self.daycost[t] = self.daycost.get(t, 0.0) + dc
                if self.refund == 'prop':
                    H = np.flatnonzero(self.S > 0)
                    E = self.cash + self.cash_late + float((self.S[H] * D.CF[t, H]).sum())
                    f = 1.0 + dc / E
                    self.S[H] *= f; self.cash *= f; self.cash_late *= f
                    for jj in list(self.pend): self.pend[jj] = int(round(self.pend[jj] * f))
                    self.refunded += dc
            self._dcost = 0.0
        def rebalance(self, t, pk, tr):
            if self.dolog:
                H = np.flatnonzero(self.S > 0)
                hb = {int(j): float(self.S[j]) for j in H}
                avail = float(self.cash + sum(self.S[j] * D.O[t, j] for j in H if tr[j]))
            self._ctx = 'R'
            super().rebalance(t, pk, tr)
            if self.dolog:
                H2 = np.flatnonzero(self.S > 0)
                self.rebs.append((t, [int(x) for x in pk], hb, {int(j): float(self.S[j]) for j in H2}, avail, float(self.cash),
                                  {int(j): int(d) for j, d in self.pend.items()}))
            self._endday(t)
        def deferred(self, t, tr):
            self._ctx = 'P'
            super().deferred(t, tr)
            self._ctx = 'R'
            self._endday(t)
        def run(self, t1=None):
            X.TAX = 0.003 if self.sw['tax'] else 0.0
            try:
                return super().run(t1)
            finally:
                X.TAX = 0.003
    return SensSim

# ---------- 名單與設定 ----------
def load_plans():
    return pd.read_csv(TFC / 'data/plans_real_allseeds.csv.gz', dtype={'picks': str, 'inserted': str})
def to_plan(g, D, close=False):
    out = []
    for e, pk, sg in zip(g.exec_idx, g.picks, g.signal):
        s = D.TPOS[pd.Timestamp(sg)]; picks = np.array([D.CI[c] for c in pk.split(',')], int)
        out.append((s, picks, s - 1) if close else (int(e), picks, s))
    return sorted(out, key=lambda x: x[0])
def cfg_for(E, tier, cost):
    c = dict(tier=TIERS[tier], mode='GROW', model='K5T', cost=cost); c.update(E['BASE']); return c
def cash0(E, tier): return TIERS[tier] * E['D'].cpi_scale(2006)

# ---------- 統計 ----------
XS_START = {'Y19on', 'POSTODD', 'V1520', 'H2126'}   # x_stats 口徑：起點＝期間內第一個交易日收盤（CAGR_Y19on 自 2019-01-02 收盤）；其餘期間＝期間前最後一個收盤（integrate_o19 pstats 口徑）
def seg(nav, a, b, xs=False):
    x = nav[(nav.index >= a) & (nav.index <= b)]
    if xs: return x.index[0], x.index[-1]
    prev = nav[nav.index < x.index[0]]; s0 = prev.index[-1] if len(prev) else x.index[0]
    return s0, x.index[-1]
def winret(S, nav, j):
    e = S.end_day('AB_V7_B10', j); sd = S.SIGD[('AB_V7_B10', j)]; sd = sd[sd < e].append(pd.DatetimeIndex([e]))
    r = nav.reindex(sd); rb = S.E50.reindex(sd)
    return pd.DataFrame({'r': (r.shift(-1) / r - 1).iloc[:-1], 'r50': (rb.shift(-1) / rb - 1).iloc[:-1]})
def pstats(S, nav, j, pers=None):
    """各期間：日曆年化（起點收盤→終點收盤，(b−a).days/365.25）、0050 同期、超額、最大回撤、對 0050 t（訊號日窗，窗起點在期間內）。"""
    e = S.end_day('AB_V7_B10', j); nav = nav.loc[:e].dropna(); w = winret(S, nav, j); o = {}
    for k in (pers or PER):
        a, b = PER[k]
        s0, s1 = seg(nav, a, b, xs=(k in XS_START)); sg = nav.loc[s0:s1]
        ww = w[(w.index >= a) & (w.index <= b)]
        cg = S.cagr(nav, s0, s1); c5 = S.cagr(S.E50, s0, s1)
        o[k] = dict(cagr=cg, cagr50=c5, exc=cg - c5, mdd=float((sg / sg.cummax() - 1).min() * 100), t=S.tstat(ww.r - ww.r50), s0=str(s0.date()), s1=str(s1.date()))
    return o

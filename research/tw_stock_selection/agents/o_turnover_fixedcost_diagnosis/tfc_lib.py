#!/usr/bin/env python3
"""o_turnover_fixedcost_diagnosis 引擎包裝（研究員用；複製員不得 import）。
只讀 import 正式 V7X 引擎：o_engine_delist_consideration/cz_v7x.CSim（consideration × cons_main 結算）← o_exec_engine_rewrite_v7x/x_exec.Sim。
設定＝integrate_o19/run_v7x.py 的 R4：orig_cfg＋f1,f2,f3=0.5,f4；GROW；K5T。
LogSim＝CSim 子類別：
  (a) 逐筆成交日誌（log=True）：每筆 fill 的日期索引 t、欄位 j、買賣、股數、整股／零股／處置賣出、開盤價、成交價、
      跳動點占比 tk、K5T 平方根衝擊 sq（BASE＋κσ√(q/OA)）、實際衝擊率 im、零股價差率、手續費、證交稅、名目金額、上限前要求量、是否順延。
  (b) Shapley 開關 sw（全開＝正式引擎；斷言全開時淨值與 CSim 逐位相同）：
      tax：證交稅 0.3%（含 fixed_sell 結算）；fee：手續費（含最低 20 元）；tick：跳動點／價差類（K5T 的跳動點下限、零股價差 OCB/OCS、處置賣出收盤減一跳）；
      imp：K5T 平方根衝擊（BASE＋κσ√(q/OA)）；caps：參與上限（整股 20%×OA、零股量上限、處置 20% 日量上限）；
      timing：'open'＝成交日 e 開盤（正式）｜'close'＝訊號日 d 收盤（需在 worker 載入前把 x_data.O 換成 C，見 setup_close_timing）。
  K5T 每邊整股衝擊＝max(sq, tk)（tick&imp）；只 tick＝tk；只 imp＝sq；皆關＝0；壓力成本另加 0.4%（不屬任何開關，Shapley 只跑一般成本）。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/tfc')

def setup_close_timing():
    """只能在 import cz_v7x 之前呼叫（worker initializer）：把 V7X 資料層的開盤價換成收盤價，衍生陣列（TKO、ODDP…）隨之以收盤價計算。"""
    V7X = AG / 'o_exec_engine_rewrite_v7x'
    sys.path.insert(0, str(V7X)); import x_data as D; sys.path.pop(0)
    assert 'x_exec' not in sys.modules
    D.O = D.C.copy()

_E = {}
def engine():
    if _E: return _E
    sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
    D = CV.D
    sys.path.insert(0, str(CV.V7X)); import x_stats as S; import x_exec as XE; sys.path.pop(0)
    VR = json.loads((CV.V7X / 'results.json').read_text())
    BASE = dict(VR['PREREG']['orig_cfg']); BASE.update(dict(f1=True, f2=True, f3=0.5, f4=True))
    R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
    dd_ = R2['PREREG']['events_v2']['summary']['d']['d_used']
    E2 = pd.read_csv(AG / 'o_engine_delist_consideration/events_v2.csv', dtype={'code': str, 'successor': str})
    ST = CE.build_settlement(E2, 'main', d=dd_, TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
    _E.update(CV=CV, CE=CE, D=D, S=S, XE=XE, BASE=BASE, ST=ST)
    _E['LogSim'] = _make_class(CV, XE, D)
    return _E

TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
ALLON = dict(tax=1, fee=1, tick=1, imp=1, caps=1, timing='open')

def _make_class(CV, X, D):
    BASE_, KAPPA = X.BASE, X.KAPPA
    class LogSim(CV.CSim):
        def __init__(self, pl, cfg, settle=None, cmode='consideration', sw=None, log=False):
            super().__init__(pl, cfg, settle=settle, cmode=cmode)
            self.sw = dict(ALLON); self.sw.update(sw or {})
            self.dolog = log; self.fills = []; self.rebs = []; self._ctx = 'R'; self._req = None
        # ---------- 成本元件 ----------
        def _fee(self, t, v):
            if not self.sw['fee']: return 0.0
            return max(20.0, self.feerate(t) * v)
        def _sq(self, t, j, L):
            r = BASE_; oa = X.OASH[t, j] * D.V[t, j]; sg = D.SIGp[t, j]
            if L > 0 and np.isfinite(oa) and oa > 0 and np.isfinite(sg):
                q = L * 1000.0 * D.PC[t, j]; r = r + KAPPA * sg * math.sqrt(q / oa)
            return r
        def imp(self, t, j, L):
            assert self.c['model'] == 'K5T'
            tk = X.TKO[t, j]; sq = self._sq(t, j, L)
            a, b = self.sw['tick'], self.sw['imp']
            r = max(sq, tk) if (a and b) else (tk if a else (sq if b else 0.0))
            return r + self.extra
        def oddpx(self, t, j):
            if self.sw['timing'] == 'close': return D.C[t, j]
            return X.ODDP[t, j]
        def oddc(self, t, j, side):
            c = super().oddc(t, j, side)
            if not self.sw['tick']:
                # 移除零股價差（保留壓力成本的額外加碼部分）
                base = (X.OCB1 if self.c.get('rank_right') else X.OCB)[t, j] if side > 0 else (X.OCS1 if self.c.get('rank_right') else X.OCS)[t, j]
                if side > 0 and self.c['cost'] == 'stress': base = base * 2
                c = c - base
            return c
        def oddcap(self, t, j):
            if not self.sw['caps']: return None
            return super().oddcap(t, j)
        def lotcap(self, t, j):
            if not self.sw['caps']: return 10 ** 9
            return super().lotcap(t, j)
        def cost_est(self, t, j, delta, disp):
            if delta == 0: return 0.0
            side = 1 if delta > 0 else -1; a = abs(delta)
            L, q = int(a // 1000), int(a % 1000)
            tot = 0.0; TAX = X.TAX
            if disp and side < 0:
                px = D.C[t, j] - (X.TKC[t, j] if self.sw['tick'] else 0.0)
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
        # ---------- 成交（與 x_exec 同式，加開關與日誌） ----------
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
            if self.dolog:
                self.fills.append((t, j, side, 1000 * L, 'W', o, o * (1 + side * im), X.TKO[t, j], self._sq(t, j, L), im, 0.0, fe, tx, 1000.0 * L * o, 1000 * req, self._ctx))
            return L * 1000
        def fill_odd(self, t, j, q, side, cashlim=None):
            if q <= 0: return 0
            req = q
            cap = self.oddcap(t, j)
            if cap is not None and q > cap:
                q = max(cap, 0); self.acc['n_zcap_hit'] += 1
                if q == 0:
                    if self.dolog: self.fills.append((t, j, side, 0, 'O', D.O[t, j], np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, req, self._ctx))
                    return 0
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
            if self.dolog:
                self.fills.append((t, j, side, q, 'O', D.O[t, j], px * (1 + side * c), X.TKO[t, j], np.nan, 0.0, c, fe, tx, q * px, req, self._ctx))
            return q
        def fill_disp_sell(self, t, j, n):
            if n <= 0: return 0
            req = n
            if self.c.get('f2') and self.sw['caps']:
                capn = math.floor(0.20 * np.nan_to_num(D.V[t, j]) / D.C[t, j])
                if n > capn: n = max(capn, 0); self.acc['n_disp_cap_hit'] += 1
                if n == 0: return 0
            tkc = X.TKC[t, j] if self.sw['tick'] else 0.0
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
            if self.dolog:
                self.fills.append((t, j, -1, n, 'D', D.C[t, j], px, tkc / D.C[t, j], np.nan, tkc / D.C[t, j], 0.0, fes, txs, n * D.C[t, j], req, self._ctx))
            return n
        def rebalance(self, t, pk, tr):
            if self.dolog:
                H = np.flatnonzero(self.S > 0)
                hv = {int(j): float(self.S[j] * (D.O[t, j] if tr[j] else D.CF[t - 1, j])) for j in H}
                avail = float(self.cash + sum(v for j, v in hv.items() if tr[j]))
                self._ctx = 'R'
            super().rebalance(t, pk, tr)
            if self.dolog:
                H2 = np.flatnonzero(self.S > 0)
                self.rebs.append((t, [int(x) for x in pk], sorted(hv), [int(x) for x in H2], avail, float(self.cash), {int(j): int(d) for j, d in self.pend.items()}))
        def deferred(self, t, tr):
            self._ctx = 'P'
            super().deferred(t, tr)
            self._ctx = 'R'
        def run(self, t1=None):
            X.TAX = 0.003 if self.sw['tax'] else 0.0
            try:
                return super().run(t1)
            finally:
                X.TAX = 0.003
    return LogSim

FILL_COLS = ['t', 'j', 'side', 'shares', 'kind', 'open', 'px', 'tk', 'sq', 'im', 'oddc', 'fee', 'tax', 'notional', 'req', 'ctx']

def to_plan(g, D, close=False):
    """plans CSV 的一組 (variant,j,seed) → 引擎 plan。close=True：在訊號日 d 收盤成交（t0 取 d−1 讓首次換股被執行）。"""
    out = []
    for e, pk, sg in zip(g.exec_idx, g.picks, g.signal):
        s = D.TPOS[pd.Timestamp(sg)]; picks = np.array([D.CI[c] for c in pk.split(',')], int)
        out.append((s, picks, s - 1) if close else (int(e), picks, s))
    return sorted(out, key=lambda x: x[0])

def cfg_for(E, tier, cost):
    c = dict(tier=TIERS[tier], mode='GROW', model='K5T', cost=cost); c.update(E['BASE']); return c

def load_plans():
    return pd.read_csv(HERE / 'data/plans_real_allseeds.csv.gz', dtype={'picks': str, 'inserted': str})

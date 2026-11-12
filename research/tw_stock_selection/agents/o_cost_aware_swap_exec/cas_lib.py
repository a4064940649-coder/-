#!/usr/bin/env python3
"""o_cost_aware_swap_exec 引擎包裝（研究員用；複製員不得 import）。
只讀 import 正式 V7X 引擎 o_engine_delist_consideration/cz_v7x.CSim（consideration × cons_main）← o_exec_engine_rewrite_v7x/x_exec.Sim，
並在載入時斷言兩檔與 v7x_lib.py 的 sha256。設定＝integrate_o19 R4：orig_cfg＋f1,f2,f3=0.5,f4；GROW；K5T。

CASim(CSim)：把 run／rebalance／deferred／成交函式原樣複製，加上本題開關（全部關閉時淨值與 CSim 逐位相同，見 selftest）：
  opt['ca']      ：成本感知遞補，τ 百分位（0.80／0.90；鄰域 0.75／0.85／0.95）。新進名額依排名逐一檢查 ĉ，ĉ>τ 則改選
                   官方序列第 11～20 名中「不在前月名單」、ĉ≤τ 的下一名；找不到就保留原候選。換出、強制出場照常；不續抱舊股。
                   τ＝過去 36 個月（訊號日）全部前 10 名候選在當時資料下的 ĉ（以目前部位規模重算）的百分位；歷史 < 12 個換股日不啟動。
  opt['mt']      ：執行層最小交易金額倍數（1＝F*＝20 元 ÷ 當時手續費率；鄰域 0.5／2）。續抱股再平衡單名目 < F* 不下；
                   買進或續抱調整單的零股尾數（有 ≥1 張時）名目 < F* 不下；換出一律全賣。
  opt['nr']      ：不做等權再平衡：續抱股股數不動，可用資金（2% 現金摩擦後扣續抱股市值）平均分給新進股。
  opt['dc']      ：順延買單上限（交易日數 n；5，鄰域 3／10）：換股後第 n＋1 個交易日起取消尚未成交的買單；賣單照常。
  opt['dcmode']  ：'C' 取消資金留現金到下次換股；'E' 取消資金當日收盤買 0050（E50 含息），下次換股日以前一日收盤賣出，
                   0050 成本＝手續費（最低 20 元）＋每邊價差 0.05%＋賣出 ETF 交易稅 0.1%。
  opt['a20']     ：參照臂（純排名緩衝，已關閉題）：前月名單中官方序列前 20 名者續抱，其餘由官方序列依序補滿。
  opt['pr']      ：安慰劑：新進名額以機率 p 隨機改成第 11～20 名的下一名新候選（不看成本）。opt['ps']：安慰劑：再平衡單／零股尾數以機率 p 隨機不下。
  sw            ：成本開關 tax／fee／tick／imp（全關＝成本前；ĉ、F* 的決策規則不受開關影響）。
ĉ（事前預估來回成本，只用訊號日 d 收盤前資料）：部位 P＝d 收盤淨值 ×(1−2%)／10；價格 p＝d 收盤價；張數 L、零股 q 依 P／p；
  每邊整股率＝max(BASE＋κ·σ_{≤d}·√(1000·L·p／(OA 比例×60 日均值_{≤d})), tick(p)/p)；零股價差＝OCB／OCS（價格級距×成交值名次，皆 ≤ d）；
  手續費＝max(20, 費率×名目)，整股與零股分開；賣出加證交稅 0.3%。一律用一般成本口徑。ĉ＝買＋賣合計 ÷ P。"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ocas')
SCR.mkdir(parents=True, exist_ok=True)
ENGINE_SHA = {
    'o_engine_delist_consideration/cz_v7x.py': None,
    'o_exec_engine_rewrite_v7x/x_exec.py': None,
    'o_exec_engine_rewrite_v7x/x_data.py': None,
    'o_delist_rev_backfill_allpit/v7x_lib.py': None,
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(o): return hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
PIN = HERE / 'engine_sha.json'
def check_engine():
    cur = {k: sha(AG / k) for k in ENGINE_SHA}
    if PIN.exists():
        ref = json.loads(PIN.read_text())
        assert ref == cur, ('引擎檔 sha256 改變', ref, cur)
    else:
        PIN.write_text(json.dumps(cur, indent=1))
    return cur

TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
ALLON = dict(tax=1, fee=1, tick=1, imp=1)
OFFC = dict(tax=0, fee=0, tick=0, imp=0)
E50_SPREAD = 0.0005; E50_TAX = 0.001

_E = {}
def engine():
    if _E: return _E
    check_engine()
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
    _E['CASim'] = _make_class(CV, XE, D)
    return _E

def cfg_for(E, tier, cost):
    c = dict(tier=TIERS[tier], mode='GROW', model='K5T', cost=cost); c.update(E['BASE']); return c

def load_cands(variant=None, seed=None):
    out = []
    for ch in pd.read_csv(HERE / 'data/cands_allseeds.csv.gz', dtype={'cand40': str, 'gapup': str, 'picks': str}, chunksize=20000):
        if variant is not None: ch = ch[ch.variant.isin([variant] if isinstance(variant, str) else variant)]
        if seed is not None: ch = ch[ch.seed.isin([seed] if np.isscalar(seed) else seed)]
        out.append(ch)
    return pd.concat(out, ignore_index=True)

def to_plan(g, D):
    pl = []; cands = {}
    for e, pk, sg, c40, gu in zip(g.exec_idx, g.picks, g.signal, g.cand40, g.gapup):
        s = D.TPOS[pd.Timestamp(sg)]; picks = np.array([D.CI[c] for c in pk.split(',')], int)
        pl.append((int(e), picks, s))
        cands[int(e)] = ([D.CI[c] for c in c40.split(',')], set(D.CI[c] for c in gu.split(',')) if isinstance(gu, str) and gu else set())
    return sorted(pl, key=lambda x: x[0]), cands

def _make_class(CV, X, D):
    BASE_, KAPPA = X.BASE, X.KAPPA
    SP = X.SP
    CASHF = SP['frictions']['cash']
    class CASim(CV.CSim):
        def __init__(self, pl, cfg, settle=None, cmode='consideration', cands=None, opt=None, sw=None, rng_seed=0, track=False):
            super().__init__(pl, cfg, settle=settle, cmode=cmode)
            self.sw = dict(ALLON); self.sw.update(sw or {})
            self.o = dict(ca=None, mt=None, nr=False, dc=None, dcmode='C', a20=False, pr=None, ps=None)
            self.o.update(opt or {})
            self.cands = cands or {}
            self.sig = {e: s for e, pk, s in pl}
            self.prev_pk = None; self.hist_t = []; self.hist_j = []
            self.rng = np.random.default_rng(rng_seed)
            self.e50v = 0.0; self._eqm = None; self._lastreb = None
            self.track = track
            self.st = dict(n_new=0, n_ca_repl=0, n_ca_over=0, n_ca_nofind=0, n_ca_days=0, n_mt_skip=0, mt_skip_notional=0.0, n_mt_oddcut=0,
                           n_dc_cancel=0, dc_cancel_amt=0.0, n_pr=0, n_ps=0, n_rw_orders=0, n_small_elig=0)
            T = len(D.IDX)
            if track:
                self.dc_ = {k: np.zeros(T, np.float32) for k in ('tax', 'fee', 'tick', 'imp', 'odd', 'disp', 'turn', 'ord')}
        # ---------- 成本元件 ----------
        @property
        def _tax(self): return 0.003 if self.sw['tax'] else 0.0
        def _fee(self, t, v):
            if not self.sw['fee']: return 0.0
            return max(20.0, self.feerate(t) * v)
        def imp(self, t, j, L):
            tk = X.TKO[t, j]
            r = BASE_
            oa = X.OASH[t, j] * D.V[t, j]; sg = D.SIGp[t, j]
            if L > 0 and np.isfinite(oa) and oa > 0 and np.isfinite(sg):
                q = L * 1000.0 * D.PC[t, j]
                r = r + KAPPA * sg * math.sqrt(q / oa)
            a, b = self.sw['tick'], self.sw['imp']
            rr = max(r, tk) if (a and b) else (tk if a else (r if b else 0.0))
            return rr + self.extra
        def oddc(self, t, j, side):
            c = super().oddc(t, j, side)
            if not self.sw['tick']:
                base = (X.OCB1 if self.c.get('rank_right') else X.OCB)[t, j] if side > 0 else (X.OCS1 if self.c.get('rank_right') else X.OCS)[t, j]
                if side > 0 and self.c['cost'] == 'stress': base = base * 2
                c = c - base
            return c
        def _trk(self, t, j, kind, notional, fe, tx, tickc, impc, oddc_, dispc):
            if not self.track: return
            eq = self._eqm if self._eqm else 1.0
            d = self.dc_
            d['tax'][t] += tx / eq; d['fee'][t] += fe / eq; d['tick'][t] += tickc / eq; d['imp'][t] += impc / eq
            d['odd'][t] += oddc_ / eq; d['disp'][t] += dispc / eq; d['turn'][t] += notional / eq; d['ord'][t] = 1.0
        def cost_est(self, t, j, delta, disp):
            if delta == 0: return 0.0
            side = 1 if delta > 0 else -1; a = abs(delta)
            L, q = int(a // 1000), int(a % 1000)
            tot = 0.0; TAX = self._tax
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
        def fill_whole(self, t, j, L, side):
            if L <= 0: return 0
            cap = self.lotcap(t, j)
            if L > cap: L = cap; self.acc['n_cap_hit'] += 1
            im = self.imp(t, j, L)
            oa = X.OASH[t, j] * D.V[t, j]
            if np.isfinite(oa) and oa > 0:
                self.part.append(1000.0 * L * D.PC[t, j] / oa)
                if np.isfinite(D.V60p[t, j]) and D.V60p[t, j] > 0: self.trk['oa_ratio'].append(D.V[t, j] / D.V60p[t, j])
            o = D.O[t, j]; v = 1000.0 * L * o * (1 + side * im)
            fe = self._fee(t, v); tx = self._tax * v if side < 0 else 0.0
            self.cash += -side * v - fe - tx
            self.S[j] += side * 1000.0 * L
            self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['impact'] += 1000.0 * L * o * im; self.acc['n_orders'] += 1
            if self.track:
                nt = 1000.0 * L * o; tk = min(im - self.extra, X.TKO[t, j]) if self.sw['tick'] else 0.0
                self._trk(t, j, 'W', nt, fe, tx, nt * tk, nt * max(0.0, im - tk), 0.0, 0.0)
            return L * 1000
        def fill_odd(self, t, j, q, side, cashlim=None):
            if q <= 0: return 0
            cap = self.oddcap(t, j)
            if cap is not None and q > cap:
                q = max(cap, 0); self.acc['n_zcap_hit'] += 1
                if q == 0: return 0
            px = self.oddpx(t, j); c = self.oddc(t, j, side)
            if t < X.I_ODD:
                vv = X.ZSH[t, j] * D.V60p[t, j]
                if np.isfinite(vv) and vv > 0: self.trk['odd_pre_ratio'].append(q * px / vv)
            v = q * px * (1 + side * c)
            fe = self._fee(t, v); tx = self._tax * v if side < 0 else 0.0
            if cashlim is not None and side > 0 and v + fe > cashlim: return 0
            self.cash += -side * v - fe - tx
            self.S[j] += side * q
            self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['odd_spread'] += q * px * c; self.acc['n_orders'] += 1
            if self.track: self._trk(t, j, 'O', q * px, fe, tx, 0.0, 0.0, q * px * c, 0.0)
            return q
        def fill_disp_sell(self, t, j, n):
            if n <= 0: return 0
            if self.c.get('f2'):
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
                    v = m * px; fe = self._fee(t, v); tx = self._tax * v
                    proceeds += v - fe - tx; fes += fe; txs += tx
                    self.acc['fee'] += fe; self.acc['tax'] += tx; self.acc['disp_cost'] += m * tkc; self.acc['n_orders'] += 1
            if self.c.get('f2'): self.cash_late += proceeds
            else: self.cash += proceeds
            self.S[j] -= n; self.acc['n_disp_sells'] += 1
            if self.track: self._trk(t, j, 'D', n * D.C[t, j], fes, txs, 0.0, 0.0, 0.0, n * tkc)
            return n
        # ---------- ĉ ----------
        def chat(self, t, js, P):
            """t＝成交日 e（資料只取 ≤ d＝e−1）；js 欄位陣列；P 部位金額。回傳每檔預估來回成本率。"""
            js = np.asarray(js, int); tt = np.full(len(js), t, int) if np.isscalar(t) else np.asarray(t, int)
            return self._chat_vec(tt, js, P)
        def _chat_vec(self, tt, js, P):
            s = tt - 1
            p = D.C[s, js]
            p = np.where(np.isfinite(p), p, D.PC[tt, js])
            fr = np.where(tt < X.I_FEE, 0.001425 * 0.6, 0.001425 * 0.28)
            L = np.floor(P / (1000.0 * p) + 1e-12); q = np.floor((P - 1000.0 * L * p) / p + 1e-9)
            vw = 1000.0 * L * p; vo = q * p
            oa = X.OASH[tt, js] * D.V60p[tt, js]; sg = D.SIGp[tt, js]
            ok = np.isfinite(oa) & (oa > 0) & np.isfinite(sg) & (L > 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                sq = BASE_ + np.where(ok, KAPPA * np.nan_to_num(sg) * np.sqrt(np.where(ok, vw / np.where(ok, oa, 1.0), 0.0)), 0.0)
                tk = D.tick(p) / p
            im = np.maximum(sq, tk)
            rr = self.c.get('rank_right')
            ocb = (X.OCB1 if rr else X.OCB)[tt, js]; ocs = (X.OCS1 if rr else X.OCS)[tt, js]
            fw = np.where(vw > 0, np.maximum(20.0, fr * vw), 0.0); fo = np.where(vo > 0, np.maximum(20.0, fr * vo), 0.0)
            buy = vw * im + fw + vo * ocb + fo
            sell = vw * im + fw + vo * ocs + fo + 0.003 * (vw + vo)
            c = (buy + sell) / P
            return np.where(np.isfinite(c), c, np.nan)
        # ---------- 名單層 ----------
        def _pick(self, t, pk):
            if t not in self.cands:
                return pk
            o = self.o
            cand, gu = self.cands[t]
            held = set(self.prev_pk) if self.prev_pk is not None else set()
            F = [x for x in cand if not (x in gu and x not in held)]
            base = F[:10]
            if not (o['ca'] or o['a20'] or o['pr']):
                out = list(pk)
                self.st['base_mismatch'] = self.st.get('base_mismatch', 0) + int(base != [int(x) for x in pk])
            elif o['a20']:
                keep = [x for x in cand[:20] if x in held]
                out = list(keep)
                for x in F:
                    if len(out) >= 10: break
                    if x not in out: out.append(x)
            else:
                out = list(base)
                newpos = [i for i, x in enumerate(out) if x not in held]
                self.st['n_new'] += len(newpos)
                R = [x for x in F[10:20] if x not in held]
                if o['pr']:
                    for i in newpos:
                        if R and self.rng.random() < o['pr']:
                            out[i] = R.pop(0); self.st['n_pr'] += 1
                else:
                    s = self.sig[t]
                    eqd = self._eqm if self._eqm else self.c['tier'] * D.cpi_scale(2006)
                    P = eqd * (1 - CASHF) / 10.0
                    # τ：過去 36 個月前 10 名候選
                    lo = np.searchsorted(D.DATES, np.datetime64(D.IDX[s] - pd.DateOffset(months=36)), side='right')
                    ht = np.array(self.hist_t, int); hj = np.array(self.hist_j, int)
                    m = ht - 1 >= lo
                    ndays = len(set(ht[m].tolist()))
                    if ndays >= 12 and newpos:
                        self.st['n_ca_days'] += 1
                        tau = float(np.nanpercentile(self._chat_vec(ht[m], hj[m], P), 100 * o['ca']))
                        cn = self.chat(t, [out[i] for i in newpos], P)
                        cr = dict(zip(R, self.chat(t, R, P))) if R else {}
                        for i, c in zip(newpos, cn):
                            if np.isfinite(c) and c > tau:
                                self.st['n_ca_over'] += 1
                                y = next((r for r in R if np.isfinite(cr[r]) and cr[r] <= tau), None)
                                if y is None: self.st['n_ca_nofind'] += 1; continue
                                R.remove(y); out[i] = y; self.st['n_ca_repl'] += 1
                    self.hist_t.extend([t] * len(base)); self.hist_j.extend(base)
            self.prev_pk = list(out)
            return np.array(out, int)
        # ---------- 換股（x_exec.Sim.rebalance 原樣＋開關） ----------
        def rebalance(self, t, pk, tr):
            pk = self._pick(t, pk)
            self._lastreb = t
            c = self.c; k = len(pk); o = self.o
            H = np.flatnonzero(self.S > 0)
            cur = H[tr[H]]
            U = np.union1d(cur, pk)
            curS = np.floor(self.S[U] + 1e-9)
            ispk = np.isin(U, pk)
            dmask = D.DISP[t, U] & ~ispk
            avail = self.cash + (self.S[cur] * D.O[t, cur]).sum()
            hi = avail / k * (1 - SP['frictions']['cash'])
            p = D.PC[t, U]
            heldpk = ispk & (curS > 0)
            if o['nr']:
                trU = tr[U]
                hv = float((curS[heldpk & trU] * D.O[t, U[heldpk & trU]]).sum())
                nnew = int((ispk & ~heldpk).sum())
                hi = max(0.0, (avail * (1 - SP['frictions']['cash']) - hv) / nnew) if nnew > 0 else 0.0
            Fst = (20.0 / self.feerate(t)) * o['mt'] if o['mt'] else None
            psmask = None
            if o['ps']:
                psmask = self.rng.random(len(U)) < o['ps']
            def deltas(T):
                L = np.floor(T / (1000.0 * p) + 1e-12)
                q = np.floor((T - 1000.0 * L * p) / p + 1e-9)
                tg = np.where(ispk, 1000.0 * L + np.maximum(q, 0), 0.0)
                dl = tg - curS
                if o['nr']:
                    dl = np.where(heldpk, 0.0, dl)
                if Fst is not None or psmask is not None:
                    a = np.abs(dl); s = np.sign(dl); Lp = np.floor(a / 1000.0); qp = a - 1000.0 * Lp
                    if Fst is not None:
                        small_rw = heldpk & (a * p < Fst)
                        cut = ispk & ~small_rw & (Lp >= 1) & (qp > 0) & (qp * p < Fst)
                    else:
                        small_rw = heldpk & psmask
                        cut = ispk & ~heldpk & (Lp >= 1) & (qp > 0) & psmask
                    dl = np.where(small_rw, 0.0, np.where(cut, s * 1000.0 * Lp, dl))
                return dl
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
            if Fst is not None or psmask is not None:
                d0 = None
                # 統計：被略過的再平衡單（以未修改的 deltas 比較）
                L0 = np.floor(T / (1000.0 * p) + 1e-12); q0 = np.floor((T - 1000.0 * L0 * p) / p + 1e-9)
                d0 = np.where(ispk, 1000.0 * L0 + np.maximum(q0, 0), 0.0) - curS
                if o['nr']: d0 = np.where(heldpk, 0.0, d0)
                sk = (d0 != 0) & (dl == 0); ct = (d0 != 0) & (dl != 0) & (dl != d0)
                self.st['n_mt_skip'] += int(sk.sum()); self.st['mt_skip_notional'] += float((np.abs(d0[sk]) * p[sk]).sum()); self.st['n_mt_oddcut'] += int(ct.sum())
                self.st['n_rw_orders'] += int((heldpk & (d0 != 0)).sum())
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
        # ---------- 順延（x_exec.Sim.deferred 原樣＋DC） ----------
        def deferred(self, t, tr):
            if self.o['dc'] is not None and self._lastreb is not None and (t - self._lastreb) > self.o['dc']:
                for j in [j for j, d in self.pend.items() if d > 0]:
                    d = self.pend.pop(j)
                    amt = d * D.CF[t - 1, j]
                    self.st['n_dc_cancel'] += 1; self.st['dc_cancel_amt'] += float(amt)
                    if self.o['dcmode'] == 'E':
                        A = min(max(self.cash, 0.0), amt)
                        if A > 0:
                            fe = self._fee(t, A); sp = E50_SPREAD * A if self.sw['tick'] else 0.0
                            if A - fe - sp > 0:
                                self.cash -= A; self.e50v += A - fe - sp
                                if self.track: self._trk(t, j, 'E', A, fe, 0.0, sp, 0.0, 0.0, 0.0)
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
        def _sell_e50(self, t):
            if self.e50v > 0:
                v = self.e50v
                fe = self._fee(t, v); sp = E50_SPREAD * v if self.sw['tick'] else 0.0; tx = E50_TAX * v if self.sw['tax'] else 0.0
                self.cash += v - fe - sp - tx; self.e50v = 0.0
                if self.track: self._trk(t, -1, 'E', v, fe, tx, sp, 0.0, 0.0, 0.0)
        # ---------- 主迴圈（cz_v7x.CSim.run 原樣＋0050 殘餘資金、d 收盤淨值記錄、證交稅開關） ----------
        def run(self, t1=None):
            c = self.c; t0 = self.t0
            t1 = len(D.IDX) - 1 if t1 is None else t1
            TAXV = self._tax
            self.S = np.zeros(D.N); self.cash = c['tier'] * D.cpi_scale(2006); self.cash_late = 0.0
            self.pend = {}; mpend = []; claims = []
            unit = 1.0; eq_prev = self.cash
            self._eqm = self.cash
            out = np.empty(t1 - t0 + 1); out[0] = 1.0
            cons = self.cmode == 'consideration'
            for t in range(t0 + 1, t1 + 1):
                r50 = D.E50[t] / D.E50[t - 1] if np.isfinite(D.E50[t - 1]) and np.isfinite(D.E50[t]) else 1.0
                for m in mpend:
                    if not (t in self.reb and m[1] < t): m[0] *= r50
                isreb = t in self.reb
                if self.e50v and not isreb: self.e50v *= r50
                H = np.flatnonzero(self.S > 0)
                if len(H): self.S[H] *= D.ADJ[t, H]
                if self.cash_late:
                    self.cash += self.cash_late; self.cash_late = 0.0
                tr = D.TR[t] & np.isfinite(D.O[t])
                if cons:
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
                                pr = 0.0 if cl.get('zero') else cl['value'] * (1 - TAXV - self.feerate(t))
                                self.cash += pr; cl['ev']['parts'].append(('writeoff' if cl.get('zero') else 'fixed_sell', str(D.IDX[t].date()), float(pr))); continue
                        keep.append(cl)
                    claims = keep
                if isreb:
                    pk = self.reb[t]
                    self.pend = {}
                    self._sell_e50(t)
                    H = np.flatnonzero(self.S > 0)
                    frz = H[~tr[H]]
                    for j in frz:
                        if cons:
                            if j in D.MCASH or j in D.XR:
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
                        raise NotImplementedError
                    fr = self.S[trH] - np.floor(self.S[trH] + 1e-9)
                    fr = np.where(fr > 0, fr, 0.0)
                    self.cash += (fr * D.O[t, trH]).sum(); self.S[trH] -= fr
                    self.rebalance(t, pk, tr)
                    H = np.flatnonzero(self.S > 0)
                    eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + sum(cl['value'] for cl in claims) + self.e50v
                    unit *= eq / (eq_pre + flow)
                else:
                    if self.pend:
                        self.deferred(t, tr); self.acc['deferred_days'] += 1
                    H = np.flatnonzero(self.S > 0)
                    eq = self.cash + self.cash_late + (self.S[H] * D.CF[t, H]).sum() + sum(m[0] for m in mpend) + sum(cl['value'] for cl in claims) + self.e50v
                    unit *= eq / eq_prev
                eq_prev = eq; self._eqm = eq
                out[t - t0] = unit
            return pd.Series(out, index=D.IDX[t0:t1 + 1])
    return CASim

# ---------------- 統計共用 ----------------
def t_end_idx(D, date):
    return int(np.searchsorted(D.DATES, np.datetime64(pd.Timestamp(date)), side='right') - 1)

def avg_nav(navs):
    R = pd.concat([n.pct_change() for n in navs], axis=1).mean(1).fillna(0)
    return (1 + R).cumprod()

def cagr(n, a, b):
    n = n.dropna(); a = n.index[n.index >= pd.Timestamp(a)][0]; b = n.index[n.index <= pd.Timestamp(b)][-1]
    yrs = (b - a).days / 365.25
    return float((n[b] / n[a]) ** (1 / yrs) - 1) * 100

def cagr_td(n, a, b):
    n = n.dropna(); s = n[(n.index >= pd.Timestamp(a)) & (n.index <= pd.Timestamp(b))]
    r = s.pct_change().dropna()
    return float(np.exp(np.log1p(r).mean() * 252) - 1) * 100

def monthly(n):
    me = n.groupby(n.index.to_period('M')).last()
    return me.pct_change()

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def dumps(o): return json.dumps(o, ensure_ascii=False, indent=1, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

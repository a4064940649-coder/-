# 本檔由 s0_make_ecore_tick.py 自 raw/exec/ecore.py（o_exec_capacity_tiers/ecore.py 的複製）產生；只改資料路徑並新增 impact='tick'（每邊一個跳動單位）。
#!/usr/bin/env python3
"""o_exec_capacity_tiers 核心：資料讀取、執行量模型套用、逐日成交模擬器、統計。
只讀本題 data/ 與 raw/（由 gen_plans.py、build_exec_data.py、fetch_5m.py 產生）；執行時不 import 其他資料夾。

模擬器 simulate() 兩種模式：
  ideal＝分數股、全部在成交日開盤、成本＝引擎一般成本（買 0.6 折手續費＋0.1%、賣再加 0.3% 稅），二分法求每檔目標金額
        → 必須逐日重現 O3.simulate（AB_V4 每個 seed 的零衝擊淨值；斷言 < 1e-9）。
  exec ＝凍結規格（整股＋零股、零股制度史、最低手續費、開盤集合競價衝擊與 20% 參與上限、順延）。
"""
import json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RAW = HERE / 'raw' / 'exec', HERE / 'raw' / 'exec'   # 本題：輸入全部在 raw/exec/
def tick_size(p):
    """證交所股票升降單位（2005 年起六級）。"""
    if p < 10: return 0.01
    if p < 50: return 0.05
    if p < 100: return 0.1
    if p < 500: return 0.5
    if p < 1000: return 1.0
    return 5.0
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

TIERS = {'30萬': 3e5, '100萬': 1e6, '300萬': 3e6, '1000萬': 1e7}
FEE_FULL = 0.001425
FEES = {'A': dict(rate=FEE_FULL * 0.28, min_lot=20.0, min_odd=20.0),
        'B': dict(rate=FEE_FULL * 0.6, min_lot=20.0, min_odd=20.0),
        'C': dict(rate=FEE_FULL * 0.28, min_lot=20.0, min_odd=1.0)}
TAX = 0.003
ENG_FEE = FEE_FULL * 0.6
ODD_INTRADAY = pd.Timestamp('2020-10-26')
CAP = 0.20
A_, B_ = '2006-04-01', '2026-10-12'
PER = {'S1': ('2006-04-01', '2013-12-31'), 'S2': ('2014-01-01', '2020-12-31'), 'S3': ('2021-03-01', '2026-10-12'),
       'SEL': ('2006-04-01', '2018-12-31'), 'VAL': ('2019-01-01', '2023-12-31'), 'HOLD': ('2024-01-01', '2026-10-12'),
       'ALL': ('2006-04-01', '2026-10-12'), 'FROM2008': ('2008-01-01', '2026-10-12'),
       'PRE_ODD': ('2006-04-01', '2020-10-25'), 'POST_ODD': ('2020-10-26', '2026-10-12'),
       'Y06_18': ('2006-04-01', '2018-12-31'), 'Y19_26': ('2019-01-01', '2026-10-12')}

_CPI = json.loads((RAW / 'cpi_tw.json').read_text(encoding='utf-8'))['index_2006_100']
CPI = {int(k): v for k, v in _CPI.items()}
def cpi_scale(ts):
    """把 2026 幣值換成 ts 當年幣值的倍數。"""
    return CPI[min(max(ts.year, 2006), 2026)] / CPI[2026]

# ---------------- 資料 ----------------
class Data:
    def __init__(self):
        O = pd.read_parquet(DATA / 'px_open.parquet')
        self.IDX = O.index; self.CODES = list(O.columns); self.CI = {c: j for j, c in enumerate(self.CODES)}
        self.O = O.to_numpy(); self.C = pd.read_parquet(DATA / 'px_close.parquet').to_numpy()
        self.G = pd.read_parquet(DATA / 'px_grow.parquet').to_numpy()
        self.TR = pd.read_parquet(DATA / 'px_traded.parquet').to_numpy().astype(bool)
        self.PC = pd.read_parquet(DATA / 'prevclose.parquet').to_numpy()
        self.CF = pd.DataFrame(self.C, index=self.IDX).ffill().to_numpy()      # 最後收盤（跨停牌）
        F = pd.read_parquet(DATA / 'feat_daily.parquet')
        def w(col):
            return F.pivot(index='date', columns='code', values=col).reindex(index=self.IDX, columns=self.CODES).to_numpy()
        self.V60, self.SIG, self.VRANK = w('V60'), w('sig60'), w('vrank')
        xr = json.loads((DATA / 'xrules.json').read_text(encoding='utf-8'))
        self.XR = {self.CI[c]: (v[0], int(np.searchsorted(self.IDX, pd.Timestamp(v[1]))),
                                int(np.searchsorted(self.IDX, pd.Timestamp(v[3]))) if v[3] else 10 ** 9) for c, v in xr.items()}
        B = pd.read_parquet(DATA / 'bench.parquet')
        self.E50 = B['E50']; self.AB = B['AB_V4|normal']; self.ABEW = B['AB_V4|EW']; self.ABS = B['AB_V4|stress']; self.V4 = B['V4|normal']
        P = pd.read_csv(DATA / 'plans_ab.csv', dtype={'picks': str})
        self.plans = {}
        for s, g in P.groupby('seed'):
            self.plans[int(s)] = [(int(e), [self.CI[c] for c in pk.split(',')], sg) for e, pk, sg in zip(g.exec_idx, g.picks, g.signal)]
        P4 = pd.read_csv(DATA / 'plan_v4.csv', dtype={'picks': str})
        self.plans[-1] = [(int(e), [self.CI[c] for c in pk.split(',')], sg) for e, pk, sg in zip(P4.exec_idx, P4.picks, P4.signal)]
        self.sig_days = [pd.Timestamp(x) for x in P4.signal] + [pd.Timestamp(B_)]
        self.t0 = int(self.IDX.get_loc(pd.Timestamp(P4.signal.iloc[0])))
        self.t1 = int(self.IDX.get_loc(pd.Timestamp(B_)))
        # 調整因子（除權息、減資）：持有股數每個有成交日乘 G×前收/收 → 等同股利再投資（與引擎的市值×G 等價）
        with np.errstate(invalid='ignore', divide='ignore'):
            adj = np.where(self.TR, self.G * self.PC / self.C, 1.0)
        self.ADJ = np.where(np.isfinite(adj), adj, 1.0)
        self.GAP = np.where(self.TR & np.isfinite(self.O), self.O / self.PC - 1, np.nan)

# ---------------- 執行量模型（參數由 iterate.py 選版、PREREG 凍結）----------------
class ExecModel:
    """oa：開盤集合競價成交值 ÷ V60 的回歸 ln(ratio)＝b·x；drift：9:10 價格 ÷ 開盤 −1（依跳空分桶的平均）；
    odd：零股買／賣成本（相對同時點整股價），依價位×成交值名次分桶的中位數。"""
    def __init__(self, spec):
        self.spec = spec
        self.oa_coef = np.array(spec['oa']['coef']); self.oa_vars = spec['oa']['vars']; self.oa_kappa = spec['oa']['kappa']
        self.dr_edges = np.array(spec['drift']['edges']); self.dr_vals = np.array(spec['drift']['vals'])
        self.odd_pe = np.array(spec['odd']['price_edges']); self.odd_re = np.array(spec['odd']['rank_edges'])
        self.odd_buy = np.array(spec['odd']['buy']); self.odd_sell = np.array(spec['odd']['sell'])
        self.oa_scale = spec.get('oa_scale', 1.0); self.odd_scale = spec.get('odd_scale', 1.0)

    def x(self, rank, price, sig, gap, cx=None):
        cols = {'pick': float(cx == 'pick'), 'drop': float(cx == 'drop'), 'const': 1.0, 'lnrank': np.log(np.clip(rank, 1, None)), 'lnprice': np.log(price), 'lnsig': np.log(np.clip(sig, 1e-4, None)),
                'absgap': np.abs(gap), 'gapup': np.clip(gap, 0, None), 'gapdn': np.clip(-gap, 0, None)}
        return np.array([cols[v] for v in self.oa_vars], dtype=float)

    def scale_at(self, ts):
        s = self.oa_scale
        if self.spec.get('oa_pre_cut') and ts < pd.Timestamp(self.spec['oa_pre_cut']): s *= self.spec.get('oa_pre_scale', 1.0)
        return s

    def oa_ratio(self, rank, price, sig, gap, cx=None):
        """回傳 開盤集合競價成交值 ÷ V60（中位數口徑：κ×exp(預測 log)；κ 把 09:05–09:10 量換成集合競價量）。"""
        return self.oa_scale * self.oa_kappa * float(np.exp(self.oa_coef @ self.x(rank, price, sig, gap, cx)))

    def drift(self, gap):
        k = int(np.searchsorted(self.dr_edges, gap, side='right'))
        return float(self.dr_vals[k])

    def odd_cost(self, price, rank, side):
        i = int(np.searchsorted(self.odd_pe, price, side='right')); j = int(np.searchsorted(self.odd_re, rank if np.isfinite(rank) else 1e9, side='right'))
        c = float((self.odd_buy if side > 0 else self.odd_sell)[i, j])
        if self.spec.get('odd_stress'): c = 2 * c if side > 0 else max(c, 0.0)
        return self.odd_scale * c

# ---------------- 模擬器 ----------------
def simulate(D, seed, tier=None, version='GROW', fee='A', model=None, mode='exec', flags=None, keep=False, collect=False):
    """回傳 (nav Series, info dict)。
    version：'GROW'（2006-04-11 以 tier×CPI 換算起始，隨淨值成長）或 'FIX'（每個換股日把帳戶重設為 tier×當年 CPI 換算；流量在開盤前）。
    flags（exec 模式拆解用；預設全開）：integer、odd_time、odd_spread、fees、impact（'sqrt'｜'flat'）、cap。
    keep：SOP 第 6 步版（續抱股不調整、只對新進股下單；敏感度）。"""
    fl = dict(integer=True, odd_time=True, odd_spread=True, fees=True, impact='sqrt', cap=True)
    if flags: fl.update(flags)
    plan = D.plans[seed]
    reb = {p[0]: p for p in plan}
    t0, t1 = D.t0, D.t1
    N = len(D.CODES)
    O, C, CF, TR, ADJ, PC = D.O, D.C, D.CF, D.TR, D.ADJ, D.PC
    if mode == 'ideal': return simulate_ideal(D, seed)
    ideal = False
    S = np.zeros(N)                 # 股數（ideal＝以「元／開盤價」計的分數股）
    E0 = tier * cpi_scale(D.IDX[t0])
    cash = E0
    nav = np.empty(t1 - t0 + 1); nav[0] = 1.0
    prev_eq = E0; unit = 1.0
    fr = FEES[fee]
    pend = {}                       # j → 待成交整股股數（正＝買、負＝賣），換股日清空
    acc = dict(tax=0.0, fee=0.0, impact=0.0, odd_spread=0.0, odd_timing=0.0, base_slip=0.0, n_orders=0, n_cap_hit=0, n_lot_orders=0,
               deferred_days=0, unfilled_at_reb=0.0, flow=0.0, _eq=E0, _tv={}, cash_frac_sum=0.0)
    rows = [] if collect else None
    eq_hist = []
    for t in range(t0 + 1, t1 + 1):
        S = S * ADJ[t]
        flow = 0.0; eq_pre = None
        if t in reb:
            e, pk, sg = reb[t]
            tr = TR[t] & np.isfinite(O[t])
            # ---------------- exec ----------------
            acc['unfilled_at_reb'] += sum(abs(v) * CF[t, j] for j, v in pend.items())
            pend = {}
            held = S > 0
            frz = held & ~tr
            for j in np.where(frz)[0]:
                if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                    cash += S[j] * CF[t - 1, j] * D.XR[j][0]; S[j] = 0.0; frz[j] = False
            if version == 'FIX':
                # 名目固定：開盤前把全部持股與現金按 target÷權益 等比例縮放（無成本的帳戶規模重設），流量記為 flow
                Vfz0 = float((S[frz] * CF[t - 1, frz]).sum()); Vtr0 = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
                eq_pre = Vtr0 + Vfz0 + cash
                target = tier * cpi_scale(D.IDX[t]); sc = target / eq_pre
                S = S * sc; cash = cash * sc; flow = target - eq_pre; acc['flow'] += flow
            # 小數股（股利再投資代理、FIX 縮放）→ 現金（開盤價、無成本）
            if fl['integer']:
                fra = np.where(tr & (S > 0), S - np.floor(S + 1e-9), 0.0); fra = np.clip(fra, 0, None)
                cash += float((fra * np.where(tr, O[t], 0.0)).sum()); S = S - fra
            Vfz = float((S[frz] * CF[t - 1, frz]).sum())
            Vtr = float((np.where(tr, S, 0.0) * np.where(tr, O[t], 0.0)).sum())
            acc['_eq'] = Vtr + Vfz + cash
            out = _rebalance(D, t, pk, S, tr, frz, cash, fl, fr, model, acc, keep, rows, sg)
            S, cash, pend = out
        elif pend:
            acc['_eq'] = prev_eq
            S, cash, pend = _work_pending(D, t, S, cash, pend, fl, fr, model, acc, rows)
        nz = S != 0
        eq = cash + float((S[nz] * CF[t, nz]).sum())
        assert cash > -1e-6 * max(1.0, eq), (t, cash)
        if eq_pre is None: unit *= eq / prev_eq
        else: unit *= (eq_pre / prev_eq) * (eq / (eq_pre + flow))      # 時間加權：流量在開盤（隔夜報酬之後）
        acc['cash_frac_sum'] += cash / eq if eq > 0 else 0.0
        nav[t - t0] = unit; prev_eq = eq
    acc['end_equity'] = prev_eq; acc['start_equity'] = E0
    return pd.Series(nav, index=D.IDX[t0:t1 + 1]), dict(acc=acc, rows=rows)

def simulate_ideal(D, seed):
    """引擎（O3.simulate，含 X 出清）逐行重寫：市值記帳、開盤調倉、二分法 60 次。"""
    plan = D.plans[seed]; reb = {p[0]: p for p in plan}
    t0, t1 = D.t0, D.t1; N = len(D.CODES)
    O, C, G, TR = D.O, D.C, D.G, D.TR
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1)
    b, s_ = ENG_FEE + 0.001, ENG_FEE + TAX + 0.001
    for t in range(t0, t1 + 1):
        if t in reb:
            pka = np.array(reb[t][1])
            tr = TR[t] & np.isfinite(O[t])
            gap = np.where(tr, G[t] * O[t] / np.where(tr, C[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if j in D.XR and D.XR[j][1] <= t < D.XR[j][2]:
                    cash += h[j] * D.XR[j][0]; h[j] = 0.0; frz[j] = False
            htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash; k = len(pka)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pka] = T
                return b * np.clip(tgt - htr, 0, None).sum() + s_ * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            newval = np.where(frz, h, 0.0); newval[pka] += lo
            oc = np.where(tr, C[t] / np.where(tr, O[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=D.IDX[t0:t1 + 1]), {}

def _book(acc, t, **kw):
    """成本記帳：絕對金額，以及除以當日開盤前權益的比例（年化成本拆解用）；traded＝成交金額比例（每日，供損益兩平近似）。"""
    eq = acc['_eq']
    for k, v in kw.items():
        if k == 'traded':
            acc['_tv'][t] = acc['_tv'].get(t, 0.0) + v / eq; continue
        acc[k] += v; acc['pct_' + k] = acc.get('pct_' + k, 0.0) + v / eq

def _px_odd(D, t, j, side, fl, model):
    """零股成交價（不含價差）與 timing 參考價。"""
    if not fl['odd_time']: return D.O[t, j]
    if D.IDX[t] < ODD_INTRADAY: return D.C[t, j]                      # 盤後零股：成交日收盤代理
    return D.O[t, j] * (1.0 + model.drift(D.GAP[t, j]))                # 盤中零股：9:10 估計價

def _fee(value, fr, odd, fl):
    if not fl['fees']: return ENG_FEE * value
    if value <= 0: return 0.0
    return max(fr['min_odd'] if odd else fr['min_lot'], fr['rate'] * value)

def _oa(D, t, j, model, tfeat, cx=None):
    r = model.oa_ratio(D.VRANK[tfeat, j], D.PC[t, j], D.SIG[tfeat, j], D.GAP[t, j] if np.isfinite(D.GAP[t, j]) else 0.0, cx) * model.scale_at(D.IDX[t]) / model.oa_scale
    v = D.V60[tfeat, j]
    return r * v if np.isfinite(v) and np.isfinite(r) else np.nan

def _lot_exec(D, t, j, lots, side, fl, fr, model, acc, tfeat, rows, tag, cx=None):
    """執行 lots 張（>0）整股，回傳 (成交張數, 現金變動)。衝擊依 0.1%＋0.5σ√(Q/OA)，參與上限 20%。"""
    p_ref = D.PC[t, j]
    oa = _oa(D, t, j, model, tfeat, cx)
    n = lots
    if fl['cap'] and np.isfinite(oa):
        mx = max(1, int(math.floor(CAP * oa / (1000.0 * p_ref))))
        if n > mx: n = mx; acc['n_cap_hit'] += 1
    Q = n * 1000.0 * p_ref
    sig = D.SIG[tfeat, j]
    if fl['impact'] == 'tick':
        imp = tick_size(D.O[t, j]) / D.O[t, j]
    elif fl['impact'] == 'sqrt' and np.isfinite(oa) and oa > 0 and np.isfinite(sig):
        imp = 0.001 + 0.5 * sig * math.sqrt(Q / oa)
    else:
        imp = 0.001
    px = D.O[t, j] * (1 + side * imp)
    val = n * 1000.0 * px
    f = _fee(val, fr, False, fl)
    tax = TAX * val if side < 0 else 0.0
    _book(acc, t, fee=f, tax=tax, impact=n * 1000.0 * D.O[t, j] * (imp - 0.001), base_slip=n * 1000.0 * D.O[t, j] * 0.001, traded=val)
    acc['n_orders'] += 1; acc['n_lot_orders'] += 1
    if rows is not None:
        rows.append((t, j, tag, side, n, 'lot', val, imp, (Q / oa) if np.isfinite(oa) and oa > 0 else np.nan, lots))
    return n, (-val - f) if side > 0 else (val - f - tax)

def _odd_exec(D, t, j, sh, side, fl, fr, model, acc, rows, tag):
    base = _px_odd(D, t, j, side, fl, model)
    c = model.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], side) if fl['odd_spread'] else 0.0
    px = base * (1 + side * c)
    val = sh * px
    f = _fee(val, fr, True, fl)
    tax = TAX * val if side < 0 else 0.0
    _book(acc, t, fee=f, tax=tax, odd_spread=sh * base * c, odd_timing=side * sh * (base - D.O[t, j]), traded=val)
    acc['n_orders'] += 1
    if rows is not None:
        rows.append((t, j, tag, side, sh, 'odd', val, c, np.nan, sh))
    return (-val - f) if side > 0 else (val - f - tax)

def _plan_shares(T, p):
    if not np.isfinite(p) or p <= 0: return 0
    lots = int(math.floor(T / (1000.0 * p) + 1e-12))
    odd = int(math.floor((T - lots * 1000.0 * p) / p + 1e-9))
    return lots * 1000 + max(odd, 0)

class _Ctx:
    """單日單檔的成本參數快取（估計現金需求用）。"""
    def __init__(self, D, t, fl, fr, model, pk=()):
        self.D, self.t, self.fl, self.fr, self.model = D, t, fl, fr, model; self.c = {}; self.memo = {}; self.pk = set(pk)
    def get(self, j):
        if j not in self.c:
            D, t, fl, model = self.D, self.t, self.fl, self.model
            oa = _oa(D, t, j, model, t - 1, 'pick' if j in self.pk else 'drop'); sig = D.SIG[t - 1, j]
            ok = fl['impact'] == 'sqrt' and np.isfinite(oa) and oa > 0 and np.isfinite(sig)
            ob = {sd: _px_odd(D, t, j, sd, fl, model) for sd in (1, -1)}
            oc = {sd: (model.odd_cost(D.PC[t, j], D.VRANK[t - 1, j], sd) if fl['odd_spread'] else 0.0) for sd in (1, -1)}
            self.c[j] = (oa, sig, ok, D.O[t, j], D.PC[t, j], ob, oc)
        return self.c[j]
    def cost(self, j, delta):
        key = (j, delta)
        if key in self.memo: return self.memo[key]
        if delta == 0: return 0.0
        oa, sig, ok, o, pc, ob, oc = self.get(j)
        fl, fr = self.fl, self.fr
        side = 1 if delta > 0 else -1; a = abs(delta)
        lots, odd = a // 1000, a % 1000
        tot = 0.0
        if lots:
            imp = (tick_size(o) / o) if fl['impact'] == 'tick' else 0.001 + (0.5 * sig * math.sqrt(lots * 1000.0 * pc / oa) if ok else 0.0)
            val = lots * 1000.0 * o * (1 + side * imp)
            tot += side * val + _fee(val, fr, False, fl) + (TAX * val if side < 0 else 0.0)
        if odd:
            val = odd * ob[side] * (1 + side * oc[side])
            tot += side * val + _fee(val, fr, True, fl) + (TAX * val if side < 0 else 0.0)
        self.memo[key] = tot
        return tot

def _est_cost(D, t, j, delta, fl, fr, model):
    """估計 delta 股的現金需求（買＞0）或淨收入（賣＜0 → 負值），用當日實際價格（與引擎同為『已知開盤』的理想化），不含參與上限。"""
    return _Ctx(D, t, fl, fr, model, ()).cost(j, delta)

def _rebalance(D, t, pk, S, tr, frz, cash, fl, fr, model, acc, keep, rows, sg):
    N = len(S); pka = list(pk)
    trade_ok = tr & ~frz
    cur = {j: int(round(S[j])) if fl['integer'] else S[j] for j in np.where(trade_ok & (S > 0))[0]}
    Vtr = sum(S[j] * D.O[t, j] for j in cur) + cash
    k = len(pka)
    def deltas(T):
        out = {}
        for j in set(cur) | set(pka):
            if j in pka:
                if keep and j in cur: continue
                tgt = _plan_shares(T, D.PC[t, j]) if fl['integer'] else T / D.O[t, j]
            else:
                tgt = 0
            d = tgt - cur.get(j, 0)
            if d != 0: out[j] = d
        return out
    ctx = _Ctx(D, t, fl, fr, model, pka)
    def need(T):
        return sum(ctx.cost(j, d) if fl['integer'] else (d * D.O[t, j] * ((1 + ENG_FEE + 0.001) if d > 0 else (1 - ENG_FEE - TAX - 0.001)))
                   for j, d in deltas(T).items())
    if keep:
        new = [j for j in pka if j not in cur]
        hi = Vtr / k
    else:
        hi = Vtr / k
    lo = 0.0
    if need(hi) <= cash: lo = hi
    else:
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if need(mid) <= cash: lo = mid
            else: hi = mid
    dl = deltas(lo)
    pend = {}
    # 先賣後買
    for j, d in sorted(dl.items(), key=lambda x: x[1]):
        side = 1 if d > 0 else -1; a = abs(d)
        if not fl['integer']:
            val = a * D.O[t, j]; f = _fee(val, fr, False, fl); tax = TAX * val if side < 0 else 0.0
            _book(acc, t, fee=f, tax=tax, base_slip=0.001 * val, traded=val)
            cash += (-val * 1.001 - f) if side > 0 else (val * 0.999 - f - tax)
            S[j] += d; continue
        lots, odd = a // 1000, a % 1000
        if side > 0 and ctx.cost(j, d) > cash + 1e-9:
            # 現金不足（前面的賣單被參與上限順延）→ 整筆順延
            pend[j] = pend.get(j, 0) + d; acc['deferred_days'] += 1; continue
        if odd:
            cash += _odd_exec(D, t, j, odd, side, fl, fr, model, acc, rows, 'reb'); S[j] += side * odd
        if lots:
            n, dc = _lot_exec(D, t, j, lots, side, fl, fr, model, acc, t - 1, rows, 'reb', 'pick' if j in pka else 'drop')
            cash += dc; S[j] += side * n * 1000
            if n < lots: pend[j] = pend.get(j, 0) + side * (lots - n) * 1000
    return S, cash, pend

def _work_pending(D, t, S, cash, pend, fl, fr, model, acc, rows):
    newp = {}
    for j, d in sorted(pend.items(), key=lambda x: x[1]):
        if not (D.TR[t, j] and np.isfinite(D.O[t, j])):
            newp[j] = d; continue
        side = 1 if d > 0 else -1; a = abs(d)
        if side < 0: a = min(a, int(round(S[j])))
        lots, odd = a // 1000, a % 1000
        acc['deferred_days'] += 1
        if side > 0:
            # 現金允許的最大張數
            unit = D.O[t, j] * 1000.0 * 1.05
            lots = min(lots, int(cash // unit)) if unit > 0 else 0
            if odd and _est_cost(D, t, j, odd, fl, fr, model) > cash - lots * unit: odd = 0
        if odd:
            cash += _odd_exec(D, t, j, odd, side, fl, fr, model, acc, rows, 'pend'); S[j] += side * odd
        n = 0
        if lots:
            n, dc = _lot_exec(D, t, j, lots, side, fl, fr, model, acc, t - 1, rows, 'pend')
            cash += dc; S[j] += side * n * 1000
        rem = a - odd - n * 1000
        if side > 0: rem = abs(d) - odd - n * 1000
        if rem > 0: newp[j] = side * rem
    return S, cash, newp

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

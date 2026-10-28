#!/usr/bin/env python3
"""o_hold_buffer_fresh 核心：換股緩衝帶與延長持有（只續抱訊號仍新鮮的舊持股）。

資料與規格＝主策略 V3（總監 3.5 節正式口徑）：
  * main_v2/v2_core.V2('gapfix', R_DEL 營收, rev_5y, exfin=True)、正式同分序、n10
  * o_disposal_rejudge 官方處置名單 E1（drj_core.load_list / Disp / excl_rule，只讀 import）
  * o_infra_v3_rebuild：P（下市股價格 apply_patch）、X（併購出清 exit_rules 'main'）、R_DEL（load_rdel），只讀 import，sha256 由 o3_core 斷言
引擎：sim() 是 o3_core.simulate 的推廣（多了：權重漂移帶 band、逐檔成本向量、換股拆解紀錄）。
      band=None 且成本為純量時，與 o3_core.simulate 逐日相同（validate.py 斷言）。
選股：planner() 是 V2.plan 的推廣（多了：排名緩衝 K、新鮮度門檻 h、分數門檻差 delta、兩期重疊）。
      全部參數為原版時，與 V2.plan 每期名單集合相同（validate.py 斷言）。
成本：一般＝滑價 0.1%（買 0.1855%、賣 0.4855%）；壓力＝一般每邊再加 0.3%（買 0.4855%、賣 0.7855%）；
      另報舊口徑 flat 壓力（買賣各 0.5%，ia_core.rates('stress')）與 1,000 萬容量情境（每邊另加 2.5%×sqrt(成交額/60 日均值)）。
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import drj_core as R          # noqa: E402  只讀 import（再 import main_v2/v2_core → independent_audit/ia_core）
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3          # noqa: E402  只讀 import
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K and O3.V is V
IDX, COLS, N = K.IDX, K.COLS, K.N

# ---------------- 期間（訊號日） ----------------
SEL_A, SEL_B = pd.Timestamp('2006-01-01'), pd.Timestamp('2018-12-31')
VAL_A, VAL_B = pd.Timestamp('2019-01-01'), pd.Timestamp('2023-12-31')
HOLD_A, HOLD_B = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-12')
PER_OOS = {'SEL': (SEL_A, SEL_B), 'VAL': (VAL_A, VAL_B), 'HOLD': (HOLD_A, HOLD_B), 'POST': (VAL_A, HOLD_B),
           'ALL': (SEL_A, HOLD_B), 'S1': (pd.Timestamp('2006-01-01'), pd.Timestamp('2013-12-31')),
           'S2a': (pd.Timestamp('2014-01-01'), pd.Timestamp('2018-12-31'))}

FEE, TAX = K.FEE, 0.003
NORMAL = K.rates(0.001)
STRESS = (NORMAL[0] + 0.003, NORMAL[1] + 0.003)
FLAT = K.rates('stress')
def rates_slip(s): return (FEE + s, FEE + TAX + s)

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

# ---------------- 資料（V3） ----------------
_ST = {}
def setup():
    if _ST: return _ST
    npatch = O3.apply_patch(require_freeze=True)
    E = O3.load_events(require_freeze=True)
    XR = O3.exit_rules(E, 'main')
    D = R.load_list(); DS = R.Disp(D)
    YL0, Y50 = R.revenue()
    YRD = O3.load_rdel()
    _ST.update(npatch=npatch, XR=XR, DS=DS, Y50=Y50, YRD=YRD)
    return _ST

def make_S(unavail=None):
    st = setup()
    S = V.V2('gapfix', st['YRD'], st['Y50'], exfin=True, unavail=unavail)
    DS = st['DS']
    def mask(d):
        m = np.zeros(N, bool)
        for j, info in DS.at(d).items():
            if R.excl_rule('E1', info): m[j] = True
        return m
    orig = S.cross_section
    def cs(d, _o=orig):
        old = K.disposal_mask; K.disposal_mask = mask
        try: return _o(d)
        finally: K.disposal_mask = old
    S.cross_section = cs
    return S

def single_yoy(S, d):
    """最新單月營收年增率（訊號月 m 的 m−1 月），排除金融（與 V2.y3 一致）。"""
    Y = S.Y5 if d >= V.SWITCH else S.YL
    m = pd.Period(d, 'M') - 1
    if m not in Y.index: return np.full(N, np.nan)
    y = Y.reindex(index=[m], columns=COLS).to_numpy()[0].astype(float)
    return np.where(V.FINMASK, np.nan, y)

# ---------------- 選股 ----------------
class Prep:
    """每個訊號日的橫斷面快取：合格、分數、52 週比、排序、隔日開盤漲停。"""
    def __init__(self, S):
        self.S = S
        self.days = []
        self.cs = {}
        for d in S.sig_days:
            cs = S.cross_section(d)
            if cs['el'].sum() < 50: continue
            e = cs['e']
            order = V.V2.order(cs, 'official')[:int(cs['el'].sum())]
            self.cs[d] = dict(e=e, el=cs['el'], sc=cs['sc'], s52=cs['s52'], y3=cs['y3'], order=order,
                              rank={int(j): r for r, j in enumerate(order)}, gu=K.gap_up_open(e))
            self.days.append(d)

def planner(P, n=10, K_=None, h=None, delta=None, days=None, keep_log=None):
    """回傳 plan=[(e, picks, pool, d)]。
    K_：舊持股排名 ≤ K_ 續抱（rank 1 起算）；h：續抱另需 52 週比 ≥ h；delta：舊持股分數加 delta 後重排（新股要高出 δ 才換）。
    K_=None 且 delta=None ＝ 原版（每月取前 n）。K_ 與 delta 同時給定：先依 K_/h 續抱，其餘名額用加 δ 後的排序補滿。"""
    days = P.days if days is None else days
    out = []; held = []
    for d in days:
        c = P.cs[d]
        order, rank, gu = c['order'], c['rank'], c['gu']
        hs = set(held)
        if K_ is not None or delta is None:
            KK = n if K_ is None else K_
            keep = [j for j in held if j in rank and rank[j] < KK and (h is None or c['s52'][j] >= h)]
            keep = sorted(keep, key=rank.get)[:n]
        else:
            keep = []
        if delta is not None:
            sc = np.round(np.nan_to_num(c['sc'], nan=-9), 10).copy()
            hm = np.zeros(N, bool); hm[list(hs)] = True
            sc = np.where(hm & c['el'], np.round(sc + delta, 10), sc)
            col = np.arange(N)
            fill = np.lexsort((col, -np.nan_to_num(c['y3'], nan=-1e18), -sc))[:int(c['el'].sum())]
        else:
            fill = order
        pick = list(keep); ps = set(pick)
        for j in fill:
            if len(pick) >= n: break
            j = int(j)
            if j in ps: continue
            if gu[j] and j not in hs: continue
            pick.append(j); ps.add(j)
        if keep_log is not None:
            keep_log.append((d, len(set(pick) & hs), len(hs)))
        out.append((c['e'], np.array(pick, int), np.where(c['el'])[0], d))
        held = pick
    return out

# ---------------- 引擎 ----------------
def sim(plan, buy, sell, t0, t1, xrules=None, band=None, cap=None, ret_override=None, log_detail=True):
    """o3_core.simulate 的推廣。
    band：續抱股（換股前已持有、本期仍入選）市值落在目標等權 T0=Vt/k 的 [1−band, 1+band] 內就不動；
          其餘入選股（新進與超出帶外的續抱股）調到共同目標 T（用二分法使現金守恆）。band=None＝全部調回等權（原版）。
    cap：1,000 萬容量情境（帳戶規模 cap 元，固定），每檔每邊另加 2.5%×sqrt(成交額/60 日均成交值)。
    回傳 nav、log（turnover＝賣出額/總值 的平均（不含第一期）；det＝每期拆解）。"""
    G = K.GROW if ret_override is None else ret_override
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; det = []; frozen_ev = []; exits = []
    rel = None
    if xrules:
        rel = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in xrules.items()}
    V60 = _V60()
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            if rel is not None:
                for j in np.where(frz)[0]:
                    if j in rel and rel[j][1] <= t < rel[j][2]:
                        f = rel[j][0]
                        exits.append((str(IDX[t].date()), COLS[j], float(f)))
                        cash += h[j] * f; h[j] = 0.0; frz[j] = False
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            fixed = np.zeros(N, bool)
            if band is not None:
                T0 = Vt / k
                inpk = np.zeros(N, bool); inpk[pk] = True
                fixed = inpk & (htr > 0) & (np.abs(htr / T0 - 1.0) <= band)
            adj = np.zeros(N, bool); adj[pk] = True; adj &= ~fixed
            ka = int(adj.sum())
            if cap is not None:
                T0 = Vt / k
                tg0 = np.where(fixed, htr, 0.0); tg0[adj] = T0
                amt = np.abs(tg0 - htr) / (Vt + Vfz) * cap
                v60 = V60[max(t - 1, 0)]
                with np.errstate(invalid='ignore', divide='ignore'):
                    ex = 0.025 * np.sqrt(np.where(np.isfinite(v60) & (v60 > 0), amt / v60, 0.0))
                bv, sv = buy + ex, sell + ex
            else:
                bv, sv = buy, sell
            base_t = np.where(fixed, htr, 0.0)
            if ka > 0:
                rest = Vt - base_t.sum()
                def cost_of(T):
                    tgt = base_t.copy(); tgt[adj] = T
                    return (bv * np.clip(tgt - htr, 0, None)).sum() + (sv * np.clip(htr - tgt, 0, None)).sum()
                lo, hi = 0.0, max(rest, 0.0) / ka
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if base_t.sum() + mid * ka + cost_of(mid) > Vt: hi = mid
                    else: lo = mid
                T = lo
                tgt = base_t.copy(); tgt[adj] = T
                spent = tgt.sum() + cost_of(T)
                left = Vt - spent       # 二分法殘差（極小）
            else:
                tgt = base_t.copy(); left = Vt - tgt.sum()
            sells = np.clip(htr - tgt, 0, None); buys = np.clip(tgt - htr, 0, None)
            tot = Vt + Vfz
            tov.append(sells.sum() / tot)
            if log_detail:
                exm = (htr > 0) & (tgt == 0); cont = (htr > 0) & (tgt > 0); new = (htr == 0) & (tgt > 0)
                det.append(dict(t=int(t), sell=float(sells.sum() / tot), buy=float(buys.sum() / tot),
                                sell_exit=float(sells[exm].sum() / tot), sell_trim=float(sells[cont].sum() / tot),
                                buy_new=float(buys[new].sum() / tot), buy_top=float(buys[cont].sum() / tot),
                                n_new=int(new.sum()), n_cont=int(cont.sum()), n_fixed=int(fixed.sum()), V=float(tot)))
            newval = np.where(frz, h, 0.0)
            newval[pk] += tgt[pk]
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = max(left, 0.0) if ka == 0 else 0.0
            if frz.any(): frozen_ev.append((str(IDX[t].date()), [COLS[j] for j in np.where(frz)[0]]))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan,
                                                   'tov': tov, 'det': det, 'frozen': frozen_ev, 'exits': exits}

_V60C = {}
def _V60():
    if 'v' not in _V60C: _V60C['v'] = V.rolling_signals('gapfix')[1]
    return _V60C['v']

def bounds(days, end_day):
    """訊號日清單 → (t0, t1, sd)；評價終點＝end_day 收盤。"""
    t0 = IDX.get_loc(days[0]); t1 = IDX.get_loc(end_day)
    return t0, t1, list(days) + [end_day]

def run(P, spec, cost, days, end_day, xr, log_detail=False):
    """spec：dict(K=, h=, delta=, band=, overlap=bool)。cost：(buy, sell) 或 'cap'。回傳 nav, log。"""
    buy, sell = (NORMAL if cost == 'cap' else cost)
    cap = 1e7 if cost == 'cap' else None
    t0, t1, sd = bounds(days, end_day)
    if not spec.get('overlap'):
        pl = planner(P, 10, spec.get('K'), spec.get('h'), spec.get('delta'), days=days)
        nav, lg = sim([(p[0], p[1]) for p in pl], buy, sell, t0, t1, xrules=xr, band=spec.get('band'), cap=cap, log_detail=log_detail)
        lg['plan'] = pl
        return nav, lg
    # 兩期重疊：兩個子組合各半資金，A 在第 0、2、4…個訊號日換股，B 在第 0、1、3、5…個訊號日換股；各自持有 10 檔兩個月
    dA = [d for i, d in enumerate(days) if i % 2 == 0]
    dB = [d for i, d in enumerate(days) if i == 0 or i % 2 == 1]
    plA = planner(P, 10, spec.get('K'), spec.get('h'), spec.get('delta'), days=dA)
    plB = planner(P, 10, spec.get('K'), spec.get('h'), spec.get('delta'), days=dB)
    nA, lA = sim([(p[0], p[1]) for p in plA], buy, sell, t0, t1, xrules=xr, band=spec.get('band'), cap=cap, log_detail=True)
    nB, lB = sim([(p[0], p[1]) for p in plB], buy, sell, t0, t1, xrules=xr, band=spec.get('band'), cap=cap, log_detail=True)
    nav = 0.5 * nA + 0.5 * nB
    # 合併周轉：每個換股日的賣出額 ÷ 合併總值
    ev = {}
    for nv, lg in ((nA, lA), (nB, lB)):
        for x in lg['det']:
            ev.setdefault(x['t'], []).append((x['sell'] * x['V'] * 0.5, x))
    tov = []
    for t in sorted(ev):
        tot = 0.5 * nA.iloc[t - t0] / 1.0 + 0.5 * nB.iloc[t - t0]       # 收盤值近似總值（只用於周轉分母）
        tov.append(sum(a for a, _ in ev[t]) / max(tot, 1e-12))
    return nav, {'turnover': float(np.mean(tov[1:])), 'tov': tov, 'plan': plA + plB, 'det': lA['det'] + lB['det']}

# ---------------- 統計 ----------------
E50 = K.E50
def stats_block(nav, sd, a, b, base=None, ew=None):
    last = min(b, nav.index[-1])
    cg, mdd = R.cagr_mdd(nav, a, last)
    e50 = E50.reindex(nav.index)
    o = dict(CAGR=cg, maxDD=mdd, vs0050=R.tvs(nav, e50, sd, a, b), E50_CAGR=R.cagr_mdd(e50, a, last)[0])
    if ew is not None: o['vsEW'] = R.tvs(nav, ew, sd, a, b)
    if base is not None: o['paired'] = R.paired(nav, base, sd, a, b)
    return o

def window_diff(nav, base, sd, a, b):
    ra, rb = R.win_returns(nav, sd), R.win_returns(base, sd)
    dd = (ra - rb); return dd[(dd.index >= pd.Timestamp(a)) & (dd.index <= pd.Timestamp(b))].dropna()

def drop_best(dd, k):
    x = dd.sort_values(ascending=False).iloc[k:]
    return dict(mean_pp=float(x.mean() * 100), t=R.tstat(x), n=int(len(x)))

def p_two(t):
    if not np.isfinite(t): return float('nan')
    return float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))

def p_one(t):
    if not np.isfinite(t): return float('nan')
    return float(1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))

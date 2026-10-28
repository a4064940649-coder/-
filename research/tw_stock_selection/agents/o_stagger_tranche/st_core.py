#!/usr/bin/env python3
"""o_stagger_tranche 核心：換股日分批錯開（時點運氣、營收新鮮度、單日開盤容量）。

資料＝主策略 V3（總監 o3 報告 3.5 節正式口徑），全部只讀 import：
  * main_v2/v2_core（再 import independent_audit/ia_core）：訊號（gapfix 窗）、正式同分序、排除金融、AB 延後機率表。
  * o_infra_v3_rebuild/o3_core：P（下市股價格 apply_patch）、X（併購出清 exit_rules 'main'）、R_DEL 營收（load_rdel）；sha256 由 o3_core 斷言。
  * o_disposal_rejudge/drj_core：官方處置名單 E1（全部排除；訊號日定義）。
本題自己的部分：
  * 錨定日曆：營收月 m−1 的法定期限 D_m＝第 m 月 10 日起第一個交易日（台股加權指數日曆）；
    特例：2019-01 營收延到 2019-02-15、2023-03 營收延到 2023-04-12（皆取該日起第一個交易日）。
    版本 J{j}：第 m 月的訊號日＝D_m 之後第 j 個交易日（j＝0 即 D_m 當天）；若該日不在面板（或有值檔數 < 300），該月跳過（續抱）。
    成交日＝訊號日之後第一個「有值檔數 ≥ 300」的面板日開盤。
  * AB（只用訊號日收盤前確定已公告的營收）：
      訊號日 < D_m：m−1 月營收未公告 → 改用 m−2..m−4 的 3 月平均年增率（＝最近已公告值）。
      訊號日 = D_m：依 rev_announce 的產業延後機率 p 抽樣（seed 決定）；金融 p=1。
      2020-01 營收（疫情可延到 2/15）：D_m < 訊號日 ≤ 2020-02-14 也視同 D=S 抽樣。
      2026-01～08 營收：INS15 名單期限為 15 日（< 期限未公告；= 期限抽樣）。
      2021 年後 rev_5y 沒有逐檔公告日（rev_hist 的 ann_date 只有 2026 年 10.6%），所以全期都用上面的規則。
  * 分批（tranche）：k 批各持 1/k 資金、各自在自己的 j 日換成前 10 名（持有到下個月同一個 j 日）。
      a0＝其他批持有的股票照樣買（重複）；a1＝跳過其他批目前持有的股票，改買下一名。
      b0＝每批完整重排（AB 營收＝最近已公告值）；b1＝j>1 的批次營收腿百分位凍結在同一循環 j=+1 那批的橫斷面（只更新價格腿）。
      c0＝批與批之間不再平衡（自然漂移）；c1＝每月官方換股日（成交日開盤）把各批調回等額（只對淨買賣計成本）。
引擎：tsim() 是 o3_core.simulate 的推廣（k 批共用帳戶、c1 調回等額）；k=1 時與 o3_core.simulate 逐日相同（validate 斷言）。
"""
import sys, json, math, hashlib, zlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
sys.path.insert(0, str(AG / 'o_disposal_rejudge'))
import drj_core as R          # noqa: E402  只讀 import
sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_infra_v3_rebuild'))
import o3_core as O3          # noqa: E402  只讀 import
sys.path.pop(0)
K, V = R.K, R.V
assert O3.K is K and O3.V is V
IDX, COLS, N = K.IDX, K.COLS, K.N

FEE, TAX = K.FEE, 0.003
NORMAL = K.rates(0.001)                      # 買 0.1855%、賣 0.4855%
STRESS = (NORMAL[0] + 0.003, NORMAL[1] + 0.003)   # 一般每邊再加 0.3%
FLAT = K.rates('stress')                     # BRIEF 壓力：買賣各 0.5%（來回 1%）
GROSS = (0.0, 0.0)
def rates_slip(s): return (FEE + s, FEE + TAX + s)

def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()

# ---------------- 期間（以錨定月 m 分段） ----------------
P = lambda s: pd.Period(s, 'M')
PERIODS = {'SEL': (P('2006-01'), P('2018-12')), 'VAL': (P('2019-01'), P('2023-12')), 'HOLD': (P('2024-01'), P('2026-10')),
           'POST': (P('2019-01'), P('2026-10')), 'ALL': (P('2006-01'), P('2026-10')),
           'S1': (P('2006-01'), P('2013-12')), 'S2': (P('2014-01'), P('2020-12')), 'S2a': (P('2014-01'), P('2018-12')),
           'S3': (P('2021-03'), P('2026-10')), 'S4': (P('2024-01'), P('2026-10')), 'Y2008': (P('2008-01'), P('2008-12'))}
SEL_KEYS = ['SEL', 'S1', 'S2a', 'Y2008']

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

def e1_mask(d):
    DS = setup()['DS']
    m = np.zeros(N, bool)
    for j, info in DS.at(d).items():
        if R.excl_rule('E1', info): m[j] = True
    return m

def official_S(unavail=None):
    """主策略 V3 正式物件（官方日曆、R_DEL、E1）。"""
    st = setup()
    S = V.V2('gapfix', st['YRD'], st['Y50'], exfin=True, unavail=unavail)
    orig = S.cross_section
    def cs(d, _o=orig):
        old = K.disposal_mask; K.disposal_mask = e1_mask
        try: return _o(d)
        finally: K.disposal_mask = old
    S.cross_section = cs
    return S

# ---------------- 日曆 ----------------
_TX = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv', parse_dates=['date'])['date']
TCAL = pd.DatetimeIndex(sorted(set(_TX)))
GOODSET = set(IDX[K.GOOD])
GPOS = np.where(K.GOOD)[0]
def next_good(t):
    k = np.searchsorted(GPOS, t, side='right')
    return int(GPOS[k]) if k < len(GPOS) else None

SPECIAL_DL = {'2019-02': pd.Timestamp('2019-02-15'), '2023-04': pd.Timestamp('2023-04-12')}   # 錨定月 → 延長期限
def first_td(ts):
    k = TCAL.searchsorted(ts); return TCAL[k] if k < len(TCAL) else None

def deadline(m):
    """錨定月 m（營收月 m−1）的法定期限（交易日）。"""
    s = str(m)
    if s in SPECIAL_DL: return first_td(SPECIAL_DL[s])
    return first_td(pd.Timestamp(m.year, m.month, 10))

def deadline15(m):
    return first_td(pd.Timestamp(m.year, m.month, 15))

def day_for(m, j):
    D = deadline(m)
    if D is None: return None
    k = TCAL.get_loc(D) + j
    if k < 0 or k >= len(TCAL): return None
    s = TCAL[k]
    if s not in GOODSET: return None
    t = IDX.get_loc(s)
    if next_good(t) is None: return None
    return s

# 錨定月清單：與主策略 V3 官方訊號日相同的月份（V2 的營收可用性與名額條件）
_ANCH = {}
def anchors():
    if 'a' in _ANCH: return _ANCH['a']
    S = official_S()
    a = [pd.Period(d, 'M') for d in S.sig_days]
    _ANCH['a'] = a; _ANCH['off_days'] = list(S.sig_days); _ANCH['off_end'] = S.end_day
    return a

def official_days():
    anchors(); return _ANCH['off_days'], _ANCH['off_end']

# ---------------- AB ----------------
PCOL = V.PCOL; IS15 = V.IS15
def _rng(seed, d, tag=0):
    return np.random.default_rng([int(seed), int(d.strftime('%Y%m%d')), tag])

def unavail_for(m, s, seed=0):
    """回傳 bool 陣列（True＝m−1 月營收在訊號日 s 收盤前未確定公告）。"""
    D = deadline(m)
    u = np.zeros(N, bool)
    if s < D: u[:] = True
    elif s == D or (str(m) == '2020-02' and s <= pd.Timestamp('2020-02-14')):
        u = _rng(seed, s) .random(N) < PCOL
    rm = str(m - 1)
    if '2026-01' <= rm <= '2026-08':
        D15 = deadline15(m)
        if s < D15: u |= IS15
        elif s == D15: u |= IS15 & (_rng(seed, s, 15).random(N) < PCOL)
    return u

# ---------------- 橫斷面 ----------------
_Y3C = {}
def y3_month(m_eff, src):
    """訊號月 m_eff 的 3 月平均年增率（m_eff−1..m_eff−3），排除金融。src：'L'（R_DEL）或 '5'（rev_5y）。"""
    key = (str(m_eff), src)
    if key not in _Y3C:
        st = setup()
        Y = st['Y50'] if src == '5' else st['YRD']
        y = K.yoy3_for_month(Y, m_eff)
        _Y3C[key] = np.where(V.FINMASK, np.nan, y)
    return _Y3C[key]

def src_for(m):
    return '5' if m >= pd.Period(V.SWITCH, 'M') else 'L'

_BASE = {}
def base_day(d):
    """與營收無關的部分：前 300 流動性池、52 週比、處置、隔日可成交、成交日、成交日開盤漲停。"""
    if d in _BASE: return _BASE[d]
    S52, V60 = V.rolling_signals('gapfix')
    t = IDX.get_loc(d); e = next_good(t)
    v60 = V60[t]; px = K.Cn[t]
    base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
    vv = np.where(base, v60, -np.inf)
    order = np.argsort(-vv, kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True
    top &= base
    s52 = S52[t]
    ok = top & np.isfinite(s52) & ~e1_mask(d) & np.isfinite(K.On[e]) & K.TRADED[e]
    b = dict(t=t, e=e, ok=ok, s52=s52, v60=v60, gu=K.gap_up_open(e))
    _BASE[d] = b
    return b

def cross(d, m, seed=0, ab=True, w52=0.5):
    """訊號日 d、錨定月 m 的橫斷面（AB 營收）。回傳 dict(el, sc, y3, prr, pr52, order, ...)。"""
    b = base_day(d)
    src = src_for(m)
    y3 = y3_month(m, src)
    nold = 0
    if ab:
        U = unavail_for(m, d, seed)
        if U.any():
            y3 = np.where(U, y3_month(m - 1, src), y3)
            nold = int((U & b['ok']).sum())
    el = b['ok'] & np.isfinite(y3)
    pr52 = pd.Series(np.where(el, b['s52'], np.nan)).rank(pct=True).to_numpy()
    prr = pd.Series(np.where(el, y3, np.nan)).rank(pct=True).to_numpy()
    sc = w52 * pr52 + (1 - w52) * prr
    sc[~el] = np.nan
    return dict(d=d, m=m, t=b['t'], e=b['e'], el=el, sc=sc, y3=y3, pr52=pr52, prr=prr, s52=b['s52'], v60=b['v60'], gu=b['gu'], nold=nold)

def order_of(sc, y3, el):
    s = np.round(np.nan_to_num(sc, nan=-9), 10)
    col = np.arange(N)
    o = np.lexsort((col, -np.nan_to_num(y3, nan=-1e18), -s))
    return o[:int(el.sum())]

# ---------------- 單一換股日版本 ----------------
def version_days(j, start_m=None, end_m=None):
    """J{j} 的 [(m, d)]，m＝錨定月；另回傳評價終點（2026-10 錨定日，若超出面板則面板最後一日）。"""
    A = anchors()
    out = [(m, day_for(m, j)) for m in A]
    out = [(m, d) for m, d in out if d is not None]
    endm = pd.Period('2026-10', 'M')
    de = day_for(endm, j)
    off_days, off_end = official_days()
    if de is None or de > IDX[-1]:
        de = IDX[-1]
    out = [(m, d) for m, d in out if d < de]
    return out, de

def official_list():
    off_days, off_end = official_days()
    return [(pd.Period(d, 'M'), d) for d in off_days], off_end

def plan_single(days, seed=0, ab=True, w52=0.5, n=10):
    """V2.plan 的同一邏輯（開盤漲停且未持有→遞補）。days=[(m,d)]。回傳 plan=[(e, picks, pool, d, m)]。"""
    out = []; held = set()
    for m, d in days:
        c = cross(d, m, seed, ab, w52)
        el = c['el']
        if el.sum() < 50: continue
        cand = order_of(c['sc'], c['y3'], el)
        pick = []
        for j in cand:
            if c['gu'][j] and j not in held: continue
            pick.append(int(j))
            if len(pick) == n: break
        out.append((c['e'], np.array(pick, int), np.where(el)[0], d, m))
        held = set(pick)
    return out

# ---------------- 分批 ----------------
def plan_tranches(js, a='a0', b='b0', seed=0, n=10, max_m=None):
    """js：各批的 j（第一個必須是 +1，作為 b1 的新鮮排名）。回傳 plans（每批一個 plan 清單）。
    事件依日期排序後依序決定（a1 需要其他批「當時」的持股）。"""
    A = anchors()
    ev = []
    for i, j in enumerate(js):
        days, de = version_days(j)
        for m, d in days:
            if max_m is None or m <= max_m: ev.append((d, i, m))
    ev.sort(key=lambda x: (x[0], x[1]))
    held = [set() for _ in js]
    plans = [[] for _ in js]
    fresh = {}                       # m → j=+1 那批的橫斷面（b1 用）
    for d, i, m in ev:
        c = cross(d, m, seed, True)
        if b == 'b1' and js[i] == 1:
            fresh[m] = c
        el = c['el']
        if el.sum() < 50: continue
        sc, y3 = c['sc'], c['y3']
        if b == 'b1' and js[i] > 1 and m in fresh:
            f = fresh[m]
            # 營收腿凍結：用 j=+1 橫斷面的營收百分位；不在那個池的股票不合格
            el = el & np.isfinite(f['prr'])
            pr52 = pd.Series(np.where(el, c['s52'], np.nan)).rank(pct=True).to_numpy()
            sc = 0.5 * pr52 + 0.5 * f['prr']
            sc = np.where(el, sc, np.nan)
            y3 = np.where(el, f['y3'], np.nan)
            if el.sum() < 50: continue
        cand = order_of(sc, y3, el)
        other = set().union(*[held[q] for q in range(len(js)) if q != i]) if a == 'a1' else set()
        pick = []
        for j in cand:
            j = int(j)
            if j in other: continue
            if c['gu'][j] and j not in held[i]: continue
            pick.append(j)
            if len(pick) == n: break
        plans[i].append((c['e'], np.array(pick, int), np.where(el)[0], d, m))
        held[i] = set(pick)
    return plans

# ---------------- 引擎 ----------------
def tsim(plans, buy, sell, t0, t1, xrules=None, reeq_days=None, log=False):
    """k 批共用帳戶的日頻引擎。每批的規則與 o3_core.simulate 相同（換股日：前收→開盤、凍結、X 出清、二分法成本、開盤→收盤）。
    reeq_days：成交日 index 的集合；當天開盤（在任何批換股之前）把各批的可交易部位與現金等比例縮放，使各批總值相等；
    只對「全帳戶每檔淨買賣」計成本（買 buy、賣 sell），成本依價值比例分攤。所有批都開始後才執行。
    回傳 (nav 總和, 各批 nav DataFrame, log)。初始每批 1/k。"""
    k = len(plans)
    G = K.GROW
    reb = {}
    for i, pl in enumerate(plans):
        for p in pl: reb.setdefault(p[0], []).append((i, p[1], p[3]))
    started = max(pl[0][0] for pl in plans if pl)
    reeq = {t for t in (reeq_days or []) if t > started}
    val = np.zeros((k, N)); cash = np.full(k, 1.0 / k)
    nav = np.empty((t1 - t0 + 1, k)); tov = []; trades = []; frozen_ev = []; exits = []; reeq_cost = []
    rel = None
    if xrules:
        rel = {j: (v[0], int(np.searchsorted(IDX, v[1])), int(np.searchsorted(IDX, v[3])) if len(v) > 3 else 10 ** 9) for j, v in xrules.items()}
    for t in range(t0, t1 + 1):
        if t in reb or t in reeq:
            tr = K.TRADED[t] & np.isfinite(K.On[t])
            gap = np.where(tr, G[t] * K.On[t] / np.where(tr, K.Cn[t], 1.0), 1.0)
            oc = np.where(tr, K.Cn[t] / np.where(tr, K.On[t], 1.0), 1.0)
            act = set(i for (i, _, _) in reb.get(t, []))
            if t in reeq: act = set(range(k))
            H = val.copy()
            for i in act: H[i] = val[i] * gap
            if t in reeq:
                Tot = H.sum() + cash.sum()
                Hn = H.copy(); cn = cash.copy()
                for i in range(k):
                    trd = (H[i] * tr).sum(); fz = H[i].sum() - trd
                    base = trd + cash[i]
                    f = max((Tot / k - fz) / base, 0.0) if base > 0 else 1.0
                    Hn[i] = np.where(tr, H[i] * f, H[i]); cn[i] = cash[i] * f
                net = Hn.sum(axis=0) - H.sum(axis=0)
                cst = buy * np.clip(net, 0, None).sum() + sell * np.clip(-net, 0, None).sum()
                sc = 1 - cst / Tot
                H = Hn * sc; cash = cn * sc
                reeq_cost.append((int(t), float(cst / Tot), float(np.abs(net).sum() / 2 / Tot)))
            newH = H.copy()
            for (i, pk, d) in reb.get(t, []):
                h = H[i].copy()
                frz = (h > 0) & ~tr
                if rel is not None:
                    for j in np.where(frz)[0]:
                        if j in rel and rel[j][1] <= t < rel[j][2]:
                            f = rel[j][0]
                            exits.append((str(IDX[t].date()), i, COLS[j], float(f)))
                            cash[i] += h[j] * f; h[j] = 0.0; frz[j] = False
                Vfz = h[frz].sum()
                htr = np.where(frz, 0.0, h)
                Vt = htr.sum() + cash[i]
                kk = len(pk)
                def cost_of(T):
                    tgt = np.zeros(N); tgt[pk] = T
                    return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
                lo, hi = 0.0, Vt / kk
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if mid * kk + cost_of(mid) > Vt: hi = mid
                    else: lo = mid
                T = lo
                tgt = np.zeros(N); tgt[pk] = T
                tov.append((int(t), i, float(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz)), float((Vt + Vfz))))
                if log:
                    newb = [int(j) for j in pk if htr[j] <= 0]
                    trades.append(dict(t=int(t), i=i, d=d, T=float(T), new=newb))
                nv = np.where(frz, h, 0.0)
                nv[pk] += T
                newH[i] = nv
                cash[i] = 0.0
                if frz.any(): frozen_ev.append((str(IDX[t].date()), i, [COLS[j] for j in np.where(frz)[0]]))
            for i in range(k):
                if i in act: val[i] = newH[i] * oc
                elif t > t0: val[i] = val[i] * G[t]
        else:
            if t > t0: val = val * G[t][None, :]
        nav[t - t0] = val.sum(axis=1) + cash
    navs = pd.DataFrame(nav, index=IDX[t0:t1 + 1])
    tv = pd.DataFrame(tov, columns=['t', 'i', 'tov', 'V'])
    return navs.sum(axis=1), navs, {'tov': tv, 'trades': trades, 'frozen': frozen_ev, 'exits': exits, 'reeq_cost': reeq_cost}

def turnover_month(lg, nav_total, t0):
    """合併帳戶的月單邊周轉：每個錨定循環內各批賣出額 ÷ 帳戶總值，加總後平均（不含第一個循環）。"""
    tv = lg['tov']
    if tv.empty: return np.nan
    s = tv['tov'] * tv['V'] / nav_total.iloc[(tv['t'] - t0 - 1).clip(lower=0)].to_numpy()
    m = pd.Series(s.to_numpy(), index=IDX[tv['t'].to_numpy()].to_period('M'))
    g = m.groupby(level=0).sum()
    return float(g.iloc[1:].mean()) if len(g) > 1 else np.nan

# ---------------- 統計 ----------------
E50 = K.E50
def windows_from(days, end_day):
    """days=[(m,d)] → [(m, d_start, d_end)]。"""
    out = []
    for i, (m, d) in enumerate(days):
        de = days[i + 1][1] if i + 1 < len(days) else end_day
        out.append((m, d, de))
    return out

def wret(nav, W):
    nv = nav.reindex(pd.DatetimeIndex([w[1] for w in W] + [W[-1][2]]))
    a = np.array([nav.get(w[1], np.nan) for w in W]); b = np.array([nav.get(w[2], np.nan) for w in W])
    return pd.Series(b / a - 1, index=pd.PeriodIndex([w[0] for w in W], freq='M'))

def sel(s, key):
    a, b = PERIODS[key]
    return s[(s.index >= a) & (s.index <= b)]

def tstat(x): return R.tstat(np.asarray(x, float))

def cagr_span(nav, W, key):
    w = [x for x in W if PERIODS[key][0] <= x[0] <= PERIODS[key][1]]
    if not w: return np.nan, np.nan
    a, b = w[0][1], w[-1][2]
    s = nav[(nav.index >= a) & (nav.index <= b)]
    yrs = (b - a).days / 365.25
    eq = s / s.iloc[0]
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def block(nav, W, keys, ew=None):
    """nav 在窗 W 上的各期間：CAGR、maxDD、對 0050 月超額均值／標準差／t、追蹤誤差（年化）、對 EW t。"""
    e50 = E50.reindex(nav.index)
    r = wret(nav, W); rb = wret(e50, W)
    ex = (r - rb)
    o = {}
    for k in keys:
        x = sel(ex, k).dropna()
        cg, mdd = cagr_span(nav, W, k)
        cb, _ = cagr_span(e50, W, k)
        d = dict(CAGR=cg, maxDD=mdd, E50_CAGR=cb, ex_mean_pp=float(x.mean() * 100), ex_sd_pp=float(x.std(ddof=1) * 100),
                 TE_ann_pp=float(x.std(ddof=1) * 100 * math.sqrt(12)), t_0050=tstat(x), n=int(len(x)))
        if ew is not None:
            y = sel(r - wret(ew, W), k).dropna(); d['t_EW'] = tstat(y); d['exEW_mean_pp'] = float(y.mean() * 100)
        o[k] = d
    return o

def paired(nav_a, nav_b, W, key):
    d = sel(wret(nav_a, W) - wret(nav_b, W), key).dropna()
    nz = d[d.abs() > 1e-12]
    out = dict(mean_pp=float(d.mean() * 100), t=tstat(d), n=int(len(d)), sd_pp=float(d.std(ddof=1) * 100), n_nonzero=int(len(nz)))
    if len(d) > 3:
        x = d.sort_values(ascending=False)
        out['drop_best1'] = dict(mean_pp=float(x.iloc[1:].mean() * 100), t=tstat(x.iloc[1:]))
        out['drop_worst1'] = dict(mean_pp=float(x.iloc[:-1].mean() * 100), t=tstat(x.iloc[:-1]))
    return out

def endpoint(nav, W, key='ALL'):
    """端點敏感度：對 0050 的 t，拿掉最後 1 月、最好 1／3／5／10 月、最好年份；日曆月 t。"""
    e50 = E50.reindex(nav.index)
    ex = sel(wret(nav, W) - wret(e50, W), key).dropna()
    o = {'t': tstat(ex), 'drop_last1': tstat(ex.iloc[:-1])}
    xs = ex.sort_values(ascending=False)
    for q in (1, 3, 5, 10): o[f'drop_best{q}'] = tstat(xs.iloc[q:])
    yr = ex.groupby(ex.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = tstat(ex[ex.index.year != by])
    o['best_month'] = str(xs.index[0]); o['best_month_pp'] = float(xs.iloc[0] * 100)
    a, b = PERIODS[key]
    s = nav[(nav.index >= W[0][1])]
    me = s.groupby(s.index.to_period('M')).last(); be = e50.reindex(s.index).groupby(s.index.to_period('M')).last()
    cm = (me.pct_change() - be.pct_change()).dropna()
    cm = cm[(cm.index > a) & (cm.index <= b)]
    o['t_cal'] = tstat(cm)
    return o

def p_one(t):
    if not np.isfinite(t): return float('nan')
    return float(1 - 0.5 * (1 + math.erf(t / math.sqrt(2))))

def ew_nav(plan, t0, t1):
    """同池等權（零成本），plan 的 pool 欄。"""
    nav, _ = K.simulate([(p[0], p[2]) for p in plan], 0, 0, t0, t1)
    return nav

def capacity(trades, nav_total, t0, cap=1e7, thr=0.01):
    """1,000 萬（固定）帳戶：新買名額的金額＝cap × T／當日帳戶總值；> thr × 60 日均成交值（訊號日）的比例。"""
    n = 0; big = 0; ratios = []
    for x in trades:
        t = x['t']; tot = nav_total.iloc[t - t0 - 1] if t - t0 - 1 >= 0 else 1.0
        b = base_day(x['d'])
        for j in x['new']:
            v = b['v60'][j]
            if not np.isfinite(v) or v <= 0: continue
            amt = cap * x['T'] / tot
            n += 1; big += amt > thr * v; ratios.append(amt / v)
    r = np.array(ratios)
    return dict(n_slots=n, share_gt=float(big / n) if n else np.nan, med_ratio_pct=float(np.median(r) * 100) if n else np.nan,
                p90_ratio_pct=float(np.percentile(r, 90) * 100) if n else np.nan)

#!/usr/bin/env python3
"""市值匹配選股：橫斷面、骨架（0050 自由流通複製）、三個家族的權重產生器、計畫與淨值。
所有規格常數集中在這裡；PREREG 雜湊涵蓋本檔。"""
import json, math
import numpy as np, pandas as pd
import cmlib as L

IDX, COLS, N, CI = L.IDX, L.COLS, L.N, L.CI
SKEL = json.loads((L.DATA / 'skeleton.json').read_text(encoding='utf-8'))
PHI = pd.read_parquet(L.DATA / 'float_phi.parquet')['phi'].reindex(COLS).to_numpy() if SKEL['float_adjusted'] else np.ones(N)
MEMB_PRE = {pd.Timestamp(k): v for k, v in json.loads((L.DATA / 'membership_pre2015.json').read_text(encoding='utf-8')).items()}
ACTIVE_CAP = 0.05        # 單檔主動權重上限（絕對值，相對 0050 複製）
TSMC_BAND = 0.02         # 台積電主動權重 ±2pp（實作：持有＝複製權重，主動≈0）
S52, V60, SIG = L.rolling_signals()

_MEMC = {}
def members(day):
    day = pd.Timestamp(day)
    if day in _MEMC: return _MEMC[day]
    m = L.tw50_at(day)
    src = 'public'
    if m is None:
        ks = [k for k in MEMB_PRE if k < day]
        m = MEMB_PRE[max(ks)] if ks else []
        src = 'rule'
    _MEMC[day] = (m, src)
    return _MEMC[day]

_CS = {}
def cross_section(d):
    """訊號日 d 收盤的橫斷面。回傳 dict：t、e、mc（市值）、rank（市值名次，1＝最大）、R（0050 複製權重向量）、
    elig（可當衛星）、sc（分數，elig 內百分位）、y3、s52、fin。"""
    d = pd.Timestamp(d)
    if d in _CS: return _CS[d]
    t = IDX.get_loc(d); e = t + 1
    mc = L.mcap_row(t)
    live = L.TRADED[max(t - 5, 0):t + 1].any(axis=0)
    capuni = np.isfinite(mc) & (mc > 0) & live & ~L.DRMASK
    mcu = np.where(capuni, mc, -np.inf)
    order = np.argsort(-mcu, kind='stable')
    rank = np.full(N, 10 ** 9); rank[order] = np.arange(1, N + 1); rank[~capuni] = 10 ** 9
    mem, msrc = members(IDX[e])
    js = np.array([CI[c] for c in mem if c in CI], int)
    fa = mc[js] * PHI[js]; ok = np.isfinite(fa) & (fa > 0)
    R = np.zeros(N); R[js[ok]] = fa[ok] / fa[ok].sum()
    y3 = L.y3_ab(d); s52 = S52[t]; v60 = V60[t]; px = L.Cn[t]
    elig = (capuni & (rank <= 300) & ~L.FINMASK & ~L.FORMASK & np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
            & np.isfinite(s52) & np.isfinite(y3) & ~L.disp_mask(d) & np.isfinite(L.On[e]) & L.TRADED[e])
    def pr(x): return pd.Series(np.where(elig, x, np.nan)).rank(pct=True).to_numpy()
    sc = 0.5 * pr(s52) + 0.5 * pr(y3); sc[~elig] = np.nan
    cs = dict(t=t, e=e, mc=mc, rank=rank, capuni=capuni, R=R, members=js[ok], msrc=msrc, elig=elig, sc=sc, y3=y3, s52=s52, fin=L.FINMASK)
    _CS[d] = cs
    return cs

def score_order(cs, mask):
    """mask 內依分數（四捨五入 1e-10）高→低、同分營收高→低、再代碼序。"""
    idx = np.where(mask & np.isfinite(cs['sc']))[0]
    keys = (idx, -np.nan_to_num(cs['y3'][idx], nan=-1e18), -np.round(cs['sc'][idx], 10))
    return idx[np.lexsort(keys)]

def _waterfill(budget, b, capv):
    """依相對權重 b 分 budget，單檔不超過 capv；觸頂者固定，剩餘按比例再分。回傳權重（總和可能 < budget＝全部觸頂）。"""
    b = np.asarray(b, float); b = b / b.sum(); w = budget * b; fixed = np.zeros(len(b), bool)
    for _ in range(len(b) + 1):
        over = (w > capv + 1e-12) & ~fixed
        if not over.any(): break
        fixed |= over; w[fixed] = capv[fixed]
        free = ~fixed
        if not free.any(): break
        w[free] = (budget - w[fixed].sum()) * b[free] / b[free].sum()
    return w

def _fill(cands, budget, base, R, blocked, held, cap=ACTIVE_CAP, nmin=1):
    """把 budget 依 base 分給 cands 中的前 nmin 檔（開盤漲停且未持有者跳過、由下一名遞補），單檔主動 ≤ cap；
    若全部觸頂仍分不完，就往下加檔。回傳 ({j: w}, 被跳過的開盤漲停檔數)。"""
    if budget <= 1e-9: return {}, 0
    avail = [j for j in cands if not (blocked[j] and j not in held)]
    if not avail: return {}, 0
    k = min(nmin, len(avail))
    while True:
        pick = avail[:k]
        w = _waterfill(budget, [base[j] for j in pick], np.array([R[j] + cap for j in pick]))
        if w.sum() >= budget - 1e-9 or k >= len(avail): break
        k += 1
    last = cands.index(pick[-1]) if pick[-1] in cands else len(cands) - 1
    nblk = sum(1 for j in cands[:last + 1] if blocked[j] and j not in held)
    return {j: float(x) for j, x in zip(pick, w)}, nblk

def weights_F1(cs, K, Nsat, wmode, blocked, held):
    """F1 層替換：前 K 大成分＋全部金融成分照複製權重；其他非持有成分若複製權重 > 5pp，至少保留 R−5pp；
    其餘權重給市值名次 > K 的合格非金融股中分數最高的 Nsat 檔（等權或市值^0.5），單檔主動 ≤ 5pp（觸頂就往下加檔）。"""
    R, mc, rank = cs['R'], cs['mc'], cs['rank']
    mem = cs['members']
    memrank = sorted(mem, key=lambda j: -mc[j])
    H = set(memrank[:K]) | {j for j in mem if cs['fin'][j]}
    w = {j: R[j] for j in H}
    for j in mem:
        if j not in H and R[j] - ACTIVE_CAP > 0: w[j] = R[j] - ACTIVE_CAP
    budget = 1.0 - sum(w.values())
    cands = [j for j in score_order(cs, cs['elig'] & (rank > K)) if j not in H]
    base = (lambda j: 1.0) if wmode == 'eq' else (lambda j: math.sqrt(mc[j]))
    Rres = R.copy()
    for j, x in w.items(): Rres[j] = max(R[j] - x, 0.0)
    sat, nblk = _fill(cands, budget, {j: base(j) for j in cands}, Rres, blocked, held, nmin=Nsat)
    for j, x in sat.items(): w[j] = w.get(j, 0.0) + x
    return w, nblk, len(sat)

def weights_F2(cs, lam, blocked, held):
    """F2 分數傾斜：市值前 100 大（含金融），基礎權重＝自由流通市值；非金融合格股 ×(1+λ(2·百分位−1))（百分位在前 100 大的合格股內），
    負值截零；台積電固定＝複製權重；其餘按比例縮放到 1−R_台積電；單檔主動 ±5pp。"""
    R, mc, rank = cs['R'], cs['mc'], cs['rank']
    U = np.where(rank <= 100)[0]
    base = mc[U] * PHI[U]
    el = cs['elig'][U]
    pct = pd.Series(np.where(el, cs['sc'][U], np.nan)).rank(pct=True).to_numpy()
    tilt = np.where(el, 1 + lam * (2 * np.nan_to_num(pct, nan=0.5) - 1), 1.0)
    raw = base * np.clip(tilt, 0, None)
    raw[U == L.J2330] = 0.0
    wT = R[L.J2330]
    w = np.zeros(N); w[U] = raw / raw.sum() * (1 - wT); w[L.J2330] = wT
    # 開盤漲停且未持有的新增部位：維持原持有量（未持有→0）
    for j in U:
        if blocked[j] and j not in held and w[j] > 0: w[j] = 0.0
    w = _cap_active(w, R, keep=L.J2330)
    return {int(j): float(w[j]) for j in np.where(w > 0)[0]}, 0, int((w > 0).sum())

def _cap_active(w, R, keep):
    """把 |w−R| > 5pp 的部分夾到邊界，差額按比例分攤給其他未觸界的持股（keep 不動）。"""
    w = w.copy()
    for _ in range(50):
        lo = np.clip(R - ACTIVE_CAP, 0, None); hi = R + ACTIVE_CAP
        lo[keep] = hi[keep] = w[keep]
        over = w > hi + 1e-12; under = (w < lo - 1e-12)
        if not over.any() and not under.any(): break
        ex = (w[over] - hi[over]).sum() - (lo[under] - w[under]).sum()
        w[over] = hi[over]; w[under] = lo[under]
        free = (w > 0) & ~over & ~under; free[keep] = False
        if not free.any(): break
        w[free] += ex * w[free] / w[free].sum()
    return w

F3_TIERS = [(2, 10), (11, 50), (51, 150), (151, 300)]
def weights_F3(cs, frac, wmode, blocked, held):
    """F3 層內中性：台積電＝複製權重；其他前 300 大分 4 層（2–10、11–50、51–150、151–300），層權重＝該層自由流通市值占比×(1−R_台積電)；
    層內金融股照市值；非金融部分給層內合格股分數前 frac（至少 1 檔），層內市值加權（wmode='cap'）或等權（'eq'）；單檔主動 ±5pp（觸頂往下加檔）。"""
    R, mc, rank = cs['R'], cs['mc'], cs['rank']
    wT = R[L.J2330]
    fa = np.where(cs['capuni'], mc * PHI, 0.0)
    tot = sum(fa[(rank >= a) & (rank <= b)].sum() for a, b in F3_TIERS)
    w = {L.J2330: wT}; nblk = 0; nsat = 0; Rres = R.copy()
    for a, b in F3_TIERS:
        T = (rank >= a) & (rank <= b)
        tw = fa[T].sum() / tot * (1 - wT)
        finT = T & cs['fin']
        for j in np.where(finT)[0]: w[j] = fa[j] / tot * (1 - wT)
        budget = tw - fa[finT].sum() / tot * (1 - wT)
        for j in np.where(T & ~cs['fin'])[0]:          # 非金融成分若複製權重 > 5pp，至少保留 R−5pp（單檔主動 ±5pp）
            if R[j] - ACTIVE_CAP > 0:
                w[j] = R[j] - ACTIVE_CAP; budget -= w[j]; Rres[j] = R[j] - w[j]
        el = T & cs['elig']
        k = max(1, int(math.ceil(frac * el.sum())))
        cands = list(score_order(cs, el))
        base = {j: (fa[j] if wmode == 'cap' else 1.0) for j in cands}
        sat, nb = _fill(cands, budget, base, Rres, blocked, held, nmin=k)
        nblk += nb; nsat += len(sat)
        for j, x in sat.items(): w[j] = w.get(j, 0.0) + x
    return w, nblk, nsat

VERSIONS = {
    'F1_K1_N10_eq': ('F1', dict(K=1, Nsat=10, wmode='eq')),
    'F1_K1_N20_eq': ('F1', dict(K=1, Nsat=20, wmode='eq')),
    'F1_K5_N10_eq': ('F1', dict(K=5, Nsat=10, wmode='eq')),
    'F1_K5_N20_eq': ('F1', dict(K=5, Nsat=20, wmode='eq')),
    'F1_K10_N10_eq': ('F1', dict(K=10, Nsat=10, wmode='eq')),
    'F1_K10_N20_eq': ('F1', dict(K=10, Nsat=20, wmode='eq')),
    'F1_K5_N20_sqrt': ('F1', dict(K=5, Nsat=20, wmode='sqrt')),
    'F1_K10_N20_sqrt': ('F1', dict(K=10, Nsat=20, wmode='sqrt')),
    'F2_lam1': ('F2', dict(lam=1.0)),
    'F2_lam2': ('F2', dict(lam=2.0)),
    'F3_top30_cap': ('F3', dict(frac=0.3, wmode='cap')),
    'F3_top30_eq': ('F3', dict(frac=0.3, wmode='eq')),
}

def strip_tsmc(cs):
    """「去台積電 0050」骨架：台積電從複製權重、成分與衛星候選中移除，其餘成分權重重新正規化（P5 用）。"""
    c2 = dict(cs); R = cs['R'].copy(); R[L.J2330] = 0.0; R = R / R.sum()
    c2['R'] = R; c2['members'] = np.array([j for j in cs['members'] if j != L.J2330], int)
    el = cs['elig'].copy(); el[L.J2330] = False; c2['elig'] = el
    return c2

def target(cs, version, held):
    fam, kw = VERSIONS[version] if isinstance(version, str) else version
    kw = dict(kw)
    if kw.pop('ex2330', False): cs = strip_tsmc(cs)
    blocked = L.limit_up_open(cs['e'])
    if fam == 'F1': w, nb, ns = weights_F1(cs, blocked=blocked, held=held, **kw)
    elif fam == 'F2': w, nb, ns = weights_F2(cs, blocked=blocked, held=held, **kw)
    elif fam == 'F3': w, nb, ns = weights_F3(cs, blocked=blocked, held=held, **kw)
    elif fam == 'REP': w, nb, ns = {int(j): float(cs['R'][j]) for j in np.where(cs['R'] > 0)[0]}, 0, 0
    else: raise ValueError(fam)
    js = np.array(list(w.keys()), int); ws = np.array(list(w.values()), float)
    ok = ws > 1e-9; js, ws = js[ok], ws[ok]
    return js, ws / ws.sum(), nb, ns

def build_plan(version, days):
    """days：訊號日序列。回傳 (plan{exec_t:(idx,w)}, info list)。"""
    plan = {}; info = []; held = set()
    for d in days:
        cs = cross_section(d)
        js, ws, nb, ns = target(cs, version, held)
        plan[cs['e']] = (js, ws)
        kw = (VERSIONS[version] if isinstance(version, str) else version)[1]
        R = strip_tsmc(cs)['R'] if kw.get('ex2330') else cs['R']
        act = np.zeros(N); act[js] = ws; act = act - R
        info.append(dict(d=str(pd.Timestamp(d).date()), n_hold=int(len(js)), n_sat=ns, n_blocked=nb, w2330=float(act[L.J2330] + R[L.J2330]),
                         act2330=float(act[L.J2330]), max_abs_active=float(np.abs(act).max()), active_share=float(0.5 * np.abs(act).sum()),
                         n_members=int(len(cs['members'])), msrc=cs['msrc'], n_elig=int(cs['elig'].sum())))
        held = set(js.tolist())
    return plan, info

def run(version, days, end_day, cost='normal', G=None):
    plan, info = build_plan(version, days)
    t0 = IDX.get_loc(pd.Timestamp(days[0])); t1 = IDX.get_loc(pd.Timestamp(end_day))
    b, s = L.rates(cost)
    nav, lg = L.simulate_w(plan, b, s, t0, t1, G=G)
    return nav, lg, plan, info

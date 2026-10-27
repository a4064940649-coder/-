#!/usr/bin/env python3
"""o_disposal_rejudge 核心：主策略 v2（main_v2/v2_core.py，只讀 import）＋歷史處置名單的股池版本 E0～E5＋日頻引擎。

* 主策略：v2_core.V2('gapfix', exfin=True)，正式同分序；n10 為主、n30 附報。
* 處置狀態（換股訊號日 d）：公告日 ≤ d 且 start ≤ d ≤ end（＝現行 SOP／v2 程式的定義）。
  剩餘處置天數 rem＝面板交易日 [e, end] 的天數（e＝d 隔日成交日）；L＝該次處置的面板交易日數；
  E3 門檻 thr＝ceil(0.6×L)（10 日制＝6、5 日制＝3）。
* 版本：E0 全部不排除；E1 全部排除（現行 SOP）；E2 只排除第二次以上；E3 只排除 rem ≥ thr；
  E4 不排除，但處置股買進延到處置結束後第一個交易日開盤（期間資金停在 0050；0050 不計買進日與賣出日的隔夜／當日報酬，保守）；
  E5＝E2 ∪ E3 的排除。
* 引擎：改寫 ia_core.simulate，成本可逐檔（壓力＝一般成本＋處置中交易每邊再加 0.5%），並支援 E4 的延後進場。
  與 ia_core.simulate 在純量成本、無延後時逐日相同（validate 斷言）。
"""
import sys, json, math, hashlib
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
MV2 = HERE.parent / 'main_v2'
sys.path.insert(0, str(MV2))
import v2_core as V          # noqa: E402  （只讀 import；v2_core 會再 import independent_audit/ia_core）
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
CI = K.CI
X = K.X

SEL_A, SEL_B = pd.Timestamp('2006-04-01'), pd.Timestamp('2021-08-31')      # 挑選期（訊號日）
VAL_A, VAL_B = pd.Timestamp('2021-09-01'), pd.Timestamp('2023-12-31')      # 驗證期（已看過，只報告）
HOLD_A, HOLD_B = pd.Timestamp('2024-01-01'), pd.Timestamp('2026-10-12')    # 保留期（已看過，只報告）
NEWREG = pd.Timestamp('2026-08-10')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# ---------------- 處置名單 ----------------
def load_list(use_proxy_gapyears=False):
    A = pd.read_csv(HERE / 'disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
    A = A[A.in_panel].copy()
    A['src'] = 'A'
    B = pd.read_csv(X / 'disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    B = B[B.code.isin(set(COLS))]
    ka = set(zip(A.code, A.start))
    Bo = B[[(c, s) not in ka for c, s in zip(B.code, B.start)]].copy()
    Bo = pd.DataFrame({'announce': Bo['date'], 'code': Bo.code, 'start': Bo.start, 'end': Bo.end, 'measure': 'disposal_5y_only', 'kind': 'regular',
                       'regime': 'periodic_match', 'match_min': np.nan, 'nth': np.nan, 'src': 'B'})
    D = pd.concat([A, Bo], ignore_index=True, sort=False)
    if use_proxy_gapyears:
        fz = json.loads((HERE / 'data_freeze.json').read_text(encoding='utf-8'))
        bad = {int(y) for y, v in fz['gate_50pct'].items() if not v['pass50']}
        PX = pd.read_csv(HERE / 'proxy_events.csv', dtype={'code': str}, parse_dates=['start', 'end'])
        PX = PX[PX.start.dt.year.isin(bad) & (PX.start <= pd.Timestamp('2021-09-23'))]
        # 只補真實名單沒有的（同代碼 ±5 日曆日內沒有真實處置）
        bs = {}
        for c, s in zip(D.code, D.start): bs.setdefault(c, []).append(s)
        PX = PX[[not any(abs((s - x).days) <= 5 for x in bs.get(c, [])) for c, s in zip(PX.code, PX.start)]]
        pidx = IDX.searchsorted(PX.start.values)
        ann = IDX[np.maximum(pidx - 1, 0)]
        D = pd.concat([D, pd.DataFrame({'announce': ann, 'code': PX.code.values, 'start': PX.start.values, 'end': PX.end.values, 'measure': 'proxy',
                                        'kind': 'regular', 'regime': 'proxy', 'match_min': np.nan, 'nth': np.nan, 'src': 'P'})], ignore_index=True, sort=False)
    D = D.sort_values(['code', 'start']).reset_index(drop=True)
    # nth 缺值（disposal_5y 獨有、近似）：同代碼前一次處置的 end 在本次 start 前 30 個交易日內 → 2，否則 1
    pos = {d: i for i, d in enumerate(IDX)}
    def ti(x):
        return IDX.searchsorted(x)
    nth = D['nth'].to_numpy(dtype=float).copy()
    for c, g in D.groupby('code'):
        prev_end = None
        for i in g.index:
            if np.isnan(nth[i]):
                if prev_end is not None and ti(D.at[i, 'start']) - ti(prev_end) <= 30: nth[i] = 2
                else: nth[i] = 1
            prev_end = D.at[i, 'end'] if prev_end is None else max(prev_end, D.at[i, 'end'])
    D['nth'] = nth.astype(int)
    s_i = IDX.searchsorted(D.start.values); e_i = IDX.searchsorted(D.end.values, 'right') - 1
    D['i0'] = s_i; D['i1'] = e_i
    D['L'] = np.maximum(e_i - s_i + 1, 1)
    D['match_min'] = np.where(D.match_min.notna(), D.match_min, np.where(D.regime == 'prepay_only', 0, np.where(D.nth >= 2, 20, 5)))
    return D

class Disp:
    """訊號日 d 的處置狀態：dict j → info。"""
    def __init__(self, D):
        self.D = D
        self.cache = {}
        self.by_code = {c: g for c, g in D.groupby('code')}
    def at(self, d):
        if d in self.cache: return self.cache[d]
        t = IDX.get_loc(d); e = t + 1
        x = self.D[(self.D.start <= d) & (self.D.end >= d) & (self.D.announce <= d)]
        out = {}
        for _, r in x.iterrows():
            j = CI[r.code]
            rem = max(int(r.i1) - e + 1, 0)
            info = dict(code=r.code, nth=int(r.nth), rem=rem, L=int(r.L), thr=int(math.ceil(0.6 * int(r.L))), end_i=int(r.i1), start_i=int(r.i0),
                        match_min=float(r.match_min), kind=r.kind, regime=r.regime, src=r.src, measure=r.measure)
            if j in out:   # 重疊（連續處置）：取較晚結束、較高次數
                o = out[j]
                if info['end_i'] > o['end_i']: o.update(rem=info['rem'], end_i=info['end_i'], L=info['L'], thr=info['thr'])
                o['nth'] = max(o['nth'], info['nth'])
            else:
                out[j] = info
        self.cache[d] = out
        return out
    def in_disp_day(self, j, t, known_by=None):
        """第 t 日 j 是否在處置期間（公告日 ≤ known_by）。"""
        g = self.by_code.get(COLS[j])
        if g is None: return None
        kb = IDX[known_by] if known_by is not None else IDX[t]
        x = g[(g.i0 <= t) & (g.i1 >= t) & (g.announce <= kb)]
        return None if x.empty else int(x.i1.max())

VERSIONS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5']

def excl_rule(ver, info):
    if ver in ('E0', 'E4'): return False
    if ver == 'E1': return True
    if ver == 'E2': return info['nth'] >= 2
    if ver == 'E3': return info['rem'] >= info['thr']
    if ver.startswith('E3@'): return info['rem'] >= int(math.ceil(float(ver[3:]) * info['L']))
    if ver == 'E5': return info['nth'] >= 2 or info['rem'] >= info['thr']
    raise ValueError(ver)

_REV = {}
def revenue():
    if not _REV:
        YL, Y5, Y5f = V.load_revenue(); _REV.update(YL=YL, Y5=Y5)
    return _REV['YL'], _REV['Y5']

def trim(S, last):
    """只保留訊號日 ≤ last；評價終點＝之後第一個訊號日（收盤）。"""
    allsd = list(S.sig_days)
    keep = [d for d in allsd if d <= last]
    after = [d for d in allsd if d > last]
    S.sig_days = keep
    S.end_day = after[0] if after else S.end_day
    return S

def make_strategy(ver, DS, exfin=True):
    """回傳 V2 物件；處置遮罩依版本。"""
    YL, Y5 = revenue()
    S = V.V2('gapfix', YL, Y5, exfin=exfin)
    def mask(d, _ver=ver):
        m = np.zeros(N, bool)
        for j, info in DS.at(d).items():
            if excl_rule(_ver, info): m[j] = True
        return m
    S._mask = mask
    # 包一層 cross_section：暫時替換 K.disposal_mask（v2_core 以 K.disposal_mask 取用）
    orig = S.cross_section
    def cs(d, _orig=orig, _S=S):
        old = K.disposal_mask
        K.disposal_mask = _S._mask
        try:
            return _orig(d)
        finally:
            K.disposal_mask = old
    S.cross_section = cs
    return S

# ---------------- 日頻引擎（逐檔成本、E4 延後進場） ----------------
E50 = K.E50
G50 = np.r_[1.0, (E50 / E50.shift(1)).to_numpy()[1:]]
G50 = np.where(np.isfinite(G50), G50, 1.0)
LT = np.where(K.TRADED.any(axis=0), len(IDX) - 1 - np.argmax(K.TRADED[::-1], axis=0), -1)
B50 = K.FEE + 0.001; S50 = K.FEE + 0.001 + 0.001     # 0050：手續費＋滑價 0.1%＋ETF 證交稅 0.1%

def simulate2(plan, buy, sell, t0, t1, defer=None, extra=None, DS=None):
    """plan=[(e, picks, pool, d)]。buy/sell：純量。extra：處置中交易每邊加的成本（壓力）；需要 DS。
    defer：dict e → {j: 進場日 index}（E4）。回傳 nav、log。"""
    Gs, Cn, On, TR = K.GROW, K.Cn, K.On, K.TRADED
    reb = {p[0]: p for p in plan}
    exe = [p[0] for p in plan]
    val = np.zeros(N); park = np.zeros(N); cash = 0.0 + 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; frozen = []; ndef_enter = 0; ndef_never = 0
    pending = {}        # j → 預定進場日（E4）
    def ex_vec(t, known):
        if extra is None or DS is None: return None
        v = np.zeros(N)
        for j in np.where((val > 0) | known)[0]:
            if DS.in_disp_day(j, t, known_by=t - 1) is not None: v[j] = extra
        return v
    for t in range(t0, t1 + 1):
        if t in reb:
            e, pk, _, d = reb[t]
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, Gs[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            # 0050 停車部位：賣出日不計當日隔夜（保守）
            hp = park.sum()
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + hp * (1 - S50) + cash
            k = len(pk)
            dj = defer.get(t, {}) if defer else {}
            known = np.zeros(N, bool); known[pk] = True
            xv = ex_vec(t, known)
            bvec = np.full(N, buy) + (xv if xv is not None else 0)
            svec = np.full(N, sell) + (xv if xv is not None else 0)
            ndf = sum(1 for j in pk if j in dj)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                for j in dj: tgt[j] = 0.0
                c = (bvec * np.clip(tgt - htr, 0, None)).sum() + (svec * np.clip(htr - tgt, 0, None)).sum()
                return c + B50 * T * ndf
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            for j in dj: tgt[j] = 0.0
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz) if (Vt + Vfz) > 0 else 0)
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            park = np.zeros(N)
            for j in pending: ndef_never += 1
            pending = {}
            for j, te in dj.items():
                park[j] = T          # 當日 0050 買進，不計當日報酬（保守）
                pending[j] = te
            cash = 0.0
            if frz.any(): frozen.append((str(IDX[t].date()), [COLS[j] for j in np.where(frz)[0]], float(Vfz / (Vt + Vfz))))
        else:
            if t > t0:
                val = val * Gs[t]
                # 延後進場：今日開盤賣 0050（不計今日隔夜）、買股票
                if pending:
                    done = []
                    gu = None
                    for j, te in pending.items():
                        if t < te: continue
                        if not (TR[t, j] and np.isfinite(On[t, j])): continue
                        if gu is None: gu = K.gap_up_open(t)
                        if gu[j]: continue                     # 開盤漲停買不到 → 隔日再試
                        if DS is not None:
                            ee = DS.in_disp_day(j, t, known_by=t - 1)
                            if ee is not None:                 # 又進新的處置（已公告）→ 延到其結束後
                                pending[j] = ee + 1; continue
                        amt = park[j] * (1 - S50)
                        bj = buy + (0.0 if extra is None else 0.0)
                        v0 = amt / (1 + bj)
                        val[j] = v0 * Cn[t, j] / On[t, j]      # 開盤買進 → 收盤（val 先前為 0）
                        park[j] = 0.0; done.append(j); ndef_enter += 1
                    for j in done: pending.pop(j)
                park = park * G50[t] if pending else park
                # 已進場者今日以外的 0050 停車部位：上式已經套用；當日進場者 park 已歸零
        nav[t - t0] = val.sum() + park.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan, 'frozen': frozen,
                                                 'defer_entered': ndef_enter, 'defer_never': ndef_never}

def defer_map(plan, DS):
    """E4：每個換股日 e，入選股中處置中者 → 預定進場日＝處置結束後第一個面板交易日。"""
    out = {}
    for (e, pk, _, d) in plan:
        info = DS.at(d)
        m = {}
        for j in pk:
            if j in info:
                te = info[j]['end_i'] + 1
                if te > e: m[int(j)] = te
        if m: out[e] = m
    return out

# ---------------- 統計 ----------------
def win_returns(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    nd = nav.reindex(D)
    return (nd.shift(-1) / nd - 1).iloc[:-1]

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3: return float('nan')
    s = x.std(ddof=1)
    return float(x.mean() / (s / math.sqrt(len(x)))) if s > 0 else float('nan')

def paired(nav_a, nav_b, sd, a, b):
    ra, rb = win_returns(nav_a, sd), win_returns(nav_b, sd)
    dd = (ra - rb)
    dd = dd[(dd.index >= pd.Timestamp(a)) & (dd.index <= pd.Timestamp(b))].dropna()
    nz = dd[dd.abs() > 1e-12]
    return dict(mean_pp=float(dd.mean() * 100), t=tstat(dd), n=int(len(dd)), n_nonzero=int(len(nz)), sd_pp=float(dd.std(ddof=1) * 100),
                mean_nonzero_pp=float(nz.mean() * 100) if len(nz) else 0.0, t_nonzero=tstat(nz))

def cagr_mdd(nav, a, b):
    s = nav[(nav.index >= pd.Timestamp(a)) & (nav.index <= pd.Timestamp(b))]
    i0 = nav.index.get_loc(s.index[0])
    base = nav.iloc[i0 - 1] if i0 > 0 else s.iloc[0]
    start = nav.index[i0 - 1] if i0 > 0 else s.index[0]
    yrs = (s.index[-1] - start).days / 365.25
    eq = s / base
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, float((eq / eq.cummax() - 1).min()) * 100

def tvs(nav, bench, sd, a, b):
    ra = win_returns(nav, sd); rb = win_returns(bench, sd)
    dd = (ra - rb); dd = dd[(dd.index >= pd.Timestamp(a)) & (dd.index <= pd.Timestamp(b))].dropna()
    return dict(t=tstat(dd), mean_pp=float(dd.mean() * 100), sd_pp=float(dd.std(ddof=1) * 100), n=int(len(dd)))

def holm(pvals):
    """pvals: dict name→p。回傳 dict name→(p, p_holm)。"""
    items = sorted(pvals.items(), key=lambda x: x[1])
    m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p); run = max(run, adj); out[k] = (p, run)
    return out

def p_from_t(t, n):
    from math import erf, sqrt
    if not np.isfinite(t) or n < 3: return float('nan')
    # 常態近似（樣本數多時夠用；小樣本偏樂觀，報告時註明）
    return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))

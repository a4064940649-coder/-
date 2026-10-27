#!/usr/bin/env python3
"""replicate_o3：獨立複製 o_infra_v3_rebuild（V3＝R＋X＋P＋D_hi）。

只讀 import main_v2/v2_core.py（凍結規格寫明 v2 部分完全照舊、引擎為 v2_core → ia_core）。
R／X／P／D 四個開關、出清規則引擎、統計整理全部自己寫；沒有讀 o_infra_v3_rebuild 的任何 .py。
"""
import sys, json, hashlib, math
from pathlib import Path
import numpy as np, pandas as pd

AG = Path('/home/user/-/research/tw_stock_selection/agents')
O3 = AG / 'o_infra_v3_rebuild'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V
sys.path.pop(0)
K = V.K

SHA = {'delist_prices.parquet': 'bc4526b4424eaf9b0d13dcc378ce1ccf54a3f3a35e57d414101ea587aa6abf04',
       'merger_consideration.csv': '262e92568a0162e99241c16f1de309df3736c670447b436b91d12b6f69cfd8fd',
       'disposal_hist.csv': '20f4f478d1f7756ecac1f364ad3ddf47cb7ab342c2d45a9131610f8031bc2a1b',
       'delist_official.csv': '41f51cb7e1aadbf1d28244f26cb6452f3ede20f2fbb594350848ed8f55a1f2bd'}
RDEL = AG / 'o_rev_survivorship' / 'yoy_long_R_DEL.parquet'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

for f, h in SHA.items():
    assert sha(O3 / f) == h, f
assert sha(RDEL).startswith('3e4b72e4')

DSWITCH = pd.Timestamp('2021-09-24')

# ---------------- P：價格修補 ----------------
PATCHED = False
def apply_patch():
    global PATCHED
    d = pd.read_parquet(O3 / 'delist_prices.parquet')
    ti = K.IDX.get_indexer(pd.DatetimeIndex(d['date']))
    ci = d['code'].map(K.CI).to_numpy()
    assert (ti >= 0).all() and not pd.isna(ci).any()
    ci = ci.astype(int)
    Cn = K.Cn.copy(); On = K.On.copy(); Hn = K.Hn.copy(); Rn = K.Rn.copy(); Vn = K.V.to_numpy().copy()
    assert np.isnan(Cn[ti, ci]).all(), '只填面板原本缺值的格子'
    def fill(A, x):
        cur = A[ti, ci]; m = np.isnan(cur)
        A[ti[m], ci[m]] = x[m]
    fill(Cn, d['close'].to_numpy(float)); fill(On, d['open'].to_numpy(float)); fill(Hn, d['high'].to_numpy(float))
    fill(Rn, d['ret'].to_numpy(float)); fill(Vn, d['value'].to_numpy(float))
    K.Cn, K.On, K.Hn, K.Rn = Cn, On, Hn, Rn
    K.C = pd.DataFrame(Cn, index=K.IDX, columns=K.COLS)
    K.V = pd.DataFrame(Vn, index=K.IDX, columns=K.COLS)
    K.TRADED = np.isfinite(Cn)
    K.GROW = np.where(K.TRADED, 1.0 + np.nan_to_num(Rn, nan=0.0), 1.0)
    K.MYADJ = pd.DataFrame(np.where(K.TRADED, np.cumprod(K.GROW, axis=0), np.nan), index=K.IDX, columns=K.COLS)
    K.NOBS = K.TRADED.sum(axis=1)
    K.GOOD = K.NOBS >= 300
    K.PREVC = K.C.ffill().shift(1).to_numpy()
    V._RS.clear(); K._RS.clear()
    PATCHED = True
    return len(d)

# ---------------- R：長期營收 ----------------
def load_rev(R):
    _, Y5, _ = V.load_revenue()
    if R:
        y = pd.read_parquet(RDEL)
        y.index = pd.PeriodIndex(y.index, freq='M')
        y = y.astype(float)
        YL = y.reindex(pd.period_range(y.index.min(), y.index.max(), freq='M'))
    else:
        YL = K.yoy_from_long()
    return YL, Y5

# ---------------- D：處置 ----------------
DH = pd.read_csv(O3 / 'disposal_hist.csv', dtype={'code': str}, parse_dates=['trigger', 'start', 'end'])
PRE = json.loads((O3 / 'results.json').read_text(encoding='utf-8'))['PREREG_params']
DHI_EXTRA = {pd.Timestamp(k): int(v) for k, v in PRE['D_hi_extra'].items()}

def hist_mask(d):
    m = np.zeros(K.N, bool)
    x = DH[(DH['start'] <= d) & (DH['end'] >= d)]
    for c in x['code']:
        if c in K.CI: m[K.CI[c]] = True
    return m

class S3(V.V2):
    """v2 策略＋處置模式 dmode ∈ {v2, lo, hi, none}。hi_tie：D_hi 額外剔除時 52 週比同分的取法。"""
    def __init__(self, YL, Y5, exfin, dmode, hi_tie='first', extra_pool=None):
        super().__init__('gapfix', YL, Y5, exfin=exfin)
        self.dmode = dmode; self.hi_tie = hi_tie; self.extra_removed = {}
        self.extra_pool = extra_pool   # callable(d, cs_dict) -> 額外放回合格池的布林陣列與營收百分位覆寫（界限用）

    def dmask(self, d):
        if self.dmode == 'none': return np.zeros(K.N, bool)
        if self.dmode == 'v2': return K.disposal_mask(d)
        if d < DSWITCH: return hist_mask(d)
        return K.disposal_mask(d)

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = K.IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        top = np.zeros(K.N, bool); top[order[:min(300, base.sum())]] = True
        top &= base
        y3 = self.y3(d)
        s52 = self.S52[t]
        el = top & np.isfinite(y3) & np.isfinite(s52)
        el &= ~self.dmask(d)
        el &= np.isfinite(K.On[e]) & K.TRADED[e]
        if self.dmode == 'hi' and DHI_EXTRA.get(d, 0) > 0 and d < DSWITCH:
            for _ in range(DHI_EXTRA[d]):
                s = np.where(el, s52, -np.inf)
                mx = s.max()
                cand = np.where(s == mx)[0]
                if self.hi_tie == 'first': j = cand[0]
                elif self.hi_tie == 'last': j = cand[-1]
                else: j = cand[0]
                el[j] = False
                self.extra_removed[d] = self.extra_removed.get(d, []) + [K.COLS[j]]
        pctoverride = None
        if self.extra_pool is not None:
            add, pctval = self.extra_pool(self, d, dict(t=t, e=e, el=el.copy(), top=top, s52=s52, y3=y3))
            if add is not None and add.any():
                el = el | add
                pctoverride = (add, pctval)
        def pr(x):
            return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
        if pctoverride is None:
            sc = 0.5 * pr(s52) + 0.5 * pr(y3)
        else:
            add, pv = pctoverride
            yy = np.where(add, np.nan, y3)
            elr = el & ~add
            ry = pd.Series(np.where(elr, yy, np.nan)).rank(pct=True).to_numpy()
            ry = np.where(add, pv, ry)
            sc = 0.5 * pr(s52) + 0.5 * ry
            y3 = np.where(add, np.inf if pv >= 1 else np.nan_to_num(np.nanmedian(np.where(elr, y3, np.nan))), y3)
        sc[~el] = np.nan
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3, nold=0)
        self._cs[d] = cs
        return cs

# ---------------- X：永久停止交易的部位 ----------------
MC = pd.read_csv(O3 / 'merger_consideration.csv', dtype={'code': str, 'successor': str},
                 parse_dates=['delist_date', 'last_trade', 'pay_date'])
OFF = pd.read_csv(O3 / 'delist_official.csv', dtype={'code': str}, parse_dates=['date'])

class XRules:
    """dead(t)：在換股日 t 沒有成交、且之後（到新證券第一筆成交前）不再成交。
    mode：main／upper（全部最後收盤、終止日後出清，經營惡化也最後收盤）／lower（最後收盤 −5%、延到終止日＋60 天；經營惡化歸零）。"""
    def __init__(self, mode='main', failure_by='csv', pay_ge=True):
        self.mode = mode; self.pay_ge = pay_ge
        TR = K.TRADED
        T = len(K.IDX)
        self.lt = np.full(K.N, -1); self.dead_until = np.full(K.N, T + 10)
        anyt = TR.any(axis=0)
        self.lt[anyt] = T - 1 - np.argmax(TR[::-1], axis=0)[anyt]
        self.term = {}
        mc = MC.set_index('code')
        for c in K.COLS:
            j = K.CI[c]
            if c in mc.index:
                self.term[c] = mc.at[c, 'delist_date']
            else:
                o = OFF[OFF['code'] == c]
                if len(o): self.term[c] = o['date'].max()
        # 代碼重用：舊證券在終止日前最後一筆下市；規則只在新證券第一筆成交前有效
        for c in ('2301', '2432'):
            j = K.CI[c]; td = OFF[OFF['code'] == c]['date'].max()
            ti = K.IDX.searchsorted(td)
            pre = np.where(TR[:ti, j])[0]
            post = np.where(TR[ti:, j])[0]
            self.lt[j] = pre[-1] if len(pre) else -1
            self.dead_until[j] = ti + post[0] if len(post) else T + 10
            self.term[c] = td
        self.mc = mc
        self.failure_by = failure_by
        self.events = []

    def is_dead(self, t):
        return (self.lt < t) & (t < self.dead_until)

    def action(self, j, t):
        """回傳 None（保持凍結）或乘數（轉為現金）。"""
        c = K.COLS[j]; d = K.IDX[t]
        typ = self.mc.at[c, 'type'] if c in self.mc.index else 'unknown'
        term = self.term.get(c)
        if term is None or pd.isna(term): term = K.IDX[self.lt[j]]
        lastc = K.Cn[self.lt[j], j]
        if self.mode == 'upper':
            return 1.0 if d > term else None
        if self.mode == 'lower':
            if typ == 'failure':
                lt_d = self.mc.at[c, 'last_trade'] if (c in self.mc.index and self.failure_by == 'csv') else K.IDX[self.lt[j]]
                return 0.0 if d > lt_d else None
            return 0.95 if d > term + pd.Timedelta(days=60) else None
        if typ == 'failure':
            lt_d = self.mc.at[c, 'last_trade'] if (c in self.mc.index and self.failure_by == 'csv') else K.IDX[self.lt[j]]
            return 0.0 if d > lt_d else None
        if typ == 'cash' and c in self.mc.index and np.isfinite(self.mc.at[c, 'cash']):
            pay = self.mc.at[c, 'pay_date']
            if pd.isna(pay): pay = term + pd.Timedelta(days=14)
            return float(self.mc.at[c, 'cash'] / lastc) if (d > pay or (self.pay_ge and d >= pay)) else None
        return 1.0 if d > term else None


def simulate(plan, buy, sell, t0, t1, xr=None):
    """ia_core.simulate（freeze=True）的重寫＋X 規則。回傳 nav, log。"""
    G = K.GROW; TR = K.TRADED; On = K.On; Cn = K.Cn; N = K.N
    reb = {p[0]: p[1] for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []; frozen_ev = []; conv = []
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            if xr is not None and frz.any():
                dead = frz & xr.is_dead(t)
                for j in np.where(dead)[0]:
                    a = xr.action(j, t)
                    if a is not None:
                        conv.append((str(K.IDX[t].date()), K.COLS[j], float(a), float(h[j])))
                        cash += h[j] * a; h[j] = 0.0; frz[j] = False
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            T = lo
            tgt = np.zeros(N); tgt[pk] = T
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0)
            newval[pk] += T
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
            if frz.any(): frozen_ev.append((str(K.IDX[t].date()), [K.COLS[j] for j in np.where(frz)[0]], float(Vfz / (Vt + Vfz))))
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=K.IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan,
                                                     'frozen': frozen_ev, 'conv': conv}

def ew_plan(plan):
    return [(e, pool, pool, d) for (e, pk, pool, d) in plan]

def stats_all(nav, ew, sd):
    out = {}
    for s, (a, b, ad) in V.SEGS.items():
        out[s] = V.seg_stats(nav, ew, a, b, sd, ad=ad)
    return out

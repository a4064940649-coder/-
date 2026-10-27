#!/usr/bin/env python3
"""o_capacity_tsmc_gap 核心：只讀 import main_v2/v2_core.py（→ independent_audit/ia_core.py）。
自己實作：名次範圍股池（V2R）、入選事件的下月淨報酬與超額、加權日頻引擎（台積電固定權重）、分段統計、月窗序列。"""
import sys, json, math, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'main_v2'))
import v2_core as V
sys.path.pop(0)
import numpy as np, pandas as pd
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
J2330 = K.CI['2330']
assert hashlib.md5((K.P / 'value.parquet').read_bytes()).hexdigest().startswith('8272e253'), 'value.parquet 不是修正版'

YL, Y5, _ = V.load_revenue()
NORMAL = K.rates(0.001)
STRESS = K.rates('stress')
SPLIT = pd.Timestamp('2019-04-11')        # 2019-04 換股日：SEL 窗起點 < SPLIT，VAL 窗起點 ≥ SPLIT
END = V.END1012
BUCKETS = [(1, 50), (51, 100), (101, 200), (201, 300)]
BNAME = ['1-50', '51-100', '101-200', '201-300']
UNIV = {'U300': (1, 300), 'U51_300': (51, 300), 'U101_300': (101, 300), 'U100': (1, 100)}

def bucket_of(rank):
    rank = np.asarray(rank)
    out = np.full(rank.shape, -1)
    for b, (lo, hi) in enumerate(BUCKETS):
        out[(rank >= lo) & (rank <= hi)] = b
    return out

class V2R(V.V2):
    """v2 排除金融、gapfix；股池改成流動性名次 [lo, hi]（名次在 base＝均值 ≥1,000 萬且收盤 ≥10 內排）。"""
    def __init__(self, lo=1, hi=300, exfin=True):
        super().__init__('gapfix', YL, Y5, exfin=exfin)
        self.lo, self.hi = lo, hi

    def cross_section(self, d):
        if d in self._cs: return self._cs[d]
        t = IDX.get_loc(d); e = t + 1
        v60 = self.V60[t]; px = K.Cn[t]
        base = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10)
        vv = np.where(base, v60, -np.inf)
        order = np.argsort(-vv, kind='stable')
        nb = int(base.sum())
        rank = np.full(N, 10 ** 6); rank[order[:nb]] = np.arange(1, nb + 1)
        top = base & (rank >= self.lo) & (rank <= self.hi)
        y3 = self.y3(d)
        s52 = self.S52[t]
        el = top & np.isfinite(y3) & np.isfinite(s52)
        el &= ~K.disposal_mask(d)
        el &= np.isfinite(K.On[e]) & K.TRADED[e]
        def pr(x):
            return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
        sc = 0.5 * pr(s52) + 0.5 * pr(y3)
        sc[~el] = np.nan
        cs = dict(t=t, e=e, el=el, sc=sc, s52=s52, y3=y3, nold=0, rank=rank, v60=v60)
        self._cs[d] = cs
        return cs

# ---------------- 加權日頻引擎（ia_core.simulate 的加權版；w 全部相等時與原版相同） ----------------
def simulate_w(plan, buy, sell, t0, t1, G=None):
    """plan=[(exec_t, idx ndarray, weight ndarray(和為 1))]。其餘邏輯（凍結、二分法成本、買進日 close/open）照 ia_core.simulate。"""
    G = K.GROW if G is None else G
    reb = {p[0]: (np.asarray(p[1], int), np.asarray(p[2], float)) for p in plan}
    val = np.zeros(N); cash = 1.0
    nav = np.empty(t1 - t0 + 1); tov = []
    TR, On, Cn = K.TRADED, K.On, K.Cn
    for t in range(t0, t1 + 1):
        if t in reb:
            pk, w = reb[t]
            tr = TR[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum()
            htr = np.where(frz, 0.0, h)
            Vt = htr.sum() + cash
            wv = np.zeros(N); np.add.at(wv, pk, w)
            def cost_of(T):
                tgt = wv * T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            tgt = wv * lo
            tov.append(np.clip(htr - tgt, 0, None).sum() / (Vt + Vfz))
            newval = np.where(frz, h, 0.0) + tgt
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc)
            cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), {'turnover': float(np.mean(tov[1:])) if len(tov) > 1 else np.nan}

def tsmc_plan(plan, w):
    out = []
    for p in plan:
        pk = np.asarray(p[1], int); k = len(pk)
        idx = np.concatenate([pk, [J2330]]) if w > 0 else pk
        wt = np.concatenate([np.full(k, (1 - w) / k), [w]]) if w > 0 else np.full(k, 1.0 / k)
        out.append((p[0], idx, wt))
    return out

# ---------------- 統計 ----------------
def segstat(nav, ew, sd, a, b, ad, bd, bench=None):
    """日頻年化／回撤：(ad, bd]；訊號窗 t：窗起點在 [a, b]。bench 預設 0050 含息。"""
    bench = K.E50 if bench is None else bench
    o = K.stats(nav, ad, bd, sd, bench, ew)
    ow = K.stats(nav, a, b, sd, bench, ew)
    oe = K.stats(ew, ad, bd, sd)
    # 月窗序列（追蹤誤差、資訊比率）
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    m = (nav.reindex(D).shift(-1) / nav.reindex(D) - 1).iloc[:-1]
    bm = (bench.reindex(D).shift(-1) / bench.reindex(D) - 1).iloc[:-1]
    sel = (m.index >= pd.Timestamp(a)) & (m.index <= pd.Timestamp(b))
    x = (m - bm)[sel].dropna()
    te = float(x.std(ddof=1) * math.sqrt(12)); ir = float(x.mean() * 12 / te) if te > 0 else np.nan
    return {'CAGR': round(o['CAGR'] * 100, 2), 'maxDD': round(o['maxDD'] * 100, 2), 'E50_CAGR': round(o['0050_CAGR'] * 100, 2),
            'EW_CAGR': round(oe['CAGR'] * 100, 2), 'ex_0050_pp': round((o['CAGR'] - o['0050_CAGR']) * 100, 2),
            't_0050': round(ow['t_0050'], 3), 't_EW': round(ow['t_EW'], 3), 'n_m': ow['n_m'], 'TE_ann': round(te * 100, 2), 'IR': round(ir, 3),
            'mean_ex_0050_m': round(float(x.mean()) * 100, 3), 'yearly': o['yearly']}

def periods(sd):
    """回傳 {名稱: (a, b, ad, bd)}；sd＝訊號日（含終點）。"""
    lastw = [d for d in sd if d < SPLIT][-1]           # SEL 最後一窗起點
    return {
        'SEL': ('2006-04-01', str(lastw.date()), '2006-04-01', str(SPLIT.date())),
        'VAL': (str(SPLIT.date()), '2026-10-11', str((SPLIT + pd.Timedelta(days=1)).date()), '2026-10-12'),
        'VAL_A': (str(SPLIT.date()), '2023-12-31', str((SPLIT + pd.Timedelta(days=1)).date()), '2023-12-31'),
        'HOLD': ('2024-01-01', '2026-10-11', '2024-01-01', '2026-10-12'),
        'S1': ('2006-04-01', '2013-12-31', '2006-04-01', '2013-12-31'),
        'S2': ('2014-01-01', '2020-12-31', '2014-01-01', '2020-12-31'),
        'S3': ('2021-03-01', '2026-10-11', '2021-03-12', '2026-10-12'),
        'ALL': ('2006-04-01', '2026-10-11', '2006-04-01', '2026-10-12'),
    }
SEL_ONLY = ('SEL',)

def window_returns(nav, sd):
    D = pd.DatetimeIndex([d for d in sd if d in nav.index])
    return (nav.reindex(D).shift(-1) / nav.reindex(D) - 1).iloc[:-1]

def paired_t(a, b, lo, hi):
    x = (a - b).dropna(); x = x[(x.index >= pd.Timestamp(lo)) & (x.index <= pd.Timestamp(hi))]
    return {'mean_pp': round(float(x.mean()) * 100, 3), 't': round(K.tstat(x), 3), 'n': int(len(x))}

# ---------------- 入選事件 ----------------
def event_table(S, plan, cost=NORMAL):
    """每個（換股期 × 入選股）一列：訊號日、代碼、名次、組、gross 期報酬、淨報酬（新買進扣買進成本、下期賣出扣賣出成本）、
    同池等權期報酬、同組等權期報酬、20 日均成交值。期報酬＝成交日開盤 → 下一成交日開盤（最後一期到 END 收盤）。"""
    t_end = IDX.get_loc(END)
    _, per = V.stock_contrib(plan, t_end)
    pool_plan = [(p[0], p[2], None, p[3]) for p in plan]
    _, perp = V.stock_contrib(pool_plan, t_end)
    pmap = {d: (pk, r) for d, pk, r in perp}
    bb, ss = cost
    rows = []
    picks_by = [set(p[1].tolist()) for p in plan]
    Vn = K.V.to_numpy()
    for i, (d, pk, r) in enumerate(per):
        cs = S.cross_section(d)
        t = cs['t']
        prev = picks_by[i - 1] if i > 0 else set()
        nxt = picks_by[i + 1] if i + 1 < len(picks_by) else None
        ppk, pr = pmap[d]
        prank = cs['rank'][ppk]; pb = bucket_of(prank)
        ew_all = float(np.mean(pr))
        ew_b = {b: float(np.mean(pr[pb == b])) if (pb == b).any() else np.nan for b in range(4)}
        adv20 = np.nanmean(Vn[t - 19:t + 1, pk], axis=0)
        for j, x, a20 in zip(pk, r, adv20):
            rk = int(cs['rank'][j]); b = int(bucket_of([rk])[0])
            new = j not in prev
            ext = (nxt is not None) and (j not in nxt)
            net = (1 + x) * (1 - bb * new) * (1 - ss * ext) - 1
            rows.append({'d': d, 'code': COLS[j], 'rank': rk, 'bucket': b, 'gross': float(x), 'net': float(net), 'new': bool(new),
                         'ew_pool': ew_all, 'ew_bucket': ew_b[b], 'adv20': float(a20), 'v60': float(cs['v60'][j]),
                         'score': float(cs['sc'][j])})
    df = pd.DataFrame(rows)
    df['ex_pool'] = df['net'] - df['ew_pool']
    df['ex_bucket'] = df['net'] - df['ew_bucket']
    return df

def group_summary(df, col='ex_pool'):
    out = {}
    for b in range(4):
        x = df.loc[df['bucket'] == b, col].dropna()
        if len(x) == 0: out[BNAME[b]] = {'n': 0}; continue
        q = np.quantile(x, 0.95)
        top = x[x >= q]
        out[BNAME[b]] = {'n': int(len(x)), 'mean_pp': round(float(x.mean()) * 100, 3), 'median_pp': round(float(x.median()) * 100, 3),
                         'win': round(float((x > 0).mean()), 3), 'top5_share_of_sum': round(float(top.sum() / x.sum()), 3) if x.sum() != 0 else None,
                         'mean_ex_top5_pp': round(float(x[x < q].mean()) * 100, 3),
                         't_month': round(K.tstat(x.groupby(df.loc[x.index, 'd']).mean()), 3)}
    return out

def mid_minus_large(df, col='ex_pool', lo=None, hi=None):
    """月序列：101～300 組平均 − 1～100 組平均（兩組都有的月份）。"""
    x = df if lo is None else df[(df['d'] >= pd.Timestamp(lo)) & (df['d'] <= pd.Timestamp(hi))]
    g = x.assign(mid=x['bucket'] >= 2).groupby(['d', 'mid'])[col].mean().unstack()
    if True not in g.columns or False not in g.columns: return pd.Series(dtype=float)
    return (g[True] - g[False]).dropna()

def mml_stats(s):
    return {'mean_pp': round(float(s.mean()) * 100, 3), 't': round(K.tstat(s), 3), 'n_months': int(len(s)),
            'win': round(float((s > 0).mean()), 3) if len(s) else None}

def clustered_reg(df, col='ex_pool'):
    """事件層級：ex = a + b·1[101～300]，月份群聚標準誤。"""
    y = df[col].to_numpy(); xm = (df['bucket'] >= 2).to_numpy().astype(float)
    X = np.column_stack([np.ones(len(y)), xm])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for _, idx in df.groupby('d').indices.items():
        s = X[idx].T @ e[idx]; meat += np.outer(s, s)
    G = df['d'].nunique(); n = len(y)
    c = G / (G - 1) * (n - 1) / (n - 2)
    V_ = c * XtX_inv @ meat @ XtX_inv
    return {'b_mid_pp': round(beta[1] * 100, 3), 't_cluster': round(beta[1] / math.sqrt(V_[1, 1]), 3), 'n_events': int(n), 'n_months': int(G)}

def dumps(o):
    return json.dumps(o, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))

def ols(y, X):
    """y、X（DataFrame），含截距；回傳係數、t、R²。"""
    Z = np.column_stack([np.ones(len(y)), X.to_numpy()])
    b = np.linalg.lstsq(Z, y.to_numpy(), rcond=None)[0]
    e = y.to_numpy() - Z @ b
    s2 = e @ e / (len(y) - Z.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Z.T @ Z)))
    r2 = 1 - (e @ e) / (((y - y.mean()) ** 2).sum())
    names = ['alpha'] + list(X.columns)
    return {'coef': {k: round(float(v), 4) for k, v in zip(names, b)}, 't': {k: round(float(v), 2) for k, v in zip(names, b / se)},
            'R2': round(float(r2), 3), 'n': int(len(y)), 'resid': pd.Series(e, index=y.index)}

TSMC_IDX = pd.Series(np.where(K.TRADED[:, J2330], np.cumprod(K.GROW[:, J2330]), np.nan), index=IDX).ffill()

def tsmc_weight_proxy(S, sig, y0, y1, t_max=None):
    from scipy.optimize import nnls
    E = K.E50.pct_change().to_numpy()
    out = {}
    for y in range(y0, y1 + 1):
        ds = [d for d in sig if d.year == y]
        if not ds: continue
        cs = S.cross_section(ds[0])
        idx = np.where(cs['rank'] <= 70)[0]
        tt = np.where(IDX.year == y)[0]
        if t_max is not None: tt = tt[tt <= t_max]
        R = K.GROW[np.ix_(tt, idx)] - 1; e = E[tt]
        ok = np.isfinite(e) & np.isfinite(R).all(axis=1)
        b, _ = nnls(R[ok], e[ok])
        out[int(y)] = {'w2330': round(float(b[list(idx).index(J2330)]), 3), 'sum_w': round(float(b.sum()), 3), 'n_days': int(ok.sum())}
    return out

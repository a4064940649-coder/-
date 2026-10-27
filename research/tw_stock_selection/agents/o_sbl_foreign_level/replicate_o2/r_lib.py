"""複製 o2 共用：主策略股池（main_v2/v2_core 正式版 gapfix、排除金融）＋自建外資持股比率／市值，產生 B1、C2 計畫。
只 import 研究中心共用基礎（main_v2/v2_core → independent_audit/ia_core），不 import 研究員 o_sbl_foreign_level 的任何程式。"""
import sys, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
sys.path.insert(0, str(AG / 'main_v2'))
import numpy as np, pandas as pd
import v2_core as V
sys.path.pop(0)
K = V.K
IDX, COLS, N = K.IDX, K.COLS, K.N
FEE = K.FEE
START = pd.Timestamp('2021-10-01')
END = V.END1012

FO = pd.read_parquet(HERE / 'my_fo.parquet').reindex(columns=COLS)
FOn = FO.to_numpy()
ANC = pd.read_csv(HERE / 'my_anchor.csv', dtype={'code': str}).set_index('code')['shares'].reindex(COLS).to_numpy()

def fo_at(d, lagdays=0):
    """訊號日 d：嚴格早於 d 的最後一筆（t−1 收盤已公布）。lagdays>0：再往前 lagdays 筆（資料自己的日曆）。"""
    i = FO.index.searchsorted(d) - 1 - lagdays
    if i < 0: return np.full(N, np.nan)
    return FOn[i]

def costs(kind):
    b, s = FEE + 0.001, FEE + 0.003 + 0.001
    if kind == 'stress': b += 0.003; s += 0.003
    return b, s

def build():
    f = HERE / 'cs_cache.pkl'
    if f.exists(): return pickle.loads(f.read_bytes())
    YL, Y5, _ = V.load_revenue()
    S = V.V2('gapfix', YL, Y5, exfin=True)
    t0, t1, sd = S.bounds()
    main_plan = S.plan(10, 'official')
    cs = {}
    for d in S.sig_days:
        if d < START: continue
        c = S.cross_section(d)
        if c['el'].sum() < 50: continue
        c = dict(c); c['gu'] = K.gap_up_open(c['e']); c['order'] = S.order(c, 'official')
        cs[d] = c
    out = dict(sig_days=[d for d in S.sig_days if d >= START], end=END, main_plan=main_plan, cs=cs, t0=t0, t1=t1)
    f.write_bytes(pickle.dumps(out))
    return out

def groups(d, c, ng=3):
    """池內（有市值、有外資比率）依近似市值分 ng 組，回傳 (g 陣列 -1=不在, fo, mcap)。"""
    t = c['t']
    fo = fo_at(d); mc = ANC * K.Cn[t]
    ok = c['el'] & np.isfinite(fo) & np.isfinite(mc)
    g = np.full(N, -1)
    idx = np.where(ok)[0]
    r = pd.Series(mc[idx]).rank(method='first').to_numpy()
    g[idx] = np.minimum((r - 1) * ng // len(idx), ng - 1).astype(int)
    return g, fo, mc

def b1_plan(D, per=10, only_group=None):
    plan = []; held = set()
    for d in D['sig_days']:
        c = D['cs'].get(d)
        if c is None: continue
        g, fo, mc = groups(d, c)
        pick = []
        for k in range(3):
            if only_group is not None and k != only_group: continue
            idx = np.where(g == k)[0]
            o = idx[np.lexsort((idx, -fo[idx]))]
            n = 0
            for j in o:
                if c['gu'][j] and j not in held: continue
                pick.append(j); n += 1
                if n == per: break
        plan.append((c['e'], np.array(pick, int), np.where(c['el'])[0], d))
        held = set(pick)
    return plan

def ew_group_plan(D, k):
    out = []
    for d in D['sig_days']:
        c = D['cs'].get(d)
        if c is None: continue
        g, _, _ = groups(d, c)
        out.append((c['e'], np.where(g == k)[0], None, d))
    return out

def c_plan(D, w=0.2, n=10):
    plan = []; held = set()
    for d in D['sig_days']:
        c = D['cs'].get(d)
        if c is None: continue
        el = c['el']
        g, fo, mc = groups(d, c)
        pr = lambda x, m: pd.Series(np.where(m, x, np.nan)).rank(pct=True).to_numpy()
        fp = np.full(N, np.nan)
        for k in range(3):
            m = g == k
            fp = np.where(m, pr(fo, m), fp)
        fp = np.where(el & ~np.isfinite(fp), 0.5, fp)
        sc = (1 - w) / 2 * pr(c['s52'], el) + (1 - w) / 2 * pr(c['y3'], el) + w * fp
        sc[~el] = np.nan
        cc = dict(c); cc['sc'] = sc
        order = V.V2.order(cc, 'official')[:int(el.sum())]
        pick = []
        for j in order:
            if c['gu'][j] and j not in held: continue
            pick.append(j)
            if len(pick) == n: break
        plan.append((c['e'], np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return plan

def sim(plan, kind, t0, t1):
    if kind == 'gross': b = s = 0.0
    else: b, s = costs(kind)
    nav, _ = K.simulate(plan, b, s, t0, t1)
    return nav

"""步驟 9：P4（事前寫死）。主策略 EXFIN n10 gapfix 正式名單持有期內遇到 MSCI 標準／小型、臺灣50 納入生效日 E
（e_k ≤ E < e_{k+1}−1）：配對差＝E 收盤 → e_{k+1} 開盤的報酬 − 同窗其他持股平均；改版＝E 收盤賣出、資金放 0050、額外成本＝權重×0.5%。
主策略名單與日頻引擎沿用共用的 main_v2/v2_core.py、independent_audit/ia_core.py（只讀 import）；事件比對與改版引擎是複製員自己寫的。"""
import sys, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L
AG = L.TOP.parent
sys.path.insert(0, str(AG / 'main_v2'))
import v2_core as V
K = V.K
sys.path.pop(0)

YL, Y5, Y5f = V.load_revenue()
S = V.V2('gapfix', YL, Y5, exfin=True)
pl = S.plan(10, 'official')
t0, t1, sd = S.bounds()
IDX, COLS = K.IDX, K.COLS
CI = {c: j for j, c in enumerate(COLS)}
G = K.GROW; Cn, On = K.Cn, K.On; E50 = K.E50.to_numpy()

e = L.load_events()
inc = e[e.type.isin(['MSCI_STD_ADD', 'MSCI_SC_ADD', 'TW50_ADD']) & e.code.isin(CI.keys())].copy()
inc = inc[inc.eff.isin(IDX)]
inc['tE'] = [IDX.get_loc(x) for x in inc.eff]
ex = [p[0] for p in pl]
rows = []
for k, (tk, pk, _, dsig) in enumerate(pl):
    tn = ex[k + 1] if k + 1 < len(ex) else t1 + 1          # 最後一期：到 t1 收盤（以 t1+1 開盤代替不存在 → 用 t1 收盤）
    last = k + 1 >= len(ex)
    held = {COLS[j]: j for j in pk}
    m = inc[(inc.tE >= tk) & (inc.tE < tn - 1) & inc.code.isin(held.keys())]
    for r in m.itertuples():
        tE = int(r.tE)
        if not K.TRADED[tE, CI[r.code]]: continue
        def R(j):
            g = np.prod(G[tE + 1:tn + 1, j]) if not last else np.prod(G[tE + 1:t1 + 1, j])
            if not last and K.TRADED[tn, j] and np.isfinite(On[tn, j]):
                g = g * On[tn, j] / Cn[tn, j]
            return g - 1
        ri = R(CI[r.code]); rp = [R(j) for c, j in held.items() if c != r.code]
        r50 = (E50[tn - 1] if not last else E50[t1]) / E50[tE] - 1
        rows.append({'code': r.code, 'name': r.name, 'type': r.type, 'eff': r.eff, 'sig': dsig, 'tE': tE, 'R_i': ri, 'R_peers': float(np.mean(rp)),
                     'diff': ri - float(np.mean(rp)), 'w': 1 / len(pk), 'r50': r50, 'src3': bool(L.SRC3[0] <= r.eff <= L.SRC3[1])})
P = pd.DataFrame(rows).drop_duplicates(['code', 'tE'])
P['month'] = P.eff.dt.to_period('M')
P.to_csv(HERE / 'p4_events_r.csv', index=False)

def st(g):
    if len(g) == 0: return {'n': 0}
    mm = g.groupby('month')['diff'].mean()
    return {'n': int(len(g)), 'n_months': int(len(mm)), 'mean_month_cluster': round(mm.mean() * 100, 2), 't_month': round(L.tstat(mm), 2),
            'mean_events': round(g['diff'].mean() * 100, 2), 'median_events': round(g['diff'].median() * 100, 2)}
M = P[~P.src3]
out = {'n_all': int(len(P)), 'n_ex_src3': int(len(M)), 'main': st(M), 'all_incl_src3': st(P),
       'seg_2006_2015': st(M[M.eff.dt.year <= 2015]), 'seg_2016_2026': st(M[M.eff.dt.year >= 2016]),
       'by_class': {c: st(g) for c, g in M.groupby('type')}}
mn = out['main']
out['adopt'] = bool(mn['mean_month_cluster'] < -1 and mn['t_month'] <= -2 and out['seg_2006_2015'].get('mean_month_cluster', 1) < 0
                    and out['seg_2016_2026'].get('mean_month_cluster', 1) < 0)

# ---- 改版引擎：複製 ia_core.simulate 的再平衡邏輯，加上「E 收盤賣出 → 0050」 ----
def sim(plan, buy, sell, t0, t1, sells=None, extra=0.005):
    sells = sells or {}
    reb = {p[0]: p[1] for p in plan}
    N = len(COLS); val = np.zeros(N); cash = 1.0; u50 = 0.0
    nav = np.empty(t1 - t0 + 1)
    for t in range(t0, t1 + 1):
        if t in reb:
            if u50 > 0: cash += u50 * E50[t - 1]; u50 = 0.0      # 0050 於換股日開盤賣（以前一日收盤近似）
            pk = reb[t]
            tr = K.TRADED[t] & np.isfinite(On[t])
            gap = np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            h = val * gap
            frz = (h > 0) & ~tr
            Vfz = h[frz].sum(); htr = np.where(frz, 0.0, h); Vt = htr.sum() + cash; k = len(pk)
            def cost_of(T):
                tgt = np.zeros(N); tgt[pk] = T
                return buy * np.clip(tgt - htr, 0, None).sum() + sell * np.clip(htr - tgt, 0, None).sum()
            lo, hi = 0.0, Vt / k
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mid * k + cost_of(mid) > Vt: hi = mid
                else: lo = mid
            newval = np.where(frz, h, 0.0); newval[pk] += lo
            oc = np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0)
            val = np.where(frz, newval, newval * oc); cash = 0.0
        else:
            val = val * G[t] if t > t0 else val
        for j in sells.get(t, ()):
            if val[j] > 0:
                u50 += val[j] * (1 - extra) / E50[t]; val[j] = 0.0
        nav[t - t0] = val.sum() + cash + u50 * E50[t]
    return pd.Series(nav, index=IDX[t0:t1 + 1])

bb, ss = K.rates(0.001)
plan2 = [(p[0], p[1]) for p in pl]
base_k, _ = K.simulate(plan2, bb, ss, t0, t1)
base = sim(plan2, bb, ss, t0, t1)
def sells_of(df):
    d = {}
    for r in df.itertuples(): d.setdefault(int(r.tE), []).append(CI[r.code])
    return d
mod_main = sim(plan2, bb, ss, t0, t1, sells_of(M))
mod_all = sim(plan2, bb, ss, t0, t1, sells_of(P))
def cg(s, a=None):
    z = s if a is None else s[s.index >= pd.Timestamp(a)]
    yrs = (z.index[-1] - z.index[0]).days / 365.25
    return round(((z.iloc[-1] / z.iloc[0]) ** (1 / yrs) - 1) * 100, 2), round(float((z / z.cummax() - 1).min()) * 100, 1)
ref = pd.read_parquet(AG / 'main_v2/navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
out['engine_check_max_abs_diff_vs_ia_core'] = float((base / base_k - 1).abs().max())
out['engine_check_vs_navs_gapfix'] = float((base_k / base_k.iloc[0] - ref.reindex(base_k.index) / ref.reindex(base_k.index).iloc[0]).abs().max())
out['nav'] = {}
for a in (None, '2008-01-02', '2010-11-30'):
    key = a or str(IDX[t0].date())
    out['nav'][key] = {'base': cg(base, a), 'sell_on_add_ex_src3': cg(mod_main, a), 'sell_on_add_all': cg(mod_all, a)}
L.dump(out, 'p4.json')
import pprint; pprint.pprint(out, width=160)
print(P[['code', 'name', 'type', 'eff', 'diff', 'src3']].to_string())

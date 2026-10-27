#!/usr/bin/env python3
"""24 版網格＋max-t bootstrap、各半合成、最大貢獻股剔除。"""
import sys, json, pickle
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import rlib as L

z = pickle.load(open(HERE / 'rep_core_both.pkl', 'rb'))
ew = z['ew']; plm = z['plm']; pl2 = z['pl2']; pl10 = z['pl2_10']
days = L.reb_days('2006-07-01', '2013-12-31'); END = pd.Timestamp('2013-12-31')
t0 = L.IDX.get_loc(days[0]); t1 = L.IDX.get_loc(END); X0 = L.IDX[t0 + 1]
WD = days[1:] + [END]
e50 = L.E50.iloc[t0:t1 + 1] / L.E50.iloc[t0]
we = L.win_ret(ew, WD); w5 = L.win_ret(e50, WD)
out = {}

# ---------- 24 版 ----------
B = {'B1': (0.3, 0.8), 'B2': (0.4, 0.85), 'B3': (0.5, 0.85), 'B4': (0.5, 0.9)}
EX = {'normal': [], 'zero': []}; names = []
for rev in ['ACC', 'NH', 'AVG']:
    for b, (lo, hi) in B.items():
        for n in [10, 30]:
            k = f'{rev}|{b}|n{n}'; names.append(k)
            pl = L.make_plan(days, n, lambda cs, d: L.leg2_score(cs, d, lo, hi, mode=rev)[0])
            row = {}
            for c in ['normal', 'zero']:
                nav, _ = L.simulate_w(L.to_w(pl), *L.rates(c), t0, t1)
                x = L.win_ret(nav, WD) - we; EX[c].append(x)
                row[c] = {'CAGR': round(L.cagr_dd(nav.loc[X0:])[0] * 100, 4), 't_EW': round(L.tstat(x), 4)}
            out[k] = row
rng = np.random.default_rng(12345)
for c in ['normal', 'zero']:
    M = np.vstack(EX[c]).T              # T × K
    T, K = M.shape
    tobs = np.array([L.tstat(M[:, j]) for j in range(K)])
    Md = M - M.mean(axis=0)
    mx = np.empty(5000)
    for i in range(5000):
        s = Md[rng.integers(0, T, T)]
        mx[i] = (s.mean(0) / (s.std(0, ddof=1) / np.sqrt(T))).max()
    out[f'maxt_{c}'] = {'t_max': float(tobs.max()), 'argmax': names[int(tobs.argmax())], 'p_maxt': float((mx >= tobs.max()).mean()), 'q95': float(np.quantile(mx, 0.95))}

# ---------- 各半合成 ----------
wm, w2 = L.to_w(plm), L.to_w(pl10)
wc = {e: 0.5 * wm[e] + 0.5 * w2[e] for e in wm}
for c in ['normal', 'stress']:
    for k, wp in [('combo', wc), ('main', wm)]:
        nav, _ = L.simulate_w(wp, *L.rates(c), t0, t1)
        cg, dd = L.cagr_dd(nav.loc[X0:])
        out[f'{k}|{c}'] = {'CAGR': cg * 100, 'maxDD': dd * 100, 't_0050': L.tstat(L.win_ret(nav, WD) - w5), 't_EW': L.tstat(L.win_ret(nav, WD) - we)}

# ---------- 最大貢獻股（毛、開盤→下一換股日開盤） ----------
G = L.G
ex = [p[0] for p in pl2]
per = []
for i, (t, pk, d) in enumerate(pl2):
    t2 = ex[i + 1] if i + 1 < len(ex) else None
    last = t2 if t2 is not None else t1
    r = L.Cn[t, pk] / L.On[t, pk]
    if last > t: r = r * np.prod(G[t + 1:last + 1, pk], axis=0)
    if t2 is not None:
        tr = L.TR[t2, pk] & np.isfinite(L.On[t2, pk])
        r = np.where(tr, r * L.On[t2, pk] / np.where(tr, L.Cn[t2, pk], 1.0), r)
    per.append((d, pk, r - 1))
con = {}
for d, pk, r in per:
    for j, x in zip(pk, r): con[L.COLS[j]] = con.get(L.COLS[j], 0) + x / len(pk)
top = max(con, key=con.get)
def gross(excl):
    rs = []
    for d, pk, r in per:
        rr = r.copy()
        if excl:
            m = np.array([L.COLS[j] == excl for j in pk])
            if m.any(): rr[m] = rr[~m].mean()
        rs.append(rr.mean())
    rs = np.array(rs); yrs = (END - per[0][0]).days / 365.25
    return (np.prod(1 + rs) ** (1 / yrs) - 1) * 100, rs
ga, ra = gross(None); gx, rx = gross(top)
# 近似毛 t：期報酬對 同池等權 窗報酬（訊號日窗；第一期起點不同，近似）
wew = L.win_ret(ew, days + [END])
out['top_contrib'] = {'code': top, 'name': L.SL['name'].get(top, ''), 'contrib_pp': con[top] * 100, 'gross_CAGR': ga, 'gross_CAGR_ex': gx, 'drop_pp': ga - gx,
                      't_gross_approx': L.tstat(ra - wew), 't_gross_ex_approx': L.tstat(rx - wew),
                      't_gross_approx_aug': L.tstat(ra[1:] - wew[1:]), 't_gross_ex_approx_aug': L.tstat(rx[1:] - wew[1:]),
                      'top5': sorted(((c, round(v * 100, 1)) for c, v in con.items()), key=lambda x: -x[1])[:5]}
json.dump(out, open(HERE / 'rep_extra.json', 'w'), ensure_ascii=False, indent=1, default=float)
for k, v in out.items(): print(k, v)

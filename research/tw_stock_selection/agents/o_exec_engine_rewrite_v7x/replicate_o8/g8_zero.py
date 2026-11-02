#!/usr/bin/env python3
"""零衝擊閘門（自寫）：以凍結名單逐 seed 重跑零衝擊引擎，對 V7 seed_navs_v7（官方日 80 seeds × 一般/壓力）與 navs_jscan_v7（13 j）比對。
目標金額用排序斷點精確解；凍結股與 X 規則依規格文字。另做處置期賣出（收盤減一跳）與併購延遲拆解，對 navs_ideal7。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import d8
from multiprocessing import Pool

D = d8.load()
IDX = D['IDX']; TR = D['TR']; On = D['On']; Cn = D['Cn']; G = D['G']; XR = D['XR']; E50 = D['E50']; DISP = D['DISP']; CF = D['CF']
N = len(D['COLS']); TPOS = D['TPOS']
FEE = 0.001425 * 0.6; TAX = 0.003
RATES = {'normal': (FEE + 0.001, FEE + TAX + 0.001), 'stress': (0.005, 0.005)}
PL = D['COLS']
O7 = d8.O7
CASHJ = {j: v for j, v in XR.items() if v[3] == 'cash' and v[4] is not None}


def load_plans(kind):
    if kind == 'ab':
        P = pd.read_csv(O7 / 'data' / 'plans_ab7.csv', dtype={'picks': str}); P['j'] = -1
    else:
        P = pd.read_csv(O7 / 'data' / 'plans_j7.csv', dtype={'picks': str})
    out = {}
    CI = D['CI']
    for (s, j, sd), g in P.groupby(['strat', 'j', 'seed']):
        out[(s, j, sd)] = [(int(e), pd.Timestamp(sg), np.array([CI[c] for c in pk.split(',')])) for e, sg, pk in zip(g.exec_idx, g.signal, g.picks)]
    return out


def solve_T(hp, k, other, V, b, s):
    """k·T + b·Σ(T−hp)+ + s·Σ(hp−T)+ + s·other = V；斷點排序後逐段線性解。"""
    xs = np.sort(hp)
    # f 在 T 時：k T + b Σ_{hp<T}(T−hp) + s Σ_{hp>T}(hp−T) + s other
    def f(T): return k * T + b * np.clip(T - hp, 0, None).sum() + s * np.clip(hp - T, 0, None).sum() + s * other
    bp = np.unique(np.concatenate([[0.0], xs[xs > 0], [V / k]]))
    fv = np.array([f(x) for x in bp])
    i = int(np.searchsorted(fv, V, side='right')) - 1
    i = min(max(i, 0), len(bp) - 2)
    x0, x1 = bp[i], bp[i + 1]; f0, f1 = fv[i], fv[i + 1]
    return x0 + (V - f0) * (x1 - x0) / (f1 - f0)


def sim_zero(plan, b, s, t0, t1, disp=False, mdelay=False):
    reb = {e: pk for e, sg, pk in plan}
    val = np.zeros(N); cash = 1.0; nav = np.empty(t1 - t0 + 1); pend = {}; nd = 0; nm = 0
    for t in range(t0, t1 + 1):
        if t in reb:
            pk = reb[t]; tr = TR[t] & np.isfinite(On[t])
            h = val * np.where(tr, G[t] * On[t] / np.where(tr, Cn[t], 1.0), 1.0)
            frz = (h > 0) & ~tr
            for j in np.where(frz)[0]:
                if mdelay and j in pend:
                    if t > pend[j][2]:
                        cash += h[j]; h[j] = 0.0; frz[j] = False; del pend[j]
                    continue
                if j in XR and XR[j][1] <= t < XR[j][2] and not (mdelay and j in CASHJ):
                    cash += h[j] * XR[j][0]; h[j] = 0.0; frz[j] = False
            htr = np.where(frz, 0.0, h)
            if not disp:
                inpk = np.zeros(N, bool); inpk[pk] = True
                V = htr.sum() + cash
                T = solve_T(htr[pk], len(pk), htr[~inpk].sum(), V, b, s)
                cash = 0.0
            else:
                rho = np.ones(N); dm = DISP[t] & tr & (htr > 0)
                if dm.any():
                    c = Cn[t]; rho = np.where(dm, (c - d8.tick(c)) / np.where(tr, On[t], 1.0), 1.0)
                inpk = np.zeros(N, bool); inpk[pk] = True
                def gap(T):
                    d_ = np.where(inpk, T, 0.0) - htr
                    return (np.clip(d_, 0, None) * (1 + b)).sum() - (np.clip(-d_, 0, None) * rho * (1 - s)).sum() - cash
                # 精確解：gap 為分段線性遞增，斷點＝htr[pk]
                bp = np.unique(np.concatenate([[0.0], htr[pk][htr[pk] > 0], [(htr.sum() * max(rho.max(), 1.0) + cash) / len(pk) + 1.0]]))
                gv = np.array([gap(x) for x in bp])
                i = int(np.searchsorted(gv, 0.0, side='right')) - 1; i = min(max(i, 0), len(bp) - 2)
                T = bp[i] + (0 - gv[i]) * (bp[i + 1] - bp[i]) / (gv[i + 1] - gv[i])
                nd += int((dm & (np.where(inpk, T, 0.0) < htr)).sum())
                cash = -gap(T)
            nv = np.where(frz, h, 0.0); nv[pk] += T
            val = np.where(frz, nv, nv * np.where(tr, Cn[t] / np.where(tr, On[t], 1.0), 1.0))
        elif t > t0:
            val = val * G[t]
        if mdelay:
            for j, v in CASHJ.items():
                if val[j] > 0 and j not in pend and t >= v[4] and v[2] > t and not TR[t, j]:
                    pend[j] = (val[j] * v[0], v[4], v[4] + 20); nm += 1
            for j, (v0, idd, ipay) in pend.items():
                val[j] = v0 * E50[t] / E50[idd]
        nav[t - t0] = val.sum() + cash
    return pd.Series(nav, index=IDX[t0:t1 + 1]), nd, nm


ENDS = json.loads((O7 / 'data' / 'jscan_end_days.json').read_text())


def job_off(a):
    (st, j, sd), plan = a
    t0 = plan[0][0] - 1; t1 = TPOS[pd.Timestamp('2026-10-12')]
    return (st, sd), {c: sim_zero(plan, *RATES[c], t0, t1)[0] for c in ('normal', 'stress')}


def job_j(a):
    (st, j, sd), plan = a
    t0 = plan[0][0] - 1; t1 = TPOS[pd.Timestamp(ENDS[f'{st}|J{j}'])]
    out = {c: sim_zero(plan, *RATES[c], t0, t1)[0].pct_change() for c in ('normal', 'stress')}
    if st == 'AB_V7_B10' or st == 'AB_V7':
        b, s = RATES['normal']
        for m in ('disp', 'mdelay'):
            r, nd, nm = sim_zero(plan, b, s, t0, t1, disp=(m == 'disp'), mdelay=(m == 'mdelay'))
            out[m] = r.pct_change(); out[m + '_n'] = nd if m == 'disp' else nm
    return (st, j, sd), out


if __name__ == '__main__':
    res = {}
    PA = load_plans('ab')
    with Pool(4) as P: R = P.map(job_off, list(PA.items()), chunksize=2)
    S = pd.read_parquet(O7 / 'seed_navs_v7.parquet')
    mx = 0.0; worst = None
    for (st, sd), d in R:
        for c, nv in d.items():
            ref = S[f'{st}|{c}|s{sd}'].dropna()
            mine = nv.reindex(ref.index)
            rel = float(((mine / mine.iloc[0]) / (ref / ref.iloc[0]) - 1).abs().max())
            if rel > mx: mx, worst = rel, f'{st}|{c}|s{sd}'
    res['official_160_maxrel'] = mx; res['official_worst'] = worst; print('official', mx, worst, flush=True)
    PJ = load_plans('j')
    with Pool(4) as P: R = P.map(job_j, list(PJ.items()), chunksize=4)
    J = pd.read_parquet(O7 / 'navs_jscan_v7.parquet'); I = pd.read_parquet(O7 / 'navs_ideal7.parquet')
    print(list(J.columns)[:8])
    agg = {}
    for (st, j, sd), d in R:
        for c, r in d.items():
            if c.endswith('_n'): agg.setdefault((st, j, c), []).append(r); continue
            agg.setdefault((st, j, c), []).append(r)
    jm = {}; dec = {}; nn = {}
    for (st, j, c), L in agg.items():
        if c.endswith('_n'): nn[f'{st}|J{j}|{c}'] = float(np.mean(L)); continue
        nv = (1 + pd.concat(L, axis=1).mean(1).fillna(0)).cumprod()
        key = 'AB' if st == 'AB_V7' else 'B10'
        if c in ('normal', 'stress'):
            cand = [x for x in J.columns if x.startswith(f'{st}|') and f'J{j}|' in x + '|' and c in x] or \
                   [x for x in J.columns if x.split('|')[0] == st and x.endswith(f'J{j}') and c in x]
            jm[f'{st}|J{j}|{c}'] = (nv, cand)
        cc = {'normal': 'base', 'disp': 'disp', 'mdelay': 'mdelay'}.get(c)
        if cc:
            ref = I[f'{key}|J{j}|{cc}'].dropna(); mine = nv.reindex(ref.index)
            dec[f'{key}|J{j}|{cc}'] = float(((mine / mine.iloc[0]) / (ref / ref.iloc[0]) - 1).abs().max())
    res['decomp_maxrel'] = dec; res['n_events'] = nn
    jr = {}
    for k, (nv, cand) in jm.items():
        st, jj, c = k.split('|'); col = f'{st}|{c}|{jj}'
        if col in J.columns:
            ref = J[col].dropna(); mine = nv.reindex(ref.index)
            jr[k] = float(((mine / mine.iloc[0]) / (ref / ref.iloc[0]) - 1).abs().max())
    res['jscan_maxrel'] = jr; res['jscan_max'] = max(jr.values()) if jr else None
    pd.to_pickle({k: v[0] for k, v in jm.items()}, HERE / 'out' / 'zero_j_navs.pkl')
    json.dump(res, open(HERE / 'out' / 'gate_zero.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])

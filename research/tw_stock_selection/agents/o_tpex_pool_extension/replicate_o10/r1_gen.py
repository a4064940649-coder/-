#!/usr/bin/env python3
"""replicate_o10 r1：自寫選股產生器。輸出 scratchpad/rep_o10/plans_<VAR>.pkl：{j: [(exec_idx, picks_idx, signal_idx), ...]}
VAR：M＝ALL400 n10 cap5；BASE＝TW300 n10；R1＝ALL500 n10；R2＝TP300 n10；N5／N15／P300／P500／L0／L20（鄰域）。"""
import sys, pickle, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import x_data as D
sys.path.append(str(HERE.parents[1] / 'o_exec_engine_rewrite_v7x'))
import x_exec as X

OUT = D.SCR
LONG = Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv')
rv = pd.read_csv(LONG, dtype={'code': str, 'market': str, 'name': str})
rv['p'] = pd.PeriodIndex(pd.to_datetime(rv['month']), freq='M')
rv = rv.sort_values(['p', 'market']).drop_duplicates(['p', 'code'], keep='first')
names = {}
for c, n in zip(rv.code, rv.name): names.setdefault(c, set()).add(str(n))
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str)
for c, n in zip(sl.code, sl.name): names.setdefault(c, set()).add(str(n))
L = rv[rv.code.isin(D.CI)].pivot_table(index='p', columns='code', values='revenue_k_twd', aggfunc='last').sort_index()
L = L.reindex(pd.period_range(L.index.min(), L.index.max(), freq='M'))
prev = L.shift(12)
YOY = ((L / prev - 1) * 100).where(prev > 0)
YOY3 = YOY.rolling(3, min_periods=3).mean().reindex(columns=D.COLS)

FIN = np.array([c.startswith('28') or c in ('5876', '5880') for c in D.COLS])
KYDR = np.array([c.startswith('91') or any(('KY' in n.upper()) or ('DR' in n.upper()) for n in names.get(c, ())) for c in D.COLS])
EXCL = FIN | KYDR
CODES = np.array(D.COLS)
DEL_AFTER = np.full(D.N, -1)
for j in D.XR: DEL_AFTER[j] = D.LASTT[j]

P = pd.read_csv(D.PNL / 'plans_j7.csv', dtype={'picks': str})
P = P[(P.strat == 'AB_V7_B10') & (P.seed == 0)]
SIG = {int(j): [(D.TPOS[pd.Timestamp(s)], int(e)) for s, e in zip(g.signal, g.exec_idx)] for j, g in P.groupby('j')}
for j, L_ in SIG.items():
    assert all(e == t + 1 for t, e in L_), j

VARS = dict(M=('ALL', 400, 10, 5, 1e7), BASE=('TW', 300, 10, 10, 1e7), R1=('ALL', 500, 10, 10, 1e7), R2=('TP', 300, 10, 10, 1e7),
            N5=('ALL', 400, 5, 2, 1e7), N15=('ALL', 400, 15, 7, 1e7), P300=('ALL', 300, 10, 5, 1e7), P500=('ALL', 500, 10, 5, 1e7),
            L0=('ALL', 400, 10, 5, 0.0), L20=('ALL', 400, 10, 5, 2e7))


def select(t, e, mk, K, n, cap, vmin, held):
    v = D.V60[t]
    m = np.isfinite(v) & (v >= vmin) & D.TR[t] & (D.C[t] >= 10)
    if mk == 'TW': m &= ~D.TPX[t]
    elif mk == 'TP': m &= D.TPX[t]
    w = np.where(m)[0]
    w = w[np.lexsort((CODES[w], -v[w]))][:K]
    ym = YOY3.loc[pd.Period(D.IDX[t], 'M') - 1].to_numpy()
    b10 = (D.IDX[t].year >= 2013) & ~np.isfinite(ym) & (DEL_AFTER > t)
    ok = ~EXCL[w] & ~D.DISP[t, w] & ~D.DISP[e, w] & np.isfinite(D.O[e, w]) & np.isfinite(D.S52[t, w]) & (np.isfinite(ym[w]) | b10[w])
    w = w[ok]
    if len(w) == 0: return np.array([], int)
    s52 = pd.Series(D.S52[t, w]).rank(pct=True).to_numpy()
    yy = ym[w]; hv = np.isfinite(yy)
    rp = np.ones(len(w)); rp[hv] = pd.Series(yy[hv]).rank(pct=True).to_numpy()
    sc = 0.5 * s52 + 0.5 * rp
    yk = np.where(hv, yy, np.inf)
    order = np.lexsort((CODES[w], -yk, -sc))
    out = []; ntp = 0
    for i in order:
        j = w[i]
        if D.O[e, j] >= X.LIMUP[e, j] - 1e-6 and j not in held: continue
        if D.TPX[t, j]:
            if ntp >= cap: continue
            ntp += 1
        out.append(j)
        if len(out) == n: break
    return np.array(out, int)


def gen(var):
    mk, K, n, cap, vmin = VARS[var]
    res = {}
    for j, L_ in SIG.items():
        held = set(); pl = []
        for t, e in L_:
            pk = select(t, e, mk, K, n, cap, vmin, held)
            pl.append((e, pk, t)); held = set(pk.tolist())
        res[j] = pl
    pickle.dump(res, open(OUT / f'plans_{var}.pkl', 'wb'))
    tpsh = [np.mean(D.TPX[t, pk]) if len(pk) else np.nan for pl in res.values() for e, pk, t in pl]
    return dict(var=var, n_windows={j: len(v) for j, v in res.items()}, tp_share_med=float(np.nanmedian(tpsh)),
                min_n=int(min(len(pk) for pl in res.values() for e, pk, t in pl)))


if __name__ == '__main__':
    vs = sys.argv[1:] or list(VARS)
    for v in vs:
        print(json.dumps(gen(v)), flush=True)

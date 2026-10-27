import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
K, V = R.K, R.V
IDX, N = R.IDX, R.N
Cn, On, G, TR = R.Cn, R.On, R.G, R.TR
ci, co = R.daily_split()
S52, V60 = V.rolling_signals('gapfix')
days = [d for d in K.rebalance_days('2005-01-01') if d <= pd.Timestamp('2026-10-12')]
DT = [IDX.get_loc(d) for d in days]

def hold_ret(e1, e2):
    """e1 開盤買 → e2 開盤賣（含息；e2 沒成交＝以最後市值計）。"""
    w = np.where(TR[e1], Cn[e1] / On[e1], np.nan)
    if e2 > e1 + 1: w = w * np.prod(G[e1 + 1:e2], axis=0)
    tr2 = TR[e2] & np.isfinite(On[e2])
    w = np.where(tr2, w * G[e2] * On[e2] / np.where(tr2, Cn[e2], 1.0), w)
    return w - 1

def pool(t):
    e = t + 1
    v = V60[t]; px = Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10)
    order = np.argsort(-np.where(base, v, -np.inf), kind='stable')
    top = np.zeros(N, bool); top[order[:min(300, base.sum())]] = True; top &= base
    el = top & np.isfinite(S52[t]) & ~K.disposal_mask(IDX[t]) & np.isfinite(On[e]) & TR[e] & ~V.FINMASK
    return el

JOINT = True
SEL_END_SIG = sys.argv[1] if len(sys.argv) > 1 else '2019-04-30'
SPECS = {'main': (273, 21), 'sec': (126, 0), 'mainA_231': (252, 21)}
sig = {}
for nm, (lk, sk) in SPECS.items():
    I, mI = R.roll_sum(ci, lk, sk); O_, mO = R.roll_sum(co, lk, sk)
    if JOINT: I = np.where(mI & mO, I, np.nan); O_ = np.where(mI & mO, O_, np.nan)
    sig[nm] = (I, O_, R.src3_frac(lk, sk))
HR = {}
rows = []
prev = {}
for i in range(len(DT) - 1):
    t, t2 = DT[i], DT[i + 1]
    el = pool(t)
    hr = hold_ret(t + 1, t2 + 1); HR[days[i]] = hr
    rec = {'sig': days[i]}
    for nm in SPECS:
        I, O_, f3 = sig[nm]
        for fac, X in (('IMOM', I), ('OMOM', O_)):
            x = np.where(el, X[t], np.nan)
            ok = np.isfinite(x) & np.isfinite(hr)
            ok0 = np.isfinite(x)
            if ok0.sum() < 50: continue
            r = pd.Series(x[ok0]).rank(pct=True).to_numpy()
            js = np.where(ok0)[0]
            q5 = js[r > 0.8]; q1 = js[r <= 0.2]
            s = np.nanmean(hr[q5]) - np.nanmean(hr[q1])
            key = f'{nm}|{fac}'
            p5, p1_ = prev.get(key, (None, None))
            tov = (len(set(q5) - set(p5)) / len(q5) + len(set(q1) - set(p1_)) / len(q1)) if p5 is not None else 2.0
            prev[key] = (q5, q1)
            rec[key] = s; rec[key + '|net'] = s - tov * 0.005; rec[key + '|n'] = int(ok0.sum())
        rec[f'{nm}|src3frac'] = f3[t]
    rows.append(rec)
df = pd.DataFrame(rows).set_index('sig')
df.to_pickle(R.HERE / 'p1_spreads.pkl')

# 主策略月超額（對 0050；訊號窗）
mn = pd.read_parquet(R.HERE / 'main_nav.parquet')
S = R.main_strategy(); t0, t1, sd = S.bounds()
mr = R.win_rets(mn.nav, sd, '2000-01-01', '2030-01-01'); br = R.win_rets(R.E50.reindex(mn.index), sd, '2000-01-01', '2030-01-01')
mex = (mr - br.reindex(mr.index))

def summ(x):
    x = x.dropna(); return {'mean_pct': round(float(x.mean()) * 100, 3), 't': round(R.tstat(x), 2), 'n': int(len(x))}
out = {}
P = {'SEL': ('2005-01-01', SEL_END_SIG), 'VAL': ('2019-05-01', '2026-09-30')}
for nm in SPECS:
    for k, (a, b) in P.items():
        x = df[(df.index >= a) & (df.index <= b)]
        o = {}
        for keep3 in ((False, True) if k == 'VAL' else (True,)):
            y = x if keep3 else x[~(x[f'{nm}|src3frac'] > 0.2)]
            tag = 'keep_src3' if keep3 else 'ex_src3'
            o[tag] = {fac: summ(y[f'{nm}|{fac}']) for fac in ('IMOM', 'OMOM')}
            o[tag]['IMOM_net'] = summ(y[f'{nm}|IMOM|net'])
            o[tag]['first_sig'] = str(y[f'{nm}|IMOM'].dropna().index.min().date())
            o[tag]['n_excluded'] = int(len(x) - len(y))
            c = pd.concat([y[f'{nm}|IMOM'], mex], axis=1, join='inner').dropna()
            o[tag]['corr_IMOM_mainexcess'] = round(float(c.corr().iloc[0, 1]), 3)
        if k == 'VAL':
            for sub, (sa, sb) in {'19-23': ('2019-05-01', '2023-12-31'), '24-26': ('2024-01-01', '2026-09-30')}.items():
                y = x[(x.index >= sa) & (x.index <= sb)]
                o['seg_' + sub] = {'ex_src3': summ(y[~(y[f'{nm}|src3frac'] > 0.2)][f'{nm}|IMOM']), 'keep': summ(y[f'{nm}|IMOM'])}
        out[f'{nm}|{k}'] = o
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(R.HERE / 'p1.json', 'w'), ensure_ascii=False, indent=1)

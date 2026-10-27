import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import rlib as R
IDX = R.IDX; Cn, On, G, TR = R.Cn, R.On, R.G, R.TR
ci, co = R.daily_split()
S = R.main_strategy(); t0, t1, sd = S.bounds()
A = np.where(TR, np.cumprod(G, axis=0), np.nan)
def run(look, joint, ret, rankby='official', n=30):
    I, mI = R.roll_sum(ci, look, 0); O_, mO = R.roll_sum(co, look, 0)
    ok = (mI & mO) if joint else np.ones_like(mI)
    CI = np.where(ok & mI, I, np.nan); CO = np.where(ok & mO, O_, np.nan); tot = CI + CO
    SH = np.where(tot > 0, CI / np.where(tot > 0, tot, 1), np.nan)
    if rankby == 'official': pl = R.plan(n)
    else:
        pl = []
        for d in S.sig_days:
            cs = S.cross_section(d)
            if cs['el'].sum() < 50: continue
            pl.append((cs['e'], S.order(cs)[:n], None, d))
    res = []
    for i in range(len(pl) - 1):
        e, pk, _, d = pl[i]; e2 = pl[i + 1][0]; t = e - 1
        if ret == 'open':
            w = Cn[e, pk] / On[e, pk] * np.prod(G[e + 1:e2, pk], axis=0)
            tr2 = TR[e2, pk] & np.isfinite(On[e2, pk])
            r = np.where(tr2, w * G[e2, pk] * On[e2, pk] / np.where(tr2, Cn[e2, pk], 1), w) - 1
        else:
            a = pd.DataFrame(A[:, pk]).ffill().to_numpy(); r = a[e2 - 1] / a[t] - 1
        sh = SH[t, pk]; m = np.isfinite(sh) & np.isfinite(r)
        if m.sum() < 6: continue
        o = np.argsort(sh[m]); rr = r[m][o]; h = len(rr) // 2
        res.append((d, rr[-h:].mean() - rr[:h].mean()))
    x = pd.Series(dict(res))
    s, v = x[x.index < '2019-05-01'], x[x.index >= '2019-05-01']
    return f'SEL {s.mean()*100:.3f} t{R.tstat(s):.2f} n{len(s)} | VAL {v.mean()*100:.3f} t{R.tstat(v):.2f} n{len(v)}'
for look in (252, 273):
    for joint in (True, False):
        for ret in ('open', 'close'):
            for rb in ('official', 'nogapblock'):
                print(look, joint, ret, rb, run(look, joint, ret, rb))
print('--- full market (含金融) top30')
YL, Y5, _ = R.V.load_revenue()
S2 = R.V.V2('gapfix', YL, Y5, exfin=False)
R._cache['pl30'] = S2.plan(30, 'official')
print(run(252, True, 'open'))
# 依年份看 VAL 差

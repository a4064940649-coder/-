import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
IDX = R.IDX; Cn, On, G, TR = R.Cn, R.On, R.G, R.TR
ci, co = R.daily_split()
I, mI = R.roll_sum(ci, 252, 0); O_, mO = R.roll_sum(co, 252, 0)
ok = mI & mO
CI = np.where(ok, I, np.nan); CO = np.where(ok, O_, np.nan)
tot = CI + CO
SH = np.where(tot > 0, CI / np.where(tot > 0, tot, 1), np.nan)
f3 = R.src3_frac(252, 0)
def hold_ret(e1, e2):
    w = np.where(TR[e1], Cn[e1] / On[e1], np.nan)
    if e2 > e1 + 1: w = w * np.prod(G[e1 + 1:e2], axis=0)
    tr2 = TR[e2] & np.isfinite(On[e2])
    w = np.where(tr2, w * G[e2] * On[e2] / np.where(tr2, Cn[e2], 1.0), w)
    return w - 1
S = R.main_strategy(); t0, t1, sd = S.bounds()
out = {}
for n, nm in ((30, 'P2'), (10, 'winners')):
    pl = R.plan(n)
    rows = []
    for i, (e, pk, pool, d) in enumerate(pl):
        e2 = pl[i + 1][0] if i + 1 < len(pl) else t1   # 最後一期：到終點收盤（只用 n10 描述）
        t = e - 1
        hr = hold_ret(e, e2) if i + 1 < len(pl) else None
        for rk, j in enumerate(pk):
            rows.append(dict(sig=d, j=j, rank=rk, sh=SH[t, j], ci=CI[t, j], co=CO[t, j], r=(hr[j] if hr is not None else np.nan), f3=f3[t]))
    df = pd.DataFrame(rows)
    if nm == 'P2':
        diffs = []
        for d, g in df.groupby('sig'):
            g = g[np.isfinite(g.sh) & np.isfinite(g.r)]
            if len(g) < 6: continue
            med = g.sh.median()
            hi_, lo_ = g[g.sh > med], g[g.sh <= med]
            # 依排序位置分兩半（等分）
            gs = g.sort_values('sh'); h = len(gs) // 2
            diffs.append(dict(sig=d, diff=gs.r.iloc[-h:].mean() - gs.r.iloc[:h].mean(), diff_med=hi_.r.mean() - lo_.r.mean(), f3=g.f3.iloc[0]))
        x = pd.DataFrame(diffs).set_index('sig')
        o = {}
        for k, (a, b) in {'SEL': ('2006-04-01', '2019-04-30'), 'VAL': ('2019-05-01', '2026-09-30')}.items():
            y = x[(x.index >= a) & (x.index <= b)]
            for col in ('diff', 'diff_med'):
                o[f'{k}|{col}'] = {'mean_pct': round(float(y[col].mean()) * 100, 3), 't': round(R.tstat(y[col]), 2), 'n': len(y)}
                if k == 'VAL':
                    z = y[~(y.f3 > 0.2)][col]
                    o[f'{k}|{col}|ex_src3'] = {'mean_pct': round(float(z.mean()) * 100, 3), 't': round(R.tstat(z), 2), 'n': len(z)}
        out['P2'] = o
    else:
        y = df[(df.sig >= '2006-04-01') & (df.sig <= '2019-04-30')]
        ys = y[np.isfinite(y.sh)]
        # 下月贏家／輸家：每月 n10 內依下月報酬
        y2 = y[np.isfinite(y.r)].copy()
        q = y2.r.quantile([0.1, 0.9])
        out['winners'] = {'n_obs': len(y), 'share_median': round(float(ys.sh.median()), 3),
                          'frac_CI_neg': round(float((y.ci.dropna() < 0).mean()), 3),
                          'CI_median': round(float(y.ci.median()), 3), 'CO_median': round(float(y.co.median()), 3),
                          'next_top10pct_share_med': round(float(y2[y2.r >= q[0.9]].sh.median()), 3),
                          'next_bot10pct_share_med': round(float(y2[y2.r <= q[0.1]].sh.median()), 3)}
print(json.dumps(out, indent=1, ensure_ascii=False))
json.dump(out, open(R.HERE / 'p2.json', 'w'), ensure_ascii=False, indent=1)

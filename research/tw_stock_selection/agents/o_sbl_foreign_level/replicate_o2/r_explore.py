import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, math
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import r_lib as L
K, IDX, COLS, N = L.K, L.IDX, L.COLS, L.N
D = L.build()
SD = D['sig_days']; END = D['end']
ex_days = [D['cs'][d]['e'] for d in SD] + [IDX.get_loc(END)]

def oo_ret(i):
    """第 i 個訊號日：成交日開盤 → 下一成交日開盤（最後一期到 END 收盤），全部股票。"""
    t, t2 = ex_days[i], ex_days[i + 1]
    r = K.Cn[t] / K.On[t] * np.prod(K.GROW[t + 1:t2 + 1], axis=0)
    if i + 1 < len(SD):
        tr = K.TRADED[t2] & np.isfinite(K.On[t2])
        r = np.where(tr, r * K.On[t2] / np.where(tr, K.Cn[t2], 1.0), r)
    return r - 1

def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 else float('nan')

def qspread(x, r, m, q=5):
    idx = np.where(m & np.isfinite(x) & np.isfinite(r))[0]
    if len(idx) < 2 * q: return np.nan
    rk = pd.Series(x[idx]).rank(method='first').to_numpy()
    b = np.minimum(((rk - 1) * q // len(idx)).astype(int), q - 1)
    return r[idx][b == q - 1].mean() - r[idx][b == 0].mean()

rows = []
for i, d in enumerate(SD):
    c = D['cs'][d]; r = oo_ret(i)
    g, fo, mc = L.groups(d, c)
    ok = g >= 0
    fo6 = fo - L.fo_at(d, 126); fo12 = fo - L.fo_at(d, 252)
    o = dict(d=d)
    for name, x in (('fo', fo), ('d6', fo6), ('d12', fo12)):
        s = [qspread(x, r, g == k) for k in range(3)]
        o[f'{name}_within'] = np.nanmean(s) if np.isfinite(s).any() else np.nan
        for k in range(3): o[f'{name}_g{k}'] = s[k]
        o[f'{name}_uncond'] = qspread(x, r, ok)
    o['size'] = qspread(mc, r, ok)
    # 近似 0050：池內近似市值前 50
    rk = pd.Series(np.where(ok, mc, np.nan)).rank(ascending=False).to_numpy()
    in50 = ok & (rk <= 50)
    o['fo_out50'] = qspread(fo, r, ok & ~in50); o['fo_in50'] = qspread(fo, r, in50)
    m = ok
    o['corr_mc'] = spearmanr(fo[m], mc[m]).correlation
    m2 = ok & np.isfinite(c['s52']); o['corr_s52'] = spearmanr(fo[m2], c['s52'][m2]).correlation
    m3 = ok & np.isfinite(c['y3']); o['corr_rev'] = spearmanr(fo[m3], c['y3'][m3]).correlation
    o['n_pool'] = int(c['el'].sum()); o['n_ok'] = int(ok.sum())
    rows.append(o)
T = pd.DataFrame(rows).set_index('d')
R = {}
for lab, (a, b) in {'SEL27': ('2021-10-01', '2023-12-31'), 'SEL26': ('2021-10-01', '2023-11-30'),
                    'VAL31': ('2024-03-01', '2026-09-30'), 'ALL': ('2021-10-01', '2026-09-30')}.items():
    x = T.loc[a:b]
    R[lab] = {c: (dict(mean=round(x[c].mean() * 100, 3), t=round(tstat(x[c]), 2), n=int(x[c].notna().sum())) if not c.startswith('corr') and not c.startswith('n_')
                  else round(float(x[c].mean()), 3)) for c in x.columns}

# P9：主策略入選股入選前 6 個月外資比率是否上升
mp = [p for p in D['main_plan'] if p[3] >= SD[0]]
ex_m = [p[0] for p in D['main_plan']]
recs = []
for i, (t, pk, _, d) in enumerate(D['main_plan']):
    if d < SD[0]: continue
    t2 = ex_m[i + 1] if i + 1 < len(ex_m) else IDX.get_loc(END)
    r = K.Cn[t, pk] / K.On[t, pk] * np.prod(K.GROW[t + 1:t2 + 1, pk], axis=0)
    if i + 1 < len(ex_m):
        tr = K.TRADED[t2, pk] & np.isfinite(K.On[t2, pk])
        r = np.where(tr, r * K.On[t2, pk] / np.where(tr, K.Cn[t2, pk], 1.0), r)
    up = (L.fo_at(d) - L.fo_at(d, 126))[pk]
    prev = set(D['main_plan'][i - 1][1]) if i > 0 else set()
    for j, x, u in zip(pk, r - 1, up):
        recs.append(dict(d=d, code=COLS[j], ret=x / len(pk), up=u, new=j not in prev))
Rc = pd.DataFrame(recs)
p9 = {}
for lab, a in (('ALL', SD[0]), ('from2022-04', pd.Timestamp('2022-04-01'))):
    x = Rc[Rc.d >= a]; xv = x[np.isfinite(x.up)]
    con = x.groupby('code').ret.sum().sort_values(ascending=False)
    top20 = con.index[:20]
    t20 = xv[xv.code.isin(top20)]
    p9[lab] = dict(all_picks_up=round(float((xv.up > 0).mean()), 3), n=len(xv), top20_up=round(float((t20.up > 0).mean()), 3),
                   new_picks_up=round(float((xv[xv.new].up > 0).mean()), 3),
                   top20_first_up=round(float((t20.sort_values('d').drop_duplicates('code').up > 0).mean()), 3),
                   top20_new_up=round(float((t20[t20.new].up > 0).mean()), 3))
R['P9'] = p9
(L.HERE / 'my_explore.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
for k, v in R.items():
    print(k)
    for kk, vv in (v.items() if isinstance(v, dict) else []): print('  ', kk, vv)

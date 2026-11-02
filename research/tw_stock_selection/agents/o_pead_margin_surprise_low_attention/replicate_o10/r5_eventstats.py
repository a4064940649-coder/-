"""事件層級統計（挑選期與樣本外）。"""
from pathlib import Path
import json
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
D = HERE / 'data'
E = pd.read_parquet(D / 'events_all.parquet')
P = np.load(D / 'panel.npz', allow_pickle=True)
codes = list(P['codes']); T = len(P['dates'])
dt = pd.to_datetime(P['dates'])
fin = pd.read_parquet(D / 'fin.parquet')
fin['gm'] = fin.gp / fin.rev; fin['om'] = fin.opi / fin.rev
fs = fin.set_index(['stock_id', 'qi'])
E['dgm'] = E.set_index(['stock_id', 'qi']).index.map(lambda k: fs.gm.get(k, np.nan) - fs.gm.get((k[0], k[1] - 4), np.nan))
E['dom'] = E.set_index(['stock_id', 'qi']).index.map(lambda k: fs.om.get(k, np.nan) - fs.om.get((k[0], k[1] - 4), np.nan))
# 0050 同窗報酬
j50 = codes.index('0050')
r50 = np.nan_to_num(P['ret'][:, j50]); I50 = np.cumprod(1 + r50); o50 = P['open'][:, j50] * I50 / pd.Series(P['close'][:, j50]).ffill().values
e = E.A_pos.values + 1
E['b0'] = np.where(e + 60 < T, o50[np.minimum(e + 60, T - 1)] / o50[np.minimum(e, T - 1)] - 1, np.nan)


def tq(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (x.mean(), x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), len(x)) if len(x) > 2 else (np.nan, np.nan, len(x))


def fm(df, cols, y='r0'):
    co = []
    for k, g in df.groupby('kq'):
        g = g.dropna(subset=cols + [y])
        if len(g) < 20:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in cols])
        b = np.linalg.lstsq(X, g[y].values, rcond=None)[0]
        co.append(b[1:])
    co = np.array(co)
    return [round(tq(co[:, i])[1], 2) for i in range(len(cols))]


def ar_top(df, grp_mask, y='r0', top='top'):
    """前 1/5 對同組同季平均：每季 (top 平均 − 組平均)，季分群 t。"""
    g = df[grp_mask]
    s = g.groupby('kq').apply(lambda x: x.loc[x[top], y].mean() - x[y].mean())
    m, t, n = tq(s.values)
    return round(m * 100, 2), round(t, 2), int(g[top].sum())


def ls(df, mask, y='r0'):
    g = df[mask]
    s = g.groupby('kq').apply(lambda x: x.loc[x.top, y].mean() - x.loc[x.bot, y].mean())
    m, t, n = tq(s.values)
    return round(m * 100, 2), round(t, 2)


def diffLH(df, y='r0'):
    s = []
    for k, x in df.groupby('kq'):
        a = x[x.att == 'L']; b = x[x.att == 'H']
        s.append((a.loc[a.top, y].mean() - a[y].mean()) - (b.loc[b.top, y].mean() - b[y].mean()))
    m, t, n = tq(s)
    return round(m * 100, 2), round(t, 2)


out = {}
for per, (q0, q1) in {'sel': ('2016Q1', '2019Q1'), 'oos': ('2019Q2', '2026Q2')}.items():
    X = E[E.period.between(q0, q1)].copy()
    n50 = ~X.is50
    r = dict(n=len(X), nL=int((X.att == 'L').sum()), nOTC=int((X.mkt == 2).sum()))
    r['fm_all_ex50'] = fm(X[n50], ['sperp']); r['fm_all_incl50'] = fm(X, ['sperp'])
    r['fm_L'] = fm(X[X.att == 'L'], ['sperp']); r['fm_H'] = fm(X[X.att == 'H'], ['sperp'])
    r['fm_with_rev'] = fm(X[n50], ['sperp', 'prev'])
    r['ls_L'] = ls(X, X.att == 'L'); r['ls_all_ex50'] = ls(X, n50); r['ls_all'] = ls(X, np.ones(len(X), bool))
    for gname in ['L', 'M', 'H', 'X50']:
        r['ar_' + gname] = ar_top(X, X.att == gname)
    r['ar_ex50'] = ar_top(X, n50)
    r['LminusH'] = diffLH(X)
    r['jdecay_L'] = [ar_top(X, X.att == 'L', y=f'r{j}')[0] for j in range(10)]
    r['jdecay_ex50'] = [ar_top(X, n50, y=f'r{j}')[0] for j in range(10)]
    r['corr_sperp_dgm'] = round(X[['sperp', 'dgm']].corr(method='spearman').iloc[0, 1], 3)
    r['corr_sperp_dom'] = round(X[['sperp', 'dom']].corr(method='spearman').iloc[0, 1], 3)
    r['corr_sperp_prev'] = round(X[['sperp', 'prev']].corr().iloc[0, 1], 3)
    r['corr_sperp_dgm_pearson'] = round(X[['sperp', 'dgm']].clip(-1, 1).corr().iloc[0, 1], 3)
    r['sel_rate_by_q'] = [round(float(X[X.kq == k].top.mean()), 3) for k in sorted(X.kq.unique())]
    if per == 'oos':
        val = X.A < pd.Timestamp('2024-01-01')
        r['fm_val'] = fm(X[n50 & val], ['sperp']); r['fm_hold'] = fm(X[n50 & ~val], ['sperp'])
        r['ar_ex50_val'] = ar_top(X, n50 & val); r['ar_ex50_hold'] = ar_top(X, n50 & ~val)
        r['ar_twse'] = ar_top(X, n50 & (X.mkt == 1)); r['ar_otc'] = ar_top(X, n50 & (X.mkt == 2))
        # 日分群 t（依進場日）
        g = X[n50].copy(); g['ab'] = g.r0 - g.groupby('kq').r0.transform('mean')
        dd = g[g.top].groupby('A').ab.mean()
        r['ar_ex50_tdate'] = round(tq(dd.values)[1], 2)
        # 拿掉貢獻最大 5 檔
        tot = g[g.top].groupby('stock_id').ab.sum().sort_values()
        g2 = X[n50 & ~X.stock_id.isin(tot.index[-5:])]
        r['ar_ex50_drop5'] = ar_top(g2, np.ones(len(g2), bool))
        # 對 0050 淨報酬
        h = X[n50 & X.top]
        s = h.groupby('kq').apply(lambda x: (x.r0 - x.b0).mean())
        r['vs0050'] = [round(tq(s.values)[0] * 100, 2), round(tq(s.values)[1], 2)]
        hh = h[h.A >= pd.Timestamp('2024-01-01')]
        s = hh.groupby('kq').apply(lambda x: (x.r0 - x.b0).mean())
        r['vs0050_hold'] = [round(tq(s.values)[0] * 100, 2), round(tq(s.values)[1], 2)]
        # CAR
        car = {}
        for hz in [10, 20, 40, 60, 120]:
            c = X[n50].copy(); c['ab'] = c[f'c{hz}'] - c.groupby('kq')[f'c{hz}'].transform('mean')
            car[hz] = round(c[c.top].groupby('kq').ab.mean().mean() * 100, 2)
        r['car_ex50'] = car
    out[per] = r
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
json.dump(out, open(HERE / 'r5_eventstats.json', 'w'), indent=1, ensure_ascii=False, default=float)

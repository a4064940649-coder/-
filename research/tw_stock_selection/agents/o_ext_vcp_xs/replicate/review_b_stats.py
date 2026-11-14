# -*- coding: utf-8 -*-
"""審查 B/D/E/F：事件 t 的分群穩健性、前視截斷測試、原始價對還原價差異歸因、消融差異檢定、最有利口徑。
python3 -I -B review_b_stats.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import rep_lib as L
HERE = os.path.dirname(os.path.abspath(__file__)) + '/'
D = L.Data()
sig = L.signals(D)
out = {}

# ---------------- B. 事件 t：月／季／半年分群，以及月份和的 Newey-West
ev = L.event_study(D, sig)
def ct(x, g):
    r = L.cluster_t(x, g); return r['mean'], r['t'], r['G']
def nw_month(x, ym, lags):
    # 以月份為單位的分群和，加上相鄰月份的自共變異（Bartlett）
    x = np.asarray(x); n = len(x); m = x.mean(); e = x - m
    months = pd.period_range(min(ym), max(ym), freq='M').astype(str)
    s = pd.Series(e).groupby(np.asarray(ym)).sum().reindex(months).fillna(0.0).values
    v = (s ** 2).sum()
    for l in range(1, lags + 1):
        v += 2 * (1 - l / (lags + 1)) * (s[l:] * s[:-l]).sum()
    return m / np.sqrt(v / n ** 2)
ev['q'] = ev.date.dt.to_period('Q').astype(str); ev['hy'] = ev.date.dt.year.astype(str) + np.where(ev.date.dt.month <= 6, 'H1', 'H2'); ev['yr'] = ev.date.dt.year.astype(str)
y = ev.date.dt.year
B = {}
for nm, m in (('all', np.ones(len(ev), bool)), ('sel_2006_2014', (y >= 2006) & (y <= 2014)), ('val_2015_2020', (y >= 2015) & (y <= 2020)), ('from2019', y >= 2019)):
    for h in (20, 60):
        d = ev[m & ev['r%d' % h].notna()]
        for bn, bcol in (('etf', 'etf%d' % h), ('pool', 'pool%d' % h)):
            x = (d['r%d' % h] - d[bcol]).values
            B['%s_h%d_%s' % (nm, h, bn)] = dict(n=int(len(x)), mean=float(x.mean()), t_month=ct(x, d.ym.values)[1], t_quarter=ct(x, d.q.values)[1], t_halfyear=ct(x, d.hy.values)[1],
                                                 t_year=ct(x, d.yr.values)[1], t_nw_month_lag3=float(nw_month(x, d.ym.values, 3)), G_month=ct(x, d.ym.values)[2], G_quarter=ct(x, d.q.values)[2],
                                                 naive_t=float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), median=float(np.median(x)),
                                                 trimmed_mean_1pct=float(np.sort(x)[int(len(x) * .01):len(x) - int(len(x) * .01)].mean()))
out['event_t_robust'] = B
for k, v in B.items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
# 挑選→驗證 差異檢定
for bn in ('etf', 'pool'):
    a = B['sel_2006_2014_h60_' + bn]; b = B['val_2015_2020_h60_' + bn]
    for tk in ('t_month', 't_quarter'):
        se = np.sqrt((a['mean'] / a[tk]) ** 2 + (b['mean'] / b[tk]) ** 2)
        out['flip_h60_%s_%s' % (bn, tk)] = dict(diff=a['mean'] - b['mean'], t=(a['mean'] - b['mean']) / se)
        print('flip', bn, tk, out['flip_h60_%s_%s' % (bn, tk)])
# 同窗 0050 與同池的差（風格）：
d = ev[ev.r60.notna()]
for nm, m in (('sel', (d.date.dt.year >= 2006) & (d.date.dt.year <= 2014)), ('val', (d.date.dt.year >= 2015) & (d.date.dt.year <= 2020))):
    out['pool_minus_etf_h60_' + nm] = float((d.pool60 - d.etf60)[m].mean())
print({k: v for k, v in out.items() if k.startswith('pool_minus')})

# ---------------- D1. 前視：截斷測試
def trunc_test(cut):
    D2 = L.Data()
    ci = int(np.searchsorted(D2.dates.values, np.datetime64(cut), side='right'))
    for k in ('C', 'H', 'L', 'V', 'A', 'AH', 'AL', 'O'):
        a_ = getattr(D2, k).copy(); a_[ci:] = np.nan; setattr(D2, k, a_)
    D2.IDXC = D2.IDXC.copy(); D2.IDXC[ci:] = np.nan
    D2._pack()
    s2 = L.signals(D2)
    a = set(zip(sig.code[sig.di < ci], sig.date[sig.di < ci])); b = set(zip(s2.code, s2.date))
    return dict(cut=cut, n_full_before=len(a), n_trunc=len(b), identical=(a == b))
out['lookahead_truncation'] = [trunc_test(c) for c in ('2012-06-29', '2019-05-16', '2024-06-28')]
print(out['lookahead_truncation'])

# ---------------- D2. 原始價對還原價差異歸因
sigA = L.signals(D, mode='adj')
ka = set(zip(sig.code, sig.di)); kb = set(zip(sigA.code, sigA.di))
Fp = D.pack(D.F)
def fvar(code_di, look):
    res = []
    pos = {c: i for i, c in enumerate(D.codes)}
    for c, di_ in code_di:
        j = pos[c]; i = D.barno[di_, j] - 1
        w = Fp[max(i - look, 0):i + 1, j]
        res.append(bool(np.nanmax(w) / np.nanmin(w) - 1 > 1e-9))
    return float(np.mean(res)) if res else np.nan
D2o = {}
for nm, st in (('raw_only', ka - kb), ('adj_only', kb - ka), ('both', ka & kb)):
    D2o[nm] = dict(n=len(st), F_changed_in_250bars=fvar(st, 250), F_changed_in_75bars=fvar(st, 76), F_changed_in_26bars=fvar(st, 26))
# 哪個條件翻的：對方模式下只拿掉單一條件是否就出訊號
conds = ('trend', 'rs', 'contract', 'cross')
abl = {m: {c: set(zip(*[L.signals(D, mode=m, ablate=(c,))[k] for k in ('code', 'di')])) for c in conds} for m in ('raw', 'adj')}
def attrib(st, other_mode):
    cnt = {c: 0 for c in conds}; multi = 0
    for k in st:
        hit = [c for c in conds if k in abl[other_mode][c]]
        if len(hit) == 0: multi += 1
        for c in hit: cnt[c] += 1
    return dict(single_condition=cnt, need_multiple=multi)
D2o['raw_only']['blocked_in_adj_by'] = attrib(ka - kb, 'adj')
D2o['adj_only']['blocked_in_raw_by'] = attrib(kb - ka, 'raw')
out['raw_vs_adj'] = D2o
print(D2o)

# ---------------- E. 消融差異檢定（20 日、60 日對同池超額；月份分群 OLS）
def cl_ols_diff(y_, d_, g_):
    X = np.column_stack([np.ones(len(y_)), d_.astype(float)])
    b = np.linalg.lstsq(X, y_, rcond=None)[0]
    u = y_ - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    df = pd.DataFrame({'g': g_, 'u0': u * X[:, 0], 'u1': u * X[:, 1]}).groupby('g').sum().values
    G = len(df)
    V = XtXi @ (df.T @ df) @ XtXi * G / (G - 1)
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1]))
full_keys = set(zip(ev.code, ev.di))
E = {}
for ab in ('trend', 'contract', 'voldry', 'breakvol', 'rs', 'minvol'):
    s = L.signals(D, ablate=(ab,))
    e2 = L.event_study(D, s)
    infull = np.array([k in full_keys for k in zip(e2.code, e2.di)])
    for h in (20, 60):
        m = e2['r%d' % h].notna().values
        for bn in ('pool', 'etf'):
            x = (e2['r%d' % h] - e2['%s%d' % (bn, h)]).values[m]
            dif, t = cl_ols_diff(x, infull[m], e2.ym.values[m])
            ex = x[~infull[m]]
            ce = L.cluster_t(ex, e2.ym.values[m][~infull[m]])
            E['%s_h%d_%s' % (ab, h, bn)] = dict(n_full=int(infull[m].sum()), n_extra=int((~infull[m]).sum()), mean_full=float(x[infull[m]].mean()), mean_extra=float(ex.mean()), t_extra=ce['t'],
                                                 diff_full_minus_extra=dif, t_diff=t)
    print(ab, {k: {a: round(b, 4) for a, b in v.items()} for k, v in E.items() if k.startswith(ab + '_h20')})
out['ablation_diff'] = E

# ---------------- F. 最有利口徑：還原價訊號＋空位 0050＋E5／E6（另加壓力成本、分段、拿掉最好的交易）
etf = pd.Series(D.ETF, index=D.dates)
F = {}
def seg(nav):
    o = {}
    for nm, a, b in (('2006_2012', '2006-01-01', '2012-12-31'), ('2013_2018', '2013-01-01', '2018-12-31'), ('2019_', '2019-01-01', '2026-10-22'),
                     ('sel_2006_2014', '2006-01-01', '2014-12-31'), ('val_2015_2020', '2015-01-01', '2020-12-31'), ('hold_2021_', '2021-01-01', '2026-10-22'), ('2024_', '2024-01-01', '2026-10-22')):
        o[nm] = float(L.ann(nav, a, b) - L.ann(etf, a, b))
    return o
def drop_best_years_t(nav, k=3):
    df = pd.DataFrame({'s': nav, 'b': etf.reindex(nav.index)})
    me = df.groupby([df.index.year, df.index.month]).tail(1)
    r = me.pct_change().dropna(); x = r['s'] - r['b']
    yr = x.groupby(x.index.year).sum().sort_values(ascending=False)
    keep = ~x.index.year.isin(yr.index[:k])
    z = x[keep]
    return float(z.mean() / (z.std(ddof=1) / np.sqrt(len(z)))), [int(v) for v in yr.index[:k]]
for sm, s in (('raw', sig), ('adj', sigA)):
    for ex in ('E3', 'E5', 'E6'):
        for stress in (0.0, 0.003):
            r = L.run_portfolio(D, s, ex, 'etf', stress=stress)
            p = L.perf(D, r)
            tt, yrs = drop_best_years_t(r['nav'])
            tr = r['trades']; pnl = tr.proceeds - tr.cost
            key = '%s_%s_etf%s' % (sm, ex, '_stress' if stress else '')
            F[key] = dict(full_ann=p['full']['ann'], full_ex=p['full']['ex'], full_t=p['full']['t'], y19_ann=p['2019']['ann'], y19_ex=p['2019']['ex'], y19_t=p['2019']['t'], mdd=p['mdd'],
                          seg=seg(r['nav']), t_drop_best3y=tt, best3y=yrs, n_trades=int(len(tr)), win=float((tr.ret > 0).mean()), top5_pnl_share=float(pnl.sort_values(ascending=False).head(5).sum() / pnl.sum()))
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in F[key].items()})
            if sm == 'adj' and ex == 'E5' and not stress:
                # 拿掉 PnL 最大的 5 筆交易的訊號再跑
                top = tr.loc[pnl.sort_values(ascending=False).head(5).index]
                print(top[['code', 'sig_date', 'entry', 'exit', 'ret', 'cost', 'proceeds']].to_string())
                kill = set(zip(top.code, top.sig_date))
                s2 = s[[k not in kill for k in zip(s.code, s.date)]]
                p2 = L.perf(D, L.run_portfolio(D, s2, ex, 'etf'))
                F[key + '_drop_top5_trades'] = dict(full_ex=p2['full']['ex'], full_t=p2['full']['t'], y19_ex=p2['2019']['ex'], y19_t=p2['2019']['t'])
                print('  drop top5:', F[key + '_drop_top5_trades'])
                yy = pd.DataFrame({'s': r['nav'], 'b': etf.reindex(r['nav'].index)}); ye = yy.groupby(yy.index.year).tail(1).pct_change().dropna()
                F[key + '_yearly_ex'] = {int(i.year): float(v) for i, v in (ye.s - ye.b).items()}
                print('  yearly ex', {k: round(v, 3) for k, v in F[key + '_yearly_ex'].items()})
out['best_case'] = F
json.dump(out, open(HERE + 'review_b_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)

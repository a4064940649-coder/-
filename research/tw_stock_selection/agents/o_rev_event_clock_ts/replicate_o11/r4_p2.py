import pandas as pd, numpy as np, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
SCR = '/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/'
exec(open(os.path.join(HERE, 'r3_gate_sum.py')).read().split("def load(fn)")[0])  # 只取交易日曆與 aday（自己寫的）
D = pd.read_pickle(os.path.join(HERE, 'gateD.pkl'))
r5 = pd.read_csv(SCR + 'extras/rev_5y.csv.gz', dtype={'code': str})
rh = pd.read_csv(SCR + 'extras/rev_hist.csv.gz', dtype={'code': str})
rev = pd.concat([r5[['code', 'ym', 'revenue', 'yoy']], rh[['code', 'ym', 'revenue', 'yoy']]]).drop_duplicates(['code', 'ym'], keep='first')
print('rev ym range', rev.ym.min(), rev.ym.max())
yo = rev.pivot(index='ym', columns='code', values='yoy')
def rev3(m, codes):
    p = pd.Period(m, 'M'); ms = [str(p - k) for k in (1, 2, 3)]
    sub = yo.reindex(ms)[[c for c in codes if c in yo.columns]]
    return sub.mean(skipna=False).reindex(codes)
D['rev3'] = np.nan
for m, g in D.groupby('anchor'):
    D.loc[g.index, 'rev3'] = rev3(m, list(g.code)).values
D['rev_pct'] = D.groupby('anchor').rev3.rank(pct=True)
res = pd.read_csv('../data/firstseen_hist.csv', dtype={'code': str})
cmp = D[['anchor', 'code', 'rev_pct']].merge(res[['anchor', 'code', 'rev_pct']], on=['anchor', 'code'], suffixes=('', '_res'))
c2 = cmp[cmp.anchor >= '2021-04'].dropna()
print('rev_pct vs researcher (2021-04+): n', len(c2), 'corr', c2.rev_pct.corr(c2.rev_pct_res), 'mean abs diff', (c2.rev_pct - c2.rev_pct_res).abs().mean())
D['a'] = D.t_T.map(aday)
D['rel'] = [pos[pd.Timestamp(dl)] - pos[a] if pd.notna(a) else np.nan for dl, a in zip(D.deadline, D.a)]
out = {}
def p2(mask, tag):
    G = D[mask & D.rev_pct.notna() & D.rel.notna()].copy()
    e = G[G.rel >= 3]; l = G[G.rel <= 0]
    pm = G.groupby('anchor').apply(lambda g: g[g.rel >= 3].rev_pct.mean() - g[g.rel <= 0].rev_pct.mean()).dropna()
    t = pm.mean() / (pm.std(ddof=1) / np.sqrt(len(pm)))
    G['yr'] = G.anchor.str[:4]
    yrs = {}
    for y, gy in G.groupby('yr'):
        q = gy.groupby('anchor').apply(lambda g: g[g.rel >= 3].rev_pct.mean() - g[g.rel <= 0].rev_pct.mean()).dropna()
        if len(q) > 1: yrs[y] = dict(mean=q.mean(), t=q.mean() / (q.std(ddof=1) / np.sqrt(len(q))), n=len(q))
    out[tag] = dict(n=len(G), n_early=len(e), early_pct=len(e) / len(G) * 100, n_late=len(l), late_pct=len(l) / len(G) * 100,
                    pooled=e.rev_pct.mean() - l.rev_pct.mean(), anchor_mean=pm.mean(), t=t, n_anchors=len(pm),
                    n_valid_before_dropna=int(mask.sum()), by_year=yrs)
p2(D.t_valid, 'gate')
p2(D.t_reason.notna() & ~D.t_old, 'tasi_all_firstseen_excl_old')
D.to_pickle(os.path.join(HERE, 'gateD2.pkl'))
print(json.dumps(out, indent=1, default=float))
json.dump(out, open(os.path.join(HERE, 'p2_rep.json'), 'w'), indent=1, default=float)

import sys, json; sys.path.insert(0, '.')
from spa import *
exec(open('famdef.py').read())
mm = pd.read_csv(AG / 'o_main_spa_breaks/data/missing_members.csv')
print(mm.span.value_counts().to_dict(), mm.similar_member.value_counts().to_dict())
S = pd.read_parquet(OUT / 'scan_ret.parquet')
b50 = win_ret(e0050())
segs = {'long': [(0, 157), (157, 245)], 'recent': [(179, 213), (213, 245)]}
res = {}
for ver in ('report', 'pess'):
    Z = {}
    for _, r in mm.iterrows():
        src = r.similar_member
        base = (S[src].to_numpy() - b50) if src in S.columns else X50[src].to_numpy()
        z = np.full(NW, np.nan)
        for (a, b), t in zip(segs[r.span], (r.t1, r.t2)):
            if ver == 'pess': t = max(t, 2.0)
            seg = base[a:b]; seg = seg - np.nanmean(seg)
            z[a:b] = seg + t * np.nanstd(seg, ddof=1) / math.sqrt(np.isfinite(seg).sum())
        Z['SYN::' + r.id] = z
    Z = pd.DataFrame(Z, index=X50.index)
    cov = Z.isna().sum() <= 6
    cols = FL + list(Z.columns[cov.values])
    D = pd.concat([X50[FL], Z.loc[:, cov.values]], axis=1).fillna(0.0).to_numpy()
    idx = sb_indices(NW, 6, 10000, 20261028 + 6)
    o = spa(D, cols.index(FOCUS), 6, idx, stepm=False, unstd=False)
    res[ver] = dict(n_syn=int(cov.sum()), p=o['p_SPA_c_focus'], t_max=o['t_max'], best=cols[o['argmax']])
    print(ver, res[ver])
json.dump(res, open(OUT / 'synth.json', 'w'), indent=1)

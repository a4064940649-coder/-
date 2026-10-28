import sys; sys.path.insert(0, '.')
from rlib import *
S = pd.read_parquet(OUT / 'scan_ret.parquet'); En = pd.read_parquet(OUT / 'ens_ret.parquet')
M = pd.concat([S, En], axis=1)
sig = M.round(12).T.apply(lambda r: hash(tuple(np.where(np.isnan(r.values), -999, r.values))), axis=1)
M = M.loc[:, (~sig.duplicated(keep='first')).values]
fold = pd.Series([('strategy_ensemble' if k.startswith('ENS::') else k.split('/')[0]) for k in M.columns], index=M.columns)
print('members', M.shape[1], 'folders', fold.nunique())
v4 = pd.read_parquet(AG / 'o_v4_revenue_patch_official/navs_v4.parquet')
b50 = win_ret(e0050()); bew = win_ret(v4['AB_V4|EW'])
X50 = M.sub(b50, axis=0); XEW = M.sub(bew, axis=0)
cov = {}
for w, s0 in WSTART.items():
    n = NW - s0; tol = max(2, 0.025 * n)
    miss = M.iloc[s0:].isna().sum()
    cov[w] = (miss <= tol)
    print(w, n, 'covered', int(cov[w].sum()))
C = pd.DataFrame(cov); C['folder'] = fold
X50.to_parquet(OUT / 'X50.parquet'); XEW.to_parquet(OUT / 'XEW.parquet'); C.to_csv(OUT / 'cover.csv')
print(C[C.W_long].folder.value_counts().to_string())

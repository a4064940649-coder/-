import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
P = rlib.load_panel(); close = P['close_raw'].ffill()
S = rlib.survivors(); F = rlib.full_build('V2'); ADV = rlib.adv(120); A = rlib.anchors()
ME = rlib.month_last_td('2006-01-01', '2013-12-31')
rows = []; imp_list = []
for d in ME:
    codes = S.loc[d][S.loc[d] & (close.loc[d] > 0)].index
    m = F.loc[d, codes] * close.loc[d, codes]
    r = rlib.impute_ratio(m, ADV.loc[d, codes], pd.Series(True, index=codes))
    imp = m.isna()
    m[imp] = ADV.loc[d, codes][imp] * r
    top = m.nlargest(100).index
    ii = [x for x in top if imp[x]]
    imp_list += [(str(d.date()), x) for x in ii]
    rows.append(dict(date=d, n_imp=len(ii)))
R = pd.DataFrame(rows).set_index('date')
print('mean', R.n_imp.mean(), 'max2008+', R.loc['2008':].n_imp.max())
print(R.groupby(R.index.year).n_imp.mean())
print(pd.DataFrame(imp_list, columns=['d', 'code']).code.value_counts())
# extrapolated > 1 year from nearest anchor, among top100
TD = rlib.td_index(); AA = rlib.anchor_arrays(A)
ext = {}
for d in ME:
    codes = S.loc[d][S.loc[d] & (close.loc[d] > 0)].index
    m = (F.loc[d, codes] * close.loc[d, codes]).dropna()
    r = rlib.impute_ratio(F.loc[d, codes] * close.loc[d, codes], ADV.loc[d, codes], pd.Series(True, index=codes))
    mm = F.loc[d, codes] * close.loc[d, codes]; mm = mm.fillna(ADV.loc[d, codes] * r)
    for x in mm.nlargest(100).index:
        if x in AA:
            dts = AA[x][3]
            dist = np.abs((dts - np.datetime64(d)) / np.timedelta64(1, 'D')).min()
            if dist > 366:
                ext[x] = ext.get(x, 0) + 1
print('extrap>1y', ext)

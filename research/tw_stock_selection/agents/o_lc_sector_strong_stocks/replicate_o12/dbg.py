import sys; sys.path.insert(0, '.')
import pandas as pd, numpy as np
import rlib as R
ds = sys.argv[1]; codes = sys.argv[2].split(',')
t = R.POS[pd.Timestamp(ds)]
U, mc = R.universe(t); ind = R.ind_cached(R.IDX[t]); fin = R.finmask(ind)
sc, rv = R.scores(t, U, fin)
o = R.order_key(sc, rv, U)
p1 = R.pct(R.S52[t], U); p2 = R.pct(R.RS[t], U)
df = pd.DataFrame({'code': [R.COLS[u] for u in U], 'ind': ind[U], 'fin': fin[U], 's52': R.S52[t, U], 'p1': p1, 'rs': R.RS[t, U], 'p2': p2, 'rev': rv, 'score': sc})
df['rank'] = np.nan; df.loc[o, 'rank'] = np.arange(len(o))
print(df.sort_values('rank').head(12).round(3).to_string())
pd.set_option("display.precision", 6); print(df[df.code.isin(codes)].to_string()); print("nU", len(U), "n s52", np.isfinite(R.S52[t,U]).sum(), "n rs", np.isfinite(R.RS[t,U]).sum(), "n rev", np.isfinite(rv[~fin[U]]).sum(), "nfin", fin[U].sum())

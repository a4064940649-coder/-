import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
P = rlib.load_panel(); TD = rlib.td_index(); close = P['close_raw'].ffill(); S = rlib.survivors()
AR = rlib.anchors_raw(); A = rlib.anchors(); F = rlib.full_build('V2'); ADV = rlib.adv(120)
AR = AR[AR.date.between('2006-01-01', '2013-12-31')]
# D1b: large caps = full-build mcap top-N on that date among survivors
def topN(d, n):
    codes = S.loc[d][S.loc[d] & (close.loc[d] > 0)].index
    return set((F.loc[d, codes] * close.loc[d, codes]).nlargest(n).index)
tm = AR[AR.src == 'tejm'].set_index(['code', 'date']).shares
for other in ['gq', 'ga']:
    o = AR[AR.src == other].set_index(['code', 'date']).shares
    j = pd.concat([tm.rename('a'), o.rename('b')], axis=1, join='inner').reset_index()
    j['e'] = np.log(j.a / j.b).abs()
    for n in [100, 150, 200, 1000]:
        tops = {d: topN(d, n) for d in j.date.unique()}
        jj = j[[c in tops[d] for c, d in zip(j.code, j.date)]]
        print('D1b tejm vs', other, 'top', n, 'med %.5f n %d' % (jj.e.median(), len(jj)))
# coverage: ADV top-150 at month-end, distance to nearest anchor
ME = rlib.month_last_td('2006-01-01', '2013-12-31')
AA = rlib.anchor_arrays(A)
dl = pd.read_csv(os.path.join(rlib.RAW, 'center', 'delist_official.csv'), dtype=str)
lm = pd.read_csv(os.path.join(rlib.RAW, 'center', 'listing_master.csv'), dtype=str)
delisted = set(lm[pd.to_datetime(lm.delist_date, errors='coerce') < '2021-01-01'].code)
for advn in [120, 60]:
    AD = rlib.adv(advn)
    rec = []
    for d in ME:
        codes = S.loc[d][S.loc[d]].index
        top = AD.loc[d, codes].nlargest(150).index
        for x in top:
            dist = np.inf
            if x in AA:
                dist = np.abs((AA[x][3] - np.datetime64(d)) / np.timedelta64(1, 'D')).min()
            rec.append((d, x, dist, x in delisted))
    C = pd.DataFrame(rec, columns=['d', 'code', 'dist', 'dl'])
    print('coverage ADV%d n %d <=1q %.3f <=1y %.3f surv<=1q %.3f dl<=1q %.3f dl share %.3f' % (advn, len(C), (C.dist <= 92).mean(), (C.dist <= 366).mean(), (C[~C.dl].dist <= 92).mean(), (C[C.dl].dist <= 92).mean(), C.dl.mean()))
# P2: o12 old backcast error at checkpoints 2006-2012, truth anchor at c, rank by truth mcap
o12 = pd.read_parquet(os.path.join(rlib.RAW, 'center', 'o12_shares_full.parquet')); o12.index = pd.DatetimeIndex(o12.index).as_unit('ns')
spec = json.load(open(os.path.join(rlib.TOP, 'data', 'GATE_SPEC.json')))
CK = [pd.Timestamp(x) for x in spec['D1a_2006']['checkpoints'] if x < '2013']
rec = []
for c in CK:
    codes = S.loc[c][S.loc[c] & (close.loc[c] > 0)].index
    tr = A[A.date == c].set_index('code').shares
    sh = F.loc[c, codes].copy(); ii = [x for x in codes if x in tr.index]; sh[ii] = tr[ii]
    m = (sh * close.loc[c, codes]).dropna().sort_values(ascending=False)
    rk = pd.Series(np.arange(1, len(m) + 1), index=m.index)
    for x in m.index[:120]:
        if x in tr.index and pd.notna(o12.loc[c, x]):
            rec.append((c, x, rk[x], np.log(o12.loc[c, x] / tr[x])))
R = pd.DataFrame(rec, columns=['d', 'code', 'rk', 'e']); R['a'] = R.e.abs()
print('P2 rank', R[R.rk <= 20].a.median(), R[(R.rk > 20) & (R.rk <= 59)].a.median(), R[(R.rk >= 60)].a.median())
print('P2 year', R.groupby(R.d.dt.year).a.median().round(4).to_dict())
print(R.loc[R.e.idxmax()].to_dict(), R.loc[R.e.idxmin()].to_dict())

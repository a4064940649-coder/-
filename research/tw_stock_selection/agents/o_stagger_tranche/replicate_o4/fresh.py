import sys, pathlib, json; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tranche import *
NV = pd.read_parquet(pathlib.Path(__file__).parent / 'scan_all_navs.parquet')
E = C.E50
def exc_common(col):
    nav = NV[col].dropna(); return C.excess(nav, [d for d in SD if d >= nav.index[0]])
def exc_own(col, j):
    nav = NV[col].dropna()
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]; wd = [(m, s) for m, s in days if s in C.POS]
    x = C.excess(nav, [s for _, s in wd] + [C.END]); am = {s: m for m, s in wd}
    x.index = [am[d] for d in x.index]; return x
def seg(idx, name):
    if isinstance(idx[0], pd.Period): m = pd.PeriodIndex(idx)
    else: m = pd.PeriodIndex([pd.Period(d, 'M') for d in idx])
    P = lambda a, b: (m >= pd.Period(a, 'M')) & (m <= pd.Period(b, 'M'))
    return {'SEL': P('2006-04', '2018-12'), 'POST': P('2019-01', '2026-10'), 'S4': P('2024-01', '2026-10'), 'ALL': P('2006-04', '2026-10'),
            'S1': P('2006-01', '2013-12'), 'S2': P('2014-01', '2020-12'), 'S3': P('2021-03', '2026-10'), 'Y2008': P('2008-01', '2008-12'), 'VAL': P('2019-01','2023-12'), 'HOLD': P('2024-01','2026-10')}[name]
res = {}
for kind in ('common', 'own'):
    for tag in ('w0.5|gross', 'w0.5|normal', 'w1.0|normal', 'w0.0|normal'):
        X = pd.DataFrame({j: (exc_common(f'{tag}|J{j:+d}') if kind == 'common' else exc_own(f'{tag}|J{j:+d}', j)) for j in range(-8, 13)})
        fr = X[[1, 2, 3, 4, 5]].mean(axis=1) - X[list(range(-8, 0))].mean(axis=1)
        post = X[[0, 1, 2, 3]].mean(axis=1) - X[[9, 10, 11, 12]].mean(axis=1)
        for p in ('SEL', 'POST', 'S4', 'ALL'):
            y = fr[seg(fr.index, p)].dropna(); z = post[seg(post.index, p)].dropna()
            res[f'{kind}|{tag}|fresh|{p}'] = (round(float(y.mean() * 100), 3), round(C.tstat(y), 2), len(y))
            res[f'{kind}|{tag}|J0-3minusJ9-12|{p}'] = (round(float(z.mean() * 100), 3), round(C.tstat(z), 2))
        if tag == 'w0.0|normal':
            mm = X[seg(X.index, 'SEL')].mean() - 0  # 營收腿組合挑選期對 0050 月超額
            res[f'{kind}|revonly_SEL_ex_by_j'] = (round(float(mm.min() * 100), 3), round(float(mm.max() * 100), 3))
        if tag == 'w0.5|normal':
            sdv = X.std(axis=1, ddof=1)
            for p in ('SEL', 'S1', 'S2', 'Y2008', 'S3', 'S4'):
                res[f'{kind}|disp|{p}'] = round(float(sdv[seg(sdv.index, p)].mean() * 100), 3)
            if kind == 'common':
                res['corr_J1_J6'] = round(float(X[1].corr(X[6])), 3); res['corr_J1_J11'] = round(float(X[1].corr(X[11])), 3)
for k, v in res.items(): print(k, v)
json.dump(res, open(pathlib.Path(__file__).parent / 'fresh.json', 'w'), indent=1)

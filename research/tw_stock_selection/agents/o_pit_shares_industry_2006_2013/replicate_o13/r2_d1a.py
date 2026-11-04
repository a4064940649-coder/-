import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
ver = sys.argv[1] if len(sys.argv) > 1 else 'V2'
spec = json.load(open(os.path.join(rlib.TOP, 'data', 'GATE_SPEC.json')))
CK = pd.DatetimeIndex(spec['D1a_2006']['checkpoints'])
P = rlib.load_panel(); TD = rlib.td_index()
close = P['close_raw'].ffill()
S = rlib.survivors(); A = rlib.anchors(); AA = rlib.anchor_arrays(A)
F = rlib.full_build(ver) if ver == 'V2' else rlib.full_build(ver)
ADV = rlib.adv(120)
B = rlib.Builder(rlib.events(), ver)
TDv = TD.values
rows = []; errs = []
for c in CK:
    pc = TD.get_loc(c)
    surv = S.loc[c]; cl = close.loc[c]; adv = ADV.loc[c]
    codes = surv[surv & (cl > 0)].index
    tr = A[A.date == c].set_index('code').shares
    tr_codes = [x for x in codes if x in tr.index]
    truth_sh = pd.Series(np.nan, index=codes)
    truth_sh[tr_codes] = tr[tr_codes].values
    noanc = [x for x in codes if x not in tr.index]
    truth_sh[noanc] = F.loc[c, noanc].values
    loo_sh = truth_sh.copy()
    for x in tr_codes:
        apos, aL, src, dts = AA[x]
        keep = (np.abs((dts - np.datetime64(c)) / np.timedelta64(1, 'D')) > 20) & (np.abs(apos - pc) <= 520)
        loo_sh[x] = np.exp(B.value_at(x, pc, apos[keep], aL[keep]))
    def mc(sh):
        m = sh * cl[codes]
        r = rlib.impute_ratio(m, adv[codes], pd.Series(True, index=codes))
        imp = m.isna()
        m[imp] = adv[codes][imp] * r
        return m, imp
    mt, it = mc(truth_sh); ml, il = mc(loo_sh)
    t100 = mt.nlargest(100).index; l100 = ml.nlargest(100).index
    na = [x for x in t100 if x not in tr.index]
    e = np.log(loo_sh[[x for x in t100 if x in tr.index]] / truth_sh[[x for x in t100 if x in tr.index]])
    for x, v in e.items():
        errs.append((str(c.date()), x, v, A[(A.code == x) & (A.date == c)].src.iloc[0], bool(il[x])))
    rows.append(dict(date=str(c.date()), overlap=len(set(t100) & set(l100)), n_noanchor_top100=len(na), n_imp_truth_top100=int(it[t100].sum()), n_err=len(e), med=float(e.abs().median()), nloo_nan=int(e.isna().sum())))
R = pd.DataFrame(rows); E = pd.DataFrame(errs, columns=['date', 'code', 'err', 'src', 'loo_imputed'])
print(R.to_string())
ae = E.err.abs()
print(ver, 'pooled med %.4f p90 %.4f n %d nan %d' % (ae.median(), ae.quantile(0.9), ae.notna().sum(), ae.isna().sum()))
print('overlap median', R.overlap.median(), 'min', R.overlap.min(), 'noanchor mean', R.n_noanchor_top100.mean())
print(E.assign(a=ae).groupby('src').a.describe())
print(E.loc[ae.sort_values(ascending=False).index[:15]])
R.to_csv(os.path.join(rlib.OUT, 'd1a_%s.csv' % ver)); E.to_csv(os.path.join(rlib.OUT, 'd1a_err_%s.csv' % ver))

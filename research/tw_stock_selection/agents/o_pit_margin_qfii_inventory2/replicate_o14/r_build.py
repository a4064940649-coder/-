import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import numpy as np, pandas as pd
sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_margn.py')).read().split("if __name__")[0])

NANMODE = os.environ.get('NANMODE', 'strict')  # strict: need F(t) notna ; loose: any t in FB range
M = raw_v1()
have1 = np.zeros(D.shape, bool)
for v in M.values():
    have1 |= v.notna().values
R = M['tej'].combine_first(M['snap']).combine_first(M['bb']).combine_first(M['mh'])
anc2 = M['tej'].combine_first(M['snap'])          # V2 anchors: TEJ or official monthly snapshot
fb, _ = S.finbrain()
F = fb.reindex(index=idx, columns=cols)
fbr = (idx >= fb.index.min()) & (idx <= fb.index.max())
Fv = F.notna().values
C = F.fillna(0).cumsum().values
# zero anchor at first listing day for listings >= 2007-04-23 (accumulate inclusive of first day)
fl = first.reindex(cols)
zero = np.zeros(D.shape, bool)
for j, c in enumerate(cols):
    d = fl[c]
    if pd.notna(d) and d >= pd.Timestamp('2007-04-23') and d <= fb.index.max() and d >= idx[0]:
        i = idx.searchsorted(d)
        zero[i, j] = True
Cprev = np.vstack([np.zeros((1, C.shape[1])), C[:-1]])
A = anc2.values.copy()
# anchor value / cum-at-anchor matrices (anchor on raw day a: base = A[a] - C[a]; zero anchor: base = -Cprev[a])
base_raw = np.where(~np.isnan(A), A - C, np.nan)
base_zero = np.where(zero, -Cprev, np.nan)


def first_anchor_base(base_list):
    # earliest anchor row among combined anchors; base value of that anchor, carried forward
    B = np.full(D.shape, np.nan)
    rowmin = np.full(D.shape[1], np.inf)
    for b in base_list:
        r = np.where(~np.isnan(b).all(0), np.argmax(~np.isnan(b), axis=0).astype(float), np.inf)
        upd = r < rowmin
        for j in np.where(upd)[0]:
            B[:, j] = np.nan
            B[int(r[j]):, j] = b[int(r[j]), j]
        rowmin = np.minimum(rowmin, r)
    return B


def last_anchor_base(base_list):
    B = np.full(D.shape, np.nan)
    for b in base_list:
        B = np.where(~np.isnan(b), b, B)
    return pd.DataFrame(B).ffill().values


def build(B, have_raw, Rv):
    val = B + C
    ok = ~np.isnan(B) & fbr[:, None]
    if NANMODE == 'strict':
        ok &= Fv
    V = np.where(have_raw, Rv, np.where(ok, val, np.nan))
    return V


Rv = R.values
B2 = first_anchor_base([base_zero, base_raw])
V2 = build(B2, have1, Rv)
# V3: reset at every raw balance day (any raw source) + zero anchor
base_all = np.where(~np.isnan(Rv), Rv - C, np.nan)
B3 = last_anchor_base([base_zero, base_all])
V3 = build(B3, have1, Rv)
# fill gaps of <= 2 consecutive missing (forward only)
V3d = pd.DataFrame(V3, idx, cols)
isn = V3d.isna().values
run = np.zeros(isn.shape, int)
for i in range(isn.shape[0]):
    run[i] = np.where(isn[i], (run[i - 1] if i else 0) + 1, 0)
V3f = V3d.where(~(isn & (run <= 2)), V3d.ffill())
V3 = V3f.values

res = {}
for nm, V in [('V1', np.where(have1, Rv, np.nan)), ('V2', V2), ('V3', V3)]:
    h = ~np.isnan(V)
    r = report(h, nm)
    neg = (V < 0) & Dm
    r['neg_pct_of_valued_in_den'] = round(100 * neg.sum() / (h & Dm).sum(), 4)
    r['neg_pct_all_valued'] = round(100 * (V < 0).sum() / h.sum(), 4)
    r['neg_n'] = int(neg.sum())
    res[nm] = r
    print(nm, {s: (r[s]['all'], r[s]['pool'], r[s]['delisted'], r[s]['surv']) for s in SEG}, r['neg_pct_of_valued_in_den'], r['neg_pct_all_valued'], r['neg_n'])
# V2 vs raw accuracy check where both (diagnostic)
json.dump(res, open(f'{HERE}/my_build_{NANMODE}.json', 'w'), ensure_ascii=False, indent=1)

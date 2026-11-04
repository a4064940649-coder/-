import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rlib, numpy as np, pandas as pd
P = rlib.load_panel(); S = rlib.survivors(); A = rlib.anchors(); AA = rlib.anchor_arrays(A); TD = rlib.td_index()
ME = rlib.month_last_td('2006-01-01', '2013-12-31')
lm = pd.read_csv(os.path.join(rlib.RAW, 'center', 'listing_master.csv'), dtype=str)
dl = set(lm[pd.to_datetime(lm.delist_date, errors='coerce') < '2021-01-01'].code)
for advn in [120, 60]:
    AD = rlib.adv(advn); rec = []
    for d in ME:
        top = AD.loc[d][S.loc[d]].nlargest(150).index; pd_ = TD.get_loc(d)
        for x in top:
            dd = dt = np.inf
            if x in AA:
                dd = np.abs((AA[x][3] - np.datetime64(d)) / np.timedelta64(1, 'D')).min(); dt = np.abs(AA[x][0] - pd_).min()
            rec.append((dd, dt, x in dl))
    C = pd.DataFrame(rec, columns=['dd', 'dt', 'dl'])
    for lab, q, y in [('91d', C.dd <= 91, C.dd <= 365), ('63td', C.dt <= 63, C.dt <= 252), ('90d', C.dd <= 90, C.dd <= 365)]:
        print(advn, lab, 'all %.3f/%.3f surv %.3f dl %.3f' % (q.mean(), y.mean(), q[~C.dl].mean(), q[C.dl].mean()))

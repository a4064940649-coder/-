import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pickle, numpy as np
from r_trades import *
ev = load_events()
out = {}
rs = all9(ev, stress=True); pickle.dump(rs, open('res_stress.pkl', 'wb'))
for p in ['VAL', 'FULL', 'NEW']: print('stress', p, summ(rs, p))
for sl in [3e4, 3e5, 1e6]:
    r = all9(ev, slot=sl); pickle.dump(r, open(f'res_{int(sl)}.pkl', 'wb'))
    print('slot', sl, 'VAL', summ(r, 'VAL')['med'], 'FULL', summ(r, 'FULL')['med'])
    x = sub(r[('C1', 'CL')], 'FULL'); print('  C1xCL full', len(x), x.net_x.mean() * 100)
# ideal gross close-to-close: stock LA change - nothing; use per-trade entry/exit index
res = pickle.load(open('res_10w.pkl', 'rb'))
for p in ['FULL', 'VAL']:
    ms = []
    for k, df in res.items():
        x = sub(df, p)
        g = [np.exp(LA[r.exit_i, CI[r.code]] - LA[r.entry_i, CI[r.code]]) - 1 for r in x.itertuples()]
        ms.append(np.mean(g))
    print('gross close-to-close (raw stock ret)', p, np.median(ms) * 100)

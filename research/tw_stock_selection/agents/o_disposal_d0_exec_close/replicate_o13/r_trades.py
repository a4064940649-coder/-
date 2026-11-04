# per-trade run over 9 timings, holding rule per timing
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, numpy as np, pandas as pd
from r_exec import *
def trades(ev, ent, ex, slot=1e5, **kw):
    hold = {}; rows = []
    for r in ev.itertuples():
        if not (r.adopt and r.ann_ok): continue
        if r.code in hold and r.si <= hold[r.code]: continue
        if r.bear or not r.r4: continue
        o = run_trade(r, ent, ex, slot, **kw)
        if o is None: continue
        hold[r.code] = o['exit_i']
        o.update(code=r.code, s=r.s, e=r.e, second=r.second, match_min=r.match_min, regime=r.regime, ent=ent, ex=ex, fo=r.fo)
        rows.append(o)
    return pd.DataFrame(rows)
def all9(ev, **kw):
    return {(a, b): trades(ev, a, b, **kw) for a in ENTRIES for b in EXITS}
def tmonth(df, col='net_x'):
    m = df.groupby(df.s.dt.to_period('M'))[col].mean()
    return m.mean() / m.std(ddof=1) * np.sqrt(len(m)) if len(m) > 1 else np.nan
PER = dict(SEL=('2021-10-01', '2023-12-31'), VAL=('2024-01-02', '2026-10-22'), FULL=('2006-04-01', '2026-10-22'), NEW=('2026-08-10', '2026-10-22'),
           S1=('2006-04-01', '2012-12-31'), S2=('2013-01-01', '2018-12-31'), S3=('2019-01-01', '2026-10-22'), A=('2006-04-01', '2013-12-31'), B=('2014-01-01', '2026-10-22'))
def sub(df, p):
    a, b = PER[p]; return df[(df.s >= a) & (df.s <= b)]
def summ(res, p, col='net_x'):
    means = []; ts = []; ns = []
    for k, df in res.items():
        x = sub(df, p); means.append(x[col].mean()); ts.append(tmonth(x, col)); ns.append(len(x))
    return dict(med=np.median(means) * 100, p10=np.percentile(means, 10) * 100, tmed=np.median(ts), n=ns, means=np.round(np.array(means) * 100, 2).tolist())
if __name__ == '__main__':
    ev = load_events()
    import pickle
    res = all9(ev)
    pickle.dump(res, open('res_10w.pkl', 'wb'))
    for p in ['SEL', 'VAL', 'FULL', 'NEW', 'S1', 'S2', 'S3', 'A', 'B']:
        print(p, summ(res, p))
    for p in ['FULL', 'VAL']:
        x = sub(res[('C1', 'CL')], p); print('C1xCL', p, len(x), x.net_x.mean() * 100, tmonth(x))
    print('net50 VAL', summ(res, 'VAL', 'net50_x')['med'], 'FULL', summ(res, 'FULL', 'net50_x')['med'])

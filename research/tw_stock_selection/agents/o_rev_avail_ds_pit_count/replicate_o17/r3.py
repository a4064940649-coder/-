import pickle, pandas as pd, numpy as np, sys
RAW = '/home/user/-/research/tw_stock_selection/agents/o_rev_avail_ds_pit_count/raw/'
o = pickle.load(open('c0.pkl', 'rb')); F = o[False].copy()
U = pd.read_csv(RAW + 'disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
print(len(U), U.take_R1.value_counts().to_dict(), U.start.min(), U.end.max())
for filt in ('all', 'take_R1'):
    u = U if filt == 'all' else U[U.take_R1]
    G = {c: g for c, g in u.groupby('code')}
    def inn(c, d):
        g = G.get(c); d = pd.Timestamp(d)
        return g is not None and bool(((g.start <= d) & (d <= g.end)).any())
    F['sd'] = [inn(c, s) for c, s in zip(F.code, F.sig)]
    F['td'] = [inn(c, s) for c, s in zip(F.code, F.exec)]
    print('==', filt)
    for nm, a, b in (('2006-2013', '2006', '2014'), ('2014-2021-09-23', '2014', '2021-09-24'), ('2021-09-24+', '2021-09-24', '2099')):
        s = F[(F.sig >= a) & (F.sig < b)]; den = len(s)
        for k in ('sd', 'td'):
            x = s[s[k]]; n = len(x)
            top = x.code.value_counts()
            drop = (n - top.iloc[0]) / den if n else 0
            print(f'{nm} {k}: {n}/{den}={n/den:.4%} days={x.sig.nunique()} iss={x.code.nunique()} drop={drop:.4%}', x.sig.value_counts().head(2).to_dict() if n else '')
            if nm == '2006-2013' and k == 'sd': print(x[['sig', 'code']].values.tolist())

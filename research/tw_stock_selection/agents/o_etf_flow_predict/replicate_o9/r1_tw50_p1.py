import sys, json
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_etf_flow_predict/replicate_o9')
import pandas as pd, numpy as np
import rlib as R

ev = R.events()
ev = ev[ev.etf == 'TW50']
revs = R.tw50_reviews()
VERS = {'v1_pit': dict(lag=0, fix4763=False), 'v2_lag63': dict(lag=63, fix4763=False),
        'v3_liq': dict(lag=0, fix4763=False, liq=True), 'v4_fix4763': dict(lag=0, fix4763=True),
        'v5_v4lag63': dict(lag=63, fix4763=True)}
out = {}
rows = []
for vn, kw in VERS.items():
    TPa = Pa = Aa = TPd = Pd = Ad = 0
    miss_ranks = []
    for rv, tf, C in revs:
        adds, dels, rk, mem = R.predict_tw50(C, **kw)
        e = ev[ev.review == rv]
        aa = set(e[e.action == 'add'].code); ad = set(e[e.action == 'del'].code)
        TPa += len(set(adds) & aa); Pa += len(adds); Aa += len(aa)
        TPd += len(set(dels) & ad); Pd += len(dels); Ad += len(ad)
        if vn == 'v4_fix4763':
            for x in aa - set(adds):
                miss_ranks.append((rv, x, int(rk[x]) if x in rk.index else None))
            for x in adds:
                rows.append(dict(review=rv, C=str(C.date()), code=x, rank=int(rk[x]), hit=x in aa, kind='add'))
            for x in dels:
                rows.append(dict(review=rv, C=str(C.date()), code=x, rank=int(rk[x]) if x in rk.index else None, hit=x in ad, kind='del'))
    out[vn] = dict(n_rev=len(revs), add_prec=TPa / Pa, add_rec=TPa / Aa, del_prec=TPd / Pd, del_rec=TPd / Ad,
                   TPa=TPa, Pa=Pa, Aa=Aa, TPd=TPd, Pd=Pd, Ad=Ad)
    if vn == 'v4_fix4763':
        out[vn]['missed_add_ranks'] = miss_ranks
    print(vn, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[vn].items()})
pd.DataFrame(rows).to_csv(R.AG + '/o_etf_flow_predict/replicate_o9/r1_tw50_v4_preds.csv', index=False)
json.dump(out, open(R.AG + '/o_etf_flow_predict/replicate_o9/r1_tw50_p1.json', 'w'), indent=1, default=str)

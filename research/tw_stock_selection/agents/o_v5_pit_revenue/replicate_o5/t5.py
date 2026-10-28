import sys, pickle; sys.path.insert(0, '.')
import numpy as np, pandas as pd
import rcore5 as R
S = pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o5/s1.pkl','rb'))
nav = S['NAV']['AB_V5']; pl = S['P']['V5']; W = R.wins(pl)
AD = {'S1':'2006-04-01','S2':'2014-01-01','S3':'2021-03-12','S4':'2024-01-01','SEL':'2006-04-01','VAL':'2019-01-01'}
def cg(s, ad, b):
    b = min(pd.Timestamp(b), R.END)
    pp = s[s.index < ad].index; prev = pp[-1] if len(pp) else s.index[0]; end = s[s.index <= b].index[-1]
    return (s[end]/s[prev])**(365.25/(end-prev).days)-1
for k,(a,b) in R.SEGS.items():
    ex = R.excess(nav, W); x = ex[(ex.index>=a)&(ex.index<=b)]
    print(k, '%.1f / %.2f (%.1f)'%(cg(nav, AD[k], b)*100, R.tstat(x), cg(R.E50[R.E50.index>=nav.index[0]], AD[k], b)*100))
# bootstrap variants
v = R.excess(nav, W).to_numpy()
import math
def bp(x, mode, B=10000, blk=12, seed=7):
    n=len(x); rng=np.random.default_rng(seed); t0=R.tstat(x); z=x-x.mean(); nb=math.ceil(n/blk); c=0; cm=0
    for _ in range(B):
        st = rng.integers(0, n, nb) if mode!='noncirc' else rng.integers(0, n-blk+1, nb)
        ii=(st[:,None]+np.arange(blk)[None,:]).ravel()[:n]%n
        zz=z[ii]
        if R.tstat(zz)>=t0: c+=1
        if zz.mean()>=x.mean(): cm+=1
    return c/B, cm/B
print('boot circ', bp(v,'circ'), 'noncirc', bp(v,'noncirc'))

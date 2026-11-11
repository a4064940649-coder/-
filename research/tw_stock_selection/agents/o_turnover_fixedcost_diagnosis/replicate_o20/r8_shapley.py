import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from lib import *
from itertools import permutations
c=pd.read_csv(D+'shapley_coalitions.csv.gz',keep_default_na=False)
print(c.coalition.unique()[:70], c.groupby(['variant','tier']).size())
F=['T','X','F','K','I','C']
def key(s): return ''.join(sorted(s,key=F.index))
c['key']=c.coalition.apply(lambda s: key([ch for ch in s if ch in F]))
def shap(val):
    orders=list(permutations(F)); M={f:[] for f in F}; cc=[]; o19=None
    for o in orders:
        cur=[]; m={}
        for f in o:
            m[f]=-(val[key(cur+[f])]-val[key(cur)]); cur.append(f)
        for f in F: M[f].append(m[f])
        cc.append(sum(m[f] for f in 'XFKIC'))
    res={f:(np.mean(M[f]),min(M[f]),max(M[f])) for f in F}; res['cost']=(np.mean(cc),min(cc),max(cc))
    return res
def o19(val):
    v=lambda s:val[key(list(s))]
    return dict(T=-(v('T')-v('')),X=-(v('TX')-v('T')),F=-(v('TXF')-v('TX')),rest=-(v('TXFKIC')-v('TXF')),cost=-(v('TXFKIC')-v('T')))
for tier,col in [('100萬','CAGR_Y19on'),('100萬','CAGR'),('30萬','CAGR_POSTODD')]:
  for var in ['REAL_opt','REAL_p90','REAL_carry']:
    x=c[(c.tier==tier)&(c.variant==var)]
    val=x.groupby('key')[col].median().to_dict()
    print(tier,col,var,'none',round(val[''],2),'all',round(val['TXFKIC'],2))
    r=shap(val); print('  ',{k:tuple(round(q,2) for q in v) for k,v in r.items()})
    if var=='REAL_opt': print('   o19 order',{k:round(v,2) for k,v in o19(val).items()})
    # per-j then median
    pj=[]
    for j,g in x.groupby('j'):
        vj=g.set_index('key')[col].to_dict(); s=shap(vj); pj.append((s['cost'][0],s['T'][0]))
    pj=np.array(pj); print('   perj median cost',round(np.median(pj[:,0]),2),'T',round(np.median(pj[:,1]),2))

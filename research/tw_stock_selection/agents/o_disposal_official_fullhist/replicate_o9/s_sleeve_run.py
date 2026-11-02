import sys,json; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from s_sleeve import *
def nw_t(x,L):
    x=np.asarray(x); m=x.mean(); u=x-m; n=len(x); s=u@u/n
    for l in range(1,L+1): s+=2*(1-l/(L+1))*(u[l:]@u[:-l])/n
    return m/np.sqrt(s/n)
res={}
for tier in [1e6,3e6,1e7]:
    for mode in ['FIX','GROW']:
        for st in [False,True]:
            mr,dd=sleeve(tier,mode,st); s,ex=stats(mr)
            s['nw3']=round(nw_t(ex.values,3),3); s['nw6']=round(nw_t(ex.values,6),3)
            e19=ex[ex.index>=pd.Period('2019-01','M')]; s['nw3_19']=round(nw_t(e19.values,3),3)
            if mode=='FIX' and tier==1e6 and not st:
                b=e50m.reindex(mr.index); bl=0.8*b+0.2*mr
                s['blend80_20_cagr']=round(((1+bl).prod()**(12/len(bl))-1)*100,2)
                ex.to_csv('focus_ex.csv')
            key=f'{mode}_{int(tier/1e4)}w_{"stress" if st else "normal"}'; res[key]=s; print(key,s)
json.dump(res,open('sleeve.json','w'),indent=1,default=str)

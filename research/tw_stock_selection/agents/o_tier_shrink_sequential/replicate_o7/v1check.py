import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
X,cl=load_family()
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet')
F=(fw.sub(fw.r0050,axis=0)*100)
meds=cl[cl.is_medoid].member_id.tolist()
M=X[meds]; P=M.mean(1)
vP=hacvar_mean(P.values); print('vP',vP)
xh=M.mean().values; se=np.sqrt([hacvar_mean((M[c]-P).values) for c in meds])
g,r=g2_fit(xh,se); print(r.success, r.fun, 'prior mean',(GRID*g).sum())
for f in ['AB','B10','EXEC_J']:
    s=np.sqrt(hacvar_mean((F[f]-P).values))
    p=post_grid(g,F[f].mean(),s); gr,pc=conv_normal(p,vP)
    print(f, F[f].mean(), s, summ(gr,pc))

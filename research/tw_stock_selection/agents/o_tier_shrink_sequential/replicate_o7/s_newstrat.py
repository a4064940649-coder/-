import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
S=setup_g2()
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet')
c=pd.read_parquet(RAW+'newstrat/capmatched_navs_oos.parquet')
def newclus(x, label):
    x=np.asarray(x,float); ok=np.isfinite(x); x=np.where(ok,x,0)
    s_i=np.sqrt(hacvar_mean(x-S['P'].values)); s_full=np.sqrt(hacvar_mean(x))
    p=post_grid(S['g'],x.mean(),s_i); gr,pc=conv_normal(p,S['vP'])
    sm=summ(gr,pc); print(label,'n_nan',(~ok).sum(),'raw',round(x.mean(),3),'Phi(t_HAC)',round(stats.norm.cdf(x.mean()/s_full),3),'post',round(sm['mean'],3),'P',round(sm['p_pos']+0.5*pc[np.isclose(gr,0)].sum(),3))
for col in ['F1_K5_N20_sqrt|normal','C50','C70','REP']:
    x=(wret(c[col])-fw.r0050.values)*100
    newclus(x.values,col)

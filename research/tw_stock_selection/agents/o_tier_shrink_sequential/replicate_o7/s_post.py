import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
S=setup_g2(); sw=sigma_w(S['X'],S['cl']); print('sigma_w',sw)
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet')
F=(fw.sub(fw.r0050,axis=0)*100)
for sww in (sw,0.1906):
  for f in ['AB','B10','B10_stress']:
    gr,p,info=v5_post(S,F[f].values,sw=sww)
    s=summ(gr,p); print(round(sww,4),f,F[f].mean(),info, {k:round(float(v),4) for k,v in s.items()}, 'ratio',s['mean']/F[f].mean(), 'p>=0',p[gr>=0].sum())

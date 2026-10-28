import pandas as pd, numpy as np, glob, json
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
OUT=B+'replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
h=pd.read_parquet(B+'raw/tw5m_head.parquet')
h0=h[h.hm=='09:00']; vd=h0.groupby('date').volume.apply(lambda v:(v>0).mean())
days=sorted([d for d,v in vd.items() if v>=0.99 and '2024-02-15'<=d<='2025-06-30'])
print(len(days))
cal=sd[sd.per=='CAL']; rho=np.median((cal.q05/cal.q10)[cal.q10>0]); print('rho',rho)
# ca_dec in the 14 days: use universe stock-days (pool, nonfin) — sd excludes some; use sd (with open check) and broader
sd['ca_obs']=(sd.vol_off-sd.vol_day)/sd.vol_off
cad=sd[sd.date.isin(days)].groupby('dec').ca_obs.median(); print('ca_dec (sd universe)',cad.round(4).to_dict(), sd.date.isin(days).sum())
u=sd[sd.nbar==53].copy(); u['OAb']=(u.vol_off-u.vol_day-u.dec.map(cad)*u.vol_off)-rho*u.q05
u['sh']=u.OAb/u.vol_off
for nm,g in [('ALL',u),('CAL',u[u.per=='CAL']),('VAL',u[u.per=='VAL'])]:
    s=g.groupby('dec').sh.median(); print(nm,len(g),'min',round(s.min(),4),'max',round(s.max(),4),s.round(4).to_dict())
# OA_b vs MIS median abs log error can't be matched (different dates) – skip

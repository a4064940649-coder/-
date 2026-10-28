# 獨立複製：建立股日樣本（只依凍結規格）
import pandas as pd, numpy as np
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
OUT=B+'replicate_o6/'
h=pd.read_parquet(B+'raw/tw5m_head.parquet')
d=pd.read_parquet(B+'raw/tw5m_daily.parquet')
o=pd.read_parquet(B+'data/official_daily.parquet'); o['date']=o.date.dt.strftime('%Y-%m-%d')
f=pd.read_parquet(B+'data/feat_all_2023.parquet'); f['date']=f.date.dt.strftime('%Y-%m-%d')
f=f.sort_values(['code','date'])
g=f.groupby('code')
for c in ['V60','sig60','vrank','close']:
    f[c+'_l']=g[c].shift(1)
f['pdate']=g['date'].shift(1)
# wide head
hms=['09:00']+[f'09:{m:02d}' for m in range(5,60,5)]
C=h.pivot_table(index=['code','date'],columns='hm',values='close',aggfunc='first')[hms]
Vq=h.pivot_table(index=['code','date'],columns='hm',values='volume',aggfunc='first')[hms]
O0=h[h.hm=='09:00'].set_index(['code','date'])['open']
full=C.notna().all(1)&Vq.notna().all(1)
C=C[full]; Vq=Vq[full]
lr=np.log(C.values[:,1:]/C.values[:,:-1])
sd=pd.DataFrame({'y':np.abs(lr).mean(1)},index=C.index)
for i,hm in enumerate(hms): sd['q'+hm[3:]]=Vq[hm].values
for i,hm in enumerate(hms[1:]): sd['ar'+hm[3:]]=np.abs(lr[:,i])
sd['open_y']=O0.reindex(sd.index).values
sd=sd.reset_index()
sd=sd.merge(d[['code','date','vol_day','nbar']],on=['code','date'],how='left')
sd=sd.merge(o[['code','date','open','vol']].rename(columns={'open':'open_off','vol':'vol_off'}),on=['code','date'],how='left')
sd=sd.merge(f[['code','date','V60_l','sig60_l','vrank_l','close_l','prevc','fin','pdate']],on=['code','date'],how='left')
print('rows',len(sd))
m=(sd.vrank_l<=300)&(sd.fin==False)&(sd.nbar>=50)&sd.sig60_l.notna()&sd.V60_l.notna()&(sd.sig60_l>0)
m&=(sd.open_y/sd.open_off-1).abs()<=0.005
bad=['2024-09-20','2025-01-21','2025-03-21','2025-04-07','2025-04-10','2025-11-24']
m&=~sd.date.isin(bad)
sd=sd[m].copy()
sd['per']=np.where((sd.date>='2024-02-15')&(sd.date<='2025-06-30'),'CAL',np.where((sd.date>='2025-07-01')&(sd.date<='2026-04-01'),'VAL','X'))
sd['dec']=np.ceil(sd.vrank_l/30).astype(int)
print(sd.per.value_counts())
print(sd.groupby(['per','dec']).size().unstack(0))
# prevc vs close_l check
print('prevc==close_l', np.mean(np.isclose(sd.prevc,sd.close_l)))
sd.to_parquet(OUT+'sd.parquet')

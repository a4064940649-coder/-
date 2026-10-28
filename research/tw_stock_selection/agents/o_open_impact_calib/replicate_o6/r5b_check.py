import pandas as pd, numpy as np, glob
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
OUT=B+'replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
h=pd.read_parquet(B+'raw/tw5m_head.parquet')
h0=h[h.hm=='09:00']; vd=h0.groupby('date').volume.apply(lambda v:(v>0).mean())
print(vd.sort_values(ascending=False).head(26).round(3).to_dict())
cal=sd[sd.per=='CAL']
r=(cal.q05/cal.q10).values
print('rho nanmedian',np.nanmedian(r),'q10>0',np.median(r[cal.q10.values>0]), 'all incl inf', np.median(np.where(np.isnan(r),np.inf,r)))
sl=pd.read_csv(B+'raw/screener/stock_list.csv',dtype={'code':str})
hist=pd.read_csv(B+'raw/screener/history.csv.gz',dtype={'code':str})
rows=[]
for fn in sorted(glob.glob(B+'raw/screener/vol_curve/*.csv')):
    dt=fn.split('/')[-1][:10]; v=pd.read_csv(fn,dtype={'code':str}).pivot_table(index='code',columns='time',values='vol_lots',aggfunc='first')
    for c in v.index:
        g=v.loc[c]; rows.append(dict(date=dt,code=c,v901=g.get('09:01'),v904=g.get('09:04')))
m=pd.DataFrame(rows).merge(hist[['date','code','volume']],on=['date','code'],how='left').merge(sl[['code','market']],on='code',how='left')
m=m.dropna(subset=['v901','v904','volume'])
m['sh']=(m.v901-(m.v904-m.v901)/3)*1000/m.volume
print(m.groupby('market').sh.agg(['size','median']))
f=pd.read_parquet(B+'data/feat_all_2023.parquet'); f['date']=f.date.dt.strftime('%Y-%m-%d'); f=f.sort_values(['code','date']); f['vrank_l']=f.groupby('code').vrank.shift(1)
m=m.merge(f[['code','date','vrank_l']],on=['code','date'],how='left')
print('in pool<=300',(m.vrank_l<=300).sum(), 'TWSE in pool',((m.vrank_l<=300)&(m.market=='TWSE')).sum())
print('dates',m.date.nunique())

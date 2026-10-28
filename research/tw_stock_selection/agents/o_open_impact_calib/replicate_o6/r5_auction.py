# 開盤／收盤集合競價占比、Yahoo÷官方、MIS 真值
import pandas as pd, numpy as np, glob, json
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
OUT=B+'replicate_o6/'
sd=pd.read_parquet(OUT+'sd.parquet')
h=pd.read_parquet(B+'raw/tw5m_head.parquet')
d=pd.read_parquet(B+'raw/tw5m_daily.parquet')
o=pd.read_parquet(B+'data/official_daily.parquet'); o['date']=o.date.dt.strftime('%Y-%m-%d')
f=pd.read_parquet(B+'data/feat_all_2023.parquet'); f['date']=f.date.dt.strftime('%Y-%m-%d'); f=f.sort_values(['code','date'])
f['vrank_l']=f.groupby('code').vrank.shift(1)
res={}
# Yahoo 全日量 ÷ 官方（前 300 池、非金融；每日中位數）
x=d.merge(o[['code','date','vol']],on=['code','date']).merge(f[['code','date','vrank_l','fin']],on=['code','date'])
x=x[(x.vrank_l<=300)&(~x.fin)&(x.vol>0)]
x['r']=x.vol_day/x.vol
dm=x.groupby('date').r.median()
print('Yahoo/official: median of daily medians',dm.median(),'pooled median',x.r.median(), 'ndays',len(dm))
print('days with daily median<0.75 or >1.0:',list(dm[(dm<0.75)|(dm>1.0)].index))
res['yahoo_over_off_daymed']=float(dm.median())
# ρ_y
cal=sd[sd.per=='CAL']
rr=(cal.q05/cal.q10).replace([np.inf],np.nan).dropna(); rr=rr[rr>0]
rho=np.median(cal.q05/cal.q10); print('rho_y (CAL median q05/q10)',rho)
# 09:00 有量的日子
h0=h[h.hm=='09:00']; vd=h0.groupby('date').volume.apply(lambda v:(v>0).mean())
days900=sorted(vd[vd>0.5].index); print('days with 09:00 volume',len(days900),days900)
# ca_dec：這些天 (官方−Yahoo)/官方 依十分位中位數
x['dec']=np.ceil(x.vrank_l/30)
cad=x[x.date.isin(days900)].assign(c=lambda t:(t.vol-t.vol_day)/t.vol).groupby('dec').c.median()
print('ca_dec',cad.round(4).to_dict())
# OA_b
u=sd.copy(); u=u[u.nbar==53]
u=u.merge(x[['code','date','r']],on=['code','date'],how='left')
u['ca']=u.dec.map(cad)
u['OAb']=(u.vol_off-u.vol_day-u.ca*u.vol_off)-rho*u.q05
u['share']=u.OAb/u.vol_off
sh=u.groupby('dec').share.median(); print('OA_b/official share median by decile',sh.round(4).to_dict(),'n',len(u))
for per in ['CAL','VAL']:
    print(per, u[u.per==per].groupby('dec').share.median().round(4).to_dict())
res['oa_share_dec']=sh.to_dict()
# MIS
hist=pd.read_csv(B+'raw/screener/history.csv.gz',dtype={'code':str})
rows=[]
for fn in sorted(glob.glob(B+'raw/screener/vol_curve/*.csv')):
    dt=fn.split('/')[-1][:10]; v=pd.read_csv(fn,dtype={'code':str}).pivot_table(index='code',columns='time',values='vol_lots',aggfunc='first')
    for c in v.index:
        g=v.loc[c]; rows.append(dict(date=dt,code=c,v901=g.get('09:01'),v904=g.get('09:04'),v1321=g.get('13:21'),v1324=g.get('13:24')))
m=pd.DataFrame(rows).merge(hist[['date','code','volume']],on=['date','code'],how='left')
m['OA']=(m.v901-(m.v904-m.v901)/3)*1000
m['oshare']=m.OA/m.volume
mm=m.dropna(subset=['v901','v904','volume'])
print('MIS stockdays with OA & official',len(mm),'median OA share',mm.oshare.median())
mm2=m.dropna(subset=['v901','v904','v1324','v1321','volume'])
print('with close too',len(mm2), mm2.oshare.median())
m['CA']=m.volume-m.v1324*1000-(m.v1324-m.v1321)*1000/3
print('MIS close auction share median',(m.CA/m.volume).median(), m.CA.notna().sum())
json.dump(res,open(OUT+'r5_auction.json','w'),indent=1)

import pandas as pd, numpy as np
D='../data/'
t=pd.read_csv(D+'tradelog_100n_REAL_opt_seed0.csv.gz',dtype={'code':str})
print(t.kind.value_counts(), t.cls.value_counts(), t.ctx.value_counts())
t['date']=pd.to_datetime(t.date)
rate=np.where(t.date<'2015-01-01',0.001425*0.6,0.001425*0.28)
for basis in ['notional','pxsh']:
  b=t.notional if basis=='notional' else t.px*t.shares
  f=np.maximum(20,np.round(b*rate,6)); print(basis,'fee max abs diff',np.nanmax(np.abs(f-t.fee)), (np.abs(f-t.fee)>0.01).mean())
  for tr in [0.003]:
    tx=np.where(t.side<0,b*tr,0); print(' tax diff',np.nanmax(np.abs(tx-t.tax)),(np.abs(tx-t.tax)>0.01).mean())
print(t[t.side<0].assign(r=lambda x:x.tax/x.notional).groupby('kind').r.describe())
print(t[t.shares==0].shape, t.groupby('kind')[['im','sq','tk','oddc']].describe().T)
# px vs open
w=t[t.kind=='W']; print(((w.px/w.open-1)*w.side - w.im).abs().describe())
o=t[t.kind=='O']; print(o[['open','px','notional','shares','oddc']].head())
d=t[t.kind=='D']; print(d.head())

# 獨立複製 o_lc_sector_rotation_pit：基礎資料（宇宙、市值、產業、訊號日）
import pandas as pd, numpy as np, pickle, os
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
D='/home/user/-/research/tw_stock_selection/agents/o_lc_sector_rotation_pit/'
OUT=D+'replicate_o12/'
close=pd.read_parquet(P+'close_raw.parquet'); adj=pd.read_parquet(P+'adj.parquet')
opn=pd.read_parquet(P+'open.parquet'); val=pd.read_parquet(P+'value.parquet')
good=close.notna().sum(axis=1)>=300
G=close.index[good.values]
close,adj,opn,val=[x.loc[G] for x in (close,adj,opn,val)]
codes=close.columns
sh=pd.read_parquet(D+'data/shares_full.parquet').reindex(columns=codes)
sh=sh.reindex(sh.index.union(G)).ffill().reindex(G)
# listing status from segments
lm=pd.read_csv(D+'raw/listing_master.csv',dtype=str).drop_duplicates('code').set_index('code')
listed=pd.DataFrame(False,index=G,columns=codes)
for c in codes:
    if c not in lm.index or pd.isna(lm.loc[c,'segments']): continue
    for seg in lm.loc[c,'segments'].split(';'):
        a,b=seg.split('~'); listed.loc[(G>=a)&(G<=b),c]=True
ok_code=np.array([not(c.startswith('91') or c.startswith('00')) for c in codes])
traded6=(val.fillna(0)>0).astype(float).rolling(6,min_periods=1).max()>0
adv120=val.rolling(120,min_periods=60).mean()
mcap=close*sh
ratio=(mcap/adv120).where(adv120>0)
med=ratio.median(axis=1)
elig=listed & traded6 & close.notna() & pd.DataFrame(np.tile(ok_code,(len(G),1)),index=G,columns=codes)
mc_imp=mcap.where(mcap.notna(), adv120.mul(med,axis=0))
mc_used=mc_imp.where(elig)
rk=mc_used.rank(axis=1,ascending=False,method='first')
top100=rk<=100
# industry
ip=pd.read_csv(D+'raw/industry_pit.csv',dtype=str)
bounds=sorted(set(ip.start.dropna())|set(ip.end.dropna()))
regimes=[b for b in bounds]
def mapping_at(day):
    day=str(day)[:10]
    r=ip[(ip.start<=day)&((ip.end.isna())|(ip.end>day))]
    return dict(zip(r.code,r.industry))
maps={b:{k:(v if isinstance(v,str) else "未分類") for k,v in mapping_at(b).items()} for b in regimes}
shidx=pd.read_parquet(D+'data/shares_full.parquet',columns=['2330']).index
GX=G.union(shidx)
mc_used=mc_used.reindex(GX).ffill(limit=30); top100=top100.reindex(GX).ffill(limit=30).astype(bool)
G=GX
pickle.dump(dict(G=G,mc=mc_used,top100=top100,maps=maps,regimes=regimes,imp=mcap.isna()&elig),open(OUT+'base.pkl','wb'))
print(len(G),regimes,top100.sum(axis=1).describe())
print('imputed in top100 per day', (top100&mcap.isna()).sum(axis=1).loc['2014':].describe())

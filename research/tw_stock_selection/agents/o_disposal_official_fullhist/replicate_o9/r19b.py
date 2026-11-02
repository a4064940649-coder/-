import sys; sys.path.insert(0,'.')
from s_sleeve import *
n=pd.read_parquet(A+'o_disposal_official_fullhist/navs_fixcorrected.parquet')['C1xCL|100萬|FIX|normal']
rm=n.resample('ME').last(); rm=rm/ rm.shift(1) - 1; rm.iloc[0]=n.resample('ME').last().iloc[0]/1-1
rm.index=rm.index.to_period('M')
mr,dd=sleeve(1e6,'FIX',False,sell_first=False)
c=pd.DataFrame({'mine':mr,'res':rm,'b':e50m.reindex(mr.index)})
c['dif']=c.mine-c.res
print((c.groupby(c.index.year)[['mine','res','b']].sum()*100).round(2))
print(((c.res-c.b).mean()*100), (c.res-c.b).mean()/((c.res-c.b).std()/np.sqrt(len(c))))
print(c.sort_values('dif',key=abs).tail(10))

# 用我自算的 V60、σ60（面板，t−1）與凍結 M3V 參數重算每筆整股衝擊金額，對照 cap_rows.impact_val
import pandas as pd, numpy as np
B='/home/user/-/research/tw_stock_selection/agents/o_open_impact_calib/'
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
OUT=B+'replicate_o6/'
r=pd.read_parquet(B+'data/cap_rows.parquet')
cols=pd.read_parquet(B+'data/px_traded.parquet').columns; idx=pd.read_parquet(B+'data/px_traded.parquet').index
val=pd.read_parquet(P+'value.parquet'); ret=pd.read_parquet(P+'ret.parquet'); cr=pd.read_parquet(P+'close_raw.parquet')
good=val.notna().sum(1)>=300
V60=val[good].rolling(60,min_periods=60).mean().reindex(val.index).ffill()
S60=ret[good].rolling(60,min_periods=60).std().reindex(ret.index).ffill()
V60=V60.reindex(idx)[cols]; S60=S60.reindex(idx)[cols]; C=cr.reindex(idx)[cols]
# t−1 值（以 exec index t 的前一列）
ti=r.t.values; jj=r.j.values
v=V60.values[ti-1,jj]; s=S60.values[ti-1,jj]; pc=C.values[ti-1,jj]
Qm=r.lot.values*0+r.n.values*1000*pc
print('Q match (n*1000*prevclose)',np.mean(np.isclose(Qm,r.Q.values,rtol=1e-6)))
d,b=0.48,0.48389678726280794
imp=b*s*(r.Q.values/v)**d
mine=imp*r.Q.values
rel=mine/r.impact_val.values-1
ok=np.isfinite(rel)
print('rows',len(r),'finite',ok.mean(),'median |rel|',np.median(np.abs(rel[ok])),'share within 1%',np.mean(np.abs(rel[ok])<0.01),'within 5%',np.mean(np.abs(rel[ok])<0.05))
print('sum impact mine / theirs', np.nansum(mine[ok])/r.impact_val.values[ok].sum())
# 若 impact_val 含 0.001 基礎
rel2=(imp+0.001)*r.Q.values/r.impact_val.values-1; print('with base: median |rel|',np.nanmedian(np.abs(rel2)))
r['mine']=mine
print(r.assign(rel=rel).groupby(['strat','tier','ver']).apply(lambda g:pd.Series(dict(sum_ratio=g.mine.sum()/g.impact_val.sum(),med=np.nanmedian(np.abs(g.rel))))).round(4))

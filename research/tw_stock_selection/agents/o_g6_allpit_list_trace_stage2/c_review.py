# -*- coding: utf-8 -*-
"""C 覆核（同一子代理另寫，不讀 B 的程式；只讀共用面板 parquet、官方清單、規格 json 與 K／複製員輸出的持股紀錄）。
python3 -I -B c_review.py"""
import json, gzip, math, numpy as np, pandas as pd
AG='/home/user/-/research/tw_stock_selection/agents'; O=AG+'/o_g6_allpit_list_trace_stage2/out'
PAN='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
cl=pd.read_parquet(PAN+'close_raw.parquet'); op=pd.read_parquet(PAN+'open.parquet'); val=pd.read_parquet(PAN+'value.parquet')
good=cl.notna().sum(axis=1)>=300; G=good[good].index
SPEC=json.load(open(AG+'/o_v7_listing_exec_engine/data/model_spec7.json'))
def tick(p): return 0.01 if p<10 else 0.05 if p<50 else 0.1 if p<100 else 0.5 if p<500 else 1.0 if p<1000 else 5.0
out={}
# (a) 3057 停牌與恢復
s=cl['3057']['2013-09-02':'2013-10-22']; nan_days=s[s.isna()].index; tr=s.dropna().index
v60=val.loc[G,'3057'].rolling(60,min_periods=60).mean()
out['3057']=dict(suspended_first=str(nan_days.min().date()),suspended_last=str(nan_days.max().date()),resume=str(tr[tr>nan_days.max()][0].date()),
  v60_on_1021=None if np.isnan(v60.get(pd.Timestamp('2013-10-21'),np.nan)) else float(v60[pd.Timestamp('2013-10-21')]),
  opens_after_resume=[(str(d.date()),float(op.loc[d,'3057'])) for d in tr[tr>nan_days.max()][:7]], open_1022=float(op.loc['2013-10-22','3057']))
K=json.load(gzip.open(O+'/k_orig_J6_s0_normal_1000000.json.gz','rt')); nav=pd.Series(K['nav']); nav.index=pd.to_datetime(nav.index)
# 原引擎讀法：10-22 換股日才賣；複製員讀法：恢復後每日 1 張（V60 缺 → 整股上限 1 張、零股上限 0）
pre=[r for r in K['rec'] if r['date']=='2013-09-18'][0]['S']['3057']
opens=[p for _,p in out['3057']['opens_after_resume']]
lots=int(pre//1000); rep_proceeds=sum(1000*p*(1-max(0.001,tick(p)/p))*(1-0.003-0.001425*0.28) for p in opens[:lots])
orig_value_1022=lots*1000*out['3057']['open_1022']*(1-0.001-0.003-0.001425*0.28)
eq_1022=None
out['3057'].update(held_0918=pre, lots=lots, rep_reading_proceeds=rep_proceeds, orig_reading_value_1022=orig_value_1022,
  diff_value=orig_value_1022-rep_proceeds)
# (b) 3007 表外永久停牌
dl=pd.read_csv(AG+'/o_infra_v3_rebuild/delist_official.csv',dtype=str); r=dl[dl.code=='3007'].iloc[0]
lt=cl['3007'].last_valid_index()
out['3007']=dict(delist_official_date=r['date'],last_trade=str(lt.date()),lt_plus14=str((lt+pd.Timedelta(days=14)).date()),
  in_merger_table=bool((pd.read_csv(AG+'/o_infra_v3_rebuild/merger_consideration.csv',dtype=str).code=='3007').any()),
  j6_rebalances=[x['date'] for x in K['rec'] if x['kind']=='reb' and '2007-04-01'<=x['date']<='2007-06-30'],
  shares=[x for x in K['rec'] if x['date']=='2007-04-18'][0]['pre'].get('3007'), last_close=float(cl['3007'][lt]))
# (c) 6414 零股上限（2018-03-23，F3：2020-10-26 前 floor(0.5×zcap_dec×V60_{t−1}/收盤)）
d=pd.Timestamp('2018-03-23'); dprev=G[G<d][-1]
v=val.loc[G,'6414'].rolling(60,min_periods=60).mean()[dprev]
# vrank：V60 ≥ 1e7 且收盤 ≥ 10 依 V60 排名（t−1）
V60all=val.loc[G].rolling(60,min_periods=60).mean().loc[dprev]; elig=V60all[(V60all>=1e7)&(cl.loc[dprev]>=10)]
rk=int((elig>elig['6414']).sum()+1); dec=min(10,math.ceil(rk/30))
zs=SPEC['zcap']['share_by_dec'][str(dec)]; cap_odd=math.floor(0.5*zs*v/cl.loc[d,'6414']+1e-9)
oa=SPEC['oa_share_by_dec'][str(dec)]; cap_lot=max(1,math.floor(0.2*oa*v/(1000*cl.loc[dprev,'6414'])+1e-12))
R=json.load(gzip.open(O+'/rep_J8_s0_N1000.json.gz','rt')); K8=json.load(gzip.open(O+'/k_orig_J8_s0_normal_10000000.json.gz','rt'))
r_pre=[x for x in R['rec'] if x['date']=='2018-03-06'][0]['S'].get('6414'); k_pre=[x for x in K8['rec'] if x['date']=='2018-03-23'][0]['pre'].get('6414')
out['6414']=dict(vrank=rk,dec=dec,v60_tm1=float(v),odd_cap=cap_odd,lot_cap=cap_lot,orig_pre=k_pre,rep_post_0306=r_pre,
  rep_sellable=min(int(r_pre//1000),cap_lot)*1000+min(int(r_pre%1000),cap_odd), orig_sellable=min(int(k_pre//1000),cap_lot)*1000+min(int(round(k_pre%1000)),cap_odd),
  value_left_rep=(r_pre-(min(int(r_pre//1000),cap_lot)*1000+min(int(r_pre%1000),cap_odd)))*float(cl.loc[d,'6414']))
print(json.dumps(out,ensure_ascii=False,indent=1,default=float))
json.dump(out,open(O+'/c_review.json','w'),ensure_ascii=False,indent=1,default=float)

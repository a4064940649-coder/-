import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
d=load_events()
# trading-day second variant
d=d.sort_values(['code','start','announce','ord'])
sec2=[];sec3=[]
for code,gg in d.groupby('code',sort=False):
    pe=[];pend=[]
    for r in gg.itertuples():
        sec2.append(any((x>=r.si-30)&(x<r.si) for x in pe))
        sec3.append(any((x>=r.start-pd.Timedelta(days=30))&(x<r.start) for x in pend))
        pe.append(r.ei); pend.append(r.end)
d['sec_td']=sec2; d['sec_raw']=sec3
tr=run(d); o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')].copy()
dd=d.set_index(['code','start'])
for col in ['second','sec_td','sec_raw']:
    sc=np.array([dd.loc[(c,s0),col] if not isinstance(dd.loc[(c,s0),col],pd.Series) else dd.loc[(c,s0),col].iloc[0] for c,s0 in zip(o.code,o.start)])
    o['net2']=o.gross-np.where(sc,0.012,0.008)
    print(col,sc.sum(),summ(o,'net2'))

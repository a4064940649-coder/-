import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from ev import *
d=load_events(); tr=run(d)
t=pd.read_csv(A+'o_disposal_official_fullhist/trades_all.csv.gz',dtype={'code':str},parse_dates=['date','exit_date'])
o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')]
to=t[(t.date>='2006-04-01')&(t.date<='2021-09-30')]
print(len(to), to.net.mean(), summ(to.rename(columns={'date':'s'})))
m=o.merge(to,left_on=['code','s'],right_on=['code','date'],how='outer',indicator=True,suffixes=('','_r'))
print(m._merge.value_counts())
mm=m[m._merge=='both']
for c,cr in [('ret','raw'),('ew','ew_r'),('b50','e50'),('cost','cost_r'),('net','net_r')]:
    dif=(mm[c]-mm[cr]).abs(); print(c,dif.max(), (dif>1e-6).sum())
x=mm[(mm.net-mm.net_r).abs()>1e-6]
print(x[['code','s','x','exit_date','second','second_r','ret','raw','ew','ew_r','cost','cost_r','delist','delisted']].to_string())
print(m[m._merge!='both'][['code','s','date','_merge']])
tr.to_csv('my_trades.csv.gz',index=False)

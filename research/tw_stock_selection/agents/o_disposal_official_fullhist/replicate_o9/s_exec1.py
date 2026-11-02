import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_disposal_official_fullhist/replicate_o9')
from xexec import *
tr=pd.read_csv('my_trades.csv.gz',dtype={'code':str},parse_dates=['s','x','e'])
out=[]
for r in tr.itertuples():
    ev=dict(code=r.code,si=r.si,ei=r.ei,regime=r.regime,mm=r.mm)
    a=exec_trade(ev,r.si,r.xi); b=exec_trade(ev,r.si,r.xi,stress=True)
    out.append((a['net_x'],b['net_x'],a['fill']))
tr['net_x'],tr['net_xs'],tr['fill']=zip(*out)
t=pd.read_csv(A+'o_disposal_official_fullhist/trades_all.csv.gz',dtype={'code':str},parse_dates=['date'])
m=tr.merge(t[['code','date','net_x','net_x_stress','fill10','cost_x']],left_on=['code','s'],right_on=['code','date'],suffixes=('','_r'))
print('n',len(m),'maxdiff net_x',(m.net_x-m.net_x_r).abs().max(),'mean abs',(m.net_x-m.net_x_r).abs().mean(),'stress',(m.net_xs-m.net_x_stress).abs().max(),'fill',(m.fill-m.fill10).abs().max())
m['dd']=(m.net_x-m.net_x_r)
print(m.sort_values('dd',key=abs,ascending=False)[['code','s','x','e','regime','mm','fill','fill10','net_x','net_x_r','net_xs','net_x_stress','cost_x']].head(12).to_string())
for nm,a,b in [('OOS','2006-04-01','2021-09-30'),('full','2006-04-01','2026-10-22')]:
    o=tr[(tr.s>=a)&(tr.s<=b)]; print(nm,summ(o,'net_x'),summ(o,'net_xs'))
tr.to_csv('my_trades_x.csv.gz',index=False)

import sys; sys.path.insert(0,"/home/user/-/research/tw_stock_selection/agents/o_turnover_fixedcost_diagnosis/replicate_o20")
from swaps import *
for tier,per in [('100n','2019-01-01'),('30n','2020-10-26')]:
    t=load_tl(tier); t['cs']=np.maximum(20,t.pxsh*0.001425*0.6)+t.tax+t.tick+t.imp+t.odd+0.004*t.notional
    a=t.groupby(['j','k','code','cls']).agg(cs=('cs','sum'),n=('notional','sum')).reset_index(); a['rs']=a.cs/a.n
    ex=pd.read_pickle(f'/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/ex{tier[:-1]}.pkl')
    ex=ex.merge(a[a.cls=='exit'][['j','k','code','rs']],on=['j','k','code'])
    en=a[a.cls=='entry'].groupby(['j','k']).rs.mean().rename('rs_en'); ex=ex.join(en,on=['j','k']); ex['rtcs']=ex.rs+ex.rs_en
    y=ex[ex.signal>=per]; print(tier,'stress approx <50',round(y[y.price<50].rtcs.median()*100,3),'>=100',round(y[y.price>=100].rtcs.median()*100,3))

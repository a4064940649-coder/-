import sys; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
W=pd.read_parquet(RAW+'newstrat/regime_W_full.parquet')
v4=pd.read_parquet(RAW+'navs/navs_v4.parquet')['AB_V4|normal'].dropna()
e=pd.read_csv(RAW+'navs/etf_0050.csv',parse_dates=['date']).set_index('date').adj
MB,MS,EBu,ES=0.001855,0.004855,0.001355,0.002355
dec=list(W.index)  # decision dates
S2=W.top5-W.top5.shift(6)
days=v4.index
rm=v4.pct_change(); re=e.reindex(days).ffill().pct_change() if False else e.pct_change().reindex(days)
# need 0050 return on main nav days: use asof levels
el=e.asof(days); re=el/el.shift(1)-1
def build(start='2006-04-11', mode='rot', wbar=0.4791666666666667):
    decs=[d for d in dec if d>=pd.Timestamp(start)]
    execd={d: days[days.searchsorted(d,side='right')] for d in decs if days.searchsorted(d,side='right')<len(days)}
    nav=pd.Series(np.nan,index=days); i0=days.get_loc(pd.Timestamp(start)); nav.iloc[i0]=1.0
    st=None; wm=we=0.0; tot=1.0; switches=0
    # holdings start in cash until first exec
    for i in range(i0+1,len(days)):
        d=days[i]
        wm*= (1+rm.iloc[i]); we*=(1+re.iloc[i])
        if st is None: tot=1.0
        else: tot=wm+we
        if d in execd.values():
            T=[k for k,v in execd.items() if v==d][0]
            if mode=='rot':
                new = 'E' if S2.loc[T]>0 else 'M'
                if st is None:
                    tot=tot/(1+(EBu if new=='E' else MB))
                elif new!=st:
                    switches+=1
                    if new=='E': tot=tot*(1-MS)/(1+EBu)
                    else: tot=tot*(1-ES)/(1+MB)
                st=new; wm,we=(tot,0.0) if st=='M' else (0.0,tot)
            else:
                if st is None:
                    tot=tot/(1+(1-wbar)*MB+wbar*EBu)
                else:
                    dm=wm-(1-wbar)*tot
                    c= dm*(MS+EBu) if dm>0 else -dm*(ES+MB)
                    tot=tot-c
                st='S'; wm=(1-wbar)*tot; we=wbar*tot
        nav.iloc[i]=tot if st is not None else 1.0
    return nav.dropna(), switches
rot,sw=build(); sta,_=build(mode='static')
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet')
xr=(wret(rot)-fw.r0050.values)*100; xd=(wret(rot)-wret(sta))*100
idx=fw.index; m=(idx>='2019-01-01')&(idx<='2022-12-31')
rr=wret(rot)[m]; print('SEL CAGR 12/n',(np.prod(1+rr)**(12/len(rr))-1), 'switches total',sw)
rs=wret(sta)[m]; print('SEL static', (np.prod(1+rs)**(12/len(rs))-1))
S=setup_g2()
for lab,x in [('rot vs 0050',xr.values),('rot - static d',xd.values)]:
    s_i=np.sqrt(hacvar_mean(x-S['P'].values)); s_full=np.sqrt(hacvar_mean(x))
    p=post_grid(S['g'],x.mean(),s_i); gr,pc=conv_normal(p,S['vP']); sm=summ(gr,pc)
    print(lab,'raw',round(x.mean(),3),'Phi(t_HAC)',round(stats.norm.cdf(x.mean()/s_full),3),'G2post',round(sm['mean'],3),round(sm['p_pos'],3))
    # plain normal prior N(0,0.5^2) for d
    if 'd' in lab:
        v=1/(1/0.25+1/s_full**2); mu=v*x.mean()/s_full**2; print('  N(0,.5^2) update', round(mu,3), round(stats.norm.cdf(mu/np.sqrt(v)),3))
x=xr.values; s_full=np.sqrt(hacvar_mean(x)); s_i=np.sqrt(hacvar_mean(x-S['P'].values))
p=post_grid(S['g'],x.mean(),s_full); print('full se no conv',summ(GRID,p)['mean'],summ(GRID,p)['p_pos'])
gr,pc=conv_normal(p,S['vP']); print('full se conv',summ(gr,pc)['p_pos'])
print('s_i',s_i,'s_full',s_full)

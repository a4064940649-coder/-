import numpy as np, pandas as pd
exec(open('r1_engine_navs.py').read().split('res={}')[0])
def sigdays(th):
    df=pd.Series(eidx,index=eidx); out=[]
    for (y,m),g in df.groupby([eidx.year,eidx.month]):
        g=g[g.index.day>=th]
        if len(g): out.append(g.index[0])
    return pd.DatetimeIndex(out)
for th in [10,11]:
  sd=sigdays(th)
  for cell in ['N1000','S100','O30']:
    tl=[];tl2=[]
    for j in range(13):
        s=pd.Series(z[f'{cell}|V0|J{j}'],index=eidx); a0,a1=jspan(j); s=s[(s.index>=a0)&(s.index<=a1)].ffill()
        pos=[eidx.get_loc(d)+j for d in sd if eidx.get_loc(d)+j<len(eidx)]
        rd=eidx[pos]; rd=rd[(rd>=a0)&(rd<=a1)]
        if rd[-1]!=a1: rd=rd.append(pd.DatetimeIndex([a1]))
        r=s[rd].pct_change().dropna(); rb=e[rd].pct_change().dropna(); x=r-rb
        tl.append(x.mean()/x.std()*np.sqrt(len(x)))
        x2=x.iloc[:-1] if True else x; tl2.append(x2.mean()/x2.std()*np.sqrt(len(x2)))
    print(th,cell,round(np.median(tl),4),round(np.median(tl2),4))

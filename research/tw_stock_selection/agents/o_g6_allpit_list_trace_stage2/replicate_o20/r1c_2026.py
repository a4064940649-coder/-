import numpy as np, pandas as pd, json
exec(open('r1b_t.py').read().split('for th in')[0])
sd=sigdays(10)
def tj(s,j,a0,a1):
    pos=[eidx.get_loc(d)+j for d in sd if eidx.get_loc(d)+j<len(eidx)]
    rd=eidx[pos]; rd=rd[(rd>=a0)&(rd<=a1)]
    if rd[-1]!=a1: rd=rd.append(pd.DatetimeIndex([a1]))
    x=(s[rd].pct_change()-e[rd].pct_change()).dropna(); return x.mean()/x.std()*np.sqrt(len(x))
out={}
for cell in ['N1000','S100','O30']:
  for v in ['V0','V3']:
    tl=[];y26_trunc=[];y26_full=[];y26b_trunc=[];y26b_full=[]
    for j in range(13):
        raw=pd.Series(z[f'{cell}|{v}|J{j}'],index=eidx); a0,a1=jspan(j)
        s=raw[(raw.index>=a0)&(raw.index<=a1)].ffill(); tl.append(tj(s,j,a0,a1))
        flat=raw.copy(); flat[flat.index>a1]=np.nan; flat=flat.ffill()
        full=raw.ffill()
        y0=pd.Timestamp('2025-12-31')
        y26_trunc.append(100*(flat.iloc[-1]/flat[flat.index<=y0].iloc[-1]-1)); y26_full.append(100*(full.iloc[-1]/full[full.index<=y0].iloc[-1]-1))
    d=np.array(y26_full)-np.array(y26_trunc)
    out[f'{cell}|{v}']=dict(t_med=float(np.median(tl)),y26_full_minus_flat_perj=[float(x) for x in np.round(d,3)],y26_full_med=float(np.median(y26_full)),y26_flat_med=float(np.median(y26_trunc)))
    print(cell,v,'t_med',round(np.median(tl),4),'2026 full-flat median-of-years diff',round(np.median(y26_full)-np.median(y26_trunc),3), 'perj range',d.min().round(3),d.max().round(3))
json.dump(out,open('r1c_2026.json','w'),indent=1)

#!/usr/bin/env python3
"""獨立逐筆迴圈重算 CRSI，對照 run.py 事件檔中的 crsi（抽樣 30 筆）。執行：python3 -I check_crsi.py"""
import numpy as np, pandas as pd
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel'
adj=pd.read_parquet(P+'/adj.parquet'); flag=pd.read_parquet(P+'/flag.parquet')
good=adj.notna().sum(axis=1)>=300; adj=adj.loc[good].mask(flag.loc[good]==5)
ev=pd.read_csv('events_F200_c10_E70_B.csv.gz',dtype={'code':str},parse_dates=['sig']).sample(30,random_state=1)
def rsi(x,n):
    ag=al=None; out=[]
    for i in range(len(x)):
        if i==0 or not np.isfinite(x[i]) or not np.isfinite(x[i-1]): out.append(np.nan); 
        d=x[i]-x[i-1] if i>0 else np.nan
        if not np.isfinite(d): continue
        g,l=max(d,0),max(-d,0)
        ag=g if ag is None else ag+(g-ag)/n; al=l if al is None else al+(l-al)/n
        out[-1:]=[]; out.append(100.0 if al==0 and ag>0 else 50.0 if al==0 else 100-100/(1+ag/al))
    return out
diffs=[]
for r in ev.itertuples():
    a=adj[r.code].ffill(limit=10); a=a[a.first_valid_index():]; x=a.to_numpy(); t=a.index.get_loc(r.sig)
    # RSI3：需全序列 Wilder 遞推（從第一個有效值起）
    d=np.diff(x); ag=al=None; r3=np.nan
    for i,dd in enumerate(d[:t]):
        if not np.isfinite(dd): continue
        g,l=max(dd,0),max(-dd,0); ag=g if ag is None else ag+(g-ag)/3; al=l if al is None else al+(l-al)/3
    r3=100.0 if al==0 and ag>0 else (50.0 if al==0 else 100-100/(1+ag/al))
    st=[0.0]
    for i in range(1,t+1):
        dd=x[i]-x[i-1]; p=st[-1]
        st.append((p+1 if p>0 else 1) if dd>0 else ((p-1 if p<0 else -1) if dd<0 else 0))
    sd=np.diff(np.array(st)); ag=al=None
    for dd in sd[1:]:
        g,l=max(dd,0),max(-dd,0); ag=g if ag is None else ag+(g-ag)/2; al=l if al is None else al+(l-al)/2
    rs=100.0 if al==0 and ag>0 else (50.0 if al==0 else 100-100/(1+ag/al))
    ret=x[1:]/x[:-1]-1; cur=ret[t-1]; prev=ret[t-101:t-1]; pr=100*np.mean(prev<cur)
    c=(r3+rs+pr)/3; diffs.append((r.code,str(r.sig.date()),round(r.crsi,3),round(c,3)))
for z in diffs: print(z)
print('max abs diff',max(abs(a-b) for _,_,a,b in diffs))

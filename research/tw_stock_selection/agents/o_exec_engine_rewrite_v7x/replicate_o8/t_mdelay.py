import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, json
import g8_zero as Z
from multiprocessing import Pool
D=Z.D; N=Z.N; TR=Z.TR; On=Z.On; Cn=Z.Cn; G=Z.G; XR=Z.XR; E50=Z.E50; CASHJ=Z.CASHJ
def sim(plan,b,s,t0,t1,var,ge):
    reb={e:pk for e,sg,pk in plan}; val=np.zeros(N); cash=1.0; nav=np.empty(t1-t0+1); pend={}
    for t in range(t0,t1+1):
        if t in reb:
            pk=reb[t]; tr=TR[t]&np.isfinite(On[t])
            h=val*np.where(tr,G[t]*On[t]/np.where(tr,Cn[t],1.0),1.0)
            frz=(h>0)&~tr
            for j in np.where(frz)[0]:
                if j in pend:
                    v0,base,ipay=pend[j]
                    if (t>=ipay if ge else t>ipay):
                        cash+=v0*E50[t-1]/E50[base]; h[j]=0; frz[j]=False; del pend[j]
                    continue
                if j in CASHJ and t>=CASHJ[j][4]:
                    v=CASHJ[j]
                    if var=='A': pend[j]=(h[j]*v[0],v[4],v[4]+20)
                    else: pend[j]=(h[j]*v[0],t-1,v[4]+20)
                    continue
                if j in XR and XR[j][1]<=t<XR[j][2]:
                    cash+=h[j]*XR[j][0]; h[j]=0; frz[j]=False
            htr=np.where(frz,0.0,h)
            for j in pend: htr[j]=0
            inpk=np.zeros(N,bool); inpk[pk]=True
            T=Z.solve_T(htr[pk],len(pk),htr[~inpk].sum(),htr.sum()+cash,b,s); cash=0.0
            nv=np.where(frz,h,0.0); nv[pk]+=T
            val=np.where(frz,nv,nv*np.where(tr,Cn[t]/np.where(tr,On[t],1.0),1.0))
        elif t>t0: val=val*G[t]
        for j,(v0,base,ipay) in pend.items(): val[j]=v0*E50[t]/E50[base]
        nav[t-t0]=val.sum()+cash
    return pd.Series(nav,index=D['IDX'][t0:t1+1])
PJ=Z.load_plans('j'); items=[(k,v) for k,v in PJ.items() if k[0]=='AB_V7_B10']
def job(a):
    (st,j,sd),plan=a; t0=plan[0][0]-1; t1=Z.TPOS[pd.Timestamp(Z.ENDS[f'{st}|J{j}'])]; b,s=Z.RATES['normal']
    return j,{(v,ge):sim(plan,b,s,t0,t1,v,ge).pct_change() for v in 'AB' for ge in (0,1)}
if __name__=='__main__':
    with Pool(4) as P: R=P.map(job,items,chunksize=4)
    I=pd.read_parquet(Z.O7/'navs_ideal7.parquet')
    for key in [(v,ge) for v in 'AB' for ge in (0,1)]:
        out={}
        for j in range(13):
            nv=(1+pd.concat([r[1][key] for r in R if r[0]==j],axis=1).mean(1).fillna(0)).cumprod()
            ref=I[f'B10|J{j}|mdelay'].dropna(); m=nv.reindex(ref.index)
            out[j]=float(((m/m.iloc[0])/(ref/ref.iloc[0])-1).abs().max())
        print(key, {k:f'{v:.1e}' for k,v in out.items()})

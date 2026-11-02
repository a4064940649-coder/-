import sys;sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_0050_tilt_pit/replicate_o8')
from rcore import *
W=World();W.build_pre()
O=pd.read_parquet(O8+'/data/oos_series.parquet')['T1|0'].dropna(subset=['T']);O.index=pd.to_datetime(O.index)
theirs=(O.T_g-O.B_g)
sd=W.signal_days(0,'2014-01','2026-09');pers=list(sd);dates=[sd[p] for p in pers]
hi=pd.read_parquet(V7+'/high.parquet').reindex(index=W.days,columns=W.cl.columns)
good=W.cl.notna().sum(1)>=300
adj=(1+W.rt.fillna(0)).cumprod().where(W.clf.notna())
# variant 52w: adj close / max adj close 250
variants={}
base_r52=W.r52.copy(); base_rev=W.rev.copy()
def run(name):
    out=[]
    for p,d in zip(pers,dates):
        i=W.days.get_loc(d); d1=W.days[min(i+21,len(W.days)-1)]
        nxt=sd.get(p+1); d1=nxt if nxt is not None else W.days[-1]
        w=W.weights(d,p,'T1')
        rr=((1+W.rt.loc[d:d1].iloc[1:].fillna(0)).prod()-1)
        out.append(((w['T']-w['B'])*rr.reindex(w['B'].index).fillna(0)).sum())
    s=pd.Series(out,index=dates);ix=s.index.intersection(theirs.index)
    dd=s[ix]-theirs[ix]
    print(name,'corr %.4f mine %.4f theirs %.4f diff/yr'%(np.corrcoef(s[ix],theirs[ix])[0,1],s[ix].mean()*100,theirs[ix].mean()*100),(dd.groupby(dd.index.year).mean()*100).round(2).to_dict(),flush=True)
run('base')
ag=adj[good]; W.r52=(ag/ag.rolling(250,min_periods=250).max()).reindex(W.days).where(W.cl.notna()); run('52w_250_strict')
W.r52=base_r52
R=pd.read_parquet(O8+'/raw/agents/yoy_long_R_FIX.parquet');R.index=pd.PeriodIndex(R.index,freq='M');R.columns=R.columns.astype(str)
W.rev=R.reindex(columns=base_rev.columns); run('rev_Ronly')
W.rev=base_rev
# rev: single latest month yoy
class _:
    pass
orig=W.rev
W.rev=orig.copy()
import types
def weights_last(self,d,per,which,**k):
    return World.weights(self,d,per,which,**k)

import pandas as pd, numpy as np, json
B='/home/user/-/research/tw_stock_selection/agents/'
D=B+'o_turnover_fixedcost_diagnosis/data/'
JEND={j:pd.Timestamp(v) for j,v in ((int(k.split('J')[1]),v) for k,v in json.load(open(B+'o_v7_listing_exec_engine/data/jscan_end_days.json')).items() if k.startswith('V7_B10|'))}
PER={'FULL':('2006-04-01','2100-01-01'),'Y19on':('2019-01-01','2100-01-01'),'POSTODD':('2020-10-26','2100-01-01'),'S0618':('2006-04-01','2018-12-31')}
def load_tl(tier):
    t=pd.read_csv(D+f'tradelog_{tier}_REAL_opt_seed0.csv.gz',dtype={'code':str})
    t['date']=pd.to_datetime(t.date)
    t['pxsh']=t.px*t.shares
    rate=np.where(t.date<'2015-01-01',0.001425*0.6,0.001425*0.28)
    t['fee_min_ex']=t.fee-t.pxsh*rate
    W=t.kind=='W'; O=t.kind=='O'; Dd=t.kind=='D'
    t['tick']=np.where(W,t.notional*np.minimum(t.im,t.tk),np.where(Dd,t.notional*t.im,0.0))
    t['imp']=np.where(W,t.notional*np.maximum(0,t.im-t.tk),0.0)
    t['odd']=np.where(O,t.notional*t.oddc,0.0)
    t['cost']=t.fee+t.tax+t.tick+t.imp+t.odd
    sq=t.sq.fillna(0)
    for h,n in [(0.5,'h05'),(1.0,'h10')]:
        t['cost_'+n]=t.fee+t.tax+t.notional*(h*t.tk+np.where(W,sq,0))
    return t
def years(per,j):
    a,b=PER[per]; a=max(pd.Timestamp(a),pd.Timestamp('2006-04-01')); b=min(pd.Timestamp(b),JEND[j])
    return (b-a).days/365.25
def drag(t,per,cols):
    a,b=PER[per]; x=t[(t.date>=a)&(t.date<=b)]
    out=[]
    for j,g in x.groupby('j'):
        y=years(per,j); out.append({c:(g[c]/g.avail).sum()/y*100 for c in cols}|{'j':j})
    return pd.DataFrame(out)

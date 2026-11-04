# 獨立複製 o14 OC：只依凍結規格重寫
import json, sys, numpy as np
P='/home/user/-/research/tw_stock_selection/agents/o_g8_protocol_fin_erratum/G8_PROTOCOL.json'
d=json.load(open(P)); R=np.array(d['data']['resid_post2019']); R24=np.array(d['data']['resid_2024'])
SD=6.141373712658076; C=1.0596378779547428; B=1.6787025976148637; CD=0.8946228207485066
H=48; UP=(12,24); DN=tuple(range(12,H+1,6)); LOOKS=sorted(set(UP)|set(DN))
STRESS=-0.5769618693715921

def boot(rng,res,L,Pn):
    n=len(res); p=1.0/L
    idx=np.empty((Pn,H),dtype=np.int64)
    idx[:,0]=rng.integers(0,n,Pn)
    new=rng.random((Pn,H))<p; jump=rng.integers(0,n,(Pn,H))
    for t in range(1,H):
        idx[:,t]=np.where(new[:,t],jump[:,t],(idx[:,t-1]+1)%n)
    return res[idx]

def gen(dgp,alpha,Pn,rng):
    if dgp=='D1': e=SD*rng.standard_normal((Pn,H))
    elif dgp=='D2': e=SD*rng.standard_t(4,(Pn,H))/np.sqrt(2)
    elif dgp=='D3': e=boot(rng,R,6,Pn)
    elif dgp=='D3_2024': e=boot(rng,R24,6,Pn)
    elif dgp=='D3_L3': e=boot(rng,R,3,Pn)
    elif dgp=='D3_L12': e=boot(rng,R,12,Pn)
    elif dgp=='D4_o8':
        st=rng.random(Pn)<0.0319/(0.0319+0.06); u=rng.random((Pn,H)); m=np.empty((Pn,H))
        for t in range(H):
            if t>0: st=np.where(st,u[:,t]>=0.06,u[:,t]<0.0319)
            m[:,t]=np.where(st,1.4,0.7)
        e=SD*m*rng.standard_t(4,(Pn,H))/np.sqrt(2)
    return alpha+e

def judge(x, up_absorb=True, refit=True, ddof=1):
    Pn=x.shape[0]; S=x.cumsum(1)
    sig=np.full(Pn,C*SD); refd=np.zeros(Pn,bool)
    up_t=np.full(Pn,999); dn_t=np.full(Pn,999)
    for n in LOOKS:
        if refit:
            s=x[:,:n].std(1,ddof=ddof); trig=(~refd)&(s>1.5*SD)
            sig=np.where(trig,np.maximum(C*SD,s),sig); refd|=trig
        z=S[:,n-1]/(sig*np.sqrt(n))
        alive=(dn_t==999)&((up_t==999) if up_absorb else True)
        if n in UP:
            thr=B*np.sqrt(24/n); hit=alive&(up_t==999)&(z>=thr); up_t[hit]=n
        if n in DN:
            alive=(dn_t==999)&((up_t==999) if up_absorb else True)
            hit=alive&(z<=-CD); dn_t[hit]=n
    lit=S[:,11]>0
    return up_t,dn_t,lit

def seed(base,alpha,k): return np.random.default_rng([base,int(round((alpha+3)*100)),k])
IDX={'D1':1,'D2':2,'D3':3,'D3_2024':4,'D3_L3':5,'D3_L12':6,'D4_o8':7}
ALPHAS=(-1,-0.5,0,0.5,1,2)

def summarize(dgp,base,shift=0.0,Pn=40000,**kw):
    out={}
    for a in ALPHAS:
        x=gen(dgp,a+shift,Pn,seed(base,a,IDX[dgp])); u,dn,lit=judge(x,**kw)
        out[a]=dict(up24=(u<=24).mean(),dn24=(dn<=24).mean(),dn48=(dn<=48).mean(),lit=lit.mean(),
                    cup=((u<=24)&lit).mean(),cfail=((~lit)|(dn<=24)).mean())
    s=dict(false_up=out[0]['up24'],false_up_se=np.sqrt(out[0]['up24']*(1-out[0]['up24'])/Pn),
           down_power=out[-1]['dn24'],false_down=out[0]['dn24'],down_m05=out[-0.5]['dn24'],
           up_p05=out[0.5]['up24'],up_p1=out[1]['up24'],up_p2=out[2]['up24'],
           literal_g8_a0=out[0]['lit'],literal_g8_p05=out[0.5]['lit'],literal_g8_m1=out[-1]['lit'],
           combined_up_a0=out[0]['cup'],combined_up_p1=out[1]['cup'],combined_fail_a0=out[0]['cfail'],
           combined_fail_m1=out[-1]['cfail'],down_a0_48=out[0]['dn48'],down_m1_48=out[-1]['dn48'])
    return {k:float(v) for k,v in s.items()}

if __name__=='__main__':
    base=int(sys.argv[1]) if len(sys.argv)>1 else 99001
    res={'primary':{},'sens':{},'stress':{}}
    for g in ('D1','D2','D3'): res['primary'][g]=summarize(g,base)
    for g in ('D3_2024','D3_L3','D3_L12','D4_o8'): res['sens'][g]=summarize(g,base+1)
    for g in ('D1','D2','D3'): res['stress'][g]=summarize(g,base+2,shift=STRESS)
    json.dump(res,open(f'oc_seed{base}.json','w'),indent=1)
    print('done',base)

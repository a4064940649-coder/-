import sys, json, pickle; sys.path.insert(0,'/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7')
from rlib import *
OUT='/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/replicate_o7/out/'
PP=pickle.load(open(OUT+'post.pkl','rb')); post=PP['post']
Z=np.load('/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/data/forward_priors.npz')
REG=json.load(open('/home/user/-/research/tw_stock_selection/agents/o_tier_shrink_sequential/prereg_forward.json'))['registration']
th=Z['theta']
# my own L1 priors: 0.5 KAPPA + 0.5 TICK cell posterior, projected to theta grid
fw=pd.read_parquet(RAW+'o6shrink/focus_windows.parquet'); F=(fw.sub(fw.r0050,axis=0)*100)
A=pd.read_parquet(RAW+'navs/navs_abavg.parquet'); EB=pd.read_parquet(RAW+'exec/bench.parquet'); T=pd.read_parquet(OUT+'navs_tick_tickstress_main.parquet')
z=wret(EB['AB_V4|normal'])*100
def to_grid(v,w):
    p=np.zeros(len(th)); i=np.clip(np.round((v-th[0])/0.02).astype(int),0,len(th)-1); np.add.at(p,i,w); return p/p.sum()
lines={}
for t in ['30萬','100萬','300萬','1000萬']:
    for m in ['FIX','GROW']:
        sK=F.B10.values+(wret(A[f'main|{t}|{m}|A'])*100-z).values; sT=F.B10.values+(wret(T[f'tick|{t}|{m}'])*100-z).values
        pK=to_grid(*mix_shift(*post['B10'],np.mean(sK-F.B10.values)+PP['jB'])); pT=to_grid(*mix_shift(*post['B10'],np.mean(sT-F.B10.values)+PP['jB']))
        p=0.5*pK+0.5*pT
        sig=0.5*(np.sqrt(lrv(sK))+np.sqrt(lrv(sT)))
        key=f'L1_MAIN_B10_EXEC_AT_{t}__{m}'; reg=REG['lines_L1_variants'][f'L1_MAIN_B10_EXEC@{t}|{m}']
        pz=Z[key]/Z[key].sum()
        print(key,'mine mean',round((th*p).sum(),4),'P',round(p[th>0].sum()+0.5*p[th==0].sum(),4),'sigma',round(sig,3),'| npz mean',round((th*pz).sum(),4),'reg sigma',round(reg['sigma_pp'],3),'maxabs pmf diff',np.abs(p-pz).max())
        lines[key]=dict(prior=p,sigma=sig,amin=0.5*max((th*p).sum(),0),up_ok=True)
for k,v in REG['lines'].items():
    if k.startswith('L1'): continue
    pz=Z[v['prior_key']]; pz=pz/pz.sum()
    lines[k]=dict(prior=pz,sigma=v['sigma_pp'] if k!='L2_MAIN80_DISP20' else v['sigma_pp'],amin=v['alpha_min'],up_ok=not k.startswith('L4'))
def run(L,alpha,B=4000,H=120,seed=1):
    rng=np.random.default_rng(seed)
    p=L['prior']; s2=L['sigma']**2; am=L['amin']
    pos=th>0; neg=th<am
    pp=p*pos; pp/=pp.sum(); pn=p*neg; pn/=pn.sum()
    S=np.zeros(B); logW=np.zeros(B)
    up=np.full(B,10**9); dn=np.full(B,10**9); rv=np.full(B,10**9)
    for n in range(1,H+1):
        x=rng.normal(alpha,np.sqrt(s2),B); S+=x; logW+=np.log1p(x/100)
        llr=(S[:,None]*th[None,:]-n*th[None,:]**2/2)/s2  # log LR(theta vs 0)
        mx=llr.max(1,keepdims=True)
        Eup=np.exp(mx[:,0])*(np.exp(llr-mx)@pp)
        lla=(S*am-n*am**2/2)/s2
        Edn=np.exp(mx[:,0]-lla)*(np.exp(llr-mx)@pn)
        post_=np.exp(llr-mx)*p[None,:]; post_/=post_.sum(1,keepdims=True)
        P=post_[:,pos].sum(1)+0.5*post_[:,th==0].sum(1)
        if L['up_ok']: up=np.where((up>n)&(Eup>=20)&(n>=24),n,up)
        dn=np.where((dn>n)&((Edn>=10)|((P<0.25)&(n>=18))),n,dn)
        rv=np.where((rv>n)&((np.exp(logW)<0.75)|(Edn>=20)),n,rv)
    return up,dn,rv
res={}
for k,L in lines.items():
    out={}
    for a in [0.0,0.2,0.5,0.76,-1.0]:
        up,dn,rv=run(L,a)
        out[a]=dict(up36=(up<=36).mean(),up120=(up<=120).mean(),dn24=(dn<=24).mean(),dn60=(dn<=60).mean(),dn120=(dn<=120).mean(),rv24=(rv<=24).mean())
    res[k]=out
    print(k,'a=0 up120',out[0.0]['up120'],'| a=.2 up36',out[0.2]['up36'],'a=.5 up36',out[0.5]['up36'],'a=.76 up120',out[0.76]['up120'],'| a=-1 dn24/60/120',out[-1.0]['dn24'],out[-1.0]['dn60'],out[-1.0]['dn120'],'rv24',out[-1.0]['rv24'],flush=True)
json.dump({k:{str(a):v for a,v in o.items()} for k,o in res.items()},open(OUT+'seqsim.json','w'),ensure_ascii=False,indent=1)

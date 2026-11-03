# 獨立複製：H52 產業指數、權重建構、紙上引擎
import pandas as pd, numpy as np, pickle, sys, json
P='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
D='/home/user/-/research/tw_stock_selection/agents/o_lc_sector_rotation_pit/'
OUT=D+'replicate_o12/'
IDXMEM=sys.argv[1] if len(sys.argv)>1 else 'top100'   # 產業指數成分：top100 或 all
WF=sys.argv[2] if len(sys.argv)>2 else 'I1'          # I1: over=min(A/k,3B)；I2: 3B 上限水位法
K=int(sys.argv[3]) if len(sys.argv)>3 else 3
A=float(sys.argv[4]) if len(sys.argv)>4 else 0.15
ENG=sys.argv[5] if len(sys.argv)>5 else 'cc'
TURN=sys.argv[6] if len(sys.argv)>6 else 'open'
b=pickle.load(open(OUT+'base.pkl','rb'))
G=b['G']; mc=b['mc']; top=b['top100']; maps=b['maps']; regimes=b['regimes']
close=pd.read_parquet(P+'close_raw.parquet').reindex(G); adj=pd.read_parquet(P+'adj.parquet').reindex(G)
opn=pd.read_parquet(P+'open.parquet').reindex(G)
codes=close.columns
adjf=adj.ffill()
aopen=(opn*adj/close)
aopen=aopen.where(aopen.notna(),adjf)
def regime_of(day):
    r=[x for x in regimes if x<=str(day)[:10]]; return r[-1]
# reset dates: first trading day with day>=11 each month
ser=pd.Series(G,index=G)
resets=ser[ser.dt.day>=11].groupby([ser[ser.dt.day>=11].dt.year,ser[ser.dt.day>=11].dt.month]).first().values
resets=pd.DatetimeIndex(resets)
gi={d:i for i,d in enumerate(G)}
# industry index per regime
A_=adjf.values
idx_cache={}
def build_index(reg):
    mp=maps[reg]
    inds=sorted(set(mp.values()))
    ind_of=np.array([mp.get(c,'未分類') for c in codes])
    M=np.zeros((len(codes),len(inds)))
    for j,s in enumerate(inds): M[:,j]=(ind_of==s)
    M[list(codes).index('2330'),:]=0
    I=np.full((len(G),len(inds)),np.nan)
    lvl=np.ones(len(inds))
    rp=[gi[r] for r in resets]
    for a,bb in zip(rp,rp[1:]+[len(G)-1]):
        if IDXMEM=='top100': mem=top.values[a]
        else: mem=mc.notna().values[a]
        w=np.nan_to_num(mc.values[a])*mem
        base=A_[a]
        okb=np.isfinite(base)&(w>0)
        gr=np.where(okb,A_[a:bb+1]/np.where(okb,base,1),0)
        gr=np.nan_to_num(gr)
        W=(w*okb)[:,None]*M
        tot=W.sum(0)
        seg=gr@W
        segn=np.where(tot>0,seg/np.where(tot>0,tot,1),1.0)
        I[a:bb+1]=segn*lvl
        lvl=I[bb].copy()
    df=pd.DataFrame(I,index=G,columns=inds)
    h52=df/df.rolling(252,min_periods=252).max()
    return df,h52
for reg in regimes: idx_cache[reg]=build_index(reg)
# signal dates
def D_m(y,m):
    day=15 if (y,m) in [(2019,2),(2023,4)] else 10
    return pd.Timestamp(y,m,day)
anchors=[(y,m) for y in range(2012,2027) for m in range(1,13) if (y,m)<=(2026,10)]
sigd={}
for j in range(13):
    lst=[]
    for (y,m) in anchors:
        p=G.searchsorted(D_m(y,m))
        if p+j>=len(G): continue
        lst.append(((y,m),G[p+j]))
    # drop duplicated dates (keep later anchor)
    seen={}
    for am,d in lst: seen[d]=am
    sigd[j]=[(am,d) for am,d in lst if seen[d]==am]
FIN='金融保險業'
def weights(d,prev_set,k=K,A=A,cap_mult=3.0,buf=2,sig='H52'):
    i=gi[d]; reg=regime_of(d); mp=maps[reg]
    mem=top.values[i]; m=np.nan_to_num(mc.values[i])*mem
    bw=m/m.sum()
    ind=np.array([mp.get(c,'未分類') for c in codes])
    is2330=(codes=='2330')
    sect=pd.Series(bw[mem],index=ind[mem]); cnt=pd.Series(1,index=ind[mem][~is2330[mem]]).groupby(level=0).sum()
    Bs=pd.Series(bw[mem&~is2330],index=ind[mem&~is2330]).groupby(level=0).sum()
    h=idx_cache[reg][1].loc[d]
    rot=[s for s in Bs.index if s not in (FIN,'未分類') and cnt.get(s,0)>=2 and np.isfinite(h.get(s,np.nan))]
    if len(rot)<=k: 
        return bw, set()
    sc=pd.DataFrame({'s':h[rot].values,'b':Bs[rot].values},index=rot).sort_values(['s','b'],ascending=False)
    order=list(sc.index); rank={s:r+1 for r,s in enumerate(order)}
    keep=[s for s in order if s in prev_set and rank[s]<=k+buf]
    sel=list(keep)
    for s in order:
        if len(sel)>=k: break
        if s not in sel: sel.append(s)
    sel=sel[:k] if len(keep)<=k else sorted(keep,key=lambda s:rank[s])[:k]
    over={}
    if WF=='I1':
        for s in sel: over[s]=min(A/k,cap_mult*Bs[s])
    else:
        caps={s:cap_mult*Bs[s] for s in sel}; rem=A; free=list(sel); over={s:0.0 for s in sel}
        while free and rem>1e-12:
            q=rem/len(free); nf=[]
            for s in free:
                add=min(q,caps[s]-over[s]); over[s]+=add; rem-=add
                if caps[s]-over[s]>1e-12: nf.append(s)
            if len(nf)==len(free): break
            free=nf
    O=sum(over.values())
    others=[s for s in rot if s not in sel]; Bo=Bs[others].sum()
    if O>Bo:
        sc_=Bo/O; over={s:v*sc_ for s,v in over.items()}; O=Bo
    W=bw.copy()
    for s in rot:
        msk=mem&(ind==s)&~is2330
        tgt=Bs[s]+over[s] if s in sel else Bs[s]*(1-O/Bo)
        W[msk]=bw[msk]/Bs[s]*tgt
    return W,set(sel)
RATES={'normal':(0.000855+0.001,0.000855+0.003+0.001),'stress':(0.005,0.005),'zero':(0,0)}
def run(j,mode):
    sd=sigd[j]; recs=[]; prev=set(); hold=None
    for t,(am,d) in enumerate(sd):
        i=gi[d]
        if i+1>=len(G): break
        d1=G[i+1]
        dn=sd[t+1][1] if t+1<len(sd) else G[-1]
        if mode=='B': W,_=weights(d,set(),A=0.0)
        else: W,prev=weights(d,prev)
        # overnight of drifted holdings
        if hold is None: on=0.0; pre=W.copy()
        else:
            g=np.nan_to_num(aopen.values[i+1]/adjf.values[i],nan=1.0)
            g=np.where(hold>0,g,1.0)
            on=(hold*g).sum()-1; pre=hold*g/(hold*g).sum()
        if TURN=='close' and hold is not None: pre=hold
        tb=np.clip(W-pre,0,None).sum(); ts=np.clip(pre-W,0,None).sum()
        res={}
        for cm,(cb,cs) in RATES.items():
            res[cm]=cb*tb+cs*ts
        ii=gi[dn]
        endv=aopen.values[ii+1] if (ENG=='oo' and ii+1<len(G)) else adjf.values[ii]
        if ENG in ('oo','oc'): on=0.0
        g2=np.nan_to_num(endv/aopen.values[i+1],nan=1.0)
        g2=np.where(W>0,g2,1.0)
        rin=(W*g2).sum()-1
        hold=W*g2/(W*g2).sum()
        rec=dict(am=f'{am[0]}-{am[1]:02d}',d=d,dn=dn,turn=(tb+ts)/2)
        for cm in RATES: rec[cm]=(1+on)*(1-res[cm])*(1+rin)-1
        recs.append(rec)
    return pd.DataFrame(recs)
e=pd.read_csv(D+'raw/bench/etf_0050.csv',parse_dates=['date']).set_index('date').adj
out={}
for j in range(13):
    s=run(j,'S'); bb=run(j,'B')
    s['B_normal']=bb.normal.values; s['B_zero']=bb.zero.values; s['B_stress']=bb.stress.values; s['B_turn']=bb.turn.values
    s['e0050']=[e.asof(dn)/e.asof(d)-1 for d,dn in zip(s.d,s.dn)]
    out[j]=s
tag=f'{IDXMEM}_{WF}_k{K}_A{int(A*100)}'+('' if ENG=='cc' else '_'+ENG)+('' if TURN=='open' else '_t'+TURN)
pickle.dump(out,open(OUT+f'win_{tag}.pkl','wb'))
print('done',tag)

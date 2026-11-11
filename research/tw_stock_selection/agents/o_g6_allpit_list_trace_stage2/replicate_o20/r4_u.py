# 獨立複製員 o20：依 ER2_SPEC 文字重建 U_opt／U_p90／U_carry（2006～2013），strict 與 posthoc（名稱加 delist_official）
import json, sys, numpy as np, pandas as pd
MODE=sys.argv[1]  # strict | posthoc
src=open('r3_c0.py').read().split("allidx=c.index")[0].replace("OPT=dict(pool_order=sys.argv[1] if len(sys.argv)>1 else 'top300_then_filter')","OPT=dict(pool_order='filter_then_top300')")
exec(src)
if MODE=='posthoc':
    do=pd.read_csv(A+'o_infra_v3_rebuild/delist_official.csv',dtype={'code':str})
    for k,n in zip(do.code,do.name): names.setdefault(k,set()).add(str(n))
    FOR={k for k in cols if foreign(k)}
# 營收水準（期間）
lvl=R  # 期間×代碼
has=lvl.notna()
first={k:lvl.index[has[k]].min() for k in lvl.columns if has[k].any()}
last={k:lvl.index[has[k]].max() for k in lvl.columns if has[k].any()}
lastpx=c.apply(lambda s: s.last_valid_index())
P0912,P1212,P1301=pd.Period('2009-12','M'),pd.Period('2012-12','M'),pd.Period('2013-01','M')
PRI=['TRUNC','IFRS13','PRE12','HOLE','NEWLIST','BASE_LE0']
def cause(k,m):
    if k in FOR: return 'FOREIGN'
    if k not in first: return 'NEVER_DOM'
    cs=[]
    for x in [m-1,m-2,m-3,m-13,m-14,m-15]:
        v=lvl[k].get(x,np.nan) if x in lvl.index else np.nan
        if pd.isna(v):
            lp=lastpx[k].to_period('M') if lastpx[k] is not None else None
            if x>last[k] and lp is not None and P0912<=lp<=P1212: cs.append('TRUNC')
            elif x>=P1301 and last[k]>=P1212: cs.append('IFRS13')
            elif x>last[k]: cs.append('PRE12')
            elif x<first[k]: cs.append('NEWLIST')
            else: cs.append('HOLE')
        elif x in (m-13,m-14,m-15) and v<=0: cs.append('BASE_LE0')
    cs=[p for p in PRI if p in cs]
    return cs[0] if cs else 'OTHER'
GOV={'TRUNC','IFRS13','PRE12','HOLE','NEVER_DOM'}
allidx=c.index
def run(variant):
    held=set(); slots=[]; lists={}; changed=0
    for d in sd:
        v=V60.loc[d]; cl=c.loc[d]; nd=allidx[allidx.get_loc(d)+1]; m=pd.Period(d,'M')
        ok=v[(v>=1e7)&(cl>=10)].dropna().sort_values(ascending=False,kind='mergesort'); pool=list(ok.index[:300])
        s=S52.loc[d]; yy=y3(d)
        base=[k for k in pool if pd.notna(s.get(k)) and pd.notna(op.loc[nd,k]) and k not in fin]
        el=[k for k in base if k not in FOR and pd.notna(yy.get(k,np.nan))]
        Aff=[k for k in base if pd.isna(yy.get(k,np.nan)) and cause(k,m) in GOV]
        y=yy.reindex(el+Aff).copy()
        if variant=='carry':
            keep=[]
            for k in Aff:
                val=np.nan
                for b in range(1,25):
                    ms=[m-b-1,m-b-2,m-b-3]
                    if k in yoy.columns:
                        x=yoy[k].reindex(ms)
                        if x.notna().all(): val=x.mean(); break
                if pd.notna(val): y[k]=val; keep.append(k)
            Aff2=keep; elp=el+keep
            df=pd.DataFrame({'s':s[elp].values,'y':y[elp].values},index=elp)
            df['rp']=df.y.rank(pct=True,method='average')
        else:
            elp=el+Aff; df=pd.DataFrame({'s':s[elp].values,'y':y[elp].values},index=elp)
            nonA=df.loc[el,'y']; rk=nonA.rank(method='average')/len(elp)
            df['rp']=np.nan; df.loc[el,'rp']=rk
            if variant=='opt': df.loc[Aff,'rp']=1.0; df.loc[Aff,'y']=np.inf
            else: df.loc[Aff,'rp']=0.9; df.loc[Aff,'y']=nonA.quantile(0.9)
            Aff2=Aff
        df['ord']=[cols.index(k) for k in df.index]
        df['sc']=0.5*df.s.rank(pct=True,method='average')+0.5*df.rp
        df=df.sort_values(['sc','y','ord'],ascending=[False,False,True])
        pick=[]
        for k in df.index:
            if len(pick)==10: break
            if k not in held:
                pc=c.loc[d,k]; o=op.loc[nd,k]
                if pd.notna(pc) and pd.notna(o) and o>=limit_up(pc,nd)-1e-9: continue
            pick.append(k)
        lists[str(d.date())]=pick; held=set(pick)
        for k in pick:
            if k in Aff2: slots.append((str(d.date()),k,cause(k,m)))
    return lists, slots
out={}
for var in ['opt','p90','carry']:
    L,S=run(var)
    n12=sum(1 for x in S if x[0]<='2012-12-31')
    out[var]=dict(n_2006_2013=len(S),n_2006_2012=n12,slots=S,lists=L)
    print(MODE,var,'U 2006-2013',len(S),'2006-2012',n12)
json.dump(out,open(f'r4_u_{MODE}.json','w'),ensure_ascii=False)
# 與官方 ER2 槽位比對
er=json.load(open(A+'o_rev_trunc_2009_erratum/ERRATUM2.json'))
off=er['U_opt_slot_list']
print('official slot list sample',off[:3] if isinstance(off,list) else type(off))

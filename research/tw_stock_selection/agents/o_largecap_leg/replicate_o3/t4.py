import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r30={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n30']}
for var in ['A','B','C']:
    bad=0; badl=[]
    for t in D.sigs:
        if t not in r30: continue
        df=D.base_frame(t)
        ok=(df.v60>=1e7)&(df.close>=10)&~df.disp&(df.nopen>0)
        if var in 'AC': ok&=df.yoy3.notna()&df.r52.notna()
        p=df[ok].sort_values('v60',ascending=False).head(300)
        if var=='B': p=p[p.yoy3.notna()&p.r52.notna()]
        if var=='C': p=p
        p=p[~p.index.isin(D.fin)]
        rk=D.rank(p)[:30] if var!='C' else None
        if var=='C':
            # 百分位在含金融的池中算，之後排除金融
            df2=df[ok].sort_values('v60',ascending=False).head(300)
            rk=[c for c in D.rank(df2) if c not in D.fin][:30]
        if rk!=r30[t]: bad+=1; badl.append(t.date())
    print(var,bad,badl[:12])

import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r30={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n30']}
for t in D.sigs:
    if t not in r30: continue
    df=D.base_frame(t)
    ok=(df.v60>=1e7)&(df.close>=10)&~df.disp&(df.nopen>0)
    p=df[ok].sort_values('v60',ascending=False).head(300)
    p=p[p.yoy3.notna()&p.r52.notna()]; p=p[~p.index.isin(D.fin)]
    rk=D.rank(p)
    if rk[:10]!=r30[t][:10]:
        ex=[c for c in rk[:10] if c not in r30[t][:10]]; mi=[c for c in r30[t][:10] if c not in rk[:10]]
        print(t.date(), 'extra',[(c,round(p.r52[c],3),round(p.yoy3[c],1), c in r30[t]) for c in ex], 'missing',[(c,round(p.r52[c],3) if c in p.index else None) for c in mi])

import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r10={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n10']}
held=set(); bad=[]
for t in D.sigs:
    if t not in r10: continue
    df=D.base_frame(t,disp_day='t')
    ok=(df.v60>=1e7)&(df.close>=10)
    p=df[ok].sort_values('v60',ascending=False).head(300)
    p=p[~p.disp&(p.nopen>0)]
    p=p[p.yoy3.notna()&p.r52.notna()]; p=p[~p.index.isin(D.fin)]
    pk=D.pick(D.rank(p),10,set(r10.get(t,[])) and held, D.next_day(t))
    if set(pk)!=set(r10[t]): bad.append((t.date(), sorted(set(pk)-set(r10[t])), sorted(set(r10[t])-set(pk)), [c for c in set(r10[t])-set(pk) if c not in p.index]))
    held=set(r10[t])
print(len(bad))
for b in bad: print(b)

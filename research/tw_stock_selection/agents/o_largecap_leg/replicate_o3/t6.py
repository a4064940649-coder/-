import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r30={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n30']}
for t in [pd.Timestamp('2014-05-12'),pd.Timestamp('2008-10-13'),pd.Timestamp('2022-08-11')]:
    df=D.base_frame(t)
    ok=(df.v60>=1e7)&(df.close>=10)&~df.disp&(df.nopen>0)
    p=df[ok].sort_values('v60',ascending=False).head(300)
    p=p[p.yoy3.notna()&p.r52.notna()]; p=p[~p.index.isin(D.fin)]
    rk=D.rank(p)
    print(t.date(), 'theirs', r30[t][:12]); print('mine  ', rk[:12])
    miss=[c for c in r30[t] if c not in rk[:30]]
    for c in miss:
        print(c, c in p.index, c in df.index and df.loc[c].to_dict() if c in df.index else None, rk.index(c) if c in rk else None)

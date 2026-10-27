import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r30={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n30']}
vidx=D.adj.index[D.adj.notna().sum(1)>=300]
a=D.adj.loc[vidx]
V={}
V['prev']=a/gap_roll(a,252,'max').shift(1)
V['r1']=D.r52
# rank method variants
def run(r52, method='average'):
    bad=0
    for t in D.sigs:
        if t not in r30: continue
        df=D.base_frame(t); df['r52']=r52.loc[t]
        ok=(df.v60>=1e7)&(df.close>=10)&~df.disp&(df.nopen>0)
        p=df[ok].sort_values('v60',ascending=False).head(300)
        p=p[p.yoy3.notna()&p.r52.notna()]; p=p[~p.index.isin(D.fin)].copy()
        p['score']=0.5*p.r52.rank(pct=True,method=method)+0.5*p.yoy3.rank(pct=True,method=method); p['score_r']=p.score.round(10); p['code']=p.index
        rk=list(p.sort_values(['score_r','yoy3','code'],ascending=[False,False,True]).index)
        if rk[:10]!=r30[t][:10]: bad+=1
    return bad
print('prev', run(V['prev'])); print('max',run(D.r52,'max')); print('min',run(D.r52,'min'))

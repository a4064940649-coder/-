import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
pl=json.load(open(AG+'o_rev_survivorship/plans_R_DEL.json'))
r30={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n30']}
r10={pd.Timestamp(x['sig']):x['codes'] for x in pl['EXFIN|n10']}
tg={}; held=set(); picks={}; mism=0
for t in D.sigs:
    if t not in r30: continue
    d=D.next_day(t); pk=D.pick(r30[t],10,held,d); picks[t]=pk
    if set(pk)!=set(r10[t]): mism+=1
    held=set(pk); tg[t]={c:0.1 for c in pk}
print('n10 list vs my pick w/ LU differ', mism)
nav,to=D.simulate(tg,'normal')
ref=pd.read_parquet(AG+'integrate_o2/navs.parquet')['MAIN_R']
print('turn',to,'final',nav.iloc[-1],ref.iloc[-1], 'maxreldiff', float((nav/ref-1).abs().max()))
dr=(nav.pct_change()-ref.pct_change()).abs(); print(dr.sort_values().tail(8))
pickle.dump({'tg':tg,'picks':picks,'nav':nav},open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_main.pkl','wb'))

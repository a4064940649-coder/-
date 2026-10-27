import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle; from rlib import *
SC='/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
D=pickle.load(open(SC+'rep_o3_data.pkl','rb')); PL=pickle.load(open(SC+'rep_o3_plans.pkl','rb'))
ss=PL['sigs']
for k in ['main','L','poolE']:
    ev=[]; 
    for i in range(1,len(ss)):
        d=D.next_day(ss[i])
        for c in PL[k][ss[i-1]]:
            if not (D.adjopen.at[d,c]>0):
                ev.append((ss[i].date(),c, D.adj[c].loc[d:].notna().sum()))
    print(k,len(ev),[e for e in ev if e[2]==0][:20], len([e for e in ev if e[2]==0]))

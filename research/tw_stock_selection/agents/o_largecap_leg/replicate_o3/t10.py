import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
for c,t in [('2497','2022-02-11'),('6442','2024-06-11'),('3047','2024-07-11'),('6215','2025-02-11'),('2429','2025-09-11')]:
    t=pd.Timestamp(t); df=D.base_frame(t)
    print(c, df.loc[c].to_dict() if c in df.index else 'not in panel')
    m=pd.Period(t,'M'); print('  r5', [D.r5.at[str(m-k),c] if c in D.r5.columns else None for k in (1,2,3)])

import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
b=bench0050(D.nidx)
def cg(n,a,e):
    a=n.index[n.index>=a][0]; e=n.index[n.index<=e][-1]; y=(e-a).days/365.25; print(a.date(),e.date(), round(((n[e]/n[a])**(1/y)-1)*100,2))
cg(b,'2019-01-11','2023-12-31'); cg(b,'2019-01-11','2024-02-15'); cg(b,'2023-12-29','2026-10-12');cg(b,'2024-01-01','2026-10-12');cg(b,'2024-02-15','2026-10-12')

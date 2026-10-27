import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_largecap_leg/replicate_o3')
import pickle, json; from rlib import *
D=pickle.load(open('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o3_data.pkl','rb'))
for t,cs in [('2016-10-11',['4551','2548']),('2023-04-11',['4119','1513']),('2024-12-11',['2059','3706']),('2025-12-11',['4739','2406']),('2026-01-12',['2548','2449']),('2026-07-13',['6834','6139']),('2026-08-11',['3022','2383'])]:
    t=pd.Timestamp(t); df=D.base_frame(t,'t')
    ok=(df.v60>=1e7)&(df.close>=10)&~df.disp&(df.nopen>0)
    p=df[ok].sort_values('v60',ascending=False).head(300); p=p[p.yoy3.notna()&p.r52.notna()]; p=p[~p.index.isin(D.fin)]
    rk=D.rank(p)
    for c in cs:
        dd=D.dsp[(D.dsp.code==c)&(D.dsp.end>=t-pd.Timedelta(days=20))&(D.dsp.start<=t+pd.Timedelta(days=20))]
        m=pd.Period(t,'M')
        print(t.date(), c, rk.index(c) if c in rk else None, 'v60rank', list(df[ok].sort_values('v60',ascending=False).index).index(c) if c in df[ok].index else None, round(df.r52[c],3), [D.r5.at[str(m-k),c] if c in D.r5.columns and str(m-k) in D.r5.index else None for k in (1,2,3)], D.lu_blocked(c,D.next_day(t)), dd.values.tolist())

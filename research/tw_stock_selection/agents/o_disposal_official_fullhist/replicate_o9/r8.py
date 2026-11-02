import sys; sys.path.insert(0,'.')
from ev import *
d=load_events().sort_values(['code','start','announce','ord'])
t=pd.read_csv(A+'o_disposal_official_fullhist/trades_all.csv.gz',dtype={'code':str},parse_dates=['date'])
tr=pd.read_csv('my_trades.csv.gz',dtype={'code':str},parse_dates=['s','start'])
V={}
for name in ['td_prev','td_maxe','cal_prev','cal_maxe','td_any','cal_any']:
    out=[]
    for code,gg in d.groupby('code',sort=False):
        pe=[]
        for r in gg.itertuples():
            unit_td=name.startswith('td')
            lo = r.si-30 if unit_td else r.s_date-pd.Timedelta(days=30)
            cur = r.si if unit_td else r.s_date
            vals=[x[0] if unit_td else x[1] for x in pe]
            if 'prev' in name: vals=vals[-1:]
            if 'maxe' in name: vals=[max(vals)] if vals else []
            out.append(any((v>=lo)&(v<cur) for v in vals))
            pe.append((r.ei,r.e_date))
    d[name]=out
key=d.set_index(['code','start'])
tr=tr.merge(t[['code','date','second']].rename(columns={'date':'s','second':'sr'}),on=['code','s'])
for name in ['td_prev','td_maxe','cal_prev','cal_maxe','td_any','cal_any']:
    m=tr.merge(d[['code','start',name]].drop_duplicates(['code','start']),on=['code','start'])
    print(name,len(m),(m[name]!=m.sr).sum(), m.sr.sum(), m[name].sum())

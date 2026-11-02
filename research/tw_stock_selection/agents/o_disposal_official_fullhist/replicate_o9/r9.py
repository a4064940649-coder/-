import sys; sys.path.insert(0,'.')
from ev import *
d=second_td_prev(load_events()); tr=run(d)
o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')]
nanw=[np.isnan(R[r.si+1:r.xi+1,cix[r.code]]).sum() for r in o.itertuples()]
o=o.assign(nanw=nanw)
print(o[(o.nanw>0)|o.delist][['code','s','x','e','nanw','delist','net']])
o=o.assign(noRe=[not okR[r.ei,cix[r.code]] for r in o.itertuples()], dfr=o.xi>o.ei)
print(o[o.noRe|o.dfr][['code','s','e','x','noRe','delist','net']])
t=pd.read_csv(A+'o_disposal_official_fullhist/trades_all.csv.gz',dtype={'code':str},parse_dates=['date'])
to=t[(t.date>='2006-04-01')&(t.date<='2021-09-30')]
print(to.delisted.sum(), to.delayed.sum(), to.px.isna().sum())
print(to[to.delisted|to.delayed][['code','date','exit_date','delisted','delayed','net']])

import sys,itertools,json; sys.path.insert(0,'.')
from ev import *
tr=pd.read_csv('my_trades.csv.gz',dtype={'code':str},parse_dates=['s'])
o=tr[(tr.s>='2006-04-01')&(tr.s<='2021-09-30')]
n=o.net.sort_values()
for k in [20,21]: print(k, n.iloc[:len(n)-k].mean()*100)
print('q95<', o.net[o.net<o.net.quantile(0.95)].mean()*100, o.net[o.net<=o.net.quantile(0.95)].mean()*100)
miss=json.load(open('event_results.json'))['missing_events']
for c in itertools.combinations(range(len(miss)),4):
    mm=[miss[i] for i in c]
    mo=pd.concat([o[['s','net']],pd.DataFrame({'s':[pd.Timestamp(x[1]) for x in mm],'net':[-0.30-(0.012 if x[2] else 0.008)*0 for x in mm]})])
    s=summ(mo)
    if abs(s['t']-1.95)<0.03: print([m[:2] for m in mm],s)

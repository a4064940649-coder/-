import sys, itertools, time; sys.path.insert(0, '.')
from rlib import *
sys.path.insert(0, str(AG / 'strategy_ensemble'))
import elib as E
FA = {'S_long': '2006-05-11', 'S_recent': '2021-02-26'}
ref = pd.read_parquet(AG / 'strategy_ensemble/ensemble_nav.parquet')
x = E.simulate(E.PR['sets']['S_long'], E.METHODS['E1'], FA['S_long'])
print('S_long|E1 max diff', float((x - ref['S_long|E1'].reindex(x.index)).abs().max()))
R = {}; t0 = time.time()
for setname, names in E.PR['sets'].items():
    for k in range(2, len(names) + 1):
        for c in itertools.combinations(names, k):
            for m in ('E1', 'E2'):
                nav = E.simulate(c, E.METHODS[m], FA[setname])
                R[f'ENS::{setname}|{"+".join(c)}|{m}'] = win_ret(nav)
print(len(R), round(time.time() - t0, 1))
pd.DataFrame(R, index=G[:-1]).to_parquet(OUT / 'ens_ret.parquet')

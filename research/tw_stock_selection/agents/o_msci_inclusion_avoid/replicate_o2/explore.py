import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import rlib as L
d = L.load_events()
d = L.add_metrics(d)
d['src3'] = (d['E'] >= L.SRC3[0]) & (d['E'] <= L.SRC3[1])
d['per'] = np.where(d['E'] <= pd.Timestamp('2018-12-31'), 'select', 'oos')
ok = d[d.liq_ok]
res = {}
for cls in ('STD', 'SC'):
    for per in ('select', 'oos'):
        x = ok[(ok.cls == cls) & (ok.per == per) & ~ok.src3]
        res[f'{cls}|{per}'] = L.clus(x)
    res[f'{cls}|src3'] = L.clus(ok[(ok.cls == cls) & ok.src3])
for cls in ('TW50', 'HDETF'):
    res[f'{cls}|all_nosrc3'] = L.clus(ok[(ok.cls == cls) & ~ok.src3])
    res[f'{cls}|all'] = L.clus(ok[(ok.cls == cls)])
# 價格路徑
for per in ('select', 'oos'):
    x = ok[(ok.cls == 'STD') & (ok.per == per) & ~ok.src3]
    res[f'path|{per}'] = {c: L.clus(x, c)['mean'] for c in ('pre10', 'arA', 'A_to_E', 'A1_to_Em1', 'A_to_Em1', 'A1_to_E', 'arE')}
for k, v in res.items(): print(k, v)
d.to_csv(L.HERE / 'events_rep.csv', index=False)
json.dump(res, open(L.HERE / 'explore_rep.json', 'w'), indent=1, ensure_ascii=False)
# 分群改用 E 日
d2 = ok.copy(); d2['cluster'] = d2['cls'] + '|' + d2['E'].astype(str)
for cls in ('TW50', 'HDETF'):
    print('byE', cls, L.clus(d2[(d2.cls == cls) & ~d2.src3]), L.clus(d2[(d2.cls == cls)]))

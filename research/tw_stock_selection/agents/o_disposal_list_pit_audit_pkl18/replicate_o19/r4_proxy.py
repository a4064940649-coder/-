"""近似事件精確率與強觸發（純價格條件連續 3 日）無對應處置的比例。"""
import json, sys
import numpy as np, pandas as pd
OUT = '/home/user/-/research/tw_stock_selection/agents/o_disposal_list_pit_audit_pkl18/replicate_o19/'
P = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/'
adj = pd.read_parquet(P + 'adj.parquet'); cal = adj.index[adj.notna().sum(axis=1) >= 300]
pos = {d: i for i, d in enumerate(cal)}
W = int(sys.argv[1]) if len(sys.argv) > 1 else 10
C = pd.read_csv(OUT + 'r1_universe.csv', dtype={'code': str})
byc = {}
for c, s in zip(C.code, C.start):
    s = pd.Timestamp(s)
    byc.setdefault(c, []).append(int(cal.searchsorted(s)))
def matched(code, d):
    i = int(cal.searchsorted(pd.Timestamp(d)))
    return any(abs(k - i) <= W for k in byc.get(code, []))
out = {}
for nm, f in (('proxy', 'r3_events.csv'), ('strong', 'r3_strong.csv')):
    E = pd.read_csv(OUT + f, dtype={'code': str}); E = E[E.astart.notna() & (E.astart != '')]
    E['m'] = [matched(c, d) for c, d in zip(E.code, E.astart)]
    E['y'] = E.astart.str[:4].astype(int)
    g = E.groupby('y').m.agg(['sum', 'count'])
    out[nm] = {int(y): dict(n=int(r['count']), matched=int(r['sum']), prec=round(r['sum'] / r['count'], 3),
                            unmatched_share=round(1 - r['sum'] / r['count'], 3)) for y, r in g.iterrows()}
for y in range(2004, 2027):
    p, s = out['proxy'].get(y, {}), out['strong'].get(y, {})
    print(y, 'proxy prec', p.get('prec'), p.get('n'), '| strong n', s.get('n'), 'unmatched', s.get('n', 0) - s.get('matched', 0), s.get('unmatched_share'))
json.dump(out, open(OUT + 'r4_proxy_w%d.json' % W, 'w'), indent=1)

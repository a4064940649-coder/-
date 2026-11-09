#!/usr/bin/env python3
"""登記前（名單層面計數，不算報酬）：V7X 名單「持有跨越下市」事件 × v3／cons 結算規則。
持有區間＝[exec_idx, 下一個 exec_idx)，規則 lt 落在區間內者。報各事件被多少 (j, seed) 名單持有、v3 規則是否和 cons_main 不同、lo 與 hi 是否不同。
python3 -I -B v7x_held.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o18')
sys.path.insert(0, str(AG / 'o_engine_delist_consideration')); import cz_v7x as CV, cz_engine as CE; sys.path.pop(0)
sys.path.insert(0, str(AG / 'o_delist_terms_dispmask_fix')); import cz3; sys.path.pop(0)
D = CV.D
E2, E3 = cz3.load_E()
kw = dict(TR=D.TR, IDX=D.IDX, Cn=D.C, CI=D.CI, On=D.O)
R2 = json.loads((AG / 'o_engine_delist_consideration/results.json').read_text(encoding='utf-8'))
dd = R2['PREREG']['events_v2']['summary']['d']['d_used']
ST = {'cons_main': CE.build_settlement(E2, 'main', d=dd, **kw), 'v3_main': cz3.build(E2, E3, 'main', **kw), 'v3_lo': cz3.build(E2, E3, 'lo', **kw), 'v3_hi': cz3.build(E2, E3, 'hi', **kw)}
RK = {eg: {(r['code'], r['lt']): r for rs in ST[eg].values() for r in rs} for eg in ST}
def pj(r): return None if r is None else json.dumps([{k: (round(v, 9) if isinstance(v, float) else v) for k, v in p.items()} for p in r['parts']], sort_keys=True, default=str)
PL = pd.read_csv(SCR / 'plans_v7x_i18.csv', dtype={'picks': str})
NI = len(D.IDX)
out = {}
for v, G in PL.groupby('variant'):
    cnt = {}
    for (j, s), g in G.groupby(['j', 'seed']):
        g = g.sort_values('exec_idx'); ex = list(g.exec_idx) + [NI]
        for i, pk in enumerate(g.picks):
            e, e2 = ex[i], ex[i + 1]
            for c in pk.split(','):
                jj = D.CI[c]
                for eg in ('v3_main', 'cons_main'):
                    for r in ST[eg].get(jj, []):
                        if e <= r['lt'] < e2: cnt.setdefault((c, r['lt']), set()).add((j, s))
    rows = []
    for (c, lt), ss in sorted(cnt.items()):
        g = {eg: RK[eg].get((c, lt)) for eg in ST}
        r3 = E3[E3.code == c]
        rows.append(dict(code=c, lt=str(D.IDX[lt].date()), n_plans=len(ss), v3=(f"{r3.text_status.iloc[0]}/{r3.settle.iloc[0]}" if len(r3) else None),
                         type_cons=(g['cons_main'] or {}).get('type'), type_v3=(g['v3_main'] or {}).get('type'),
                         v3_ne_cons=pj(g['v3_main']) != pj(g['cons_main']), v3_lo_ne_hi=pj(g['v3_lo']) != pj(g['v3_hi'])))
    out[v] = dict(n_events=len(rows), n_v3_ne_cons=sum(r['v3_ne_cons'] for r in rows), n_bound_sensitive=sum(r['v3_lo_ne_hi'] for r in rows),
                  n_verified_text=sum((r['v3'] or '').startswith('VERIFIED_TEXT') for r in rows), rows=rows)
    print(v, {k: out[v][k] for k in out[v] if k != 'rows'})
(HERE / 'v7x_held.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for r in out['REAL_opt']['rows']: print(r)

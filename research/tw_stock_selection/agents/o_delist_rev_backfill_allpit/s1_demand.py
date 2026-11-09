#!/usr/bin/env python3
"""o18 題 1 步驟 2(a)：需求表（不算任何報酬）。
U13 opt（pclib.VU('opt', lo=2013-01-01, GOV)）每個訊號日的受影響股（aff：池內 GOV 原因缺營收者），
以及放回入選槽（251）。需求格＝(發行人, 營收月)：訊號月 m 的 m−1..m−3 與 m−13..m−15 中 tw_revenue 缺值的月份
（訊號日 ≥ 2021-03-01 用 rev_5y yoy 欄 m−1..m−3；這部分需要 yoy 而非水準）。
輸出 data/demand_slots.csv、data/demand_aff.csv、data/demand_cells.csv、demand.json。python3 -I -B s1_demand.py"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count'))
import pclib as P
sys.path.pop(0)
V, K = P.V, P.K
TOP10 = ['2325', '2311', '1704', '3454', '1507', '6286', '1729', '3474', '6452', '5305']
S = P.VU('opt', lo='2013-01-01', hi='2026-12-31', which='GOV')
L = P.u_lists(S)
slots, affr, cells = [], [], {}
for d, o in L.items():
    D = pd.Timestamp(d); m = pd.Period(D, 'M')
    for c in o['aff']:
        affr.append(dict(sig=d, code=c, cause=o['causes'].get(c), picked=c in o['picks']))
        if D < V.SWITCH:
            for k in (1, 2, 3, 13, 14, 15):
                x = m - k; v = P.LV[c].get(x, np.nan) if c in P.LV.columns else np.nan
                if not np.isfinite(v):
                    cells.setdefault((c, str(x)), dict(code=c, ym=str(x), kind='level', sigs=set()))['sigs'].add(d)
        else:
            for k in (1, 2, 3):
                x = m - k
                v = P.Y5COL[c].get(x, np.nan) if c in P.Y5COL.columns else np.nan
                if not np.isfinite(v):
                    cells.setdefault((c, str(x)), dict(code=c, ym=str(x), kind='yoy', sigs=set()))['sigs'].add(d)
    for c in o['picks']:
        if c in o['aff']:
            slots.append(dict(sig=d, code=c, cause=o['causes'].get(c), top10=c in TOP10))
SL = pd.DataFrame(slots); AF = pd.DataFrame(affr)
CE = pd.DataFrame([dict(code=v['code'], ym=v['ym'], kind=v['kind'], n_sigs=len(v['sigs']), first_sig=min(v['sigs']), last_sig=max(v['sigs']),
                        top10=v['code'] in TOP10, in_slot_need=False) for v in cells.values()])
# 入選槽需要的格
need = set()
for r in slots:
    D = pd.Timestamp(r['sig']); m = pd.Period(D, 'M'); c = r['code']
    ks = (1, 2, 3, 13, 14, 15) if D < V.SWITCH else (1, 2, 3)
    for k in ks:
        if (c, str(m - k)) in cells: need.add((c, str(m - k)))
CE['in_slot_need'] = [(a, b) in need for a, b in zip(CE.code, CE.ym)]
CE = CE.sort_values(['code', 'ym']); SL.to_csv(HERE / 'data/demand_slots.csv', index=False)
AF.to_csv(HERE / 'data/demand_aff.csv', index=False); CE.to_csv(HERE / 'data/demand_cells.csv', index=False)
seg = lambda s: '2013_2018' if s <= '2018-12-31' else '2019_2026'
out = dict(n_slots=len(SL), slots_by_seg=SL.sig.map(seg).value_counts().to_dict(),
           top10_slots=int(SL.top10.sum()), n_issuers_slots=SL.code.nunique(),
           slots_by_issuer=SL.code.value_counts().to_dict(),
           n_aff_rows=len(AF), n_aff_issuers=AF.code.nunique(), aff_by_cause=AF.cause.value_counts().to_dict(),
           n_cells=len(CE), cells_by_kind=CE.kind.value_counts().to_dict(), n_cells_slot_need=int(CE.in_slot_need.sum()),
           cells_by_year=CE.ym.str[:4].value_counts().sort_index().to_dict(),
           issuers=sorted(AF.code.unique()),
           delist_date={c: str(P.delist_date(c).date()) if pd.notna(P.delist_date(c)) else None for c in sorted(AF.code.unique())},
           last_rev={c: str(P.LASTREV.get(c)) for c in sorted(AF.code.unique())},
           name={c: P.NAME.get(c) for c in sorted(AF.code.unique())},
           lists_sha=hashlib.sha256(json.dumps(L, sort_keys=True).encode()).hexdigest())
assert out['n_slots'] == 251 and out['slots_by_seg'] == {'2013_2018': 227, '2019_2026': 24}, out['slots_by_seg']
P.jdump(out, HERE / 'demand.json')
print(json.dumps({k: v for k, v in out.items() if k not in ('issuers',)}, ensure_ascii=False, indent=0)[:4000])

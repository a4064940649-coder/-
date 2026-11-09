#!/usr/bin/env python3
"""G6 複製員 r1：自己從 pclib.VU('opt', lo=2013-01-01, GOV) 重算放回入選槽、池內受影響列與需求格（不讀研究員 *.py）。
python3 -I -B r1_demand.py"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
AG = HERE.parent.parent
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o18')
sys.path.insert(0, str(AG / 'o_rev_avail_ds_pit_count'))
import pclib as P
sys.path.pop(0)
V, K = P.V, P.K
COLS = P.COLS
S = P.VU('opt', lo='2013-01-01', hi='2026-12-31', which='GOV')
pl = S.plan(10, 'official')
slots, aff = [], []
for e, pk, el, d in pl:
    if d < pd.Timestamp('2013-01-01'): continue
    A = set(S.aff.get(d, []))
    pks = {COLS[j] for j in pk}
    for c in sorted(A):
        aff.append(dict(sig=str(d.date()), code=c, cause=S.causes[d][c], picked=c in pks))
    for c in sorted(pks & A):
        slots.append(dict(sig=str(d.date()), code=c, cause=S.causes[d][c]))
SL = pd.DataFrame(slots); AF = pd.DataFrame(aff)
# 需求格：d < SWITCH → 水準 m−1..m−3、m−13..m−15（D=S／D>S 日加 m−4、m−16，A_UB 回退）；d ≥ SWITCH → 年增 m−1..m−3（＋m−4）
cells = {}
for r in AF.itertuples():
    d = pd.Timestamp(r.sig); m = pd.Period(d, 'M')
    ds = V.deadline_case(d) in ('D=S', 'D>S')
    ks = [1, 2, 3] + ([4] if ds else [])
    if d < V.SWITCH:
        need = [(m - k, 'level') for k in ks] + [(m - k - 12, 'level') for k in ks]
    else:
        need = [(m - k, 'yoy') for k in ks]
    for x, kd in need:
        cells.setdefault((r.code, str(x), kd), set()).add(r.sig)
CE = pd.DataFrame([dict(code=c, ym=y, kind=k, n_sigs=len(v)) for (c, y, k), v in sorted(cells.items())])
SL.to_csv(SCR / 'r1_slots.csv', index=False); AF.to_csv(SCR / 'r1_aff.csv', index=False); CE.to_csv(SCR / 'r1_cells.csv', index=False)
out = dict(n_slots=len(SL), n_1318=int((SL.sig <= '2018-12-31').sum()), n_19on=int((SL.sig >= '2019-01-01').sum()),
           n_issuers_slots=int(SL.code.nunique()), by_issuer=SL.code.value_counts().to_dict(),
           n_aff=len(AF), n_aff_issuers=int(AF.code.nunique()), aff_by_cause=AF.cause.value_counts().to_dict(),
           n_cells=len(CE), cells_by_kind=CE.kind.value_counts().to_dict())
top10 = list(SL.code.value_counts().index[:10]); out['top10'] = top10; out['top10_slots'] = int(SL.code.isin(top10).sum())
# 與研究員輸出（資料檔，非 *.py）比對
RD = HERE.parent / 'data'
rs = pd.read_csv(RD / 'demand_slots.csv', dtype=str); ra = pd.read_csv(RD / 'demand_aff.csv', dtype=str); rc = pd.read_csv(RD / 'demand_cells.csv', dtype=str)
out['cmp'] = dict(slots_equal=sorted(zip(rs.sig, rs.code)) == sorted(zip(SL.sig, SL.code)),
                  aff_equal=sorted(zip(ra.sig, ra.code, ra.cause)) == sorted(zip(AF.sig, AF.code, AF.cause)),
                  cells_equal=sorted(zip(rc.code, rc.ym, rc.kind)) == sorted(zip(CE.code, CE.ym, CE.kind)),
                  cells_only_mine=len(set(zip(CE.code, CE.ym, CE.kind)) - set(zip(rc.code, rc.ym, rc.kind))),
                  cells_only_theirs=len(set(zip(rc.code, rc.ym, rc.kind)) - set(zip(CE.code, CE.ym, CE.kind))))
(HERE / 'r1_demand.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))

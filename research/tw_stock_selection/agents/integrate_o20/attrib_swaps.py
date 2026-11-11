#!/usr/bin/env python3
"""integrate_o20 J1 歸因（只分類，不算報酬）：每個「M2 換進」的股票（基準遮罩排除、M2 不排除），判定基準為何排除它：
 dOnly＝≤2021-09-23 disposal_official_universe 處置期涵蓋 d 但不涵蓋 e（處置在訊號日結束，成交日已解除）；
 extra＝基準 disp_mask_fn(trade) 的表（disposal_hist_rejudge in_panel ∪ disposal_5y 獨有列）有「公告 ≤ d 且涵蓋 e」、但官方 universe_announce 沒有同 (code,start) 的列；
 other＝以上皆非（遞補連鎖等）。紙上 M1→M2 同法（紙上另有 ia_core disposal_5y 涵蓋 d 一項 d5_dcover）。python3 -I -B attrib_swaps.py"""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert hashlib.sha256(json.dumps(R['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == R['PREREG_sha256']
IDX = pd.DatetimeIndex(pd.read_csv(HERE / 'data/paper_daily_navs.csv.gz', usecols=[0], index_col=0, parse_dates=True).index)
U = pd.read_csv(AG / 'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
DU = pd.read_csv(AG / 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
A = pd.read_csv(AG / 'o_disposal_rejudge/disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end']); A = A[A.in_panel]
D5 = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
ka = set(zip(A.code, A.start)); Bo = D5[[(c, s) not in ka for c, s in zip(D5.code, D5.start)]]
T = pd.concat([A[['code', 'announce', 'start', 'end']], pd.DataFrame({'code': Bo.code, 'announce': Bo.date, 'start': Bo.start, 'end': Bo.end})], ignore_index=True)
ku = set(zip(U.code, U.start))
def why(code, d, paper=False):
    d = pd.Timestamp(d); e = IDX[IDX.get_loc(d) + 1]; r = []
    if d <= pd.Timestamp('2021-09-23'):
        x = DU[(DU.code == code) & (DU.start <= d) & (DU.end >= d) & ~((DU.start <= e) & (DU.end >= e))]
        if len(x): r.append('dOnly')
    y = T[(T.code == code) & (T.announce <= d) & (T.start <= e) & (T.end >= e)]
    if len(y) and not any((c, s) in ku for c, s in zip(y.code, y.start)): r.append('extra')
    if paper:
        z = D5[(D5.code == code) & (D5.start <= d) & (D5.end >= d)]
        if len(z) and not len(U[(U.code == code) & (U.announce <= d) & (U.start <= e) & (U.end >= e)]): r.append('d5_dcover')
    return '+'.join(r) or 'other'
VD = pd.read_csv(HERE / 'data/v7x_plan_diff_M2.csv.gz', dtype=str)
ev = VD.groupby(['signal', 'into']).size().reset_index(name='n_plans')
rows = []
for s, into, n in zip(ev.signal, ev['into'], ev.n_plans):
    for c in str(into).split(','):
        if c and c != 'nan': rows.append(dict(signal=s, code=c, n_plans=int(n), reason=why(c, s)))
V = pd.DataFrame(rows)
LD = R['RESULTS_PAPER']['list_diffs']; prow = []
for k, v in LD.items():
    if not k.endswith('M1->M2'): continue
    for x in v:
        for c in x['into']: prow.append(dict(variant=k.split('|')[0], signal=x['signal'], code=c, reason=why(c, x['signal'], paper=True)))
Pp = pd.DataFrame(prow)
V.to_csv(HERE / 'data/v7x_M2_swap_reasons.csv', index=False)
res = dict(v7x=dict(n_distinct=int(len(V)), by_reason=V.groupby('reason').size().to_dict(), plans_by_reason=V.groupby('reason').n_plans.sum().to_dict(),
                    by_year=V.assign(y=V.signal.str[:4]).groupby('y').size().to_dict()),
           paper=Pp.to_dict('records'))
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); R['ATTRIB_J1'] = res
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps(res, ensure_ascii=False, indent=0))

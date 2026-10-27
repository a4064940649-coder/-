#!/usr/bin/env python3
"""事後描述（不參與判定）：成交值前 50 池和臺灣50 成分（同樣排除金融、KY/DR）的重疊；池等權與 L 腿對 0050 的追蹤誤差。寫 posthoc.json。"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8')); assert 'RESULTS' in R
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K = L.K
leg = L.plan_leg(50, 5)
legT = L.plan_leg(50, 5, members_fn=lambda d: L.tw50_members(d) if d >= L.TW50_START else None)
ov = {}
for a, b in zip(leg, legT):
    if a[3] < L.TW50_START: continue
    pa, pb = set(a[4].tolist()), set(b[4].tolist())
    ov[a[3]] = (len(pa & pb) / len(pb) if pb else np.nan, len(pa & pb) / len(pa))
s = pd.DataFrame(ov, index=['share_of_TW50_eligible_in_P50', 'share_of_P50_in_TW50']).T
out = {'overlap': {lab: s[(s.index >= lo) & (s.index <= hi)].mean().round(3).to_dict() for lab, lo, hi in
                   (('2015-04~2018', '2015-04-01', '2018-12-31'), ('2019~2023', '2019-01-01', '2023-12-31'), ('2024~2026', '2024-01-01', '2026-12-31'))}}
# 2024～2026 池內不在臺灣50 的例子（入選 L 腿者）
ex = {}
for a, b in zip(leg, legT):
    if a[3] < pd.Timestamp('2024-01-01'): continue
    for j in a[1]:
        if j not in set(b[4].tolist()): ex[L.COLS[j]] = ex.get(L.COLS[j], 0) + 1
out['L_picks_not_in_TW50_2024_2026_top'] = sorted(((c, L.V.SL['name'].get(c, ''), n) for c, n in ex.items()), key=lambda x: -x[2])[:12]
out['L_pick_slots_not_in_TW50_share_2024_2026'] = round(sum(ex.values()) / sum(len(a[1]) for a in leg if a[3] >= pd.Timestamp('2024-01-01')), 3)
(HERE / 'posthoc.json').write_text(L.dumps(out), encoding='utf-8')
print(L.dumps(out))

#!/usr/bin/env python3
"""o17 反方統計審查（只讀研究員檔案）：
1. 由 slot_avail.csv 重算 R1 上下界；
2. 下界結構上限：UNAVAIL 只可能來自 2024-06 起快照或 2026-09 MOPS → r1a/r1b 下界的可達上限；
3. V4（未收緊）規則下的下界，檢查 V6 收緊是否改變判定；
4. 處置槽位 × main_v2 最大貢獻股；處置槽位的持有期價格變動（只作量級，不是策略報酬）。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
out = {}
SA = pd.read_csv(HERE/'data/slot_avail.csv', dtype={'code': str})
C0 = pd.read_csv(HERE/'data/c0_slots.csv', dtype={'code': str})
C0f = C0[C0.formal.astype(str)=='True']
SAf = SA[SA.formal.astype(str)=='True']
def seg(df, a, b): return df[(df.sig>=a)&(df.sig<=b)]
for nm,(a,b) in {'r1a':('2006-01-01','2013-12-31'),'r1b':('2006-01-01','2026-12-31'),'r1c':('2019-01-01','2026-12-31')}.items():
    d = len(seg(C0f,a,b)); s = seg(SAf,a,b)
    up = int(s.upper.astype(str).eq('True').sum()); lo = int(s.lower.astype(str).eq('True').sum())
    # 結構上限：只有 2024-06 起的 D=S/D>S 槽可能 UNAVAIL
    cap = len(s[s.sig>='2024-06-01'])
    out[nm] = dict(denom=d, upper=up, upper_share=up/d, lower=lo, lower_cap_slots=cap, lower_cap_share=cap/d, lower_can_trigger=cap/d>0.05)
# V4 下界
sys.path.insert(0, str(HERE))
import avail
E = avail.Evidence('sig')
v4 = []
for r in SAf.itertuples():
    st,_,_ = E.classify(pd.Timestamp(r.sig), r.code, r.rev_month, version='V4')
    v4.append(st)
SAf = SAf.assign(v4=v4)
for nm,(a,b) in {'r1a':('2006-01-01','2013-12-31'),'r1b':('2006-01-01','2026-12-31'),'r1c':('2019-01-01','2026-12-31')}.items():
    s = seg(SAf,a,b); d = out[nm]['denom']
    out[nm]['V4_lower'] = int((s.v4=='UNAVAIL').sum()); out[nm]['V4_lower_share'] = out[nm]['V4_lower']/d
    out[nm]['V4_upper_share'] = int(s.v4.isin(['UNAVAIL','UNDET','NONE']).sum())/d
out['state_V6_vs_V4'] = pd.crosstab(SAf.state, SAf.v4).to_dict()
# 處置 × 貢獻股
R = json.loads((HERE/'results.json').read_text())
disp = R['ERRATUM']['R3']['counts']['2014_2021-09-23']['d_sig']['slots'] + R['ERRATUM']['R3']['counts']['2006_2013']['d_sig']['slots']
MV = json.loads((HERE.parent/'main_v2/results.json').read_text())['RUN']['gapfix']['official']['EXFIN']['10']['contrib']
top = {}
for sg,v in MV.items():
    for c,pp in v['top5']: top.setdefault(c,[]).append((sg,pp))
out['disposal_slots_of_top5_contributors'] = {x: top[x.split()[1]] for x in disp if x.split()[1] in top}
# 價格量級（ia_core 調整價；持有：成交日開盤 → 下一個訊號日的成交日開盤，只為量級）
sys.path.insert(0, str(HERE.parent/'independent_audit'))
try:
    import ia_core as K
    P = K.MYADJ
    days = sorted(C0f.sig.unique()); ex = dict(zip(C0f.sig, C0f.exec))
    mv = {}
    for x in disp + ['2021-06-11 2603']:
        s, c = x.split(); i = days.index(s)
        e0 = pd.Timestamp(ex[s]); e1 = pd.Timestamp(ex[days[i+1]]) if i+1 < len(days) else None
        if c in P.columns and e1 is not None:
            p = P[c]; a = p.loc[p.index>=e0].iloc[0]; b = p.loc[p.index>=e1].iloc[0]
            mv[x] = round(float(b/a-1), 4)
    out['disposal_slot_close_move_exec_to_next_exec'] = mv
except Exception as ex_:
    out['price_err'] = repr(ex_)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

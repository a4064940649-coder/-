#!/usr/bin/env python3
"""步驟 7：彙整判定（不改任何正式數字）＋事後描述（POSTHOC，不參與判定）。
POSTHOC：三個三分位平均的 α 斜率（≈ 主策略整體的 j 衰減）各期間；低三分位在 VAL 的衰減占整體的比重。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C, fa_stats as ST
R = C.check_prereg()
A = pd.read_parquet(C.DATA / 'alpha_all_v1.parquet')
a = A[(A.scheme == 'A') & (A.leg == 'main')]
avg = a.groupby(['m', 'j']).alpha.mean().reset_index().assign(scheme='AVG', leg='main', g=0)
PH = {}
for nm, per in (('PRE', C.PRE), ('POST', C.POST), ('VAL', C.VAL), ('HOLD', C.HOLD)):
    PH[nm] = {'avg3': ST.summ(ST.month_slopes(avg, 'AVG', 'main', 0, per=per)),
              **{f'g{g}': ST.summ(ST.month_slopes(A, 'A', 'main', g, per=per)) for g in (0, 1, 2)}}
O = R['RESULTS_OOS']; R2 = R['RESULTS_R2']
V = {'mechanism': O['mechanism'], 'R2_candidate': R2['verdict'], 'R_roll': '無法回測（時間戳閘門 33.6% < 60%）；2026-11 起前瞻紙上追蹤',
     'verdict': '不可用', 'verdict_reason': '機制不支持（I_後 方向相反、t 1.06；DDD t 0.60）；R2 候選條件 (1)(3) 不成立；R_roll 無法回測。'
     '研究員判定上限為待觀察，這裡判不可用；j0／j1／j6 與 R_roll 前瞻名單照登記繼續追蹤，只用來檢驗機制。'}
C.save_key('POSTHOC', PH); C.save_key('VERDICT', V)
for k, v in PH.items(): print(k, {g: (round(x['mean_pp'], 4), round(x['t'], 2)) for g, x in v.items()})
print(V)

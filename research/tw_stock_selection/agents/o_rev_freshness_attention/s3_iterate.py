#!/usr/bin/env python3
"""步驟 3：迭代（只限資料建構，不調訊號；只用前期 2006-04～2018-12）。
選版規則（跑之前寫死）：選 v1，除非 v1 的建構檢查失敗（E2 全 el 前 10 對 plans_j7 重現率 < 99%）；
其他版本只用來報告建構敏感度（前期 b 與 I_前 的變動），不依它們的數值選版。
  v1 基本版：el 內 V60 三分位、開盤漲停遞補（上期未持有者跳過）、AB 種子 0～4。
  v2 不做開盤漲停遞補（照買）。
  v3 三分位切點改用前 300 池（排除前，含金融／處置／缺營收），再取 el 內的成員。
  v4 AB 種子 0～9。
  v5 公告晚於收盤的處理：期限日（j＝0）全部當作未公告（用上月營收；it6 J0 worst 口徑），其餘同 v1。
輸出 iterations.json。"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fa_common as C, fa_build as B, fa_stats as ST

E = C.engine(); STC = E['STC']
months = set(pd.period_range(C.PRE[0], C.PRE[1], freq='M'))
RULE = '選 v1，除非 v1 的建構檢查失敗（E2 全 el 前 10 對 plans_j7 重現率 < 99%）；其他版本只報告建構敏感度，不依數值選版'
ex = C.jload(C.HERE / 'explore.json')
plumb = ex['E2_plumbing_top10_vs_plans_j7']['overall']

def summarize(A):
    o = {}
    for sch, leg, hi, lo in (('A', 'main', 2, 0), ('A', 'rev', 2, 0), ('A', 'px', 2, 0), ('M50', 'main', 1, 0)):
        s = ST.interaction(A, sch, leg, hi, lo, per=C.PRE)[0]
        o[f'{sch}|{leg}'] = {k: (round(s[k]['mean_pp'], 4), round(s[k]['t'], 2)) for k in ('b_hi', 'b_lo', 'I')}
    return o

VERS = {'v1': {}, 'v2': dict(block=False), 'v3': dict(tercile_on='top'), 'v4': dict(seeds=list(range(10)))}
IT = {'rule': RULE, 'n_versions': 5, 'versions': {}}
A1 = pd.read_parquet(C.DATA / 'alpha_pre_v1.parquet')
IT['versions']['v1'] = dict(spec='基本版', summary=summarize(A1), plumbing_top10=plumb)
for v, opt in VERS.items():
    if v == 'v1': continue
    A, _ = B.build(months, opt, contrib=False)
    IT['versions'][v] = dict(spec=json.dumps(opt), summary=summarize(A))
    print(v, IT['versions'][v]['summary']['A|main'], flush=True)

# v5：期限日全部當作未公告（暫時替換 unavail）
orig = STC.unavail_for
def worst(m, s, seed=0):
    u = orig(m, s, seed)
    if s == STC.deadline(m): u = np.ones_like(u)
    return u
STC.unavail_for = worst
B._S.clear()
A5, _ = B.build(months, {}, contrib=False)
STC.unavail_for = orig; B._S.clear()
IT['versions']['v5'] = dict(spec='期限日（j=0）營收一律視為未公告', summary=summarize(A5))
print('v5', IT['versions']['v5']['summary']['A|main'])
IT['selected'] = 'v1' if plumb >= 0.99 else None
IT['note'] = '五版前期 I_前 都在 |t| < 1.5 以內者視為建構穩健；選版不依 I 或 b 的數值'
C.jdump(IT, C.HERE / 'iterations.json')
print(json.dumps(C.conv(IT), ensure_ascii=False)[:2500])

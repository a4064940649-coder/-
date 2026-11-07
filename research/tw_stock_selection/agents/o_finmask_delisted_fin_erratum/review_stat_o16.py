#!/usr/bin/env python3
"""反方統計審查 o16（只讀研究員輸出；import flib／fclass 但不改任何檔）。不計算報酬。
檢查：
 1. 最壞聯合遮罩：F2（加遮 5854/6004/6012/5871）＋ 2841 全期放回（非金融）＋ 2841 只在 PIT 期放回，C0 與 U_opt 的 8 比例
 2. 事件群集：異動槽位按「代號×連續月份」群集後的有效事件數；14/810 的二項上界在群集下是否有意義
 3. panel 是否含上櫃（OTC）股：若含，delist_official（上市）不涵蓋上櫃下市金融股
 4. 沒有任何名稱來源的 panel 代號（F2 只能靠前綴）
 5. F1 在 2011-07-28 前其實是「事後最早快照」：PIT 覆蓋比例
用法：python3 -I -B review_stat_o16.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
SPEC, REG = R.check_reg('ERRATUM_SPEC_FM')
import fclass as F
import flib as L
import numpy as np, pandas as pd
CL = F.Classifier(SPEC['classifiers']['keywords'], SPEC['classifiers']['f2_manual_exclusion'], SPEC['classifiers']['name_exclusion'])
C0 = L.C0MASK
F2V = np.array([CL.f2(c) for c in L.COLS])
out = {}
R0 = {m: L.run(lambda d: C0, m) for m in ('base', 'opt')}
days = sorted(R0['base'])
j2841 = L.CI['2841']
def blk(A, B):
    tot, rows = L.sym_slots(A, B)
    t1 = sum(r['slots'] for r in rows if r['sig'] < '2013')
    return dict(s930=tot, s810=t1, sh930=tot / 930, sh810=t1 / 810, rows=[(r['sig'], r['out'], r['in_']) for r in rows])
# 1. 最壞聯合
def m_union_all(d):
    m = F2V.copy(); m[j2841] = False; return m
def m_union_pit(d):
    m = F2V.copy()
    if pd.Timestamp(d) >= pd.Timestamp('2011-07-28'): m[j2841] = False
    return m
for nm, fn in (('F2+2841_all_nonfin', m_union_all), ('F2+2841_PIT_nonfin', m_union_pit)):
    for mode in ('base', 'opt'):
        out[f'{mode}|{nm}'] = blk(R0[mode], L.run(fn, mode))
out['worst_union_max_share'] = max(max(v['sh930'], v['sh810']) for k, v in out.items() if '|' in k)
# 2. 群集
FM = json.loads((HERE / 'FM_COUNTS.json').read_text(encoding='utf-8'))
def clusters(rows):
    ev = {}
    for r in rows:
        for c in r['out'] + r['in_']:
            ev.setdefault(c, []).append(days.index(r['sig']))
    n = 0
    for c, ix in ev.items():
        ix = sorted(set(ix)); n += 1 + sum(1 for a, b in zip(ix, ix[1:]) if b - a > 1)
    return dict(n_codes_touched=len(ev), n_runs=n)
out['clusters'] = {k: clusters(v['rows']) for k, v in FM['blocks'].items()}
# 3. OTC
mk = {}
for c in L.COLS:
    r = F.LM.get(c, {})
    mk[r.get('isin_market', '') or 'NA'] = mk.get(r.get('isin_market', '') or 'NA', 0) + 1
out['panel_market_counts'] = mk
# 4. 無名稱
out['panel_no_name'] = [c for c in L.COLS if c not in F.NAMES]
# 5. PIT 覆蓋
pit = sum(1 for d in days if pd.Timestamp(d) >= pd.Timestamp('2011-07-28'))
out['signal_days_true_PIT'] = pit; out['signal_days_total'] = len(days)
(HERE / 'review_stat_o16.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: (v if not isinstance(v, dict) or 'rows' not in v else {kk: vv for kk, vv in v.items()}) for k, v in out.items()}, ensure_ascii=False, default=str)[:8000])

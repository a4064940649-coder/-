#!/usr/bin/env python3
"""統計審查（反方）：只讀 ERRATUM_SPEC_FM.json、FM_COUNTS.json，import flib／fclass 重跑敏感度；不改研究員輸出，結果寫 review_stat_fm.json。
檢查：判定口徑與分母、對稱差／2 是否等於換出數、UNKNOWN 往兩個方向處理時的最大比例、以 C0 以外的基準（F 遮罩互比）、多重口徑。
用法：python3 -I -B review_stat_fm.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freg as R
SPEC, REG = R.check_reg('ERRATUM_SPEC_FM')
import fclass as F
import flib as L
import numpy as np, pandas as pd
FM = json.loads((HERE / 'FM_COUNTS.json').read_text(encoding='utf-8'))
CL = F.Classifier(SPEC['classifiers']['keywords'], SPEC['classifiers']['f2_manual_exclusion'], SPEC['classifiers']['name_exclusion'])
C0 = L.C0MASK
out = {}
# 1. 名單長度與 sym/2
lens = {k: sorted({len(v) for v in d.values()}) for k, d in FM['lists'].items()}
out['list_lengths'] = lens
out['all_lists_size10'] = all(x == [10] for x in lens.values())
# 2. 分母
out['n_days'] = len(FM['lists']['C0|C0']); out['n_days_2006_2012'] = sum(1 for d in FM['lists']['C0|C0'] if d < '2013')
# 3. UNKNOWN 敏感度：UNKNOWN→FIN（有證據者）與 UNKNOWN→NONFIN（全部）
def mk(policy):
    cache = {}
    def f(d):
        if d in cache: return cache[d]
        m = np.zeros(len(L.COLS), bool)
        for j, c in enumerate(L.COLS):
            s, _ = CL.f1(c, d)
            if s in ('FIN', 'FIN_Q'): m[j] = True
            elif s in ('NONFIN', 'NONFIN_Q'): m[j] = False
            else: m[j] = (CL.evidence(c) if policy == 'fin_if_evidence' else False)
        cache[d] = m; return m
    return f
R0b = L.run(lambda d: C0, 'base'); R0o = L.run(lambda d: C0, 'opt')
sens = {}
for pol in ('fin_if_evidence', 'nonfin'):
    mf = mk(pol)
    for mode, R0 in (('base', R0b), ('opt', R0o)):
        Rr = L.run(mf, mode)
        tot, rows = L.sym_slots(R0, Rr)
        t1 = sum(r['slots'] for r in rows if r['sig'] < '2013')
        sens[f'{mode}|{pol}'] = dict(s930=tot, s810=t1, share_930=tot / 930, share_810=t1 / 810, rows=rows)
out['unknown_sensitivity'] = sens
out['unknown_sensitivity_max_share'] = max(max(v['share_930'], v['share_810']) for v in sens.values())
# 4. 判定比例重算（由 FM_COUNTS 名單直接算）
def ratio(a, b):
    A, B = FM['lists'][a], FM['lists'][b]
    s = [len(set(A[d]) ^ set(B[d])) / 2 for d in sorted(A)]
    return sum(s), sum(x for d, x in zip(sorted(A), s) if d < '2013')
rr = {}
for lst in ('C0', 'U_opt'):
    for nm in ('F1', 'F2'):
        a, b = ratio(f'C0|{lst}', f'{nm}|{lst}'); rr[f'{lst}|{nm}'] = dict(s930=a, s810=b)
out['recomputed_from_lists'] = rr
out['M_recomputed'] = max(max(v['s930'] / 930, v['s810'] / 810) for v in rr.values())
out['M_reported'] = FM['judgement']['M']
# 5. 二項上界：若真實異動率 = 14/810，單側 95% 上界
from math import comb
def ub(k, n, a=0.05):
    lo, hi = k / n, 1.0
    for _ in range(60):
        p = (lo + hi) / 2; cdf = sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))
        lo, hi = (p, hi) if cdf > a else (lo, p)
    return p
out['binom_ub95_14_of_810'] = ub(14, 810)
# 6. 異動的訊號日分群（同一事件 5854 連續月份）
rows = FM['blocks']['U_opt|F2']['rows']
out['U_opt_F2_changed_days'] = len(rows); out['U_opt_F2_changed_by_code_out'] = pd.Series([c for r in rows for c in r['out']]).value_counts().to_dict()
(HERE / 'review_stat_fm.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k != 'unknown_sensitivity'}, ensure_ascii=False, indent=0, default=str))
print({k: (v['s930'], v['s810'], [(r['sig'], r['out'], r['in_']) for r in v['rows']]) for k, v in sens.items()})

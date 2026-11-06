#!/usr/bin/env python3
"""步驟 7：in-sample 規則示範（不是證據）：把選定版本與 G8P1 套到 2006～2026 歷史 V7X GROW 100 萬 JMED（滾動 24 個月起點）。
斷言 PREREG 與 OC_FINAL 已存在。挑選期／驗證期／保留期起點分報。用法：python3 -I -B s7_hist_demo.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8lib2 as L

PR = L.check_reg('PREREG')
OCF = L.load_res()['OC_FINAL']
SEL = OCF['selected']
VERS = PR['versions']
G8P1 = dict(stat='FIX_REFIT', upgrade=True, c_d=L.CD_G8P1, b=L.B_G8P1, delta=0.0)
H = pd.read_csv(L.SCR / 'hist_lines_g8p2.csv', index_col=0)
H.index = pd.PeriodIndex(H.index, freq='M')


def apply(x, spec):
    x = np.asarray(x, float); n = len(x) - 24 + 1
    Xp = np.stack([x[s:s + 24] for s in range(n)])
    o = L.outcomes(L.zstats(Xp, spec), spec['b'] if spec['upgrade'] else 9.9, spec['c_d'], upgrade=spec['upgrade'])
    st = np.where(o['fail'], 'FORWARD_FAILED', np.where(o['up'], 'S1_EXCEPTION_EVIDENCE', 'FORWARD_VALIDATING_NO_REJECTION_24'))
    return st, o


out = {}
for nm, col, spec in (('V_sel_worse', 'V7X_worse', VERS[SEL]), ('V1_worse', 'V7X_worse', VERS['V1']), ('V3_worse', 'V7X_worse', VERS['V3']),
                      ('G8P1_normal', 'V7X_normal', G8P1), ('G8P1_worse', 'V7X_worse', G8P1), ('V1_normal', 'V7X_normal', VERS['V1'])):
    s = H[col].dropna()
    st, o = apply(s.to_numpy(), spec)
    starts = s.index[:len(st)]
    yrs = np.array([p.year for p in starts])
    def frac(m):
        return dict(n=int(m.sum()), S1=float(np.mean(st[m] == 'S1_EXCEPTION_EVIDENCE')), FAILED=float(np.mean(st[m] == 'FORWARD_FAILED')),
                    NO_REJ=float(np.mean(st[m] == 'FORWARD_VALIDATING_NO_REJECTION_24')), literal=float(np.mean(o['lit'][m])))
    out[nm] = dict(line=col, n_months=len(s), first=str(s.index[0]), last=str(s.index[-1]), mean_pp=float(s.mean()), sd_pp=float(s.std(ddof=1)),
                   all=frac(np.ones(len(st), bool)), sel_starts_2006_2017=frac(starts <= pd.Period('2017-04', 'M')),
                   val_starts_2017_2022=frac((starts > pd.Period('2017-04', 'M')) & (starts <= pd.Period('2021-12', 'M'))),
                   hold_starts_2022_on=frac(starts > pd.Period('2021-12', 'M')),
                   s2006_2012=frac(yrs <= 2012), s2013_2018=frac((yrs >= 2013) & (yrs <= 2018)), s2019_2024=frac(yrs >= 2019),
                   last_start=str(starts[-1]), last_status=str(st[-1]),
                   s1_starts=[str(p) for p, x in zip(starts, st) if x == 'S1_EXCEPTION_EVIDENCE'])
out['note'] = ('in-sample 規則示範，不是證據：歷史線已被看過 2,573 次以上。起點分期：挑選期起點（24 個月窗完全落在 2019-04 以前）≤ 2017-04；'
               '驗證期起點 2017-05～2021-12；保留期起點 2022-01 以後（窗含 2024 年起）。G8P1_normal 應重現 integrate_o14 的 1.35%（3/223）。')
L.save_key('HIST_DEMO', out, need=('PREREG', 'OC_FINAL'))
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 's1_starts'}) for k, v in out.items()}, ensure_ascii=False, indent=1))

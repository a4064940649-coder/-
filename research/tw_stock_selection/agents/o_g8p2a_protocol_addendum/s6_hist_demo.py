#!/usr/bin/env python3
"""步驟 6：in-sample 規則示範（PREREG 之後；不是證據）：歷史 V7X GROW100 較差成本 JMED 線 2006-04～2026-09，223 個 24 個月起點，
選定版本（v2a）與 v2 各算 S1／前瞻失敗比例（純量參考實作 refimpl）。
用法：python3 -I -B s6_hist_demo.py"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import g8alib as L
import refimpl as R

PRg = L.check_reg('PREREG', files=False)
res = L.load_res(); assert 'OC_FINAL' in res
sel = res['OC_FINAL']['summary']['selected']
PAR = dict(sigma_design=L.SIGMA_D, c=L.C_INFL, c_d=L.CD_V2, b_v2=L.B_V2, b_a=(PRg['params']['b_a'].get(sel) or 99.0))
Mn, Ms, F, _ = L.hist_windows()
w = L.jmed_line(Mn, Ms)
assert len(w) == 246 and str(w.index[0]) == '2006-04' and str(w.index[-1]) == '2026-09'
xs = w.values.astype(float); idx = [str(p) for p in w.index]
rows = []
for i in range(len(xs) - 23):
    items = [('valid', float(v)) for v in xs[i:i + 24]]
    st, s2, sn, lit = R.combined_state(items, PAR, sel)
    v2only, _ = R.seq_rule(items, R.rules(PAR, 'A1')[0])
    rows.append(dict(start=idx[i], v2a=st, v2=v2only, literal=lit))
D = pd.DataFrame(rows)
def frac(d, col):
    return dict(n=len(d), S1=float((d[col] == 'S1_EXCEPTION_EVIDENCE').mean()), FAILED=float((d[col] == 'FORWARD_FAILED').mean()),
                NO_REJ=float((d[col] == 'FORWARD_VALIDATING_NO_REJECTION_24').mean()))
out = dict(selected=sel, b_a=PAR['b_a'], line='V7X B10 K5T GROW 100 萬 較差成本 JMED（2006-04～2026-09，246 個月）', n_starts=len(D),
           v2a=frac(D, 'v2a'), v2=frac(D, 'v2'), literal_pass=float((D.literal == 'PASS').mean()),
           by_period={nm: dict(v2a=frac(d, 'v2a'), v2=frac(d, 'v2')) for nm, d in
                      (('sel_starts_2006_2017', D[D.start < '2017-05']), ('val_starts_2017_2022', D[(D.start >= '2017-05') & (D.start < '2022-01')]),
                       ('hold_starts_2022_on', D[D.start >= '2022-01']))},
           last_start=D.start.iloc[-1], last_v2a=D.v2a.iloc[-1], last_v2=D.v2.iloc[-1],
           n_v2a_better_than_v2=int(sum(R.ORDER.index(a) > R.ORDER.index(b) for a, b in zip(D.v2a, D.v2))),
           v2_protocol_ref=dict(S1=0.0, FAILED=0.6547085201793722))
L.save_key('HIST_DEMO', out, need=('PREREG', 'OC_FINAL'))
print(json.dumps(out, ensure_ascii=False, indent=1))

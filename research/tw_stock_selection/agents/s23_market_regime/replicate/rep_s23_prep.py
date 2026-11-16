#!/usr/bin/env python3
"""第 0 步核對（不算報酬）：自行重算 BRD3／LMA 訓練期分位數門檻、7 條規則各位移觸發月份與少買檔數，
對照原研究 data_check.json／PREREG.json（只讀這兩個 JSON 的數字）。python3 -I -B rep_s23_prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s23_lib as L
from rep_s23_lib import s20lib, s10lib, s4lib, stlib

t = time.time()
P = L.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
H = Path(__file__).resolve().parent
PR = json.loads((H.parent / 'PREREG.json').read_text())
prev = PR['rule_form']['rule_effect_preview（data_check，只數月數與檔數）']
out = {'panel': [P.T, P.N, str(P.IDX[-1].date())]}
for minp in (1, 3):
    df = L.regime_rows(P, U, RT, 0, brd3_minp=minp)
    tr = df[df['month'] <= L.TRAIN_LAST]
    out[f'thr_minp{minp}'] = dict(n_train=len(tr), n_valid=int((df['month'] >= L.VALID_FIRST).sum()),
                                  BRD3_q20=float(np.nanquantile(tr['BRD3'], 0.2)), BRD3_q33=float(np.nanquantile(tr['BRD3'], 1 / 3)),
                                  LMA_q20=float(np.nanquantile(tr['LMA'], 0.2)), LMA_q33=float(np.nanquantile(tr['LMA'], 1 / 3)),
                                  BRD3_train_q=[float(np.nanquantile(tr['BRD3'], q)) for q in (0.1, 0.2, 1 / 3, 0.5, 0.9)])
print(json.dumps(out, ensure_ascii=False), flush=True)
df = L.regime_rows(P, U, RT, 0)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ref = {int(x['k']): x['targets'] for x in S0['info']}
assert all([str(P.COLS[j]) for j in f1] == ref[k] for k, f1 in zip(df['k'], df['f1'])) and len(df) == len(ref)
cmp = {}
for r in L.RULES:
    trig = L.triggered(df, r)
    act = L.RULES[r]['action']
    d = {}
    for per, sel in (('train', df['month'] <= L.TRAIN_LAST), ('valid', df['month'] >= L.VALID_FIRST)):
        sub = df[sel.to_numpy()]; tg = trig[sel.to_numpy()]
        f1o = sum(len(f) for f, g in zip(sub['f1'], tg) if g)
        kept = sum(int(np.ceil(len(f) / 2)) for f, g in zip(sub['f1'], tg) if g) if act == 'half' else 0
        months = [str(m) for m, g in zip(sub['month'], tg) if g]
        o = prev[r][per]
        d[per] = dict(months_triggered=len(months), f1_orig_in_trig=f1o, f1_kept_in_trig=kept, f1_total=int(sum(len(f) for f in sub['f1'])),
                      match_months=months == o['trig_months'], match_counts=(len(months), f1o, kept) == (o['months_triggered'], o['f1_orig_in_trig'], o['f1_kept_in_trig']),
                      trig_months=months)
    cmp[r] = d
    print(r, {p: (d[p]['months_triggered'], d[p]['f1_orig_in_trig'], d[p]['f1_kept_in_trig'], d[p]['match_months'], d[p]['match_counts']) for p in d}, flush=True)
out['rule_effect_shift0'] = cmp
# 各位移訓練期觸發月數、少買檔數（對照 results_train.json rule_effect_train）
eff = {}
for s in range(4):
    dfs = L.regime_rows(P, U, RT, s)
    tr = dfs['month'] <= L.TRAIN_LAST
    for r in L.RULES:
        trig = L.triggered(dfs, r) & tr.to_numpy()
        act = L.RULES[r]['action']
        drop = sum(len(f) - (int(np.ceil(len(f) / 2)) if act == 'half' else 0) for f, g in zip(dfs['f1'], trig) if g)
        eff.setdefault(r, {})[f'+{s}'] = dict(months_triggered=int(trig.sum()), f1_dropped=int(drop))
out['rule_effect_train_by_shift'] = eff
print(json.dumps(eff, ensure_ascii=False), flush=True)
out['features_shift0'] = [dict(month=str(m), BRD=float(b), BRD3=float(b3), LMA=int(l), MA200=bool(a)) for m, b, b3, l, a in
                          zip(df['month'], df['BRD'], df['BRD3'], df['LMA'], df['MA200'])]
out['meta'] = dict(runtime_s=round(time.time() - t, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(L.REP / 'prep_rep.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(out['meta'])

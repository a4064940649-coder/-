#!/usr/bin/env python3
"""第 0 步核對（不算報酬）：特徵分布、規則改動檔數，對照原研究 data_check.json。python3 -I -B rep_s21_prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rep_s21_lib as L
from rep_s21_lib import s20lib, s10lib, s4lib, stlib

t = time.time()
P = L.load()
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
G = s20lib.growth_features(P)
RT = dict(S=G['S'], av=G['av'])
PI = L.PITInd(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
RULES = ['CUR', 'FLT_P2', 'FLT_REL', 'FILL_P2', 'SWAP_P2', 'RANK_BR', 'FILL_ALL']
out = {'panel': [P.T, P.N, str(P.IDX[-1].date())]}
for nmin in (5, 6):
    for meth in (['average', 'min', 'first', 'dense'] if nmin == 5 else ['average']):
        V, rows = L.build_variants(P, U, RT, PI, S0, 0, RULES, nmin=nmin, rank_method=meth)
        res = {}
        for per, sel in (('train', lambda m: pd.Period(m, 'M') <= L.TRAIN_LAST), ('valid', lambda m: pd.Period(m, 'M') >= L.VALID_FIRST)):
            rr = [x for x in rows if sel(x['month'])]
            f1 = [(x, j) for x in rr for j in x['f1']]
            r11 = [(x, j) for x in rr for j in x['r1120']]
            d = dict(n_months=len(rr), F1_n=len(f1), F1_months=sum(1 for x in rr if x['f1']),
                     F1_brx_missing=float(np.mean([not np.isfinite(x['BRX'][j]) for x, j in f1])),
                     F1_ind_missing=float(np.mean([x['ind'][j] < 0 for x, j in f1])),
                     F1_PEER_hist=pd.Series([min(int(x['PEER'][j]), 10) for x, j in f1]).value_counts().sort_index().to_dict(),
                     F1_BRX_q=[float(np.nanquantile([x['BRX'][j] for x, j in f1], q)) for q in (1 / 3, 1 / 2, 2 / 3)],
                     R11_20_brx_missing=float(np.mean([not np.isfinite(x['BRX'][j]) for x, j in r11])) if r11 else None)
            eff = {}
            for r in RULES[1:]:
                n0 = sum(len(x['f1']) for x in rr); n1 = sum(len(x['new'][r]) for x in rr)
                rem = sum(len(set(x['f1']) - set(x['new'][r])) for x in rr); add = sum(len(set(x['new'][r]) - set(x['f1'])) for x in rr)
                ch = sum(1 for x in rr if x['new'][r] != x['f1'])
                eff[r] = dict(n_orig=n0, n_new=n1, removed=rem, added=add, months_changed=ch, n_months=len(rr))
            d['effect'] = eff
            res[per] = d
        res['coverage_every12'] = {x['month']: dict(n_fin=x['n_fin'], share_ge2=x['share_hi']) for x in rows if x['month'].endswith('-01')}
        out[f'nmin{nmin}_{meth}'] = res
        print(nmin, meth, json.dumps({p: res[p]['effect'] for p in ('train', 'valid')}, ensure_ascii=False), flush=True)
        print('  brx_missing', res['train']['F1_brx_missing'], res['train']['R11_20_brx_missing'], res['valid']['R11_20_brx_missing'], flush=True)
out['meta'] = dict(runtime_s=round(time.time() - t, 1), maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
(L.REP / 'prep_rep.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out['nmin5_average']['train'], ensure_ascii=False, default=str)[:2500])
print(json.dumps(out['nmin5_average']['coverage_every12'], ensure_ascii=False))
print(out['meta'])

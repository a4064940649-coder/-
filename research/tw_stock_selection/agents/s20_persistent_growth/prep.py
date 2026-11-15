#!/usr/bin/env python3
"""S20 第 0 步（不算任何報酬）：建面板與創新板快取（scratchpad/s20tmp/）、檢查面板同 S17；算 μ、σ、MS＝μ÷σ、MU、NPOS12；
F1 名單（位移 0）上的特徵涵蓋率、訓練期分位數、特徵相關、各候選規則在 F1 中留下的比例（只數檔數）。輸出 data_check.json。
python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s20lib
from s20lib import s10lib, s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s20lib.load()
A, names = s4lib.innov_panel(P)
print('面板', P.T, P.N, str(P.IDX[-1].date()), '創新板', len(names), f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
G = s20lib.growth_features(P)
RT0 = s2lib.rev_tables(P)
S0 = s10lib.make_signal_s10(P, U, RT0, shift=0)
S1 = s10lib.make_signal_s10(P, U, dict(S=G['S'], av=G['av']), shift=0)
acc = dict(A1_same_sig=bool(np.array_equal(S0['sig'], S1['sig'])), A1_same_X=bool(np.array_equal(S0['X'], S1['X'])))
f1 = s20lib.f1_table(P, S1, G)
f1['mper'] = pd.PeriodIndex(f1['month'], freq='M')
tr = f1[f1['mper'] <= s20lib.TRAIN_LAST_MONTH]; va = f1[f1['mper'] >= s20lib.VALID_FIRST_MONTH]
qs = (1 / 3, 1 / 2, 2 / 3)
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), acceptance=acc, sigma0=G['n_sigma0'],
           f1=dict(n_months_total=int(f1['month'].nunique()), first=f1['month'].min(), last=f1['month'].max(),
                   train=dict(n_months=int(tr['month'].nunique()), n=int(len(tr)), first_sig=str(tr['sig_date'].min().date()), last_sig=str(tr['sig_date'].max().date())),
                   valid=dict(n_months=int(va['month'].nunique()), n=int(len(va)), first_sig=str(va['sig_date'].min().date()), last_sig=str(va['sig_date'].max().date()))))
for nm, d in (('train', tr), ('valid', va)):
    o = {}
    for f in ('MS', 'MU', 'NPOS12'):
        v = d[f].to_numpy()
        o[f] = dict(coverage=float(np.isfinite(v).mean()), mean=float(np.nanmean(v)), q=[float(np.nanquantile(v, q)) for q in qs],
                    min=float(np.nanmin(v)), max=float(np.nanmax(v)))
    o['NPOS12_hist'] = {int(k): int(v) for k, v in d['NPOS12'].value_counts().sort_index().items()}
    o['spearman'] = dict(MS_NPOS12=float(d['MS'].rank().corr(d['NPOS12'].rank())), MS_MU=float(d['MS'].rank().corr(d['MU'].rank())),
                         NPOS12_MU=float(d['NPOS12'].rank().corr(d['MU'].rank())), MS_SURGE=float(d['MS'].rank().corr(d['surge'].rank())))
    chk[f'features_{nm}'] = o
# 候選門檻（訓練期 F1 分位數，凍結成數字前的預覽；只數檔數）
qms = [float(np.quantile(tr['MS'], q)) for q in qs]
qnp = sorted(set(int(np.ceil(np.quantile(tr['NPOS12'], q) - 1e-12)) for q in qs))
prev = {}
for q, thr in zip(('1/3', '1/2', '2/3'), qms):
    prev[f'MS>={q}分位({thr:.4f})'] = dict(train=float((tr['MS'] >= thr).mean()), valid=float((va['MS'] >= thr).mean()))
prev['MU>0'] = dict(train=float((tr['MU'] > 0).mean()), valid=float((va['MU'] > 0).mean()))
for thr in qnp:
    prev[f'NPOS12>={thr}'] = dict(train=float((tr['NPOS12'] >= thr).mean()), valid=float((va['NPOS12'] >= thr).mean()))
chk['keep_frac_preview'] = prev
chk['thresholds_preview'] = dict(MS=qms, NPOS12_int=qnp)
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps(chk, ensure_ascii=False, indent=1))

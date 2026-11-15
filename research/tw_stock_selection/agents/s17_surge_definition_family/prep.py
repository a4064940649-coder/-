#!/usr/bin/env python3
"""S17 前置（第 0 步，不算任何報酬）：建面板與創新板快取（scratchpad/s17tmp/）、檢查面板和 S6～S16 相同；
算 7 個 SURGE 定義，檢查 STD 和 s2lib.rev_tables 相同、σ＝0 格數、各定義每月有值檔數；檢查 0050 停牌日。輸出 data_check.json。
python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s17lib
from s17lib import s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
P = s17lib.load()
A, names = s4lib.innov_panel(P)
print('面板', P.T, P.N, str(P.IDX[-1].date()), '創新板', len(names), f'{time.time()-t:.0f}s', round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), 'MB', flush=True)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗：面板和 S6～S16 不同'
U = stlib.universe(P, 'twse') & ~A
RT = s2lib.rev_tables(P)
F = s17lib.rev_family(P)
S0, S1 = RT['S'], F['S']['STD']
both = np.isfinite(S0.to_numpy()) & np.isfinite(S1.to_numpy())
only0 = np.isfinite(S0.to_numpy()) & ~np.isfinite(S1.to_numpy())
only1 = ~np.isfinite(S0.to_numpy()) & np.isfinite(S1.to_numpy())
d = np.abs(S0.to_numpy()[both] - S1.to_numpy()[both])
only0_cells = [(str(S0.index[i]), str(S0.columns[j]), float(S0.iat[i, j])) for i, j in zip(*np.nonzero(only0))][:20]
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), innov=len(names)),
           std_vs_s2lib=dict(n_both=int(both.sum()), max_abs_diff=float(d.max()), n_only_s2lib=int(only0.sum()), n_only_s17=int(only1.sum()), only_s2lib_cells=only0_cells),
           sigma0_cells=F['n_sigma0'])
# 各定義：訊號日（未位移）股池內有值檔數
cov = {}
for k, Sd in F['S'].items():
    n = []
    first = None
    for m in Sd.index:
        k0 = F['av'][m]
        if k0 >= P.T or k0 < 61:
            continue
        c = int((U[k0] & np.isfinite(Sd.loc[m].to_numpy())).sum())
        if c >= 50:
            n.append(c)
            first = first or str(m)
    cov[k] = dict(n_signal_months=len(n), first_month=first, median_n=float(np.median(n)), min_n=int(min(n)), max_n=int(max(n)))
chk['coverage_in_universe'] = cov
# 截面 Spearman（各定義對 STD，訊號日股池內，2016-01 起每月中位數）——只看分數，不看報酬
sp = {}
for k, Sd in F['S'].items():
    if k == 'STD':
        continue
    rs = []
    for m in S1.index:
        k0 = F['av'][m]
        if k0 >= P.T or k0 < 61:
            continue
        a, b = S1.loc[m].to_numpy(), Sd.loc[m].to_numpy()
        ok = U[k0] & np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 50:
            rs.append(pd.Series(a[ok]).rank().corr(pd.Series(b[ok]).rank()))
    sp[k] = dict(median=float(np.median(rs)), min=float(np.min(rs)), n=len(rs))
chk['spearman_vs_STD'] = sp
chk['e50_suspension_fix'] = s17lib.e50_fix_suspension(P)
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps(chk, ensure_ascii=False, indent=1))

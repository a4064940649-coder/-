#!/usr/bin/env python3
"""S21 第 0 步（不算任何報酬）：建面板快取（scratchpad/s21/）、PIT 產業對照、F1 名單與 L_MA 第 11～20 名上的共振特徵涵蓋率與分位數、
各候選規則會改動多少名單（只數檔數）。輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s21lib
from s21lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s21lib.load()
A, names = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s21lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ks = [x['k'] for x in S0['info']]
IND, inames = s21lib.industry_matrix(P, ks)
df = s21lib.f1_breadth_table(P, U, RT, S0, IND)
df['mper'] = pd.PeriodIndex(df['month'], freq='M')
df['period'] = np.where(df['mper'] <= s21lib.TRAIN_LAST_MONTH, 'train', np.where(df['mper'] >= s21lib.VALID_FIRST_MONTH, 'valid', 'pre'))
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', flush=True)

# 股池（U 且 SURGE 有值）的產業涵蓋率
cov = {}
for k in ks[::12]:
    m = next(x['month'] for x in S0['info'] if x['k'] == k)
    s = RT['S'].loc[pd.Period(m, 'M')].to_numpy()
    fin = U[k] & np.isfinite(s)
    cov[m] = dict(n_fin=int(fin.sum()), ind_cov=float((IND[k][fin] >= 0).mean()), n_ind=int(len(set(IND[k][fin & (IND[k] >= 0)]))),
                  share_surge_ge2=float((np.nan_to_num(s[fin], nan=-9) >= 2).mean()))
chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date())), industries=inames, universe_coverage_every12=cov)
qs = (1 / 3, 1 / 2, 2 / 3)
for per in ('train', 'valid'):
    o = {}
    for grp in ('F1', 'R11_20'):
        d = df[(df['period'] == per) & (df['grp'] == grp)]
        o[grp] = dict(n=int(len(d)), n_months=int(d['month'].nunique()), ind_missing=float((d['ind'] < 0).mean()), brx_missing=float(d['BRX'].isna().mean()),
                      PEER_hist={int(k): int(v) for k, v in d['PEER'].dropna().astype(int).clip(upper=10).value_counts().sort_index().items()},
                      PEER_ge1=float((d['PEER'] >= 1).mean()), PEER_ge2=float((d['PEER'] >= 2).mean()),
                      BRX_q=[float(np.nanquantile(d['BRX'], q)) for q in qs], BRX_mean=float(np.nanmean(d['BRX'])),
                      ret20_gt10=float((d['ret20'] > 0.10).mean()))
    f1 = df[(df['period'] == per) & (df['grp'] == 'F1')]
    o['F1_count_per_month'] = {int(k): int(v) for k, v in f1.groupby('month').size().value_counts().sort_index().items()}
    o['top_industries_F1'] = {inames[i] if i >= 0 else '缺值': int(v) for i, v in f1['ind'].value_counts().head(10).items()}
    chk[per] = o
tr = df[(df['period'] == 'train') & (df['grp'] == 'F1')]
chk['thresholds_preview'] = dict(BRX_F1_train=[float(np.nanquantile(tr['BRX'], q)) for q in qs])
# 各規則改動名單的程度（只數檔數，不算報酬）
eff = {}
for r in s21lib.RULES + [s21lib.CONTROL]:
    _, lg = s21lib.build_variant(P, U, RT, S0, IND, r)
    o = {}
    for per in ('train', 'valid'):
        L = [x for x in lg if (pd.Period(x['month'], 'M') <= s21lib.TRAIN_LAST_MONTH) == (per == 'train') and (per == 'train' or pd.Period(x['month'], 'M') >= s21lib.VALID_FIRST_MONTH)]
        no = sum(len(x['orig']) for x in L); nn = sum(len(x['new']) for x in L)
        rem = sum(len(set(x['orig']) - set(x['new'])) for x in L); add = sum(len(set(x['new']) - set(x['orig'])) for x in L)
        o[per] = dict(n_orig=no, n_new=nn, removed=rem, added=add, months_changed=sum(1 for x in L if x['orig'] != x['new']), n_months=len(L))
    eff[r['id']] = o
chk['rule_effect_preview'] = eff
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1))
print(json.dumps(chk, ensure_ascii=False, indent=1))

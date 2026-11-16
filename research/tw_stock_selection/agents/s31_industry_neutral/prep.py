#!/usr/bin/env python3
"""S31 第 0 步（不算任何報酬）：面板（除權息修正開）、PIT 產業涵蓋、F1 名單產業集中度、同業家數分布、各候選規則改動名單的檔數。
輸出 data_check.json。python3 -I -B prep.py"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s31lib
from s31lib import s10lib, s4lib, stlib

H = Path(__file__).resolve().parent
t = time.time()
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
P = s31lib.load()
A, _ = s4lib.innov_panel(P)
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22'), 'P0 失敗'
U = stlib.universe(P, 'twse') & ~A
del A
RT = s31lib.surge(P)
S0 = s10lib.make_signal_s10(P, U, RT, shift=0)
ks = [x['k'] for x in S0['info']]
IND, inames = s31lib.industry_matrix(P, ks)
CIX = {str(c): i for i, c in enumerate(P.COLS)}
print('載入', f'{time.time()-t:.0f}s', rss(), 'MB', P.exright_info, flush=True)
per = lambda m: 'train' if pd.Period(m, 'M') <= s31lib.TRAIN_LAST_MONTH else ('valid' if pd.Period(m, 'M') >= s31lib.VALID_FIRST_MONTH else 'pre')

chk = dict(panel=dict(T=P.T, N=P.N, last=str(P.IDX[-1].date()), exright_fix=P.exright_fix, exright_info=P.exright_info), industries=inames)
# 涵蓋與同業家數
peer = {'train': [], 'valid': []}; f1peer = {'train': [], 'valid': []}; cov = {'train': [], 'valid': []}
for x in S0['info']:
    p = per(x['month'])
    if p == 'pre':
        continue
    k = x['k']; s = RT['S'].loc[pd.Period(x['month'], 'M')].to_numpy()
    fin = U[k] & np.isfinite(s); g = IND[k]
    cov[p].append(float((g[fin] >= 0).mean()))
    vc = pd.Series(g[fin & (g >= 0)]).value_counts()
    peer[p].append(float((vc - 1 >= s31lib.MIN_PEERS).reindex(g[fin & (g >= 0)]).mean()))
    for c in x['targets']:
        j = CIX[c]
        f1peer[p].append(int(vc.get(g[j], 0)) - 1 if g[j] >= 0 else -1)
for p in ('train', 'valid'):
    a = np.array(f1peer[p])
    chk[p] = dict(n_months=len(cov[p]), ind_coverage_min=min(cov[p]), share_universe_peers_ge5=float(np.mean(peer[p])),
                  F1_n=int(len(a)), F1_peers_lt5=float((a < s31lib.MIN_PEERS).mean()), F1_peers_median=float(np.median(a)))
# 各規則改動名單（只數檔數）＋名單層集中度
eff = {}
for r in [dict(id='CUR_ID', kind='score', score='identity')] + s31lib.RULES:
    S1, lg = s31lib.build_rule(P, U, RT, S0, IND, r)
    if r['id'] == 'CUR_ID':
        acc = dict(sig_identical=bool((S1['sig'] == S0['sig']).all()), X_identical=bool((S1['X'] == S0['XV']['E0']).all()))
        assert acc['sig_identical'] and acc['X_identical'], acc
        chk['A1_identity_builder'] = acc
        continue
    o = {}
    for p in ('train', 'valid'):
        L = [x for x in lg if per(x['month']) == p]
        o[p] = dict(n_orig=sum(len(x['orig']) for x in L), n_new=sum(len(x['new']) for x in L),
                    removed=sum(len(set(x['orig']) - set(x['new'])) for x in L), added=sum(len(set(x['new']) - set(x['orig'])) for x in L),
                    months_changed=sum(1 for x in L if x['orig'] != x['new']), n_months=len(L), list_div=s31lib.list_div(L))
        if r['kind'] == 'score':
            # 出場名單（前 20）改變多少：與現行 E0 的 X 不同的格數（只在訊號列、只算股池內）
            ksP = [x['k'] for x in L]
            d = [(int(((S1['X'][k] != S0['XV']['E0'][k]) & U[k]).sum())) for k in ksP]
            o[p]['top20_changed_per_month_mean'] = float(np.mean(d)) / 2
    eff[r['id']] = o
chk['rule_effect_preview'] = eff
# 現行 F1 名單產業分布（訓練期）
cnt = {}
for x in S0['info']:
    if per(x['month']) != 'train':
        continue
    for c in x['targets']:
        g = IND[x['k']][CIX[c]]
        nm = inames[g] if g >= 0 else '缺值'
        cnt[nm] = cnt.get(nm, 0) + 1
chk['train_F1_by_industry_top10'] = dict(sorted(cnt.items(), key=lambda z: -z[1])[:10])
chk['runtime_s'] = round(time.time() - t, 1); chk['maxrss_mb'] = rss()
(H / 'data_check.json').write_text(json.dumps(chk, ensure_ascii=False, indent=1, default=str))
print(json.dumps(chk, ensure_ascii=False, indent=1, default=str))

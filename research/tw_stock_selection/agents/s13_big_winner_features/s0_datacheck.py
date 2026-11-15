#!/usr/bin/env python3
"""S13 第 0 步（PREREG 前）：只建候選與六個特徵，檢查涵蓋率、候選數、F1 名單是否與 s10lib 相同。完全不算任何報酬（不碰 ret／excess）。
python3 -I -B s0_datacheck.py → data_check.json"""
import sys, json, time, resource
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s13lib
from s13lib import s10lib, s4lib, s2lib, stlib

H = Path(__file__).resolve().parent
t0 = time.time()
P = s13lib.load()
assert (P.T, P.N, str(P.IDX[-1].date())) == (2876, 2142, '2026-10-22')
U0 = stlib.universe(P, 'twse'); INNOV, _ = s4lib.innov_panel(P); U = U0 & ~INNOV; del U0, INNOV
RT = s2lib.rev_tables(P)
Rv = s13lib.rev_matrix(P)
FG = s13lib.fin_gm()
cand = s13lib.candidates(P, U, RT, 0)
S = s10lib.make_signal_s10(P, U, RT, shift=0)
# 驗收 A1：候選中的 f1＝s10lib 的買進名單；月份相同
tg = {x['k']: set(x['targets']) for x in S['info']}
mine = cand[cand['f1']].groupby('k')['code'].apply(set).to_dict()
a1 = (sorted(tg) == sorted(cand['k'].unique().tolist())) and all(tg[k] == mine.get(k, set()) for k in tg)
cand = s13lib.add_features(P, RT, Rv, FG, cand)
cand['period'] = np.where(cand['sig_date'] <= pd.Timestamp('2021-12-31'), 'train', 'valid')
out = dict(acceptance_A1_f1_equals_s10=bool(a1), n_months=int(cand['k'].nunique()),
           first_sig=str(cand['sig_date'].min().date()), last_sig=str(cand['sig_date'].max().date()))
for p, g in cand.groupby('period'):
    d = dict(n_months=int(g['k'].nunique()), n_cand=int(len(g)), n_per_month_min=int(g.groupby('k').size().min()),
             n_f1=int(g['f1'].sum()), sig_from=str(g['sig_date'].min().date()), sig_to=str(g['sig_date'].max().date()),
             coverage={f: float(np.isfinite(g[f]).mean()) for f in s13lib.FEATURES},
             feature_quantiles={f: [float(x) for x in np.nanquantile(g[f], [0.1, 1 / 3, 0.5, 2 / 3, 0.9])] for f in s13lib.FEATURES})
    out[p] = d
cr = cand[s13lib.FEATURES].corr(method='spearman')
out['feature_spearman_all'] = {a: {b: round(float(cr.loc[a, b]), 3) for b in s13lib.FEATURES} for a in s13lib.FEATURES}
out['corr_with_surge_rank'] = {f: round(float(cand[[f, 'surge']].corr(method='spearman').iloc[0, 1]), 3) for f in s13lib.FEATURES}
out['gm_quarter_used_examples'] = cand[['sig_date', 'code', 'gm_q']].drop_duplicates('sig_date').head(8).astype(str).values.tolist()
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(H / 'data_check.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

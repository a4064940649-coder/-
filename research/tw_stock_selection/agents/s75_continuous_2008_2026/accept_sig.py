#!/usr/bin/env python3
"""S75 驗收 A2／A3（只比訊號與 SURGE，不跑帳戶、不算任何策略報酬）：連續面板上
A2 SURGE：MASK 版在營收月 2016-01 起與 B2 版逐格相同；兩版在營收月 ≤ 2012-12 逐格相同；MASK 版 2013-01～2015-12 全缺值。
A3 訊號：(a) 兩版在營收月 ≥ 2016-01（訊號日 2016-02 起）的每月訊號日與入選名單，與正式引擎（accept_is.py 存的）逐月相同；
         (b) V_BRIDGE 在訊號日 ≤ 2014-12-31 的每月入選名單與 S71 oos_signals.csv（B2）逐月相同；2015 上半年（接縫後改用官方面板那一邊）另報相同月數。
python3 -I -B accept_sig.py → accept_sig.json"""
import sys, json, time, resource
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import s75lib as K
import numpy as np, pandas as pd

t0 = time.time()
P, U = K.load_cont()
out = dict(panel=dict(T=P.T, N=P.N, first=str(P.IDX[0].date()), t0=str(P.IDX[P.t0].date()), last=str(P.IDX[-1].date()), exright_info=P.exright_info,
                      e50_open_missing=int(P.e50_open_missing), e50_missing=int(P.e50_missing), bad_ret=int(P.bad_ret)))
Z = {}; SG = {}
for v, rule in K.VERSIONS.items():
    F, RT = K.surge(P, rule); Z[v] = F['S']['STD']; SG[v] = K.signal(P, U, RT, 0)
zm, zb = Z['V_MASK'], Z['V_BRIDGE']
eq = lambda a, b: bool(np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True))
post = zm.index >= pd.Period('2016-01', 'M'); pre = zm.index <= pd.Period('2012-12', 'M'); mid = ~post & ~pre
out['A2_surge'] = dict(post2016_identical=eq(zm[post], zb[post]), pre2013_identical=eq(zm[pre], zb[pre]), mask_2013_2015_all_nan=bool(zm[mid].isna().all().all()),
                       bridge_2013_2015_n_finite=int(np.isfinite(zb[mid].to_numpy()).sum()))
ref = json.load(open(K.TMP / 'is_targets.json'))
a3 = {}
for v in K.VERSIONS:
    mine = {i['month']: i for i in SG[v]['info']}
    post_m = [m for m in mine if m >= '2016-01']
    bad = [m for m in sorted(set(post_m) | set(ref)) if m not in mine or m not in ref or mine[m]['sig_date'] != ref[m]['sig_date'] or mine[m]['targets'] != ref[m]['targets']]
    a3[v] = dict(n_signal_months=len(mine), first=SG[v]['info'][0]['sig_date'], last=SG[v]['info'][-1]['sig_date'], n_post2016=len(post_m), n_official=len(ref),
                 post2016_mismatch_months=bad, post2016_identical=bool(not bad),
                 n_signal_months_2013_02_to_2016_01=int(sum(1 for m in mine if '2013-01' <= m <= '2015-12')),
                 mean_targets_all=float(np.mean([i['n_tgt'] for i in SG[v]['info']])))
s71 = pd.read_csv(K.AG / 's71_ifrs_bridge_oos' / 'oos_signals.csv', dtype=str).fillna('')
s71 = s71[s71['rule'] == 'B2'].set_index('month')
mine = {i['month']: i for i in SG['V_BRIDGE']['info']}
cmp = [(m, mine[m]['sig_date'], ' '.join(mine[m]['targets']) == s71.at[m, 'targets'] and mine[m]['sig_date'] == s71.at[m, 'sig_date']) for m in s71.index if m in mine]
a3['bridge_vs_S71_B2'] = dict(n_S71_months=int(len(s71)), n_compared=len(cmp), missing_in_mine=[m for m in s71.index if m not in mine],
                              pre_seam=dict(n=sum(1 for c in cmp if c[1] <= '2014-12-31'), n_same=sum(1 for c in cmp if c[1] <= '2014-12-31' and c[2]),
                                            mismatch=[c[0] for c in cmp if c[1] <= '2014-12-31' and not c[2]]),
                              post_seam_2015H1=dict(n=sum(1 for c in cmp if c[1] > '2014-12-31'), n_same=sum(1 for c in cmp if c[1] > '2014-12-31' and c[2]),
                                                    mismatch=[c[0] for c in cmp if c[1] > '2014-12-31' and not c[2]]))
out['A3_signals'] = a3
out['pass'] = bool(out['A2_surge']['post2016_identical'] and out['A2_surge']['pre2013_identical'] and out['A2_surge']['mask_2013_2015_all_nan']
                   and a3['V_MASK']['post2016_identical'] and a3['V_BRIDGE']['post2016_identical'] and not a3['bridge_vs_S71_B2']['pre_seam']['mismatch'])
out['runtime_s'] = round(time.time() - t0, 1); out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
(K.HERE / 'accept_sig.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

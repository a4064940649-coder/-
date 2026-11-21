#!/usr/bin/env python3
"""第 6 項（去掉報酬率前 5% 的交易）的定義敏感度＋與 S76／S84 輸出 csv 的逐月比對。不讀 S76／S84 的 .py。"""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent
sys.path.insert(0, str(AG / 's75_continuous_2008_2026'))
import s75lib as K
from s75lib import cslib, stlib, s10lib
P, U = K.load_cont(); F, RT = K.surge(P, 'B2'); sg = K.signal(P, U, RT, 0)
nav, lg, st = K.sim(P, sg, U, 1e6)
IDX = P.IDX; D0 = IDX[P.t0]; cg = stlib.cagr
e50 = pd.Series(P.E50, index=IDX); bench = e50.loc[D0:] / e50.loc[D0] * 1e6
def sim_block(blk):
    S2, ekw = s10lib.variant_S(sg, 'E0')
    return s10lib.simulate_s10(P, S2, U, cost=K.L.COSTS['strict'], capital=1e6, cap_pct=0.05, sat_w=0.7, sizing='W10', lock=False, block=blk, **ekw)[0]
res = {}
for nm, a, b in (('early', D0, pd.Timestamp('2015-12-31')), ('late', pd.Timestamp('2015-12-31'), IDX[-1])):
    x0 = lg[(lg['ent_date'] > a if nm == 'late' else lg['ent_date'] >= a) & (lg['ent_date'] <= b)]
    for pool in ('all', 'closed'):
        x = (x0 if pool == 'all' else x0[x0['why'] != 'open_end']).sort_values('ret', ascending=False)
        for cnt in ('ceil', 'floor'):
            n = int(math.ceil(len(x) * .05 - 1e-9)) if cnt == 'ceil' else int(len(x) * .05 + 1e-9)
            top = x.head(n)
            for span in ('hold', 'entry_day'):
                blk = {}
                for j, e, xx in zip(top['j'], top['ent_t'], top['exit_t']):
                    blk.setdefault(int(j), []).append((int(e), int(xx) if span == 'hold' else int(e)))
                n2 = sim_block(blk)
                res[f'{nm}|{pool}|{cnt}|{span}'] = dict(n_pool=len(x), n_drop=n, diff_pp=(cg(n2, a, b) - cg(bench, a, b)) * 100)
            # 不重跑：把那幾筆換成同期間 0050 的報酬（損益差直接從期末淨值扣）
            alt = sum(c * ((1 + r) - e50.iloc[xt] / e50.iloc[et]) for c, r, et, xt in zip(top['cost'], top['ret'], top['ent_t'], top['exit_t']))
            yrs = (b - a).days / 365.25
            res[f'{nm}|{pool}|{cnt}|arith_to_0050'] = dict(n_pool=len(x), n_drop=n, diff_pp=(((nav.loc[b] - alt) / nav.loc[a]) ** (1 / yrs) - 1 - cg(bench, a, b)) * 100)
for k, v in res.items(): print(k, v)
(HERE / 'r76_item6_variants.json').write_text(json.dumps(res, indent=1))
# ---- 與原輸出 csv 比對
M = pd.read_csv(HERE / 'r76_signal_monthly.csv')
o = pd.read_csv(AG / 's76_early_vs_late_diagnosis' / 'signal_power_by_period.csv')
print(o[o['group'].isin(['raw10', 'raw20', 'raw50', 'ma10', 'ma50'])][['period', 'group', 'vs0050_n', 'vs0050_ann_pp', 'vs0050_t']].to_string())
print(o['group'].unique(), o['period'].unique())
EV = pd.read_csv(HERE / 'r84_new_account.csv'); sm = pd.read_csv(AG / 's84_entry_after_market_crash' / 'signal_months.csv')
assert (EV['sig_date'] == sm['sig_date']).all()
cmp = {}
for mine, theirs in (('r10', 'r10'), ('r20', 'r20'), ('gap60', 'gap60'), ('n_ma', 'n_ma'), ('n_fin', 'n_fin'), ('n_tgt', 'n_tgt'), ('ma_share_fin', 'ma_frac'),
                     ('stockfrac_exec_close', 'sf_exec'), ('acct12_cal', 'acct_12'), ('e50_12_cal', 'e50_12'), ('new12_odd', 'n12_all'), ('new12_lot', 'n12_hist'),
                     ('new_minrel_odd', 'ndeep12_all')):
    d = (EV[mine] - sm[theirs]).abs()
    cmp[mine] = dict(vs=theirs, n=int(d.notna().sum()), max_abs=float(d.max()), n_mine=int(EV[mine].notna().sum()), n_theirs=int(sm[theirs].notna().sum()))
    print(mine, cmp[mine])
print('event flag same:', bool((EV['event'] == sm['MAIN']).all()), 'first same:', bool((EV['first'] == sm['FIRST']).all()))
(HERE / 'r84_compare_signal_months.json').write_text(json.dumps(cmp, indent=1))

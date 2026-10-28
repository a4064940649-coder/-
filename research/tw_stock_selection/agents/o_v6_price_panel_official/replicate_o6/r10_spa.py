#!/usr/bin/env python3
"""複製員 o6：SPA_c 焦點 p（族群 F_long × W_long，V4 格點 245 窗；14 條 o_v4 成員換成 V6 版）。
引擎＝o_main_spa_breaks/spa_lib.py（sb_counts、mcp_tests、eff_trials；只讀 import），B＝10,000，種子 20261028＋L，Politis-Romano 變異。
用法：O6R_PANEL=compat python3 -I -B r10_spa.py"""
import os, sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
import numpy as np, pandas as pd
AG = HERE.parent.parent
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import fam_common as FC, spa_lib as SL
sys.path.pop(0)
PANEL = os.environ.get('O6R_PANEL', 'compat'); OUT = HERE / f'out_{PANEL}'
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
D = pd.read_parquet(FC.DATA / 'family_excess.parquet'); Dew = pd.read_parquet(FC.DATA / 'family_excess_ew.parquet')
MM = pd.read_csv(FC.DATA / 'family_members.csv')
mem = MM[MM['cov_W_long'] & MM['F_long']].member_id.tolist()
assert len(mem) == 2319
m = FC.wmask(D.index, '2006-04-01', '2026-10-12'); n = int(m.sum()); assert n == 245
NV = pd.read_parquet(OUT / 'navs.parquet'); NF = pd.read_parquet(OUT / 'navs_family14.parquet')
V4N = pd.read_parquet(AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
e50 = FC.win_ret(FC.e50_nav())
PFX = 'o_v4_revenue_patch_official/navs_v4.parquet::'
MAP = {'V4': 'V6', 'V4_B05': 'V6_B05', 'V4_B10': 'V6_B10', 'V4_BQ': 'V6_BQ', 'V4_SIG': 'V6_SIG', 'V4_RDEL': 'V6_RDEL', 'V4_SPEC0': 'V6_SPEC0',
       'O3REP': 'O3REP', 'AB_V4': 'AB_V6', 'AB_V4_B05': 'AB_V6_B05', 'AB_V4_B10': 'AB_V6_B10', 'AB_V4_BQ': 'AB_V6_BQ', 'ABW_V4': 'ABW_V6', 'ABW_V4_B10': 'ABW_V6_B10'}
# 核對：r6_family 的 V6/AB_V6/B10 與 r5_main 相同
for k in ('V6|normal', 'AB_V6|normal', 'AB_V6_B10|normal', 'ABW_V6|normal'):
    assert np.nanmax(np.abs(NF[k].to_numpy() - NV[k].reindex(NF.index).to_numpy())) < 1e-12, k
ewb4 = FC.win_ret(V4N['AB_V4|EW']); ewb6 = FC.win_ret(NF['AB_V6|EW'])
X = D.loc[m, mem].copy(); XE4 = Dew.loc[m, mem].copy()
for k4, k6 in MAP.items():
    c = PFX + k4 + '|normal'; assert c in X.columns, c
    wr = FC.win_ret(NF[f'{k6}|normal'])
    X[c] = (wr - e50).loc[X.index].to_numpy(); XE4[c] = (wr - ewb4).loc[X.index].to_numpy()
XE6 = XE4.add((ewb4 - ewb6).loc[X.index], axis=0)        # 共同基準改成 AB_V6|EW
st = FC.win_ret(NV['AB_V6|stress'])
STX = (st - e50).loc[X.index].to_numpy()
fAB, fB10 = mem.index(PFX + 'AB_V4|normal'), mem.index(PFX + 'AB_V4_B10|normal')
res = {'n': n, 'N': len(mem)}
X0 = D.loc[m, mem].fillna(0.0).to_numpy()
Xa = X.fillna(0.0).to_numpy(); X2 = np.hstack([Xa, STX[:, None]])
res['eff_trials_V6family'] = SL.eff_trials(Xa)
res['eff_trials_V6family+stress'] = SL.eff_trials(X2)
keep = ('p_SPA_c_focus', 'focus_rank', 'focus_t', 'p_SPA_c', 'p_SPA_l_focus', 'p_SPA_u_focus', 'p_StepM_focus', 'p_single_focus', 'N')
def run(Xm, Wc, L, f, Mb=None):
    o = SL.mcp_tests(Xm, Wc, L, 'pr', Mb=Mb, focus=f)
    return {k: (round(o[k], 4) if isinstance(o[k], float) else o[k]) for k in keep}
for L in (3, 6, 12):
    Wc = SL.sb_counts(n, L, 10000, 20261028 + L)
    res[f'L{L}|o5check_AB_V4'] = run(X0, Wc, L, fAB)
    Mb = SL.boot_means(Wc, Xa)
    res[f'L{L}|AB_V6|N2319'] = run(Xa, Wc, L, fAB, Mb)
    res[f'L{L}|AB_V6_B10|N2319'] = run(Xa, Wc, L, fB10, Mb)
    Mb2 = SL.boot_means(Wc, X2)
    res[f'L{L}|stress|N2320'] = run(X2, Wc, L, X2.shape[1] - 1, Mb2)
    res[f'L{L}|AB_V6|N2320'] = run(X2, Wc, L, fAB, Mb2)
    res[f'L{L}|AB_V6_B10|N2320'] = run(X2, Wc, L, fB10, Mb2)
    res[f'L{L}|xEW(V4EW)|AB_V6'] = run(XE4.fillna(0.0).to_numpy(), Wc, L, fAB)
    res[f'L{L}|xEW(V6EW)|AB_V6'] = run(XE6.fillna(0.0).to_numpy(), Wc, L, fAB)
    log(L, {k: v['p_SPA_c_focus'] for k, v in res.items() if k.startswith(f'L{L}|')})
json.dump(res, open(OUT / 'spa.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))

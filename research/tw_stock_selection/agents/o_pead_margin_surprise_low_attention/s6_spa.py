#!/usr/bin/env python3
"""G3：本題 11 條（8 變體＋3 鄰域；FULL 窗、100 萬 GROW、一般成本、10 個 j 平均淨值）併入 o_main_spa_breaks 的長歷史族群（F_long，2,319 條），
在共同窗（本題淨值存在的月窗）做 Hansen SPA（L=6、B=10,000、固定種子）、StepM、DSR。只讀 o_main_spa_breaks 的 data 與 spa_lib／fam_common。"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
SPA_DIR = HERE.parent / 'o_main_spa_breaks'
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


S = load('spa_lib_ro', SPA_DIR / 'spa_lib.py')
FC = load('fam_common_ro', SPA_DIR / 'fam_common.py')
R = json.load(open(HERE / 'results.json')); assert 'PREREG' in R
D = pd.read_parquet(SPA_DIR / 'data' / 'family_excess.parquet')
MM = pd.read_csv(SPA_DIR / 'data' / 'family_members.csv')
flong = MM[MM['F_long'] & MM['cov_W_long']]['member_id'].tolist()
nv = pd.read_parquet(HERE / 'data' / 'navs_pead.parquet')
e50 = pd.read_csv(FC.E50F, parse_dates=['date']).set_index('date')['adj'].sort_index()
r50 = pd.Series(FC.win_ret(e50), index=FC.WIN)
mine = {}
for k in ['MAIN', 'R1', 'R2', 'ITER_v2', 'ITER_v4', 'ITER_v5', 'ITER_v6', 'ITER_v7', 'MAIN_S10', 'MAIN_S30', 'MAIN_H40']:
    s = nv[f'{k}|FULL|100w|GROW|normal'].dropna()
    mine['PEAD_' + k] = pd.Series(FC.win_ret(s), index=FC.WIN) - r50
M = pd.DataFrame(mine)
ok = M.notna().all(axis=1)
win = M.index[ok]
X = pd.concat([D.loc[win, flong].fillna(0.0), M.loc[win]], axis=1)
n, N = X.shape
focus = X.columns.get_loc('PEAD_MAIN')
out = {'n_windows': int(n), 'N_members': int(N), 'N_F_long': len(flong), 'window': [str(win[0].date()), str(win[-1].date())]}
for L in [3, 6, 12]:
    Wc = S.sb_counts(n, L, 10000, 20261102 + L)
    o = S.mcp_tests(X.to_numpy(), Wc, L, 'pr', focus=focus)
    out[f'L{L}'] = {k: o[k] for k in ['max_t', 'p_SPA_c', 'p_SPA_c_focus', 'p_SPA_l_focus', 'p_StepM_focus', 'p_single_focus', 'focus_rank', 'focus_t', 'p_RC_focus']}
    out[f'L{L}']['argmax_member'] = X.columns[o['argmax']]
et = S.eff_trials(X.to_numpy())
sr_all = (X.mean() / X.std(ddof=1)).to_numpy()
out['eff_trials'] = et
out['DSR_N_LiJi'] = S.dsr(X['PEAD_MAIN'].to_numpy(), sr_all, et['M_eff_LiJi'])
out['DSR_N_all'] = S.dsr(X['PEAD_MAIN'].to_numpy(), sr_all, N)
out['mine_window_t'] = {c: float(M.loc[win, c].mean() / (M.loc[win, c].std(ddof=1) / np.sqrt(n))) for c in M.columns}
R['G3_SPA'] = out
json.dump(R, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

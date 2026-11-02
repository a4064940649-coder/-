"""Build family matrix (2,334 x 245) and V7X / AB_V4 generator inputs from upstream data."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_g2_feasibility_stoprule/replicate_o11')
from rlib import *

OUT = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/rep_o11'
import os; os.makedirs(OUT, exist_ok=True)

b = bench(); g = grid(); r0050 = b.r0050.values
mem = pd.read_csv(SPA_DIR + '/family_members.csv')
fl = mem[mem.F_long & mem.cov_W_long]
fe = pd.read_parquet(SPA_DIR + '/family_excess.parquet')
X1 = fe[fl.member_id.tolist()].values
print('NaN cells in F_long (kept as NaN; first/last windows):', int(np.isnan(X1).sum()))
names = fl.member_id.tolist(); folders = fl.folder.tolist(); fmain = fl.F_main.tolist()

# o10 tpex 15
tp = pd.read_parquet(AG + '/o_tpex_pool_extension/out/navs_oos.parquet')
TP = ['M', 'R1', 'R2', 'BASE', 'IT_v1_ALL300_n10', 'IT_v2_ALL400_n10', 'IT_v5_ALL400_n10_floor2k', 'IT_v6_ALL400_n15',
      'IT_v7_TP200_n10', 'NB_K300', 'NB_K500', 'NB_floor0', 'NB_floor2k', 'NB_n15', 'NB_n5']
X2 = []
for s in TP:
    cols = [f'{s}|J{j}|100萬|GROW|normal' for j in range(13)]
    nav = javg_nav(tp[cols])
    X2.append(win_ret_from_nav(nav, g) - r0050)
    names.append('o_tpex_pool_extension::' + s); folders.append('o_tpex_pool_extension'); fmain.append(False)
X = np.column_stack([X1] + X2)
print('family', X.shape)

# V7X R4 B10 K5T
v = pd.read_parquet(AG + '/o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
def vcols(cost, tier):
    return [f'B10|J{j}|K5T|{cost}|{tier}|GROW' for j in range(13)]
VJ = {}
for cost in ['normal', 'stress']:
    for tier in ['100萬', '300萬', '1000萬']:
        VJ[(cost, tier)] = per_j_winret(v[vcols(cost, tier)]) - r0050[:, None]
VJAVG = {k: win_ret_from_nav(javg_nav(v[vcols(*k)]), g) - r0050 for k in [('normal', '100萬'), ('normal', '300萬'), ('normal', '1000萬'), ('stress', '100萬')]}
xj = VJ[('normal', '100萬')]
d_j = xj - xj.mean(axis=1, keepdims=True)
c_stress = (VJ[('normal', '100萬')] - VJ[('stress', '100萬')]).mean(axis=1)
c_300 = (VJ[('normal', '100萬')] - VJ[('normal', '300萬')]).mean(axis=1)
c_1000 = (VJ[('normal', '100萬')] - VJ[('normal', '1000萬')]).mean(axis=1)
print('drags mean pp/month: stress %.3f 300 %.3f 1000 %.3f' % (c_stress.mean() * 100, c_300.mean() * 100, c_1000.mean() * 100))
print('V7X javg IR (avg daily) %.3f ; mean-of-j window IR %.3f' % (ir(VJAVG[('normal', '100萬')]), ir(xj.mean(axis=1))))

# AB_V4 + neighbors
pre = 'o_v4_revenue_patch_official/navs_v4.parquet::'
ab = fe[pre + 'AB_V4|normal'].values
nb_ids = [pre + 'AB_V4_B05|normal', pre + 'AB_V4_B10|normal', pre + 'ABW_V4|normal', 'main_v2/navs_gapfix.parquet::EXFIN|n30|normal',
          'concentration_equity/navs.parquet::EW5', 'concentration_equity/navs.parquet::EW15']
NB = fe[nb_ids].values
print('AB_V4 IR %.3f sd %.3fpp' % (ir(ab), ab.std(ddof=1) * 100), 'neighbors IR', np.round(ir(NB), 3))

# ideal V7 B10 j-scan vs V7X
js = pd.read_parquet(AG + '/o_v7_listing_exec_engine/navs_jscan_v7.parquet')
idl = per_j_winret(js[[f'V7_B10|normal|J{j}' for j in range(13)]]) - r0050[:, None]
y19 = np.array([d >= pd.Timestamp('2019-01-01') for d in b.index])
gap = {}
for tier in ['100萬', '300萬', '1000萬']:
    e = VJ[('normal', tier)]
    gap[tier] = float(np.median(ir(idl) - ir(e)))
gap['100萬_Y19'] = float(np.median(ir(idl[y19]) - ir(VJ[('normal', '100萬')][y19])))
print('ideal-exec IR gap (j median):', {k: round(v, 3) for k, v in gap.items()})

np.savez(OUT + '/inputs.npz', X=X, names=np.array(names), folders=np.array(folders), fmain=np.array(fmain),
         r0050=r0050, dates=np.array(b.index.values), ab=ab, NB=NB, d_j=d_j, xj=xj, c_stress=c_stress, c_300=c_300, c_1000=c_1000,
         v7x_javg=VJAVG[('normal', '100萬')], v7x_javg300=VJAVG[('normal', '300萬')], v7x_javg1000=VJAVG[('normal', '1000萬')],
         v7x_javg_stress=VJAVG[('stress', '100萬')])
json.dump({'gap': gap, 'drag_pp': [c_stress.mean() * 100, c_300.mean() * 100, c_1000.mean() * 100],
           'AB_V4_IR': ir(ab), 'AB_V4_sd_pp': ab.std(ddof=1) * 100, 'V7X_IR': ir(VJAVG[('normal', '100萬')])},
          open(OUT + '/inputs_meta.json', 'w'), ensure_ascii=False, indent=1)

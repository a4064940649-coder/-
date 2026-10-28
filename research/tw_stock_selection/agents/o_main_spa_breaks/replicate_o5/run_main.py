import sys, json, re; sys.path.insert(0, '.')
from spa import *
X50 = pd.read_parquet(OUT / 'X50.parquet'); XEW = pd.read_parquet(OUT / 'XEW.parquet')
C = pd.read_csv(OUT / 'cover.csv', index_col=0)
FOCUS = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
MAIN_FOLDERS = {'o_v4_revenue_patch_official', 'main_v2', 'o_infra_v3_rebuild', 'main_refine', 'revenue_longhistory', 'integrate_o1',
    'integrate_o2', 'integrate_o4', 'o_buyback_gate_core_rebal', 'o_largecap_leg', 'o_capacity_tsmc_gap', 'o_disposal_rejudge',
    'o_msci_inclusion_avoid', 'o_issuance_avoid', 'o_hold_buffer_fresh', 'o_stagger_tranche', 'o_rev_survivorship',
    'o_exec_capacity_tiers', 'o_cb_board_runup', 'independent_audit', 'exec_rebuild', 'manual_screen', 'rev_announce',
    'o_sbl_foreign_level', 'o_intraday_momentum'}
CORE_RE = re.compile(r'2330|TSMC|L_E50|E0050|0050_30|MIX_|TOP1VALUE|POOLEW|combo_n10|COMBO')
C['main'] = C.folder.isin(MAIN_FOLDERS)
C['core'] = C.main & ~pd.Series([bool(CORE_RE.search(k)) for k in C.index], index=C.index)
FL = C.index[C.W_long]
fams = {'F_long': lambda w: FL, 'F_main': lambda w: C.index[C[w] & C.main] if w != 'W_long' else C.index[C.W_long & C.main],
        'F_all': lambda w: C.index[C[w]], 'F_main_core': lambda w: C.index[C[w] & C.core]}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
res = {}
for wi, (w, s0) in enumerate(WSTART.items()):
    for L in (3, 6, 12):
        n = NW - s0
        idx = sb_indices(n, L, B, 20261028 + 1000 * wi + L)
        for fn, ff in fams.items():
            if fn == 'F_main_core' and w != 'W_long': continue
            if fn == 'F_all' and w == 'W_long': continue   # F_all = F_long on W_long
            if w != 'W_long' and L != 6 and fn != 'F_long': pass
            cols = list(ff(w)); assert FOCUS in cols
            for lossn, X in (('x0050', X50), ('xEW', XEW)):
                if lossn == 'xEW' and fn not in ('F_long', 'F_main'): continue
                D = X.loc[:, cols].iloc[s0:].fillna(0.0).to_numpy()
                o = spa(D, cols.index(FOCUS), L, idx, stepm=(fn in ('F_long', 'F_main')))
                o['best'] = cols[o.pop('argmax')]
                res[f'{fn}|{w}|L{L}|{lossn}'] = o
                print(f'{fn}|{w}|L{L}|{lossn}', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
json.dump(res, open(OUT / f'spa_main_B{B}.json', 'w'), ensure_ascii=False, indent=1)

X50 = pd.read_parquet(OUT / 'X50.parquet'); XEW = pd.read_parquet(OUT / 'XEW.parquet')
C = pd.read_csv(OUT / 'cover.csv', index_col=0)
FOCUS = 'o_v4_revenue_patch_official/navs_v4.parquet::AB_V4|normal'
MAIN_FOLDERS = {'o_v4_revenue_patch_official', 'main_v2', 'o_infra_v3_rebuild', 'main_refine', 'revenue_longhistory', 'integrate_o1',
    'integrate_o2', 'integrate_o4', 'o_buyback_gate_core_rebal', 'o_largecap_leg', 'o_capacity_tsmc_gap', 'o_disposal_rejudge',
    'o_msci_inclusion_avoid', 'o_issuance_avoid', 'o_hold_buffer_fresh', 'o_stagger_tranche', 'o_rev_survivorship',
    'o_exec_capacity_tiers', 'o_cb_board_runup', 'independent_audit', 'exec_rebuild', 'manual_screen', 'rev_announce',
    'o_sbl_foreign_level', 'o_intraday_momentum'}
C['main'] = C.folder.isin(MAIN_FOLDERS)
FL = list(C.index[C.W_long]); FM = list(C.index[C.W_long & C.main])

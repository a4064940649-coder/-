#!/usr/bin/env python3
"""SPEC v2 小型驗證（只讀；不算任何報酬）：列出規格引用的資料檔是否存在、大小、sha256、欄位。python3 -I -B verify_inputs.py"""
import hashlib, json, csv, gzip, os, sys
from pathlib import Path
AG = Path('/home/user/-/research/tw_stock_selection/agents')
SCR = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
F = {
 'panel_v7_close': AG/'o_v7_listing_exec_engine/data/close_raw.parquet',
 'panel_v7_open': AG/'o_v7_listing_exec_engine/data/open.parquet',
 'panel_v7_value': AG/'o_v7_listing_exec_engine/data/value.parquet',
 'panel_v7_ret': AG/'o_v7_listing_exec_engine/data/ret.parquet',
 'panel_v7_flag': AG/'o_v7_listing_exec_engine/data/flag.parquet',
 'panel_v7_high': AG/'o_v7_listing_exec_engine/data/high.parquet',
 'panel_v7_adj': AG/'o_v7_listing_exec_engine/data/adj.parquet',
 'panel_v7_src': AG/'o_v7_listing_exec_engine/data/src.parquet',
 'panel_paper_close': SCR/'panel/close_raw.parquet',
 'panel_paper_open': SCR/'panel/open.parquet',
 'panel_paper_value': SCR/'panel/value.parquet',
 'panel_paper_ret': SCR/'panel/ret.parquet',
 'panel_paper_flag': SCR/'panel/flag.parquet',
 'panel_paper_high': SCR/'panel/high.parquet',
 'panel_paper_adj': SCR/'panel/adj.parquet',
 'panel_paper_src': SCR/'panel/src.parquet',
 'tw_revenue': Path('/home/user/niklavsd/bottleneck-signals/data/processed/tw_revenue.csv'),
 'rev_5y': Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_5y.csv.gz'),
 'rev_hist': Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/rev_hist.csv.gz'),
 'disposal_5y': Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz'),
 'stock_list': Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv'),
 'etf_0050': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'),
 'taiex_twii': Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'),
 'R_FIX': AG/'o_v4_revenue_patch_official/yoy_long_R_FIX.parquet',
 'Y5_FIX': AG/'o_v4_revenue_patch_official/yoy_rev5y_FIX.parquet',
 'data_freeze_v4': AG/'o_v4_revenue_patch_official/data_freeze_v4.json',
 'rev_backfill_B1': AG/'o_delist_rev_backfill_allpit/data/rev_backfill_B1.csv',
 'delist_official': AG/'o_infra_v3_rebuild/delist_official.csv',
 'merger_consideration': AG/'o_infra_v3_rebuild/merger_consideration.csv',
 'delist_prices': AG/'o_infra_v3_rebuild/delist_prices.parquet',
 'o3_data_freeze': AG/'o_infra_v3_rebuild/data_freeze.json',
 'events_v2': AG/'o_engine_delist_consideration/events_v2.csv',
 'cz_results': AG/'o_engine_delist_consideration/results.json',
 'disposal_hist_rejudge': AG/'o_disposal_rejudge/disposal_hist_rejudge.csv',
 'disposal_official_universe': AG/'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv',
 'universe_announce': AG/'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv',
 'DISP_OFFICIAL_v2': AG/'o_disposal_list_pit_audit_pkl18/data/DISP_OFFICIAL_v2.csv',
 'ts_firstseen': AG/'o_rev_avail_ds_pit_count/raw/ts_tasi_upstream_firstseen.csv.gz',
 'ts_mops_ann': AG/'o_rev_avail_ds_pit_count/raw/ts_mops_ann_date.csv',
 'ts_news': AG/'o_rev_avail_ds_pit_count/raw/ts_news_mentions.csv.gz',
 'p_delay': AG/'rev_announce/p_delay.json',
 'firstseen_2609': AG/'rev_announce/firstseen_2026-09.csv',
 'foreign_codes_src': AG/'o_rev_survivorship/data_freeze.json',
 'snap1': AG/'o_pit_shares_industry_2006_2013/raw/industry_snap/grs_568e6685_tools_twse_list.csv',
 'snap2': AG/'o_pit_shares_industry_2006_2013/raw/industry_snap/grs_893c8772_grs_stock_no.csv',
 'snap3': AG/'o_pit_shares_industry_2006_2013/raw/industry_snap/goristock_c3c718ab_grs_stock_no.csv',
 'listing_master': AG/'o_v7_listing_exec_engine/listing_master.csv',
 'model_spec7': AG/'o_v7_listing_exec_engine/data/model_spec7.json',
 'cpi_tw': AG/'o_v7_listing_exec_engine/raw/exec_inputs/cpi_tw.json',
 'jscan_end_days': AG/'o_v7_listing_exec_engine/data/jscan_end_days.json',
 'plans_j7': AG/'o_v7_listing_exec_engine/data/plans_j7.csv',
 'x_cache': AG/'o_exec_engine_rewrite_v7x/data/x_cache.npz',
 'v7x_results': AG/'o_exec_engine_rewrite_v7x/results.json',
 'fs_e50': AG/'o_forward_sequential_v2/raw/finlab/data/raw/etf_0050.csv',
 'k_paper_lists': AG/'o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz',
 'v7x_plan_diff_M2': AG/'integrate_o20/data/v7x_plan_diff_M2.csv.gz',
 'plans_v7x_o18_scratch': SCR/'o18_revbackfill/plans_v7x.csv',
 'plans_o19_scratch': SCR/'integrate_o19/plans_strict.csv',
 'plans_o20_scratch': SCR/'integrate_o20/plans_base.csv',
}
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def header(p):
    try:
        if str(p).endswith('.csv.gz'):
            with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as f: return f.readline().strip()
        if str(p).endswith('.csv'):
            with open(p, encoding='utf-8', errors='replace') as f: return f.readline().strip()
    except Exception as e: return 'ERR ' + str(e)
    return None
out = {}
for k, p in F.items():
    if p.exists():
        out[k] = dict(path=str(p), exists=True, bytes=p.stat().st_size, sha256=sha(p), header=header(p))
    else:
        out[k] = dict(path=str(p), exists=False)
Path(__file__).with_name('verify_inputs.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
for k, v in out.items(): print(k, v.get('exists'), v.get('bytes'), (v.get('sha256') or '')[:16], (v.get('header') or '')[:160])

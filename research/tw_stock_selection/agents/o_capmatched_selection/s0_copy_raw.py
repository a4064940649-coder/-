#!/usr/bin/env python3
"""步驟 0a：把本題用到的全部輸入複製到 raw/（之後一律只讀 raw/），寫 raw/manifest.json（sha256、來源路徑、複製時間）。
不看任何報酬。下載／外部 clone 的資料視為不可信，只當資料讀。"""
import hashlib, json, shutil, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
AG = HERE.parent
SP = Path('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad')
X = Path('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras')
SRC = {
    # 價格面板（2019～2023 只有存活股）
    **{f'panel/{n}.parquet': SP / 'panel' / f'{n}.parquet' for n in ['adj', 'close_raw', 'open', 'high', 'value', 'ret', 'flag', 'src']},
    'bench/etf_0050.csv': Path('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv'),
    'bench/taiex_twii.csv': Path('/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv'),
    # screener extras
    'screener/stock_list.csv': X.parent / 'stock_list.csv',
    **{f'screener/{n}': X / n for n in ['etf0050_w.csv', 'ex_tsmc.csv', 'shares_hist.csv', 'shares.csv', 'exdiv.csv', 'exdiv_5y.csv.gz',
                                        'corp_actions.csv', 'pe_5y.csv.gz', 'etf_tr_m.csv.gz', 'etf_long.csv.gz', 'inst_5y.csv.gz']},
    # 臺灣50 成分與股數錨
    'agents/tw50_members.json': AG / 'o_largecap_leg' / 'tw50_members.json',
    'agents/etf_rebalance_events.csv': AG / 'etf_rebalance' / 'events.csv',
    'agents/vf_twse_foreign.parquet': AG / 'o_sbl_foreign_level' / 'raw' / 'vf_twse_foreign.parquet',
    'agents/shares_anchor.csv': AG / 'o_sbl_foreign_level' / 'shares_anchor.csv',
    # 營收（V4 只取已公告值的輸入）、處置、外國公司、終止上市
    'agents/yoy_rev5y_FIX.parquet': AG / 'o_v4_revenue_patch_official' / 'yoy_rev5y_FIX.parquet',
    'agents/yoy_long_R_FIX.parquet': AG / 'o_v4_revenue_patch_official' / 'yoy_long_R_FIX.parquet',
    'agents/disposal_hist_rejudge.csv': AG / 'o_disposal_rejudge' / 'disposal_hist_rejudge.csv',
    'agents/official_trades_twse.csv': AG / 'disposal_reconcile' / 'official_trades_twse.csv',
    'agents/rev_survivorship_data_freeze.json': AG / 'o_rev_survivorship' / 'data_freeze.json',
    'agents/delist_official.csv': AG / 'o_infra_v3_rebuild' / 'delist_official.csv',
    'agents/p_delay.json': AG / 'rev_announce' / 'p_delay.json',
    'agents/firstseen_2026-09.csv': AG / 'rev_announce' / 'firstseen_2026-09.csv',
    # 對照淨值
    'navs/navs_v4.parquet': AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet',
    'navs/navs_v6.parquet': AG / 'o_v6_price_panel_official' / 'navs_v6.parquet',
    'navs/navs_jscan_v6.parquet': AG / 'o_v6_price_panel_official' / 'navs_jscan_v6.parquet',
    'navs/integrate_o6_navs.parquet': AG / 'integrate_o6' / 'navs.parquet',
    'navs/main_v2_navs_gapfix.parquet': AG / 'main_v2' / 'navs_gapfix.parquet',
    # 可執行成本模型（o5）
    'exec/model_spec.json': AG / 'o_exec_capacity_tiers' / 'data' / 'model_spec.json',
    'exec/cpi_tw.json': AG / 'o_exec_capacity_tiers' / 'raw' / 'cpi_tw.json',
    # 誰在贏（描述用）
    'winners/allianz_monthly.csv': SP / 'winners_attr' / 'allianz_monthly.csv',
    'winners/nav_00981A.csv': SP / 'winners_attr' / 'nav_00981A.csv',
}
def sha(f): return hashlib.sha256(Path(f).read_bytes()).hexdigest()
man = {'copied_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'files': {},
       'github_search_log': [
           'search_code "NumberOfSharesIssued 2330 2019 extension:csv"：只有 wayne70211/FinLookup Data/2330/2330_Shareholding.csv（單一股票），blobless clone 確認全 repo 只有 1 檔 Shareholding，已刪除不用',
           'search_code "TaiwanStockMarketValue market_value extension:csv"：0 筆',
           'search_code "ForeignInvestmentSharesRatio NumberOfSharesIssued extension:csv"：只有同一個 2330 檔',
           'search_code "發行股數 外資持股比率 2019 extension:csv"：2 筆無關（PTT、新聞事件）',
           'search_code "已發行普通股數或TDR原股發行股數 2019"：5 筆皆為抓取程式，無封存資料',
           'search_code "t187ap03_L 實收資本額 extension:csv"：0 筆；"MI_QFIIS 2020 extension:json"：只有 ETF 籌碼頁面',
           '結論：找不到 2019～2021 全市場發行股數封存 → 依計畫用錨點回推（s2_capgate.py）。o_sbl_foreign_level 的 shares_anchor／fo_panel 實為 2025-10-07 單一快照與比率回推，不是 2021-10 起的股數。'],
       }
for dst, src in SRC.items():
    d = RAW / dst; d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d)
    man['files'][dst] = {'src': str(src), 'sha256': sha(d), 'bytes': d.stat().st_size}
(RAW / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
print(len(man['files']), 'files copied')

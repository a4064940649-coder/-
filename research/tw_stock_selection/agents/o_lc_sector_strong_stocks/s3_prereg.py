#!/usr/bin/env python3
"""凍結：寫 data/PREREG.json 與 results.json['PREREG']（規格、選定格點、題 1 狀態、程式與資料 sha256、時戳、研究員點預測）。
在此之前沒有讀過 2017-01 以後的任何報酬。python3 -I -B s3_prereg.py"""
import sys, json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lc_lib as X

MODE = sys.argv[1]; X.set_mode(MODE)
it = json.loads((HERE / f'iterations_{MODE}.json').read_text(encoding='utf-8'))
it_pre = json.loads((HERE / 'iterations_COMP.json').read_text(encoding='utf-8'))
cal = it['Q7b_SEL']['v1:IND_n2_cap_c15_SAT']['calib'] if 'v1:IND_n2_cap_c15_SAT' in it['Q7b_SEL'] else None
sel_key = it['selection']['selected']
cal_sel = it['Q7b_SEL'].get(sel_key, {}).get('calib', cal)
T1 = X.AG / 'o_lc_sector_rotation_pit'
t1_pre = None
for f in (T1 / 'data' / 'PREREG.json', T1 / 'PREREG.json'):
    if f.exists(): t1_pre = dict(path=str(f), sha256=X.sha(f))
body = dict(
    mode=MODE, mode_params=X.mode_params(),
    topic='o_lc_sector_strong_stocks：產業輪動＋強勢大型股（題 2）',
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    topic1_status=('題 1（o_lc_sector_rotation_pit）凍結檔：' + json.dumps(t1_pre, ensure_ascii=False)) if t1_pre else
        '題 1（o_lc_sector_rotation_pit）在本題凍結時尚未凍結（資料夾只有空的 raw/）。依研究計畫〇，本題探索與凍結一律用「題 1 事前預設版」：COMP、k=5，'
        '由本題自行實作（定義見 sector_rule）。題 1 之後若凍結出不同選定版，需要另跑「切換後」版本；本題沒有回頭改。',
    topic1_prereg_sha256=t1_pre['sha256'] if t1_pre else None,
    universe='訊號日 d 收盤 PIT 市值前 100（股數：2013-01-02 起 o8 shares_pit＋4763／2317 修正；2014-01 起過 o8 閘門。2006-04～2012-12 用 o7 shares_est（由 2025 年往回推，非 PIT、缺部分下市股），'
             '只作描述）。排除 91 開頭 DR；近 6 日有成交；V7 上市身分遮罩。2330 一樣參與。',
    industry='PIT 證交所產業別：2023-07-03 起用現行 stock_list；2007-07-02～2023-07-02 用 ISIN 2023-01-16 快照（觀光事業→觀光餐旅；快照外而現行屬新 4 類者→其他業）；'
             '2007-07-02 前電子 8 類→電子工業、化學／生技醫療→化學生技醫療業、油電燃氣→其他業。下市股：11 檔大型股手動、傳產依代碼前兩碼。個別公司在 2007～2023 之間的改類無法 PIT。',
    sector_rule=dict(name='題 1 事前預設版 COMP、k=5（本題實作）', min_members=X.SECT_MIN,
                     signals='REV＝成員近 3 月營收年增率（V4 已公告，縮尾 [-80%,300%]）市值加權；RS＝(63、126、252 日扣最近 21 日) 三個市值加權報酬的跨產業百分位平均；'
                             'H52＝成員 52 週比（gapfix）市值加權；NH＝近 21 個交易日創 52 週新高的成員比例；金融產業無 REV。',
                     comp='四者跨產業百分位的平均（缺值略過）；排序同分比產業市值', k=5, buffer='無'),
    stock_score='(1/3)·宇宙內 52 週比百分位＋(1/3)·6-1 月相對強度（126 日到 21 日）百分位＋(1/3)·近 3 月營收年增率百分位（V4 已公告）；金融保險股＝前兩者平均；'
                '至少 2 項有值才算分；同分先比營收、再比代碼',
    grid='IND：n∈{1,2,3} × 權重{cap,eq} × 上限{15%,25%} × 載體{PURE,SAT(0.7B+0.3PURE)} 共 24；CTRL 同格點 24；IND_ONLY（PURE、SAT）；B＝宇宙 PIT 市值加權',
    buffer='IND：已持有且產業內名次 ≤ n+buf、產業仍在前 k → 續抱；CTRL：已持有且宇宙名次 ≤ k·(n+buf) → 續抱',
    cap_rule='prop：超過上限依比例分給其他未觸頂選中股；K·cap<1 時有效上限＝1/K（等權）',
    timing='訊號日 J0～J12（營收期限日後第 j 個面板交易日），d 收盤算、d+1 開盤成交；月換股；主欄＝13 個 j 的中位數',
    periods=dict(SEL='2014-01～2016-12（D1：2006～2013 沒有可核對 PIT 股數）', VAL='2017-01～2026-10', DESC='2006-04～2013-12（非 PIT 股數，只作描述）'),
    costs='紙上：一般（買 0.1855%、賣 0.4855%）、壓力（每邊再加 0.5%）、零成本；可執行：K5T（lc_exec.py，移植 o8 s7_exec＋處置期收盤減一跳賣出），30 萬／100 萬／300 萬／1,000 萬，FIX／GROW',
    iterations=dict(n_versions_per_mode=4, modes=['COMP（切換前）', MODE], versions=list(it['versions'].keys()), n_grid_per_version=48, total_configs_evaluated=2 * 4 * 48 + 6),
    selection=it['selection'],
    preswitch=dict(mode='COMP', selection=it_pre['selection'], S2=it_pre['S2'], note='切換前（題 1 事前預設 COMP、k=5）的挑選期結果與選定，照報'),
    S2=it['S2'],
    selected_spec=it['selection']['selected_spec'],
    default_spec=dict(kind='IND', n=2, wt='cap', cap=0.15, carrier='SAT'),
    placebo_rho=dict(rho_sec=cal_sel['rho_sec'], rho_stk=cal_sel['rho_stk']) if cal_sel else dict(rho_sec=0.8, rho_stk=0.8),
    placebo='Q7 (a)(b)(c) 各 1,000 次（4×250，seed 20261103+1000c+j），PIT 段 2014 起連續模擬，驗證期取 2017 起；扣成本前與後都要 > p95',
    right_tail='拿掉 2021 年航運業持股、2025 年電子零組件業＋電腦及週邊設備業持股（以 0050 報酬替代其貢獻），以及拿掉累積貢獻最大的 10 檔',
    q4_subsamples='產業 REV 跨產業標準差高於中位數的月份；j0～j4 vs j8～j12；主題擴散期 2020～2021、2023～2025（2009～2010 非 PIT 描述）；'
                  '空頭後反彈：0050 自高點回落 ≥20% 的谷底後 3 窗；主題結束：2022-01～2022-12（航運）、2024-07～2024-08（AI）',
    g3='族群 (i) 2,334 條（o_g2_feasibility_stoprule.g2lib.family）＋本題 48 格＋IND_ONLY 2 條；(ii) F_main＋同樣格點。官方日窗格；SPA_c（平穩 bootstrap L=6、B=10,000、seed 20261103）；DSR（全族群 SR 變異數、試驗數＝族群數）',
    predictions=dict(
        plan_preregistered='Q1 差中差 >0（挑選期 t≥2；t<1 機制不成立）；Q2 前 k 產業內強勢股對產業整體超額 >0（t≥1.5）；Q3 本題對 0050 超額 > 不分產業對照，差額 t 約 1.0～1.5；'
                           'Q5 主策略前 20 個股票-年中屬前 k 產業 ≥60%；Q6 SAT 超額與主策略相關 <0.6、對 (2330−0050) β≥0；Q7 扣成本前後都贏 p95',
        researcher_point_after_SEL='挑選期（2014～2016）：切換前（COMP k5）全部格點對 0050 為負、Q1 差中差 t 0.09；切換後（題 1 H52 k3）選定版 SAT n2 對 0050 IR −0.44（t −0.77）、對 B t +0.20、Q1 差中差 t −0.13，只有 n=1 PURE 為正（IR 0.35）。研究員點預測驗證期：選定版（SAT n2）對 0050 j 中位數 t 介於 −0.5～+1.0（不顯著），'
                                   '主要正貢獻集中在 2021 航運與 2025 AI；IND−CTRL 差額 t 約 0～1.0；G1～G7 不會全過；判定「不可用」的機率 > 80%。'),
    contamination='2017 年以後大型強勢股的報酬在 o3（大型股腿）、o7 與 o9（C 候選）已部分看過；本題驗證期不是乾淨樣本外。',
    code_sha256={f: X.sha(HERE / f) for f in ('lc_lib.py', 'lc_run.py', 'lc_exec.py', 'lc_placebo.py', 's4_oos.py')},
    data_sha256={str(p): X.sha(p) for p in (X.O8 / 'data' / 'shares_pit.parquet', X.O8 / 'raw' / 'agents' / 'capmatched_shares_est.parquet',
                                             X.AG / 'o_v7_listing_exec_engine' / 'data' / 'listed_twse.parquet', X.SNAP_FILE, X.SL_FILE,
                                             X.O8 / 'raw' / 'agents' / 'yoy_long_R_FIX.parquet', X.O8 / 'raw' / 'agents' / 'yoy_rev5y_FIX.parquet',
                                             X.O8 / 'raw' / 'bench' / 'etf_0050.csv', X.AG / 'o_v7_listing_exec_engine' / 'data' / 'close_raw.parquet')
                                             + ((HERE / 'data' / 't1_sector_order.json', X.T1DIR / 'lclib.py', X.T1DIR / 'raw' / 'industry_pit.csv', X.T1DIR / 'data' / 'shares_full.parquet') if MODE == 'T1' else ())},
)
sha_body = X.canon(body)
PRE = dict(body=body, _sha256_body=sha_body)
(HERE / 'data' / 'PREREG.json').write_text(X.dumps(PRE), encoding='utf-8')
RES = HERE / 'results.json'
r = json.loads(RES.read_text(encoding='utf-8')) if RES.exists() else {}
r['PREREG'] = body; r['PREREG_sha256'] = sha_body
(RES).write_text(X.dumps(r), encoding='utf-8')
print('PREREG', sha_body, body['frozen_utc'], body['selection']['selected'])

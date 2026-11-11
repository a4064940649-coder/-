#!/usr/bin/env python3
"""integrate_o20 事前登記（寫入 results.json 的 PREREG＋sha256；之後所有 run_* 斷言雜湊）。只跑一次。
python3 -I -B prereg.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; AG = HERE.parent
RJ = HERE / 'results.json'
assert not RJ.exists(), '只跑一次'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
S2 = 'o_g6_allpit_list_trace_stage2'
FILES = [f'{S2}/results.json', f'{S2}/k_target/k_paper.py', f'{S2}/out/k_paper_stats.json', f'{S2}/out/k_paper_windows.npz', f'{S2}/out/k_paper_lists.csv.gz',
         'o_disposal_list_pit_audit_pkl18/data/universe_announce.csv', 'o_rev_avail_ds_pit_count/raw/disposal_official_universe.csv',
         'o_delist_rev_backfill_allpit/o18lib.py', 'o_delist_rev_backfill_allpit/v7x_lib.py', 'o_delist_rev_backfill_allpit/results.json',
         'o_engine_delist_consideration/cz_v7x.py', 'o_engine_delist_consideration/cz_engine.py', 'o_engine_delist_consideration/events_v2.csv',
         'o_exec_engine_rewrite_v7x/x_stats.py', 'o_exec_engine_rewrite_v7x/x_exec.py', 'o_exec_engine_rewrite_v7x/results.json',
         'o_v4_revenue_patch_official/v4lib.py', 'integrate_o19/results.json']
PREREG = dict(
    written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    topic=f'{S2}（G6 第二階段：ALLPIT 名單層獨立重建＋1,000 萬／壓力兩格逐筆追蹤＋紙上處置遮罩全期統一；待觀察）',
    nature=('題目是複製／稽核：不產生選股訊號、沒有分數、沒有獨立報酬序列（S1）。唯一會改變主策略持股的產出是「處置遮罩全期統一 M2」'
            '（只用官方表 universe_announce：公告日 ≤ d 且處置期涵蓋成交日 e，d＝訊號日、e＝下一交易日；不與舊遮罩聯集）。'
            'B 角色的引擎讀法對齊（V0～V3）是複製員引擎對原引擎的讀法差，裁定原引擎正式數字不動，不構成可整合的規則變體。G3 族群 N 維持 2,573（本整合不新增候選策略）。'),
    main_reference=dict(
        paper=('紙上主策略（回測上緣）＝ALLPIT_REAL_{opt,p90,carry} × cons_main（o18lib.build／sim，只讀 import），一般與壓力成本。'
               '基準遮罩＝M1（審查員修正：o18 原遮罩 ∪ 2021-09-23 後「公告 ≤ d 且涵蓋 e」；現行正式紙上上緣 22.06%／t 1.59 的來源）。'
               '另列 M0（o18 原管線，22.22%）與 C0（舊主策略，freeze 引擎）為對照。'),
        v7x=('V7X 主策略（正式）＝v7x_lib.make REAL_{opt,p90,carry} × 13 j × 20 seeds（B1）× cz_v7x.CSim × K5T × GROW × cons_main，R4＝orig_cfg＋f1,f2,f3=0.5,f4；'
             '基準遮罩＝v7x_lib.dm_all（disp_mask_fn(trade)：公告 ≤ d 且涵蓋 e，表＝disposal_hist_rejudge ∪ disposal_5y 獨有列；另 ≤ 2021-09-23 聯集 disposal_official_universe 涵蓋 d 或 e）。'
             '逐格取 opt／p90／carry 最差。'),
        identity_check='紙上 M0／M1／M2 名單 sha 須等於題目 k_paper_stats.json 的 lists_sha；窗報酬須等於 k_paper_windows.npz（最大絕對差 < 1e-10）。V7X 基準名單計數與 ends 須等於題 1 RESULTS_V7X_PLANS；V7X 基準 100 萬一般須重現 integrate_o19 正式表（差 < 1e-9）。'),
    combos={
        'J1_filter_unified_mask': ('濾網（主欄）：處置遮罩換成 M2 全期統一（公告 ≤ d 且涵蓋 e，只用官方 universe_announce，程式內逐列斷言）。'
                                   '紙上：ALLPIT_REAL_{opt,p90,carry} × 一般／壓力，對基準 M1；V7X：dm_all 換成 M2 重建 13 j × 20 seeds 名單，名單有變動的組重跑 7 格、其餘沿用基準淨值（決定性引擎）。'
                                   '這是「紙上與可執行遮罩口徑統一」的嚴格版，紙上與 V7X 同一函式定義。'),
        'J2_filter_reviewer_patch': '濾網並列（紙上）：M1 對 M0 的差（即 22.22 → 22.06 的歸因），只報差，不另立變體；V7X 的基準遮罩本來就是「公告 ≤ d 且涵蓋 e」為主，不另跑。',
        'J3_score_bonus': '不適用：題目沒有分數或排序變數。',
        'J4_second_leg_blend': ('固定比例 80/20、50/50（主策略／第二條腿），紙上每個訊號日窗再平衡、V7X 每個 j 的訊號日再平衡；比例事前固定，不用任何期間（含保留期）挑選。'
                                '第二條腿＝J1 串流（題目唯一可交易產出）；主策略＝基準（紙上 M1、V7X dm_all）。預期相關 ≥ 0.99（自己跟自己混）。'
                                '和 0050 固定比例混合在「不再開」清單，不做。'),
        'J5_engine_reading': '不適用：B 的 V0～V3 是複製員引擎對齊原引擎的讀法敏感度（三格分歧對齊後 2019 起 0.000pp），不是策略；只在 FINDINGS 引用，不進表。'},
    cells=dict(
        paper=['一般／壓力 × 全期、2006～2012、2013～2018、2019 起、2024 起、挑選期 ～2019-04、驗證期 2019-05 起：年化（主欄 o18 ia 口徑＝窗數/12，與正式 22.06% 同口徑；日曆年化並列）、'
               '對 0050 超額、對 0050 t、最大回撤（窗頻；全期另附日頻）、和主策略（同放回版 M1）月差 pp 與 t、年化差、相關、拿掉最好 3 年 t、逐年差、成本兩平點（單邊額外成本）'],
        v7x=['100 萬一般（全期超額 j 中位數、j p10、t、2019 起年化／超額、三段超額、2024 起、最大回撤）', '100 萬壓力', '300 萬一般', '1,000 萬一般',
             '30 萬一般（V7X 標準）', '30 萬一般整股限制（2020-10-26 前不允許零股）', '30 萬壓力',
             '每格另報：和主策略（同放回版、同 j）訊號日窗月差 pp 與 t、窗報酬相關（13 j 中位數）、日報酬相關；逐 j 差分布（中位、最大絕對值）與最差變體是否換人']),
    thresholds=dict(divergence='關鍵數字差 > 0.1pp 或 t 差 > 0.05（逐格取 opt／p90／carry 最差後比）', note='與本輪題目同門檻'),
    decision={
        'D1': ('J1：逐格最差 |Δ| ≤ 0.1pp 且 |Δt| ≤ 0.05 → 「統一遮罩不改變該格」。超過 → 該格並列「M2 統一口徑版」並附歸因（換了哪些訊號日×股票），'
               '但本整合不取代正式數字：理由 (i) 現行遮罩沒有前視（只多排除處置期在 d 結束者、或多用 disposal_5y 列；兩者都只用 d 當時可得資訊），不屬「原引擎錯」；'
               '(ii) 題目 G6 角色分離不成立，PREREG (b) 禁止以本輪結果取代正式數字。是否改採 M2 為正式口徑交研究中心核定，若改採則 M2 欄即新正式數字（附本表歸因）。'),
        'D1_lookahead_check': '另計數：基準遮罩在各決策日排除的股票中，有多少列的官方公告日 > d（前視）。> 0 且影響名單 → 改判為「原引擎錯」，依使用者規則 Δ > 0.1pp 時以 M2 取代並附歸因（仍標待不同子代理覆核）。',
        'D3': 'J4：混合對主策略的窗報酬相關 ≥ 0.99 或混合超額落在兩腿之間 → 判「不構成第二條腿（自混）」，不推薦任何比例。',
        'D4': '本整合不改變題目判定（待觀察）與主策略 G1～G8 判定；G6 於本整合結束時重新核對（題目已判 ✘，維持）。'},
    predictions={
        'P1': '紙上 M0／M1／M2 名單 sha 與窗報酬完全重現題目（確定）；M2 opt 一般 22.21%／t 1.60，對 M1 +0.15pp（已知，D1 紙上全期年化格觸發並列）',
        'P2': '紙上最差變體仍為 opt（約 95%）',
        'P3': 'V7X J1 有名單變動的組 > 0（約 85%）；100 萬一般全期超額最差格 |Δ| ≤ 0.3pp（約 70%），方向不確定',
        'P4': 'V7X J1 2019 起超額最差格 |Δ| ≤ 0.3pp（約 65%）',
        'P5': 'J4 全部混合對主策略相關 ≥ 0.99（約 95%）→ D3 自混',
        'P6': '前視計數＝0（約 90%）',
        'P7': 'G1～G8 全部不過（主策略不可投資）'},
    o18_o19_lessons=['不利端逐格取最差（opt／p90／carry）', '凍結前不算正式統計量（本 PREREG 前沒有跑任何報酬；只讀題目已公布數字）',
                     '事後新增狀態與嚴格版並列且嚴格版為主欄（本整合沒有事後新增狀態；若揭盲後新增，只並列）', 'raw/ 只放未修改原始檔（本整合不新增原始資料，故無 raw/）',
                     '假設區間（3 種放回）與抽樣區間（±1.96 SE）並報', '取代正式數字附歸因（D1 若觸發取代）', '紙上附成本兩平點', '可執行測到 30 萬（含整股限制）',
                     'G6 角色必須不同子代理（本整合不做 G6 判定，只重新核對）', '關鍵中間檔不得只放 scratchpad（V7X 名單差異與 13 j 均值淨值壓縮後存 data/）',
                     '比對報逐 j 差分布', '文字證據核對主詞', '紙上與可執行遮罩口徑統一（J1 即此）', '口徑凍結後不得換'],
    o1_lessons=['事件 t 按事件日分群', '驗證期污染', '格點邊緣贏家詛咒', '成交可行性寫進事前登記', 't 提升要拆均值與波動', '存活者偏誤'],
    readonly=dict(dirs=['live_tracking/main_forward', 'o_forward_infra_live', 'o_g8p2a_protocol_addendum'],
                  rule='前後 sha256 快照比對（start 於 22:14Z，台北 06:14）；每日 06:49 台北排程寫入以 mtime 時段區分；不呼叫 G8P-2a 判定器'),
    disk='大型中間檔放 scratchpad/integrate_o20，題目結束時刪除；scratchpad/rep_o19 不碰。',
    file_sha256={f: sha(AG / f) for f in FILES})
s = hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(dict(PREREG=PREREG, PREREG_sha256=s), ensure_ascii=False, indent=1), encoding='utf-8')
print(PREREG['written_utc'], s)

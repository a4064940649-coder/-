#!/usr/bin/env python3
"""第 0 步 PREREG（看任何全樣本數字之前寫死，記 sha256 與時間戳）→ results.json['PREREG']。
python3 -I -B prereg.py   （只能執行一次；已存在即中止）"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; RJ = HERE / 'results.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = json.loads(RJ.read_text(encoding='utf-8')) if RJ.exists() else {}
assert 'PREREG' not in R, 'PREREG 已存在'
P = dict(
 written_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 title='主策略周轉 × 每筆固定成本的結構診斷（只診斷、不改規則）',
 scope=['只描述與量測；不得提出、模擬或回測任何新規則（持有緩衝、降頻、低價股排除、「跳過這些換股會怎樣」的反事實淨值都不做）；這類工作需使用者解除 S1 另開題並計入 G3。G3 族群 N 不變（本題 0 條新候選）。',
        '正式數字不變；G1～G8 表不變；仍不可投資。本題結論只能寫成「建議使用者考慮解除 S1，另開持有緩衝／價格級距題並計入 G3」，且必須並列 turnover_buffer 反證（緩衝區連扣成本前報酬一起流失）。',
        '對手（收成本的一方）：國庫（證交稅 0.3%）、造市與當沖流動性提供者（跳動點價差、衝擊）、券商（手續費）。'],
 engine=dict(name='正式 V7X：o_engine_delist_consideration/cz_v7x.CSim（consideration × cons_main 結算）× K5T × GROW；R4＝orig_cfg＋f1,f2,f3=0.5,f4（同 integrate_o19/run_v7x.py）',
             lists='ALLPIT_REAL_{opt,p90,carry}（o_delist_rev_backfill_allpit/v7x_lib.make，B1，13 j × 20 seeds；處置＝DMT ∪ 官方全歷史清單，口徑公告日 ≤ d 且涵蓋成交日 e）',
             wrapper='tfc_lib.LogSim（全開時淨值與 CSim 逐位相同，run_logs.py 斷言）；Shapley 開關 tax/fee/tick/imp/caps/timing 定義見 tfc_lib 文件字串',
             files_sha256={f: sha(HERE / f) for f in ('tfc_lib.py', 'gen_plans.py', 'run_logs.py')}),
 main_column=dict(engine='V7X', tier='100萬', cost='一般', j='13 j 中位數', variants='opt／p90／carry 逐格取最差', period='2019 起',
                  parallel=['30 萬（2020-10-26 後盤中零股為主欄；2020-10 前整股限制另列，引用 integrate_o19 官方 wholelot 格）', '300 萬', '1,000 萬', '壓力成本', '全期 2006～2026', '2006～2012', '2013～2018', '2019 起', '排除 src 3 近似收盤區段 2024-02-15～2025-09-23 的版本並列']),
 seeds='淨值層（重現正式數字）用 20 seeds 平均；逐筆／逐槽診斷與 Shapley 用 seed 0（39 個模擬＝3 變體 × 13 j）。並報 seed 0 與 20 seeds 平均的差。',
 definitions=dict(
   fill='引擎實際成交一筆（整股 W、零股 O、處置賣出 D）；ctx R＝換股日、P＝順延日。',
   fill_class='以最近一次換股 k 的集合判定：賣出且不在本期名單＝exit；賣出且在本期名單＝rw_sell（續抱調權）；買進且換股前已持有＝rw_buy；其他買進＝entry。順延成交沿用原類別並加註。',
   swap='一筆換股＝同一換股日一賣（exit）一買（entry）的配對。配對方式在挑選期迭代後凍結（候選：B 平均配對＝每檔 exit 對同日 entry 的等權平均；A2 邊際配對＝exit 依排名由好到壞、entry 依排名由差到好逐一配對；A1 反向）。同日無 entry 的 exit、無 exit 的 entry 另列為單邊。',
   rt_cost='來回成本（占成交名目 %）＝賣方（稅＋手續費＋價差＋衝擊）÷賣出名目＋買方（手續費＋價差＋衝擊）÷買進名目。ENG＝引擎實際收取（主欄，K5T 每邊＝max(平方根衝擊, 1 跳動點)，零股用零股價差表，處置賣出＝收盤減一跳）；假設區間 H05／H10＝稅＋手續費＋h×跳動點/價格＋平方根衝擊（h＝0.5／1）。與抽樣區間（月份區塊 bootstrap）並報。',
   edge='事後紙上 edge＝新進股下一持有期（成交日 e 開盤 → 下一成交日 e′ 開盤，含息：O[e′]·Π ADJ / O[e]）報酬 − 被換出股同期報酬，扣成本前。用到事後資訊，只能當診斷，不是可交易訊號。',
   slot='持股槽＝一檔股票 × 一個持有窗（換股 k 後持有至 k+1）；紙上貢獻＝(槽開盤到開盤含息報酬 − 0050 同窗報酬)/名單檔數；該槽所付成本＝k 日買進或調權成本＋k+1 日賣出成本，占 k 日帳戶淨值。',
   cost_drag='會計成本拖累＝Σ 每筆成本（元）÷ 該換股日帳戶可用淨值，逐年加總（pp／年）。與 Shapley（淨值層）並報。',
   tick_layer='整股每筆的跳動點層＝名目 × min(im, tk)，超過跳動點的衝擊層＝名目 × max(0, im − tk)；零股價差與處置減一跳歸跳動點／價差類；手續費超過費率的部分（最低 20 元）另列。',
   turnover='年周轉（單邊）＝Σ 賣出名目 ÷ 換股日可用淨值，逐年加總；拆成 exit（強制／排名掉出）與 rw_sell，買方同法拆 entry（新進來源）與 rw_buy。'),
 dimensions=dict(price=['<20', '20~50', '50~100', '100~500', '>=500'], exit_rank=['9~10(含≤10)', '11~15', '16~30', '31~50', '>50', '強制：處置 disp／流動性池移出 pool／隔日不可交易 notrade／缺營收 rev／其他 other（金融與 KY 恆排除）'],
                 entry_source=['前一訊號日排名 > 50 或不在前 60（新進）', '11~15（升入）', '其他'], n_swaps_month=['<=2', '3~4', '>=5'], slot_contrib='十分位', lot=['整股', '零股']),
 explore_rule='探索與迭代只用挑選期（換股訊號日 2006-01～2019-04-30），只能決定分類邊界、成本歸屬、配對方式與圖表；邊界凍結（PREREG_FREEZE）後一次跑全樣本與 2019 起。事後新增分類一律與凍結版並列，凍結版為主欄。程式斷言 PREREG_FREEZE 存在才計算挑選期以外的任何統計量。',
 predictions=dict(
   P1='（複製）V7X 100 萬一般 2019 起紙上→可執行落差拆解：成本類 14.4pp（衝擊與參與上限 10.1、稅 3.4、手續費 0.9）、時點類 2.4pp（審查員 2.9～3.4）；本題用正式 V7X 引擎做 6 因子（時點、稅、手續費、跳動點、衝擊、參與上限）全部 720 個順序的 Shapley 平均與範圍；o19 順序（時點→稅→手續費→跳動點＋衝擊＋上限）另列。預期成本類在所有順序都 ≥ 12pp；任一順序 < 12pp → o19「P4 否定」改判「順序相依、未定」。與 o19 差 > 0.1pp 時拆成引擎修正（F1/F3/F4）與順序兩部分。由不同子代理獨立重算，差 ≤ 0.1pp 才算一致。',
   P2='（固定成本主導）30 萬與 100 萬帳戶每筆衝擊＋價差（占成交名目 %，2020-10-26 後）比值 ≥ 0.7；< 0.5 → 否定，改判規模型衝擊；0.5～0.7 → 混合。另報衝擊與參與上限中跳動點層占比（> 50% 才算固定成分過半）。',
   P3='（低價股）股價 < 50 元的成交筆數占比 x，在成本總額中占比 ≥ 1.5x；且 < 50 元換股的 edge < 來回成本比率比 ≥ 100 元高至少 10pp。2019 起與 30 萬盤中零股應更強，2006～2012 較弱。',
   P4='（續抱邊緣）被換出股排名 11～15 的換股，事後 edge 中位數 < 該筆來回成本，且占全部 edge < 成本換股 ≥ 40%；被換出股排名 > 50 的換股 edge 中位數 > 來回成本。月份區塊 bootstrap（2,000 次，單尾）檢定兩類 (edge − 成本) 中位數差；p ≥ 0.05 即否定 P4。',
   P5='（高周轉月份）單月換股 ≥ 5 檔的換股日，每筆事後 edge 中位數低於 ≤ 2 檔的換股日，且集中在大盤反轉月份（定義凍結於 FREEZE）。',
   P6='（小貢獻槽）紙上貢獻落在 [−2%, +2%] 的持股槽，所付成本總額 > 其紙上貢獻總和。',
   P7='（同曝險對照，描述性）主策略全期 beta ≥ 1.1；2019 起扣成本後對 beta 調整 0050（挑選期估的 β，β×0050＋(1−β)×現金 0%，不構成可交易配置）的超額比對 0050 的 −6.22pp 更負。否定：beta < 1 或調整後超額變好。'),
 verdict_rule=dict(levels=['結構成立', '結構不成立（成本均勻分布、屬訊號內生代價）', '無法判定'],
   pass_all=['(1) 關鍵成本數字由不同子代理重現，或分歧已歸因（本題交付時複製待做 → 最高只能「無法判定」以外的暫定，FINDINGS 註明待複製）',
             '(2) P3 或 P4 至少一條在 2019 起通過預登記區塊 bootstrap（p < 0.05），且 2006～2018 方向一致',
             '(3) 結論在一般與壓力成本、opt／p90／carry 下都不翻號（逐格取最差 p）'],
   fail='P3 與 P4 都在 2019 起不顯著且 edge<成本 比率在各分類間差距 < 10pp（近乎均勻）→ 結構不成立',
   else_='其餘 → 無法判定'),
 compare_main=['2019 起輸 0050 的 6.22pp 中，有多少等於付在「事後 edge < 成本」換股上的成本（純會計量，不是可實現改善）',
               '這些換股能否由換股日可觀察特徵（價格、排名、分數差、當月換股檔數）辨識：報 AUC 與分組差，不建交易規則'],
 lessons_o18_o19=['不利端逐格取最差（opt／p90／carry、一般／壓力）', '凍結前不算正式統計量', '事後新增狀態與嚴格版並列、嚴格版為主欄', 'raw/ 只放未修改原始檔（本題證交所跳動點原文下載被代理 403 擋下，raw/ 為空；跳動點級距沿用 V7X x_data.tick 的現行官方級距，不另存衍生檔到 raw/）',
                  '假設區間（半個／一個跳動點）與抽樣區間（月份區塊 bootstrap）並報', '取代正式數字須附歸因（本題不取代任何正式數字）', '紙上數字附成本兩平點', '可執行測到 30 萬',
                  'G6 角色由不同子代理擔任（本題研究員不自我複製；凍結規格與成交日誌交複製員）', '關鍵中間檔不得只放 scratchpad（名單、排名、成交日誌、逐槽表、淨值以 CSV.gz／parquet 存研究資料夾）',
                  '比對報逐 j 差分布（中位、最大絕對值）與最差變體是否換人', '文字證據核對主詞', '紙上與可執行遮罩口徑統一為「公告日 ≤ d 且涵蓋 e」', '口徑凍結後不得換（年化＝x_stats.cagr 日曆口徑，t＝訊號日窗）'],
 o1_lessons=['事件 t 按事件日分群（本題 bootstrap 以月份區塊）', '驗證期污染：2019 後台積電與隔夜效應已被看過，不視為盲測', '格點邊緣贏家詛咒（分類邊界只在挑選期定）', '成交可行性已由 V7X 處理', 't 提升拆均值與波動', '存活者偏誤：面板 2019～2023 只有存活者'],
 readonly=['live_tracking/main_forward、o_forward_infra_live 只讀，前後 sha256（ro_snap.py）；避開 06:49 台北排程與換股寫入時段', '不碰 o_g8p2a_protocol_addendum/（只做 sha256 快照，不呼叫判定器）'],
 cleanup='scratchpad/tfc 在題目結束時刪除；scratchpad/rep_o19 不刪。',
)
R['PREREG'] = P
R['PREREG_sha256'] = hashlib.sha256(json.dumps(P, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
RJ.write_text(json.dumps(R, ensure_ascii=False, indent=1), encoding='utf-8')
print(P['written_utc'], R['PREREG_sha256'])

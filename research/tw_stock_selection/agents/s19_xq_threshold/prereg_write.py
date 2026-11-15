#!/usr/bin/env python3
"""S19 事前登記：在算任何報酬之前寫 PREREG.json 與 PREREG.sha256。已存在就拒絕覆寫。python3 -I -B prereg_write.py
（prep.py 已先跑：只建面板、做不碰報酬的資料檢查 data_check.json；見 seen_before_prereg。）"""
import json, hashlib, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
f = H / 'PREREG.json'
assert not f.exists(), 'PREREG.json 已存在，拒絕覆寫'
dc = json.loads((H / 'data_check.json').read_text())
DYN = {int(y): dict(min_surge=v['min_surge'], exit_surge=v['exit_surge']) for y, v in dc['dynamic_table'].items()}

P = {
    'written_at': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
    'topic': 'S19：XQ 門檻版回測（照 20／21 號腳本的固定門檻，而不是研究版的名次）（s19_xq_threshold）',
    'question': ('使用者在 XQ 用 20＋21 號腳本回測，得到的應該接近「門檻版」；研究中心報的是「名次版」（S10 E0）。兩者差多少？差距來自訊號日（B11 對 A10）還是門檻（固定 SURGE 門檻對名次）？'
                 'XQ 回測該期待什麼數字（W10_70 版與 XQ 實際看得到的純衛星版）？門檻要不要隨時間調整？XQ 怎麼設最接近研究版？'),
    'script_facts（逐行讀腳本確認，登記前）': {
        '20 號 Input 預設': 'MinRet20 10、MaxRet20 1000、RetLen 20、UseMA60 1、MALen 60、MinAvgValue 1000（萬元）、MinPrice 10、LiqDays 60、OnlyRebalDay 1、RebalDay 11、**MinSurge 0**。',
        '20 號 MinSurge 說明': ('腳本預設 MinSurge＝0（註解寫「0＝不設（E3\'，預設，只靠排序取前 10）」）；README 第 12 節要使用者在 XQ 測衛星時把 MinSurge 設 2.5（近似「只買 L_MA 前 10 名、不遞補」）。'
                               '所以「照 README 設定」是 MinSurge 2.5（本題主格 XT），「照腳本字面預設」是 MinSurge 0（敏感度格 XT0）。題目說「README 說 2.5 近似前 10」，與腳本一致。'),
        '20 號選股條件': ('IsRebal（OnlyRebalDay＝1：DayOfMonth(Date) ≥ RebalDay 且（前一根日期 < RebalDay 或前一根日期 > 今天）＝每月第一個日期 ≥ 11 的交易日）且 Close ≥ MinPrice 且 '
                         'Average(Close × Volume, 60) ÷ 10 ≥ MinAvgValue（萬元）且（UseMA60＝1 時 CurrentBar ≥ 60 且 Close > Average(Close, 60)）且 Ret20 > MinRet20 且 Ret20 ≤ MaxRet20 且 Surge > −999 且 Surge ≥ MinSurge。'
                         'Ret20＝(Close ÷ Close[20] − 1) × 100。腳本要求用「還原日線」。'),
        '20／21 號 SURGE': ('D(k)＝月營收[k] − 月營收[k+12]，k＝0（最新月）…24；任一月營收缺值或 ≤ 0 → 無值；MeanD＝過去 24 個月（k＝1…24）平均；VarD＝(ΣD² − 24·MeanD²) ÷ 23；VarD > 0 才算 '
                           'Surge＝(D0 − MeanD) ÷ √VarD。和研究版 STD 同定義（研究版 σ ≤ 1e-9 × 窗內 |D| 平均視為 0；腳本用一次算完的變異數公式，浮點上不完全等價，需在 XQ 確認）。'),
        '21 號 Input 預設': 'ExitSurge 2.0、UseF4Exit 0、ExitSurge10 2.5、UseMA60 1、MALen 60、UseRet20Exit 0、MinRet20 10、RetLen 20、RebalDay 11。',
        '21 號出場條件': 'IsRebal（同 20 號，但不受 OnlyRebalDay 控制）且（Surge < ExitSurge（營收缺值 Surge＝−999 也算）或 UseMA60＝1 時 Close ≤ Average(Close, 60)）→ 出場訊號，隔天開盤賣。**沒有股池條件**（成交值、價格、創新板都不看）。',
        'XQ 回測設定（腳本開頭與 README 說明，非腳本程式）': '股池「上市」、排除股名帶「-創」「-KY創」、最多持有 10 檔、滿了不買、每檔 1／10 資金、買得到不到 1／10 一半不買、空位放 0050（XQ 能否設定需在 XQ 確認）。',
        '排序': '選股腳本本身不排序。XQ 回測同一天候選超過空位時的取捨順序需在 XQ 確認；本題假設依 OutputField(1) SURGE 由大到小（題目指定）。',
    },
    'differences_script_vs_research（全部列出；門檻版照腳本做）': [
        {'id': 'D1', 'item': '訊號日', 'research': 'A10：次月 10 日（遇假日順延）的下一個交易日', 'script': 'B11：第一個日期 ≥ 11 的交易日。80／129 個月同一天，49 個月早 1 天（期限日當天）', 'S19': '照腳本（B11）。營收沿用 S9 的 B11opt（期限日當天仍用上月營收＝假設當天收盤後營收已到齊；需在 XQ 確認）'},
        {'id': 'D2', 'item': '買進名單', 'research': 'L_MA 前 10 名中 RET20 > 10% 的（不遞補）', 'script': 'SURGE ≥ MinSurge 且 RET20 > 10% 且季線上的全部股票，依 SURGE 由大到小買到 10 檔滿', 'S19': '照腳本'},
        {'id': 'D3', 'item': '出場', 'research': '不在 L_MA 前 20 名（含跌破季線、營收缺值、離開股池）', 'script': 'SURGE < ExitSurge（含缺值）或收盤 ≤ 60 日均線；不看股池', 'S19': '照腳本：XT 不因成交值、價格不足而賣'},
        {'id': 'D4', 'item': '價格門檻', 'research': '訊號日原始收盤 ≥ 10', 'script': '還原日線 Close ≥ 10。XQ 回測時歷史 K 棒的還原價以最新一天為基準往回等比例還原（假設），早年還原價比原始價低（研究股池內中位數比例 2016 年 0.68、2020 年 0.81、2026 年 0.98）', 'S19': '照腳本：Cadj＝G × 最後收盤 ÷ G[最後]（需在 XQ 確認還原方式）'},
        {'id': 'D5', 'item': '成交值門檻', 'research': '官方成交金額 60 日平均 ≥ 1,000 萬（無成交日 0）', 'script': 'Average(還原 Close × 成交量, 60) ≥ 1,000 萬；還原價使早年低估', 'S19': '照腳本：Cadj × 成交股數 的 60 日平均（不在日 K 內的日子不算，同研究版）'},
        {'id': 'D6', 'item': '季線、RET20', 'research': '含息還原收盤 G 的 60 日均線、G 的 20 日報酬', 'script': '還原日線 Close 的 60 日均線、20 日報酬', 'S19': '等比例還原下兩者相同，用 G（XQ 若是減法還原會有小差，需在 XQ 確認）'},
        {'id': 'D7', 'item': '營收來源與股票範圍', 'research': 'tw_revenue.csv（沒有 KY 股、沒有部分已下市公司）', 'script': 'XQ 欄位「月營收」（可能有 KY 股、可能用更正後數字）', 'S19': '無法模擬，只列為風險'},
        {'id': 'D8', 'item': '有訊號的月份', 'research': '2016-01 營收月（2016-02 換股）起；股池內 SURGE 有值 < 50 檔的月份不換股', 'script': 'XQ 營收歷史較長，2015-07～2016-01 可能就有訊號', 'S19': '所有格都用研究版的月份集合（2016-02 起），全期數字與 XQ 會不同；2019 起不受影響'},
        {'id': 'D9', 'item': 'TDR、無成交日', 'research': '排除 91 開頭；訊號日要有成交', 'script': '腳本沒寫；XQ 上市股池可能含 TDR', 'S19': '照研究版（TDR 沒有營收資料，不影響）'},
        {'id': 'D10', 'item': '季線的最少 K 棒', 'research': '60 日均線要 60 個有效值', 'script': 'CurrentBar ≥ 60', 'S19': '視為相同'},
        {'id': 'D11', 'item': '每檔金額', 'research': '（衛星淨值）÷ 10', 'script': 'XQ「每檔 1／10 資金」是固定金額還是淨值比例需在 XQ 確認', 'S19': '用淨值 ÷ 10'},
        {'id': 'D12', 'item': '同日賣出又入選', 'research': '待賣股即使在買進名單也賣、當天不買回', 'script': '門檻版的買進（SURGE ≥ MinSurge、季線上）與出場（SURGE < ExitSurge 或季線下）在 MinSurge ≥ ExitSurge 時不會同時成立；XT0（MinSurge 0 < ExitSurge 2）可能同時成立', 'S19': '照引擎：賣、當天不買回'},
    ],
    'seen_before_prereg': ('讀過：xq_scripts/20、21 號 .xs 全文；xq_scripts/README.md 第 11～25 節；s18_live_list/FINDINGS.md 第 1～5 節；s17 replicate/REPLICATION_REPORT.md (c)～(e)（審查 C 的「2016 年 2.16、2026 年 4.35」是 NOMU 定義的第 20 名，不是現行 STD）；'
                           's10_exit_details/FINDINGS.md 開頭到第 3 節、prep.py、run.py、s10lib.py 全文；s9_signal_timing/FINDINGS.md 第 1～8 節、s9lib.signal_rows／rt_for；s17lib.py 全文；s4lib.make_signal／innov_panel、s2lib.load／rev_tables、stlib.load／universe／統計函式、cslib.Panel／rev_avail_days。'
                           '事前已知的報酬（都是名次版）：E0｜W10_70（A10）100 萬 2019 起 48.13%／全期 34.08%／2024 起 70.16%／最大回撤 −33.40%，20 條路徑中位數 46.23%／−34.01%；'
                           'S9 B11opt｜W10_70（＝本題 XR｜W10_70，只差 σ > 0 與 0050 停牌修正）49.48%／35.24%／74.19%／−30.06%，路徑中位數 45.79%／−33.63%；衛星 100%、空位現金（100 萬）41.92%／29.45%；0050 28.91%／21.14%／54.48%／−33.96%。'
                           '登記前跑過 prep.py（data_check.json，不算任何報酬）：B11 對 A10 的天數差（0 天 80 個月、早 1 天 49 個月）；每月 L_MA 第 10／20 名 SURGE（B11、研究股池）整體中位數 2.51／2.01，'
                           '年度中位數第 10 名 2016 2.06、2017 2.51、2018 2.39、2019 2.33、2020 3.16、2021 3.43、2022 2.29、2023 2.36、2024 2.60、2025 2.59、2026 3.84；第 20 名 1.71～3.23；'
                           '腳本股池比研究股池平均每月少 41 檔、多 6 檔（季線上的少 23、多 2），差異集中在早年；MinSurge 2.5 的候選每月平均 6.5 檔（中位數 3、最多 43、24 個月超過 10 檔），研究版名單平均 4.0 檔、其中 79% 也在門檻版候選內。'
                           '本題 XU、XT、XT0、XDYN、XS3 的報酬在登記前都沒有算過；XR｜SAT、E0｜SAT（70 萬）也沒有算過。'),
    'data_universe_engine': ('import s10lib（→ s9lib → … → stlib → cslib，全部不改）；公開封存的官方日 K（data/universe/daily，blobless clone 107f9073）與月營收 tw_revenue.csv；面板 T＝2876、N＝2142、最後一天 2026-10-22、淨值起算 2015-07-01。'
                             'SURGE＝標準定義且 σ > 0（逐字複製 s17lib STD）；0050 在 2025-06-11～17 停牌日開盤用前一日收盤（逐字複製 s17lib）。引擎 s10lib.simulate_s10（exec_mode＝open）不改：訊號列收盤 → 次一交易日開盤先賣（開盤 ≤ 跌停或無開盤價順延）、'
                             '（W10_70）底倉 ±10 個百分點帶狀拉回、依 rank（SURGE）由大到小買到 10 檔滿為止（滿了不買）、開盤 ≥ 漲停不買、股數 × 開盤 < 目標一半不買、單筆 ≤ 當日成交值 5%、2020-10-26 前只能整張；'
                             '嚴成本（手續費 0.1425%×0.28 每邊、最低 20 元；賣出稅 0.3%；ETF 稅 0.1%；每邊半個跳動點＋0.3% 滑價）。只上市、排除創新板（訊號日日 K 名稱符合 -(KY)?創$）。'),
    'cells（策略）': [
        {'id': 'E0', 'role': '對照：研究版', 'rule': 'A10 訊號日；研究股池；L_MA 前 10 名中 RET20 > 10%；不在 L_MA 前 20 名就賣（s10lib.make_signal_s10 的 E0，不改）'},
        {'id': 'XR', 'role': '對照：分離訊號日', 'rule': '同 E0，只把訊號日改成 B11（S9 B11opt）'},
        {'id': 'XU', 'role': '對照：分離門檻與腳本細節', 'rule': 'B11；研究股池（原始價 ≥ 10、官方成交金額）；買＝SURGE ≥ 2.5 且 RET20 > 10% 且季線上，依 SURGE 買到滿；出場＝SURGE < 2.0 或季線下或離開研究股池（和 XR 一樣會因離開股池而賣）'},
        {'id': 'XT', 'role': '主格：XQ 門檻版（照 README 第 12 節設定＋腳本字面）', 'rule': 'B11；腳本股池（D4、D5：還原價 ≥ 10、還原價 × 成交量 60 日平均 ≥ 1,000 萬；上市、排除創新板）；買＝SURGE ≥ 2.5 且 RET20 > 10% 且季線上，依 SURGE 買到 10 檔滿；出場＝SURGE < 2.0（含缺值）或收盤 ≤ 60 日均線，不看股池'},
        {'id': 'XT0', 'role': '敏感度 1：腳本字面預設', 'rule': '同 XT，但 MinSurge＝0（20 號腳本的預設值）'},
        {'id': 'XDYN', 'role': '敏感度 2：每年動態門檻（只用過去資料）', 'rule': '同 XT，但 Y 年的 MinSurge＝Y−1 年每月 L_MA 第 10 名 SURGE 的中位數、ExitSurge＝Y−1 年第 20 名中位數（B11、研究股池、位移 0；2016 年沒有前一年 → 2.5／2.0）。各位移路徑共用同一張表（使用者照表每年改一次參數）', 'table': DYN},
        {'id': 'XS3', 'role': '敏感度 3：收緊門檻', 'rule': '同 XT，但 MinSurge 3.0、ExitSurge 2.5'},
    ],
    'funding（資金格）': [
        {'id': 'W10_70', 'rule': '衛星 70%、每檔＝衛星 ÷ 10、不鎖底倉、底倉 ±10 個百分點帶狀拉回、空位與剩餘現金 ≥ 2 萬放 0050；主跑 100 萬；20 條路徑＝60／80／100／150／200 萬 × 位移 0～3（在各自訊號列上往後移）'},
        {'id': 'SAT', 'rule': '純衛星、空位現金（不放 0050，對應 XQ 只能測衛星那一塊）：sat_w＝1、每檔＝淨值 ÷ 10、起始 70 萬；20 條路徑＝42／56／70／105／140 萬 × 位移 0～3'},
    ],
    'runs': '7 策略 × 2 資金格＝14 個組合；每個組合：主跑、20 條路徑、成本加倍、禁買最賺 5 筆（各自）。',
    'windows': {'since2019': '2018 最後一個交易日 → 2026-10-22', 'y2019_2023': '2018 最後一個交易日 → 2023 最後一個交易日', 'since2024': '2023 最後一個交易日 → 2026-10-22', 'full': '2015-07-01 起',
                'half1／half2': '全期對半', 'bear_segments': {'2018': '2018-01-01～2018-12-31', '2020-03': '2020-01-01～2020-06-30', '2022': '2022-01-01～2022-12-31', '2024-07~2025-04': '2024-07-01～2025-04-30'}},
    'primary_statistics': {
        'S1': 'XT 對 E0 的 20 條路徑配對差（同資金、同位移；2019 起、2019～2023、2024 起年化，個百分點）：中位數、為正條數、最小～最大。W10_70 與 SAT 各算一次。',
        'S2': '拆解（W10_70 與 SAT 各算）：訊號日＝中位數(XR − E0)、門檻＝中位數(XU − XR)、腳本細節＝中位數(XT − XU)，都是 2019 起 20 條路徑配對差的中位數；另報中位數(XT − E0)。',
        'S3': '四段空頭：每段 XT 與 E0 的段內最大跌幅配對差（正＝XT 較淺），中位數與變淺條數。',
        'S4': '名單差異（位移 0、B11）：每月 XT／XT0／XDYN／XS3／XU 買進候選檔數、和 XR 名單（L_MA 前 10 名中 RET20 > 10%）的重疊（交集 ÷ XR 名單、Jaccard）；主跑每個成交日的實際持股檔數分布。',
        'S5': ('門檻版多買／多續抱的股票表現（XT｜W10_70 主跑）：(a) 多買＝XT 實際買進、但不在同月 XR 名單的；(b) 多續抱＝訊號日 XT 持有且不賣、但 XR 規則會賣（不在研究股池 L_MA 前 20 名）的；(c) 少買＝XR 名單裡、XT 沒買的（不論原因）。'
               '每筆從次一交易日開盤到下一個成交日開盤（還原開盤）報酬減同期 0050（還原開盤），平均、t、筆數；(a) 另報整筆交易報酬。'),
    },
    'decision_rules（判定，事前寫死）': {
        'J1_差距（W10_70 與 SAT 各判）': ('d＝XT − E0 的 2019 起 20 條路徑配對差，med＝中位數。|med| ≥ 3 個百分點且 ≥ 15／20 條同號 → 「明顯不同」（標高或低），FINDINGS 要明講「XQ 結果會和研究中心數字明顯不同」並給設定建議；'
                                        '|med| ≥ 3 但同號 < 15 條 → 「明顯不同但方向不穩」；1 ≤ |med| < 3 → 「略有不同」；|med| < 1 → 「接近」。2019～2023、2024 起用同一規則描述。'),
        'J2_主要來源': '三個拆解分量（S2）中絕對值最大的就是「主要來源」；三者絕對值都 < 1 個百分點 → 「三者都小」。W10_70 為主判，SAT 另報。',
        'J3_空頭': '每段：med（正＝XT 較淺）≥ +1 且 ≥ 15 條變淺 → 「較淺」；med ≤ −1 且 ≥ 15 條變深 → 「較深」；其他 → 「差不多」。',
        'J4_要不要動態門檻': ('「建議改動態」若且唯若：|med(XDYN − E0)| ≤ |med(XT − E0)| − 1（2019 起，W10_70），而且 2019～2023 與 2024 起兩段的 |med(XDYN − E0)| 都 ≤ |med(XT − E0)|，'
                             '而且位移 0 每月買進名單對 XR 的平均 Jaccard XDYN ≥ XT。否則「不必改動態」。'),
        'J5_XQ_設定建議': ('候選＝XT、XT0、XDYN、XS3（XQ 改參數就能做到）。closeness＝|med(格 − E0)|（2019 起、W10_70）。最小者若比 XT 小 ≥ 1 個百分點、且平均 Jaccard ≥ XT − 0.05 → 建議它；否則建議 XT（MinSurge 2.5、ExitSurge 2.0）。'
                          '另外：若 |med(XT0 − E0)| ≥ 3 → FINDINGS 要警告「20 號腳本字面預設 MinSurge 0 不是研究版，請改 2.5」。報酬最高的格不參與建議。'),
        'J6_期待數字': '「XQ 回測應期待的數字」＝XT｜SAT（純衛星、空位現金、70 萬）主跑與 20 條路徑中位數／最小／最大；XT｜W10_70 同樣列出（底倉衛星版的對應）。照實報，不判定。',
        'J7_八項': ('同 S10：2019 起、全期淨報酬贏 0050；成本加倍後 2019 起、全期贏；禁買最賺 5 筆後 2019 起、全期贏；前半、後半贏。全過 → 「值得上 XQ 測」；2019 起與全期都輸 → 「不值得」；其他 → 「有條件」。'),
    },
    'acceptance（驗收）': {
        'P0': '面板 T＝2876、N＝2142、最後一天 2026-10-22。',
        'A0': 'E0｜W10_70（A10、σ > 0、停牌修正）100 萬：2019 起 48.126%、全期 34.082%、最大回撤 −33.404%（S17 STD 公布值，誤差 < 0.025 個百分點）；E0 訊號（sig、X、rank）和 s4lib.make_signal 相同。',
        'A1': 'XR｜W10_70 對 S9 B11opt（49.48%／35.24%／−30.06%）差 < 0.10 個百分點（只差 σ > 0 與停牌修正）。',
        'A2': 'make_signal_thr 用「每月門檻剛好等於當月 L_MA 第 10 名／第 20 名的 SURGE」、研究股池、出場看研究股池 → sig 與 X 和 XR 完全相同（位移 0）。',
        'A3': 'XT 主跑：任何時候持股 ≤ 10 檔；每筆買進在訊號列都滿足 SURGE ≥ MinSurge、RET20 > 10%、季線上、腳本股池；每筆「rule」出場在訊號列 SURGE < ExitSurge 或缺值或季線下。',
    },
    'predictions（事前預測，不影響判定）': {
        'XR_vs_E0': 'S9 已知：W10_70 單一路徑 +1.35、路徑中位數 −0.44 個百分點 → 預測 |med| < 1（接近）。',
        'XT_vs_E0': '預測 XT 持股較多（多買第 11 名以後、出場門檻在高分年份偏鬆）；W10_70 的 2019 起中位數差在 −4～+1 個百分點之間，最可能「略有不同、偏低」；SAT 版因為個股比重較高，差距的絕對值比 W10_70 大。',
        'decomposition': '預測主要來源是門檻（XU − XR），訊號日分量 < 1 個百分點；腳本細節（還原價股池）只影響 2019 年以前，2019 起分量 < 1。',
        'XT0': 'MinSurge 0 等於遞補到第 11 名以後（E3\' 類），預測 W10_70 比 E0 低 2～6 個百分點。',
        'XDYN': '前一年的中位數追不上當年（例如 2022 用 2021 的 3.43，但 2022 實際 2.29），預測不會比 XT 更接近 E0（「不必改動態」）。',
    },
    'data_snooping': {'N_before': 2825, 'added': 11, 'N_after': 2836,
                      'note': '新組合：XU、XT、XT0、XDYN、XS3 × {W10_70、SAT}＝10，加 XR｜SAT＝1。XR｜W10_70＝S9 B11opt（已計入）；E0｜SAT 是已公布「衛星 100%、空位現金」換資金規模（不另計）。路徑、成本加倍、禁買是同一組合的穩健性版本，不另計。不做 SPA／DSR。'},
    'outputs': 'results.json、trades.csv.gz（14 個主跑）、navs.csv.gz（14 個主跑＋0050，四捨五入到元）、results_posthoc.json（逐年、持股檔數分布、差最多的年份與原因）、code.sha256。',
}
b = json.dumps(P, ensure_ascii=False, indent=1).encode('utf-8')
f.write_bytes(b)
(H / 'PREREG.sha256').write_text(hashlib.sha256(b).hexdigest() + '  PREREG.json\n')
print('寫入', f, hashlib.sha256(b).hexdigest())

#!/usr/bin/env python3
"""S0：在任何搜尋（本地語料 grep、WebSearch）之前寫 SPEC_draft.json（狀態表、模型候選、判定規則）。只寫一次。
python3 -I -B s0_spec_draft.py"""
import json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
F = HERE / 'SPEC_draft.json'
assert not F.exists(), 'SPEC_draft 只寫一次'
S = {
 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 'topic': 'o_disposal_list_pit_audit_pkl18：官方處置清單完整性與公告時點 PIT 核對＋o18 補掃 2,093 對 2,075 差 18 組',
 'seen_before_this_spec': [
   'disposal_official_universe.csv 逐年列數（任務文字：2018 84、2019 31、2020 187；2019-06 0 列）',
   'twse_2019.json 的 fields／count=33／total=49 與 49 列原文（S1 計數必要；未做任何外部搜尋）',
   'o_disposal_rejudge FINDINGS：每年 len(data)==total 斷言、2021 年與 audichuang 一致（336）、逐年 50% 門檻沒過的年份',
   'audichuang 實驗紀錄 §1：2013～2026-10-20 去重 2,504 列（只有總數，原始 csv 404 不存在）'],
 'status_table': {
   'CONFIRMED_MISSING': '≥ 2 個彼此獨立的公開來源，都載明「代碼或可唯一對應的名稱」＋「處置公告日或處置起日（±3 個交易日）」，標的是上市普通股（4 碼、非 0 開頭、名稱不含 DR），且 universe 內同代碼沒有 start 在 ±10 個交易日內的列',
   'SINGLE_SOURCE': '只有 1 個來源符合上列條件',
   'NOT_FOUND': '已在所有規定來源搜尋，沒有任何來源載明處置',
   'ALREADY_LISTED': '來源提到的處置已在 universe（同代碼、start ±10 個交易日內）',
   'NOT_COMMON_OR_TPEX': '來源提到的是權證、ETF、DR、上櫃股或注意股（非處置）',
   'independence': '同一發布者或轉貼（PTT [新聞] 轉載鉅亨原文、tsungmin disposition_stocks.txt 由同一 TWSE 端點重建）不算獨立；TWSE 封存本身是被稽核對象，不算交叉來源',
   'post_search_states_rule': '搜尋後若需新增狀態，一律和嚴格版並列，嚴格版（只加 CONFIRMED_MISSING）為主欄'},
 'sources_planned': ['tsungmin2001 raw twse_YYYY.json 內部一致性（累計欄 vs 列數、編號 vs count、跨年查詢重疊）',
   'tw-stock-screener extras disposal_5y（2021-09 起）、attention_5y（2021-10 起）、warnings_hist（2026）',
   'tsungmin disposition_stocks.txt（非獨立，只作格式對照）',
   '鉅亨新聞全文（o_v4_revenue_patch_official/raw/ctbc EB_SegAndRawNews 2013～2021）',
   'PTT Stock 板逐日檔（o_rev_gap_news_2013/raw/ji08kk44_BDSE26_TEAM3，2019～2022）',
   'roccqqck stock_eland（Yahoo 股市＋PTT 2016～2018，含時間）',
   'bofrankbo 逐檔新聞（55 檔）',
   'scratchpad tsec_all（先確認有無處置欄）',
   'disposal_bear_longhistory 代理事件',
   '伺服器端 WebSearch（查詢詞不得含目標數字，例如 31 或特定日期；逐次記錄 search_log_web.csv）'],
 'S1_rules': {
   'lineage_stages': ['A raw TWSE 年檔列（含跨年重疊）', 'B rejudge 去重後 3,179 列', 'C 普通股（4 碼、非 0、名稱無 DR）', 'D in_panel', 'E universe 2,218（核對與 C 的關係）'],
   'year_key': 'start 年（universe 與 core.py 的 s0_gate 相同）；另報 announce 年',
   'announce_join': '以 ordr（rejudge 列號）接回並以 (code, start) 驗證一對一；接回率 < 99% 或任一列 start ≤ announce → P1 否定',
   'trade_date_relation': '主策略訊號日 d（每月第一個 ≥ 11 日的交易日）、成交日 e＝d 的下一交易日；列出所有涵蓋 e 或 d 的 universe 列的 announce 與 d 的關係（announce < d、＝d、> d）',
   'internal_completeness': '每個年檔：各證券「累計」欄 ＝ 該證券在檔內列數？max(編號)＝count？跨年重疊列在兩年檔一致？違反即列為「封存內部不一致」'},
 'S2_model': {
   'unit': '月（以處置起日 start 的月份；敏感度：announce 月），因變數＝universe 普通股且 in_panel 的處置列數',
   'fit_sample': '2006-01～2018-12 與 2020-01～2023-12（排除 2019 全年）；2004～2005、2024～2026 只報殘差（樣本外）',
   'candidates': {
     'trigger_proxy': '面板近似注意條件：6 日累積還原報酬 |r6| > 門檻（2015-06-01 前 25%、之後 32%）且與市場（面板中位數或 TAIEX）差 > 20pp；或 6 日均量 / 60 日均量 ≥ 5；或週轉率（value/close/shares）6 日均 ≥ 某門檻',
     'cluster_proxy': '股票在 3 個連續交易日都符合近似注意條件的事件數（近似「連續三次」觸發）',
     'regime': 'R15＝2015-06-01 起（漲跌幅 10%）；R17＝2017-04-28 起（當沖降稅）；COVID＝2020-03～2020-12；R26＝2026-08-10 起（不在擬合樣本）；2014-04 處置期 5→10 日',
     'forms': ['Poisson GLM', 'NB2 GLM（離散度以 ML 估計）'],
     'offset': 'log(面板當月有價普通股數) 可選'},
   'max_versions': 4,
   'selection_criterion': '只用擬合樣本：AIC、殘差自相關、逐年殘差平方和（不含 2019）',
   'freeze_rule': '凍結（寫 PREREG）前不產生任何 2019 月份的預測值；程式斷言 PREREG 存在才算 2019 預測與逐年殘差',
   'interval': '95% 預測區間＝參數抽樣（係數多元常態 2,000 次）× NB 抽樣，報年總數；另報不含參數不確定性的區間；假設區間＝迭代版本間的範圍'},
 'decisions': {
   'P1': '接回率 ≥ 99% 且 0 列 start ≤ announce → 成立；並以公開證據界定公告發布時點（盤後）',
   'P2': '2019 年觀察值落在 95% PI 內 → H_real；低於下界且嚴格 CONFIRMED_MISSING ≥ 5 → H_inc；低於下界但嚴格補列 < 5 → 「2019 年疑似不完整、未能補正」',
   'P3': '若有缺漏：CONFIRMED_MISSING 集中在 2019-05、06、08、09 的比例 > 50% → 成立',
   'P4': '補正後主策略任一正式數字（V7X 6 格＋紙上）|Δ| > 0.3pp 或 |Δt| > 0.05 → 取代（附歸因拆解）；否則加註',
   'P5': '18 組差可用 ≤ 2 條定義規則解釋 → 結案；需逐案特判 → G5 不可驗證缺口',
   'final': ['補列 0 且模型支持 H_real：加註「官方清單完整性與公告時點已核（2019 年低列數經模型與 N 個來源支持為真實）」，G5 移除「處置清單公告時點未核」',
             '模型支持 H_inc 但嚴格補列 < 5：加註「2019 年疑似不完整、未能補正」，G5 維持 ✘']},
 'o18_lessons': ['不利端逐格取最差', '凍結前不算正式口徑統計量', '搜尋後新增狀態和嚴格版並列、嚴格版為主欄',
                 'raw/ 只放未修改原始檔，註記另放，查詢詞不得含目標數字，摘要和原文分開計數', '假設區間和抽樣區間並報',
                 '取代正式數字附歸因拆解', '紙上數字附成本兩平點', '可執行測到 30 萬', 'G6 狀態在複製後重新核對']}
txt = json.dumps(S, ensure_ascii=False, indent=1, sort_keys=True)
F.write_text(txt, encoding='utf-8')
print(F, hashlib.sha256(txt.encode()).hexdigest(), S['written_utc'])

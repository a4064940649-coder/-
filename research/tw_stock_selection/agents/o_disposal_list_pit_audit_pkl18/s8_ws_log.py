#!/usr/bin/env python3
"""把 WebSearch 工具回傳（結果標題＋URL＋摘要首段，逐字）存成 raw/websearch/ws_NN.txt；查詢紀錄寫 search_log_web.csv。
查詢詞不含任何目標數字（列數）或特定日期；年份只作情境。"""
import csv, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
Q = [
 ('2026-11-09T23:3xZ', '喬山 處置股 分盤交易 2019年', 'S3 候選 1736（2019-05 面板強觸發＋PTT 推文「準備要關禁閉」）',
  'Links: 喬山代子公司處分荷蘭子公司公告(yahoo)、cmoney 1736 論壇、goodinfo 1736 等\n摘要首段：搜尋結果中找不到喬山在2019年被列為處置股、或實施分盤交易的相關資料。', 'NOT_FOUND'),
 ('2026-11-09T23:3xZ', '錩泰 處置股 2019年', 'S3 候選 1541（2019-06 面板觸發；6 月 0 列）',
  'Links: usmart、technews 錩泰公司資料、ctee 2026、sinotrade 處置股名單0819、cgc.twse 2019 Law Enforcement Report、yahoo「泰聚亨-DR 12/2起列為處置股,預收款券、每20分鐘撮合一次」、cmoney 08/19處置股公告 亞泰金屬\n摘要首段：搜尋結果中沒有找到錩泰(1541)在2019年被列為處置股的直接紀錄。', 'NOT_FOUND'),
 ('2026-11-09T23:3xZ', '燿華 處置 分盤交易 2019年', 'S3 候選 2367（2019-09 面板強觸發＋PTT 推測）',
  'Links: uanalyze、usmart、fubon-ebrokerdj、cmoney 02/26處置股公告 華城 等\n摘要首段：搜尋結果沒有找到燿華在2019年被列為處置交易（分盤交易）的資料。找到的都是2026年的相關報導：2026年3月5日起，燿華（2367）與志聖（2467）遭處置交易', 'NOT_FOUND'),
 ('2026-11-09T23:3xZ', '證交所 處置有價證券 公告時間 盤後 收盤後公布', 'P1 公告時點證據',
  'Links: udn 證交所宣布調整處置新機制 8月10日上路、ctee、pts、fubon-ebrokerdj 115年6月25日公告全文\n摘要首段：搜尋結果沒有直接說明證交所每日公布處置名單的確切時間（例如是否為盤後或收盤後），因此無法確認這點。', 'NO_TIME_EVIDENCE'),
 ('2026-11-09T23:3xZ', '"起列為處置股" 預收款券 每5分鐘撮合 喬山', 'S3 候選 1736（Yahoo/MoneyDJ 標題格式）',
  'Links: pocket.tw、udn、cmoney 06/21處置股公告 華經(2468)、tvbs、esunsec、stockfeel\n摘要首段：搜尋結果中沒有找到喬山(代號 1736)被列為處置股的相關公告', 'NOT_FOUND'),
 ('2026-11-09T23:3xZ', '"列為處置股" 錩泰 預收款券', 'S3 候選 1541',
  'Links: yahoo 寶得利公告、cmoney 亞泰金屬、sinotrade 處置股名單0513、businesstoday、yahoo《台北股市》處置股3日 等\n摘要首段：我搜尋的結果中沒有找到「錩泰」的相關資料,無法確認它是否被列為處置股', 'NOT_FOUND'),
 ('2026-11-09T23:3xZ', '"列為處置股" 晶心科 預收款券 撮合', 'S3 候選 6533（2019-01 面板觸發＋PTT「要準備分盤交易」）',
  'Links: yahoo 寶得利、yahoo 中環處分有價證券、stockfeel、tvbs、yahoo 泰聚亨-DR、cmoney、wantrich.chinatimes 20261007/20261014\n摘要首段：我搜尋了一次,結果沒有找到晶心科(6533)被列為處置股時的完整公告…時報報導指出,4檔於前一交易日處置期屆滿…晶心科(6533)…今日正式「解禁」出關。這則報導的日期是10月14日', 'NOT_FOUND（2019）；2026-10 處置已在名單'),
 ('2026-11-09T23:3xZ', '處置股 公告 "今日收盤後" 證交所 列為處置 隔日起', 'P1 公告時點證據',
  'Links: ettoday 台股10檔「抓去關」新名單 禾伸堂遭延長處置至5/28、ettoday 5檔「抓去關」新名單 載版股景碩明起處置到7/16、ettoday 台股7檔「抓去關」新名單 下周一起處置至5/22、ltn topic 處置股、yahoo《台北股市》處置股:16日明輝報到 華經等4檔明出柙、ettoday 7檔飆股「抓去關」力積電明起處置到6/15、ettoday 台股3檔「抓去關」下周一起處置至7/31\n摘要首段：搜尋結果多為新聞報導，內容是證交所與櫃買中心在盤後公告處置股，並於隔日起生效。', 'TIME_EVIDENCE（盤後公告、明起生效；2026 年報導）'),
 ('2026-11-09T23:4xZ', '《台北股市》處置股 2019 報到 出柙', 'S3 2019 名單型報導',
  'Links: yahoo《台北股市》處置股:16日明輝報到 華經等4檔明出柙、yahoo 處置股15日1出1待退 南亞科、yahoo 處置股3日2進1出2待退 光聖、yahoo 處置股:14日計6檔 雷虎、有成精密、訊達電腦等、ettoday 6檔飆股「抓去關」大立光、貿聯-KY皆入列、yahoo 處置股:24日15檔\n摘要首段：搜尋結果裡沒有明確標示 2019 年的報導', 'NOT_FOUND（無 2019 年份）'),
 ('2026-11-09T23:4xZ', '2019 列為處置股 預收款券 每5分鐘撮合一次 上市 MoneyDJ', 'S3 2019 名單型報導',
  'Links: cmoney 03/10處置股公告 先進光(3362)、02/23 先進光、10/04 新門(5432)、07/14 京晨科(6419)、05/07 廣錠(6441)、06/13 倚天酷碁-創(2432)、07/10 台聯電(4905)、06/21 華經(2468)、pocket\n摘要首段：搜尋結果沒有找到 2019 年的處置股公告，也沒有 MoneyDJ 的來源。', 'NOT_FOUND（無 2019 年份）'),
 ('2026-11-09T23:4xZ', '上緯投控 處置股 2019', 'S3 候選 3708（2019-02 面板觸發）',
  'Links: bnext swancor、technews 2019/07/01、yahoo 3708 dividend、ltn 上緯賣子公司 等\n摘要首段：搜尋結果沒有提到上緯投控被列為「處置股」的內容。', 'NOT_FOUND'),
 ('2026-11-09T23:4xZ', '茂林-KY 處置 分盤交易 2019', 'S3 候選 4935（2019-07 面板觸發）',
  'Links: cna 201911180025、cna 201911200022、nstock、ettoday 裕民、wearn 4935、yahoo 4935、moneydj 4935\n摘要首段：搜尋結果顯示,這件事發生在2019年11月,與茂林-KY的財報爭議有關。結果中沒有直接出現「分盤交易」一詞,報導用的是「變更交易方法」和「全額交割」。…證交所因此從11月19日起將其交易方法改為全額交割。', 'NOT_DISPOSAL（變更交易方法／全額交割，非注意-處置制度）'),
 ('2026-11-10T00:0xZ', '華邦電 列為處置股 分盤交易 2020年12月', '事後（範圍外）核對 PTT 推文「2344…明分盤」',
  'Links: ettoday 記憶體4檔明起抓去關、yahoo 半導體 華邦電 DRAM訂單熱、yahoo 記憶體族群劇烈波動 旺宏、彩晶、波若威將列入處置股、yahoo《台北股市》處置股17日 1進2出2待退 華邦電、wantrich 20260110、20251216\n摘要首段：我的搜尋沒有找到2020年12月華邦電列為處置股、分盤交易的相關報導。結果都是2025年底到2026年初的事件', 'NOT_FOUND（2020-12）'),
]
with open(HERE / 'search_log_web.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['n', 'utc_approx', 'query', 'purpose', 'raw_file', 'sha256', 'outcome_note'])
    for i, (t, q, pur, txt, oc) in enumerate(Q, 1):
        p = HERE / 'raw/websearch' / f'ws_{i:02d}.txt'
        body = f'QUERY: {q}\n---- tool return (titles/URLs listed as returned; summary first paragraph verbatim) ----\n{txt}\n'
        p.write_text(body, encoding='utf-8')
        w.writerow([i, t, q, pur, str(p.relative_to(HERE)), hashlib.sha256(body.encode()).hexdigest(), oc])
print(len(Q))

#!/usr/bin/env python3
"""第 6 步：盤點表（explore.json）、建構迭代紀錄（iterations.json）、結果彙整（results.json），並把逐一檢查過的 repo 記入 search_log。"""
import json, hashlib, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
probes = [  # repo, commit, 資料型態, 欄位, 起訖, 檔數/列數, 含下市, 日期依據, 用否, 理由
 ('FinBrain-tw/twstock_database', 'a41851b3', 'MARGN', '融資買進−賣出、融券賣出−買進（張，淨額；不含現償）', '2007-04-23～2021-07-09', '寬表 3,509 日 × 1,782 檔', '是（2311、2325、5854 至 2011-11-18）', '內容日期（index）；commit 2021-07-16 一次性', '只作交叉核對；版本 V2/V3 的累加被一致性否決', '欄位是淨買賣不是餘額增減：與 TEJ 買進−賣出一致 99.90%，與 TEJ Δ餘額只一致 69.2%'),
 ('Yuan-Hsi/SALL_Agency', '715cc770', 'MARGN', 'TEJ 匯出：融資餘額、融資買進、融資賣出、融券餘額/買進/賣出（無現償、無限額）', '1994-09～2024-04-03', '982 檔普通股（stock_0406）', '否（存活股；2311、2325、5854 皆缺）', '內容日期欄；commit 2024-07-22 一次性', '是（存活股逐日餘額、交叉核對）', 'TEJ 付費資料轉貼，授權不明'),
 ('joeycchen72/taiwan-market-history', '0a270d50', 'MARGN＋QFIIS＋分母', '官方 MI_MARGN 融資/融券餘額；MI_QFIIS 發行股數、外資持股、比率；MI_INDEX 全市場日檔', 'MARGN/QFIIS：2014-01～2026-10 每月 1 日（11 日起第一個交易日）；MI_INDEX：2014-01-02～2026-11-04 逐日', '154 個形成日 × 約 1,300～1,360 檔；MI_INDEX 3,122 日', '是（按日期抓全市場）', '檔名＝查詢日；2026-10/11 GitHub Actions 回溯抓官方按日端點', '是（月快照、2014 起分母）', '只到 2014-01；日頻只有 MI_INDEX'),
 ('wenchiehlee/Python-Actions.FinMind', '495cb78e', 'MARGN', 'FinMind 個股融資融券（含現償、使用率）', '2021-01-04～2026-11-04', '126 檔', '否', '內容日期', '否', '只有 126 檔、2021 起'),
 ('voidful/tw-institutional-stocker', 'e10066f3', 'QFIIS', '發行股數、外資股數、比率', '2025-10-07～2026-10-21（o2 已用）', '約 1,088 檔', '是', '內容日期', '是（本機副本，日頻 2025-10 起）', '太短'),
 ('twcs-analysis/tw-institutional-stocker', '1d28e342', 'QFIIS', '同 voidful（fork）', '2025-10～2025-12', '—', '—', '—', '否', 'voidful 的 fork'),
 ('oid87/PersonalFiance', '4465a42d', '市場層', '大盤融資餘額（億元、張）、融券（FinMind TaiwanStockTotalMarginPurchaseShortSale）；大盤維持率重建 2022-12 起', '2008-01-02～2026-11-04', '市場總額', '不適用', '內容日期', '否（只記錄；M1 市場連鎖條件可用，但 2006～2007 缺）', '非個股；FinMind 衍生'),
 ('willychen0924/tw-margin-rate', '5a85f2af', '市場層', '大盤融資金額', '2017-07-03 起', '市場總額', '不適用', '內容日期', '否', '太短'),
 ('dylanlu0604-dot/taiwan-margin-maintenance-rate', '4f56aff3', 'MAINT', '大盤維持率（推估）', '近年', '市場', '不適用', '內容日期', '否', '非官方推估、非個股'),
 ('3B132016/tws', 'e3835b1d', 'MARGN', '個股融資、外資持股（張、比例）', '2021-09～2024-11', '112 檔', '否', '內容日期', '否', '檔數太少、太短'),
 ('acronhuang/TW_STOCK', '11f76132', '—', '只有程式（Mongo）', '—', '—', '—', '—', '否', '無資料檔'),
 ('Jarry6304/StockHelper4me', 'c5a0a166', '—', '只有程式', '—', '—', '—', '—', '否', '無資料檔'),
 ('jerry102102102/FIN_mind_data', 'e06a0d15', '—', '只有 FinMind 抓取程式與 venv', '—', '—', '—', '—', '否', '無資料檔'),
 ('galaxyh/twstock-to-db', '369e19e2', '—', '只有程式', '—', '—', '—', '—', '否', '無資料檔'),
 ('kaija/taiwan_stockloader', '637244c3', '價格', '日 K（stock-parser/data）', '2015 前後', '—', '—', '—', '否', '無融資／外資持股'),
 ('olycats/twse', '5b2f3acf', '價格', 'MI_5MINS、FMTQIK', '—', '—', '—', '—', '否', '無融資／外資持股'),
 ('williamlikesweet/twstock_database', '50f1ca94', '—', '公司基本資料 1 檔', '—', '—', '—', '—', '否', '無'),
 ('voidful/tw_stocker', 'facb2be9', '價格', '個股價格檔 1,214 個', '—', '—', '—', '—', '否', '無融資／外資持股'),
 ('end9981/tw_stocker', '4b405433', '價格', '同上（fork＋策略）', '—', '—', '—', '—', '否', '無'),
 ('appr1ciat1/tst_wocker', 'f4997f88', '價格', '同上（fork）', '—', '—', '—', '—', '否', '無'),
 ('chenyicheng2008/TWSTOCK', 'e21596fd', '價格', 'daily/weekly 價格', '—', '—', '—', '—', '否', '無'),
 ('jieterli/jtl-data', '7205d058', '籌碼', 'chips/（2026 前瞻）', '近年', '—', '—', '—', '否', '前瞻、太短'),
 ('Yuan-Hsi 以外的 TEJ 範例（YiMin1117/monday、JokerRW/ETF_python、wayne70211/FinLookup）', '—', 'MARGN/QFIIS', 'FinMind 範例單檔', '2019～2022', '1～3 檔', '否', '內容日期', '否', '單檔範例'),
 ('gsinvest005/margin-dashboard', '92c10c27', '—', '只有網頁', '—', '—', '—', '—', '否', '無'),
 ('0301kenny/wfnt-snapshot-data', '53bd579d', '—', '只有程式與報告', '—', '—', '—', '—', '否', '無'),
 ('penguin72487/stockAgent', '0551f6d2', '—', '只有下載程式', '—', '—', '—', '—', '否', '無資料檔'),
 ('Justinsheep/stock-chips', '41871dbe', '價格/營收', 'data/prices、revenue', '近年', '—', '—', '—', '否', '無融資'),
 ('david26984741-cell/tw-cb-data', 'b4c0a6da', '可轉債', 'CB 報價 2007 起', '—', '—', '—', '—', '否', '不是融資／外資持股'),
 ('ntuaha/TWFS', '32387e24', '總體', '金融統計', '—', '—', '—', '—', '否', '不是個股'),
 ('hh4832/taiwan-margin-short-0050-research', '88d9f267', '—', 'notebook', '—', '—', '—', '—', '否', '無資料檔'),
 ('BlackBoxOP/tw_stock（本機副本，o2）', '5074c181', 'MARGN', '融資餘額、前日餘額、融券各欄', '2024-01-02～2026-10-27', '約 1,069 檔', '是', '內容日期', '是（本機只讀）', '—'),
 ('tw-stock-screener extras/margin_hist（本機）', '—', 'MARGN', '融資餘額、增減、融券、使用率', '2025-09-24 起', '—', '是', '內容日期', '是（本機只讀）', '—'),
]
with open(HERE / 'raw' / 'search_log.jsonl', 'a', encoding='utf-8') as f:
    for p in probes:
        f.write(json.dumps({'ts': now, 'kind': 'probe', 'repo': p[0], 'commit': p[1], 'type': p[2], 'fields': p[3], 'range': p[4], 'rows': p[5], 'delisted': p[6], 'date_basis': p[7], 'used': p[8], 'reason': p[9]}, ensure_ascii=False) + '\n')
gate = json.load(open(HERE / 'data' / 'gate.json')); cons = json.load(open(HERE / 'data' / 'consistency.json'))
miss = json.load(open(HERE / 'data' / 'missing_counts.json')); shc = json.load(open(HERE / 'data' / 'shares_check.json'))
nq = sum(1 for l in open(HERE / 'raw' / 'search_log.jsonl') if json.loads(l).get('kind') in ('code', 'repo'))
explore = {'note': '盤點階段只看資料品質，不看任何報酬或報酬相關統計', 'n_search_queries': nq, 'n_repos_probed': len(probes), 'inventory': [dict(zip(['repo', 'commit', 'type', 'fields', 'range', 'rows', 'delisted', 'date_basis', 'used', 'reason'], p)) for p in probes],
           'semantics_finding': 'FinBrain margin_buy＝融資買進−融資賣出（與 TEJ 買進−賣出完全一致 99.90%；與 TEJ Δ餘額一致只 69.2%），不是餘額增減，累加會漏掉現金償還而漂移',
           'revision_finding': 'TEJ 存活股餘額與官方按日端點（2026 回溯抓）在 2014～2024 月快照上一致 98.42%；不一致集中在少數日期（2021 年 570 筆、2019 年 509 筆，中位比 0.99996），像官方事後更正或 TEJ 修正，T 當時公布的是哪個值無法判定',
           'consistency': cons, 'shares_check_2014plus': shc}
json.dump(explore, open(HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1)
mv = gate['MARGN']['versions']
iters = {'n_versions': 3, 'rule': '取各閘門通過數最多；同分取最簡單（凍結於 GATE_SPEC）',
 'versions': {v: {'definition': json.load(open(HERE / 'GATE_SPEC.json'))['build_iterations']['planned'][v], 'n_pass_of_10': mv[v]['n_pass_of_10'], 'verdict': mv[v]['verdict_raw'],
                  'coverage': {s: [mv[v]['segments'][s]['cov_full'], mv[v]['segments'][s]['cov_pool300'], mv[v]['segments'][s]['cov_delisted']] for s in mv[v]['segments']}, 'CONSIST': mv[v]['CONSIST']} for v in mv},
 'selected': gate['MARGN']['selected'],
 'implementation_notes': ['建構中發現 FinBrain 欄位語意是淨買賣（不是餘額增減）；V2/V3 仍照凍結規格建出並計分，由一致性副閘門（FinBrain Δ 對 TEJ Δ 69.2% < 99.9%、餘額出現負值）否決，沒有改規格',
                          '一致性檢查實作修正（閘門計算前）：TEJ 不等式與 BlackBoxOP 連續性只比「前一交易日」，避免跨缺漏日比較；結果 99.70% 與 99.30%',
                          '選版：V2 以 5/10 勝 V1 的 4/10，但 V2 多出的覆蓋來自被一致性否決的衍生值；三版判定都是「不通過」，選哪一版不影響結論'],
 'amend': []}
json.dump(iters, open(HERE / 'iterations.json', 'w'), ensure_ascii=False, indent=1)
res = {'created_utc': now, 'gate_spec_sha256': gate['gate_spec_sha256'], 'gate_json_sha256': sha(HERE / 'data' / 'gate.json'),
 'verdicts': {'MARGN（MI_MARGN 個股融資餘額）': gate['MARGN']['verdict'], 'MAINT（維持率相關欄位）': gate['MAINT']['verdict'], 'QFIIS（MI_QFIIS 外資持股比率）': gate['QFIIS']['verdict']},
 'selected_build': gate['MARGN']['selected'], 'missing_counts_V1': miss,
 'qfiis_monthly_variant_descriptive': gate['QFIIS']['monthly_variant_descriptive'],
 'prereg_draft_written': False, 'prereg_note': '三種資料都沒有「通過」，依 GATE_SPEC 不寫 PREREG_DRAFT.json；本題沒有計算任何報酬',
 'predictions': {
   'P1_MI_MARGN_2006_full_archive_exists': '否：沒有任何公開 repo 有 2006 起含下市股的完整個股融資餘額。最接近的是 FinBrain（2007-04 起、含下市，但只有淨買賣）與 TEJ 存活股（含 2006，但無下市股）',
   'P2_MI_QFIIS_2006_archive': '否：最早只到 2014-01（每月 1 日快照）；日頻只有 2025-10 起。2006～2013 段完全沒有',
   'P3_official_per_stock_maintenance': '成立：0 個來源',
   'P4_identity_99.9': '不成立：官方（BlackBoxOP）前日餘額連續性 99.30%，TEJ Δ餘額 ≤ 買進−賣出 99.70%；與本機重疊逐格一致 ≥ 99% 成立（TEJ 對 BlackBoxOP 99.24%、官方月快照對 BlackBoxOP 與 margin_hist 都 100%）',
   'P5_shares_logdiff_le_0.1pct': '2014 起存活股成立（中位 0.0001%，p90 0.021%）；2006～2013 無 MI_QFIIS，o13 D1 無法重算；下市股只有 118 對可比，77 檔下市股沒有 TEJ 錨點',
   'P6_2006_2012_worse_and_crisis_window': '成立：MARGN V1 覆蓋 S1 87.7%／S2 93.2%／S3 97.7%；2008-09～2009-03 為 87.4%，但真正的缺口是下市股（S1 只 6.4%），不是停牌'}}
json.dump(res, open(HERE / 'results.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res['verdicts'], ensure_ascii=False), nq, len(probes))

#!/usr/bin/env python3
"""disposal_v2 事前登記：在算任何報酬之前把 PREREG 寫進 results.json（含 sha256 與 UTC 時間戳）。檔案已有 PREREG 就拒絕覆寫。"""
import json, hashlib, datetime, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'results.json'

PREREG = {
 'title': '處置股策略再驗證（disposal_v2）事前登記',
 'data': {
  'TWSE': '共用面板（rlib 讀取；adj/ret/value/close_raw），進場日 2021-10-01～2026-10-22；disposal_5y 最後公告 2026-10-05；預定出場日超過面板最後一日者剔除（未完成）。',
  'TPEX': 'history.csv.gz（2025-09-24 起），用 disposal_reconcile/rlib.load_hist_market（收盤/參考價報酬、60 日均值 min_periods 20）。',
  'gap': '進場、出場落在 2024-01-02～02-14 或持有跨越該缺口 → 剔除。',
 },
 'official_rules_V_OFF': {
  'R1_chain': '依 disposal_today.py：同檔依 start 排序，start ≤ 上一筆「採用」記錄的 end + 3 個日曆日 → 不採用（再處置／重疊只做第一筆）；採用的記錄持有到該記錄的 end。不採用的記錄不開倉，即使第一筆沒買到。',
  'R2_bear': '進場日前一交易日加權指數收盤 < 最近 200 個收盤平均（含該日）→ 不開新倉。',
  'R3_entry': '進場日（start，非交易日順延）收盤報酬 ≥ 9% 或無價 → 不買。',
  'R4_liq': 'BRIEF 流動性：進場前一交易日 60 日均成交值 ≥ 1,000 萬且收盤 ≥ 10 → 才買。',
  'R5_exit': '預定出場日 ret ≤ −9% 或無價 → 延到第一個 ret > −9% 且有價日收盤賣。',
  'R6_cost': '來回 0.8%；屬「第二次處置」（前 30 個交易日內同檔有處置記錄結束）1.2%。',
  'R7_slots': 'K=10；每筆＝min(部位淨值/10, 5%×進場前一日成交值, 可用閒置)；滿槽跳過；同日依公告日、檔案順序排隊；閒置持 0050（買 0.0855%、賣 0.0855%＋稅 0.1%）為主，另報閒置現金；起始 100 萬。',
  'variants_reported_not_chosen': ['V_TOOL：R1～R3、R5～R7，不加 R4（與 disposal_today.py 字面一致）', 'V_NOBEAR：拿掉 R2', 'V_RECON：disposal_reconcile 正式版（merge_known 延長、無空頭開關）'],
 },
 'measures': {
  'primary': 'net_ab＝個股含息報酬 − 上市（上櫃用上櫃）流動性合格等權（前一日判定）− 成本',
  'also': ['raw_net＝原始報酬 − 成本', 'net_vs0050＝原始 − 0050 − 成本'],
  't': 't_month：同進場月份平均後算 t（淨額）。另報勝率、中位數。',
  'event_level': '逐筆統計用所有合格事件（不受槽位限制）；另報 10 槽實際成交的逐筆。',
 },
 'periods': {'sel': '2021-10-01～2023-12-31', 'val': '2024-01-01～2026-10-22', 'last12': '進場 2025-10-23～2026-10-22',
             'pre_change': '進場 < 2026-08-10', 'post_change': '進場 ≥ 2026-08-10'},
 'a_groups': {
  'second': '第一次／第二次（30 個交易日定義，同 R6）',
  'match_interval_inferred': '推定撮合間隔：進場 < 2026-08-10 且第一次 → 5 分；< 2026-08-10 且第二次 → 20 分；≥ 2026-08-10 → 2 分（資料沒有逐筆處置措施，依制度推定）',
  'length': '預定處置期交易日數（加權指數日曆，start～end 含頭尾）：5、6～7、8～9、10、11～12、≥13',
  'regime': '制度變更前／後（進場日 2026-08-10 為界）',
  'run6': '進場前 6 日漲幅＝adj[進場前一日]/adj[進場前 7 日]−1；依 TWSE 正式版事件全體的五分位切點（不看報酬）分 5 組',
  'market': '上市／上櫃，同窗格 2025-09-24～2026-10-22',
  'stats': '每組 n、淨平均、勝率、中位數、t_month；sel／val／全期分開',
 },
 'b_curve': {
  'def': '每筆正式版事件（上市＋上櫃），從進場收盤到第 k 個交易日收盤的累積異常（毛額、不扣成本、不含延後出場），k=1..N（N＝預定出場日與進場日的交易日差）。',
  'samples': ['全期 N=9（10 日處置）', '最近 12 個月 N=9', '最近 12 個月全部', '制度變更後 N=4（5 日處置）'],
  'test': 'early＝前 floor(N/2) 天累積，late＝其餘天累積。「獲利集中在末段」成立：late 平均 > early 平均、late > 0 且 late 的 t_month ≥ 2；只有 late > early 且 > 0 → 部分成立；否則不成立。另報「提早 2 天出場」與「只抱前一半」。',
 },
 'c_G2': {
  'def': 'G2＝近 3 月營收年增率 < 0 且 收盤/252 日最高 < 0.8（門檻固定，不調）。資訊日＝進場前一交易日 d；最新可用營收月＝d 的上一月（d 日 ≥ 11）或上兩月（d 日 ≤ 10）；年增率＝最近 3 個月營收合計／去年同 3 個月合計 −1（rev_5y，6 個月都要有值）；52 週比值＝adj[d]/過去 252 個有效列（有值檔數 ≥ 300 的日子）adj 最高。缺值 → 視為非 G2（保留），另報筆數。只做上市（上櫃沒有 252 日價格）。',
  'periods': '挑選 2021-01～2023-12（實際事件從 2021-10）、驗證 2024-01～2026-10',
  'pass_rule': '兩期都要：G2 淨平均 < 非 G2 淨平均、且 G2 淨平均 < 0；驗證期另需「G2 虛擬變數對 net_ab 的迴歸、月份群聚標準誤」t ≤ −2。通過 → SOP 採用；只挑選期方向對 → 待觀察（不採用）；否則不採用。',
 },
 'd_combo': {
  'main': 'exec_rebuild/main_nav.parquet 的 M2_n10_slip0.001（2021-03-11～2026-10-12）',
  'disposal': 'V_OFF 上市 10 槽、閒置 0050；2021-10-01 以前處置部位整筆持 0050',
  'rebalance': '每月第一個日期 ≥ 11 日的交易日收盤（同 exec_rebuild 換股日）；主策略部位增減以 買 0.0855%+0.1%／賣 0.0855%+0.3%+0.1% 計，處置部位只動閒置資金',
  'weights': '80/20 為主；90/10、70/30、100/0 只報告不挑',
  'metrics': '年化、逐年、日頻最大回撤、月窗（換股日→換股日）對 0050 超額 t；全期、2021-10 起、挑選 2021-03～2023、驗證 2024～2026-10',
 },
 'e_cost': '來回 0.8%、1.2%、2.0%（不分第一次／第二次）：逐筆（sel/val/last12/post）、10 槽、80/20。',
 'verdict_rule': {
  '可用': 'sel 與 val 的 net_ab t_month 都 ≥ 2，且最近 12 個月淨平均 > 0，且成本 1.2% 時 val t_month ≥ 2',
  '待觀察': '不符可用，但 val t_month ≥ 2 且最近 12 個月淨平均 > 0',
  '不可用': '其他',
  'cap': '可用 → 上限 30%；待觀察 → 20%，但若「制度變更後淨平均 ≤ 0」或「最近 12 個月末段集中不成立」→ 10%；不可用 → 0%',
 },
 'notes': ['程式全部 python3 -I -B；只寫入本資料夾', '網頁只用 WebSearch 摘要（WebFetch/curl 被擋），存在 raw/web_search_log.txt，視為不可信資料'],
}

if __name__ == '__main__':
    res = json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    if 'PREREG' in res:
        sys.exit('PREREG 已存在，拒絕覆寫')
    txt = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
    res['PREREG'] = PREREG
    res['PREREG_sha256'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
    res['PREREG_timestamp_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(res['PREREG_sha256'], res['PREREG_timestamp_utc'])

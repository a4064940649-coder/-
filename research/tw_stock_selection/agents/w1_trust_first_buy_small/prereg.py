#!/usr/bin/env python3
"""事前設定（在看任何回測結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '投信首買／投信認養中小型股（含投信持股比 3%~15% 變體）',
 'sources': ['https://www.cmoney.tw/notes/note-detail.aspx?nid=134188', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=24756',
             'https://www.cmoney.tw/app/itemcontent.aspx?id=1551', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=11147',
             'https://www.moneyweekly.com.tw/Magazine/Info/理財周刊/22941'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對上市等權異常報酬、t 按進場月份分群）；另報 20 槽等額組合（同時最多 20 檔）',
 'data': {
   'trust': 'extras/inst_5y.csv.gz 的 trust 欄（投信買賣超，單位張），2021-10-08~2026-10-07；某日某股無列＝0。訊號在 inst 交易日曆上計算',
   'prices': 'scratchpad/panel 上市日線；2018-01-13 類「有值檔數 < 300」的日子剔除（本期間無）；flag 5 的價格視為缺值。2024-01-02~2024-02-14 面板無資料：此期間無法進出場（訊號隔日無開盤 → 放棄；出場遇缺口 → 2024-02-15 開盤）',
   'shares': 'extras/shares_hist.csv 每檔取最新值（2026-10 snap，2025-09 起為回填同值）＝ 用 2026 年股數回推 2022~2025 市值與 0.1% 門檻，有輕微前視（增資/減資/股票股利），結果須註明；面板中無股數者（9 檔，多為 2024~2026 下市）不進主股池',
   'market_scope': '教學股池為上市＋上櫃，但上櫃股價只有 2025-09-24 起（history.csv.gz），無法做 2022~2026 回測 → 只測上市股（重大限制，FINDINGS 註明）',
 },
 'universe_t': '上市普通股（4 碼數字、非 0 開頭 ETF、非 91 開頭 DR）；t 日 60 日均成交值（上市後停牌日記 0，面板日曆滾動）>= NT$1,000 萬；原始收盤 >= 10 元；flag ∈ {0,1,2}；市值＝原始收盤 × 股數 < NT$150 億（PRIMARY）',
 'size_proxy_variant': '市值條件改為：t 日在流動性池中 60 日均成交值排名後 50%（不用股數）；0.1% 門檻仍用股數（無股數者只用 300 張）',
 'signal_first_buy_t_close': {
   '1_no_prior_buy': 't-62..t-3 共 60 個 inst 交易日中，投信買超（trust > 0）日數 = 0（允許賣超或 0）',
   '2_streak3': 't-2、t-1、t 三日投信皆買超（trust > 0）；配合條件 1，t-3 非買超 → 今天恰為連續買超第 3 天',
   '3_size': '3 日累計投信買超張數 >= min(0.1% × 流通股數(張), 300 張)',
   'signal_start': '訊號日 >= 2022-01-03（inst 資料從 2021-10-08 起，需 63 日回看）',
 },
 'signal_adopt_t_close': {
   'ratio': '估算投信持股比 r_t ＝ 2021-10-08 起累計投信買賣超張數 ÷ 股數(張)（2021-10 以前既有持股未知；教學說從 2021 起累計，資料實際從 2021-10 起）',
   'entry': 'r_t > 3% 且 r_{t-1} <= 3% 且 min(r_{t-20..t-1}) < 1%（20 日內由 <1% 升到 >3%），且 t 日符合股池',
   'signal_start': '訊號日 >= 2022-01-03（同主規則；累計起點太近的 2021 年不用）',
 },
 'execution': {
   'entry': 't+1（inst 日曆下一交易日，必須是面板日且有開盤價）開盤買（還原開盤＝open×adj/close_raw）；開盤漲停（開盤/前收−1 >= 9.4%）→ 放棄不遞補；無開盤 → 放棄',
   'dedupe': '同一檔持有中（尚未到出場開盤日）的新訊號忽略',
   'exit_first_buy': '從進場日 e 起，第一個 s 使 s-2、s-1、s 三日皆 >= e 且投信皆賣超（trust < 0）→ s+1 開盤賣；否則最長持有 60 個 inst 交易日：e+60 開盤賣',
   'exit_adopt': '從 e 起第一個 s 使 r_s > 15%，或 s-4..s 五日皆 >= e 且投信皆賣超 → s+1 開盤賣；教學未給最長持有，事前加安全上限 250 個 inst 交易日',
   'exit_price': '出場日開盤；停牌/缺口 → 之後第一個開盤；已下市 → 最後收盤',
   'censor': 'inst 資料 2026-10-07 結束：仍持有者於 2026-10-08 開盤平倉，標記 censored，事件統計照算',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'FB_MCAP：投信首買、市值 < 150 億、出場＝連 3 賣或 60 日',
   'reported_not_for_verdict': ['FB_PROXY：首買、60 日均值後 50% 代理中小型', 'FB_HOLD60：首買、市值 < 150 億、只用固定 60 日出場（看投信賣超出場的貢獻）',
                                'AD_MCAP：投信認養 1%→3%、市值 < 150 億、出場 >15% 或連 5 賣', 'AD_PROXY：認養、代理中小型'],
 },
 'portfolio_20': '每日開盤先處理出場（賣出收現金）再處理進場；同日候選依「3 日累計買超 ÷ 股數」（認養版：r_t）由大到小；空槽才買，每筆金額＝當日開盤 NAV/20（不超過現金）；不再平衡；未投入現金 0 報酬；成本按買賣金額扣',
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）上市流動性池等權（前一日符合 60 日均值/股價/flag 條件的上市普通股，開盤對開盤）→ 判定用',
   'abnormal_small_EW': 'net − 同期間上市中小型池（再加市值 < 150 億）等權 → 參考',
   'vs_0050': 'net − 0050 含息（t 收盤 → 出場前一日收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅參考',
 },
 'windows': {'select': '訊號日 2022-01-03~2023-12-31（2021 起資料的挑選期，實際因回看從 2022 起）',
             'valid': '訊號日 2024-01-01~2026-10-07', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）',
             'portfolio': '組合以日期窗計（挑選 2022-01-03~2023-12-29；驗證 2024-01-02~2026-10-22；2024~2025）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對上市等權）> 0 且 t_month >= 2，且 20 槽組合扣成本年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；兩段都不過 → 不可用；事件數 < 30 或月份 < 12 → 無法測試。變體只作穩健性參考，不得改判。',
 'known_limits': ['只測上市（上櫃無 2025-09 前股價）', '股數用 2026 年值回推（前視）', '持股比為 2021-10 起累計估算值，2022 年起才較可信',
                  '2019–2023 面板只有存活者（偏樂觀）', '2024-01-02~2024-02-14 面板無資料', 'src 3 期間開盤/收盤為 5 分 K 近似',
                  '挑選期實際只有 2 年（2022~2023）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

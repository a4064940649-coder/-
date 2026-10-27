#!/usr/bin/env python3
"""事前設定（看任何回測結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '外資連續買超＋拉回（외국인 연속 순매수 + 눌림）',
 'sources': ['https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012246166',
             'https://www.mt.co.kr/stock/2011/09/01/2011090116513055270',
             'https://newstomato.com/ReadNews.aspx?no=36162'],
 'type': '事件型：事件報酬（扣成本、對上市流動性池等權異常報酬、t 按進場月份分群）＋ 20 槽等額組合（對 0050、對同池等權）',
 'difference_vs_w1_tuyang_streak': '不要求投信同步、不要求金額遞增；改為「連買後拉回」進場時點，買超以占成交量比例標準化',
 'data': {
   'inst': 'extras/inst_5y.csv.gz foreign 欄（2021-10-08~2026-10-07，日資料，單位張）；某日某股無列＝0（不算買超）',
   'prices': 'scratchpad/panel 上市日線 adj、close_raw、open、value、flag（2021-01 起載入供暖身）；flag 5 價格視為缺值；只用普通股（4 碼數字、非 0 開頭、非 91 開頭）',
   'calendar': '交易日曆 CAL＝inst 日期 ∪ 面板日期；2024-01-02~2024-02-05 在 CAL 上但面板無資料 → 該段所有價量為 NaN',
   'gap_rule': '所有 rolling（MA5、MA20、N 日成交量、60 日均成交值）在 CAL 上以 min_periods＝窗長計算，缺口列為 NaN → 視窗不會跨過缺口（缺口後需重新累積滿窗）；有值檔數 < 300 的面板日剔除（2021 起預期無）',
   'volume': '成交量（股）＝value / close_raw；外資買超股數＝foreign（張）× 1000',
 },
 'universe_t': '上市普通股；t 日 60 日均成交值（上市後停牌日記 0，缺口列為 NaN，min_periods=60）>= NT$1,000 萬；原始收盤 >= 10 元；flag ∈ {0,1,2}',
 'signal_BASE_t_close': {
   '1_streak': 't-4..t 共 5 個 CAL 交易日 foreign > 0（每天買超 > 0 張）',
   '2_participation': 'sum_{t-4..t}(foreign×1000) >= 3% × sum_{t-4..t} 成交量（股）；5 天都必須有面板 value 與 close_raw',
   '3_pullback': '還原收盤_t < MA5_t（還原收盤 5 日均）且 還原收盤_t >= MA20_t（還原收盤 20 日均）',
   'signal_start': '訊號日 >= 2021-10-14（inst 第 5 個交易日）；最後訊號日 2026-10-07（inst 結束）',
 },
 'execution': {
   'entry': 't+1（CAL 下一交易日）開盤買，還原開盤＝open×adj/close_raw；t+1 無開盤（缺口、停牌）→ 放棄；開盤漲停（還原開盤 / 前一有值還原收盤 − 1 >= 9.4%）→ 放棄、不遞補',
   'exit_stop': '進場日 e 起（含 e 當日收盤）任一日 d 還原收盤 < MA20_d → d 的下一 CAL 交易日開盤賣（無開盤 → 之後第一個有開盤的日子）；MA20 為 NaN 的日子（如缺口後暖身期）不觸發',
   'exit_time': '否則持滿 H=10 個 CAL 交易日：e+H 日開盤賣（無開盤 → 之後第一個開盤）；兩者取較早者',
   'censor': 'CAL/面板結束（2026-10-22）仍未出場 → 最後一個收盤平倉，標 censored',
   'dedupe': '同一檔持有中（出場日前）的新訊號忽略；各變體獨立',
   'limit_down': '不模擬跌停賣不掉（exlib 也未模擬），列為限制',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'BASE：連買 5 日、參與率 3%、拉回條件、H=10（MA20 停損出場）',
   'sensitivity_reported_not_for_verdict': {
     'S3': '連買 3 日（參與率視窗也改 3 日）', 'S7': '連買 7 日（參與率視窗也改 7 日）',
     'P2': '參與率門檻 2%', 'P5': '參與率門檻 5%',
     'NOPB': '拿掉拉回條件（純連買＋參與率）',
     'H5': '持有上限 5 日', 'H20': '持有上限 20 日'},
   'note': '每個敏感度只改一個參數，其餘同 BASE；MA20 停損在所有變體都保留',
 },
 'portfolio_20': '每日開盤先處理出場（賣出收現金）再處理進場；同日候選依參與率（N 日外資買超股數 / N 日成交量）由大到小；空槽才買，每筆＝當日開盤 NAV/20（不超過現金）；不再平衡；現金 0 報酬；統計只在面板日上計算（缺口後第一天的報酬含整段缺口）',
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）上市流動性池等權（前一面板日符合股池，開盤對開盤，面板日曆，單日報酬截 [-50%,100%]）→ 判定用',
   'vs_0050': 'net − 0050 含息（訊號日收盤 → 出場前一日收盤）參考',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'windows': {'select': '訊號 2021-10-14~2023-12-31（2021 起資料的挑選期）', 'valid': '訊號 2024-01-01~2026-10-07',
             'valid_2024_2025': '訊號 2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（參考）',
             'price_split_2005_2019': '不適用：外資日資料 2021-10 才有'},
 'verdict_rule': {
   'segment_pass': '事件數 >= 30 且月份 >= 12；扣成本淨異常（對同池等權）> 0 且 t_month >= 2；且 20 槽組合扣成本年化 >= 0050 同期年化',
   'grade': '挑選期與驗證期都過＝可用；一段＝待觀察；零段＝不可用；兩段都資料不足＝無法測試',
   'overall': '整體判定＝PRIMARY（BASE）；敏感度只作穩健性參考，不得升級判定',
 },
 'known_limits': ['只測上市（上櫃無 2025-09 前股價）', '2019–2023 面板只有存活者（偏樂觀）',
                  '2024-01-02~2024-02-14 面板無資料，缺口後 60 日均成交值需重新累積 → 約 2024-05 中前驗證期無訊號',
                  'src 3 期間（2024-02-15~2025-09-23）開盤/收盤為 5 分 K 近似', '挑選期只有 2021-10~2023-12 約 2.2 年',
                  '3% 參與率、MA5/MA20、10 日為研究中心設定或常見預設，非教學原文精確數字'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

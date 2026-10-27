#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '均線多頭排列且均線同步上揚（月換股）＋日頻進出版',
 'sources': ['https://blog.wistock.ai/?p=2274', 'https://hao.cnyes.com/post/10212', 'https://www.wantgoo.com/blog/67867/post/339'],
 'source_access_note': '規則依任務敘述；教學未寫明的細節用最常見寫法，於下方固定，看結果後不改。',
 'type': '選股型（月換股）為主：組合報告；日頻版本為事件型：事件報酬＋日曆組合',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling。flag 5 價格視為缺值。',
   'universe': '上市普通股（4 碼、排除 91xx 存託憑證）；判斷日 60 日均成交值（上市後停牌日記 0、需滿 60 日）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}、當日有收盤。',
   'price_for_ma': '均線與收盤一律用 adj（含息還原），避免除權息造成假的排列破壞；教學用原始收盤，差異只在除權息附近。',
 },
 'signal': {
   'SHORT（短線版）': 'MA5 > MA10 > MA20 > MA60，且 MA5、MA10、MA20、MA60 皆 MA_n,t > MA_n,t−5，且 收盤 > MA20',
   'LONG（中長線版）': 'MA20 > MA60 > MA120，且 MA20、MA60、MA120 皆 MA_n,t > MA_n,t−5，且 收盤 > MA20',
   'rising_lag': '5 個交易日（任務給定）',
 },
 'monthly_execution': {
   'calendar': '每月最後一個交易日 d（剔除稀疏日後的日曆）收盤判斷；d+1 開盤成交（M2：exec_rebuild/exlib.sim_portfolio），持有到下個月最後交易日的隔日開盤換股。最後一期持有到資料最後一日（2026-10-22）收盤。',
   'tradable': 'd+1 有開盤價與收盤；d+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）且未持有 → 買不到：ALL 版直接剔除；TOP30 版由下一名遞補。已持有者照常保留。',
   'ALL': '全部符合者等權（無上限）',
   'TOP30': '符合 > 30 檔時依 60 日報酬（adj_d / adj_{d−60} − 1）取前 30；<= 30 檔時全取',
   'empty': '某月沒有任何符合（可成交）股票 → 該月持有現金（0 報酬）',
   'weights_turnover_cost': '換股日調回等權；周轉用漂移後市值；成本 exlib.cost_rates：買 0.1425%×0.6＋滑價 0.1%；賣 0.1425%×0.6＋稅 0.3%＋滑價 0.1%；壓力 買 0.5%＋賣 0.5%；另報毛報酬',
 },
 'daily_version': {
   'entry': '訊號（SHORT 或 LONG 完整條件）t 日成立且 t−1 不成立（首次轉為多頭排列）→ t+1 開盤買；t+1 無開盤價或開盤漲停 → 放棄',
   'exit_SHORT': 's >= e 第一個 MA5 < MA10 或 收盤 < MA20 → s+1 開盤賣',
   'exit_LONG': '（類推，事前固定）s >= e 第一個 MA20 < MA60 或 收盤 < MA20 → s+1 開盤賣',
   'cap': '安全上限 250 交易日',
   'dedupe': '持有中出現的新訊號忽略',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'event_stats': 'net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 同期間（進場開盤→出場開盤）同池等權（前一日股池、開盤對開盤）；t 按進場月份分群；另報壓力成本、毛、對 0050',
   'calendar_portfolio': '每日持有所有在場事件等權、進出各按 1/n 扣成本、無持股日現金',
 },
 'variants': {
   'PRIMARY': 'SHORT_ALL（短線版、全部符合者等權、月換股）',
   'all_reported': '月換股 {SHORT, LONG} × {ALL, TOP30}；日頻 {SHORT, LONG}；全部報告，其他只作穩健性參考，不得用來改判',
 },
 'benchmarks': {
   'EW': '同池等權：同一月換股日、同一股池（d 符合股池且 d+1 可成交）全部等權，以同一引擎模擬、不扣成本（毛）——比扣成本的策略嚴格',
   '0050': '0050 含息（finlab-txd etf_0050.csv）',
 },
 'stats': '年化（實際天數）、日頻最大回撤、波動、對 0050 與同池等權的「日曆月」超額報酬 t（btlib.tstat）、逐年報酬、平均持股數、周轉；挑選期與驗證期分開',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31'},
 'factor_overlap': '每個換股日，對股池計算：符合訊號 0/1 與 12-1 動能（adj_{d−21}/adj_{d−252}−1）、52 週高點比（adj_d / 252 日最高 adj）的橫斷面 Spearman 相關（時間平均）；Fama–MacBeth：下月報酬（d+1 開盤→下期 d+1 開盤）對 [訊號 0/1, 動能百分位, 52 週高百分位] 回歸，報訊號係數（殘差報酬）的平均與 t（月份分群），挑選期／驗證期分開。',
 'verdict_rule': 'PRIMARY（SHORT_ALL）在挑選期與驗證期都要：扣成本（滑價 0.1%）後對同池等權月超額 t >= 2（且平均超額 > 0），且淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['面板無最低價、無成交量欄', '2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨過缺口）', 'src 3 期間（2024-02-15~2025-09-23）收盤為 13:20 近似'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

#!/usr/bin/env python3
"""事前設定（看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'PSY12 心理線 12 日穿越（Monex 版為主；楽天、moomoo 對照；加權指數 PSY12 擇時 0050）',
 'sources': ['https://info.monex.co.jp/technical-analysis/indicators/007.html', 'https://www.rakuten-sec.co.jp/web/fop/special/technical_analysis/chart_guide06/',
             'https://www.oanda.jp/lab-education/technical_analysis/psychological_line/view_rsi/', 'https://www.moomoo.com/us/hans/learn/what-is-a-psychological-line',
             'https://traders.co.jp/column/article/58'],
 'rule_text_as_given': '股池：每月底 60 日均成交值 >= 1,000 萬、股價 >= 10 元的上市股。PSY12＝近 12 日「收盤 > 前日收盤」天數 ÷12×100（平盤不算上漲）。'
                       '進場（Monex）：PSY12 由 <=25 回升到 >25（前一日 <=3 勝、當日 >=4 勝），收盤訊號、隔日開盤買，開盤漲停不買。'
                       '出場：PSY12 由 >=75 跌破到 <75 的隔日開盤賣，或持有滿 20 個交易日，或從進場價跌 10% 停損，取先到。等權、每檔同金額，最多 20 檔，訊號超過時優先買 60 日報酬最低。'
                       '對照：楽天版（PSY<=25 當天就買、>=75 就賣）、moomoo 版（PSY<=8.33 才買）。另測加權指數 PSY12 擇時 0050。成本來回 0.5%（壓力 1%）。',
 'type': '事件觸發＋固定槽位組合。兩種都報：(1) 事件報酬（全部訊號、不受槽位限制；扣成本、對同池等權異常、t 按進場月份分群）；(2) 規則字面的 20 槽組合。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值；排除 91xx。',
   'price_series': '漲跌判斷只用 adj（含息還原收盤），除權息日不會被算成下跌。up_t ＝ adj_t/adj_{t-1}−1 > 1e-7（平盤＝0 不算上漲）；停牌日（adj 缺值）該日 diff 為缺值。',
   'PSY12': 'K_t ＝ 近 12 個 diff 中 up 的天數（0..12，13 種值），需 12 個 diff 全部有值（即 t-12..t 13 個收盤都是真實值），否則 K 缺值。PSY12＝K/12×100。'
            '門檻換算：PSY<=25 ⇔ K<=3；>25 ⇔ K>=4；>=75 ⇔ K>=9；<75 ⇔ K<=8；<=8.33 ⇔ K<=1。',
   'universe': '每月最後一個交易日：60 日均成交值（上市後停牌日記 0、min_periods=60）>= NT$1,000 萬且原始收盤 >= 10 元 → 下個月整月的股池（教學寫「每月底」）。'
               '訊號日另需：t 日有真實收盤、t 日 flag ∈ {0,1,2}、[t-12, t] 無 flag 3/4/5（全市場同日 flag4 >= 50 檔的面板缺口日不算）、t 與 t+1 不在處置期間（名單 2021-09 起）。後三項是中心資料衛生慣例，非教學內容。',
 },
 'signals_t_close': {
   'monex': 'K_{t-1} <= 3 且 K_t >= 4（PRIMARY）',
   'rakuten': 'K_t <= 3（當天就買，狀態型）',
   'moomoo': 'K_t <= 1（PSY <= 8.33，狀態型）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）。t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%；2015-06-01 前 7%、之後 10%）→ 不買（事件層放棄；組合層換下一個候選）。',
   'exit_psy': 'monex：K_{s-1} >= 9 且 K_s <= 8（由 >=75 跌破）→ s+1 開盤賣。rakuten、moomoo：K_s >= 9（>=75 就賣）→ s+1 開盤賣。s 從進場日 e 收盤起算。',
   'time_stop': '持有滿 20 個交易日：若 e..e+19 收盤都未觸發其他出場，e+20 開盤賣。',
   'stop_loss': '沒有 low 欄，停損以收盤判斷：adj_s <= 0.9×進場價（還原開盤）→ s+1 開盤賣。三種出場取最早者；同日同時觸發時記為 stop>psy。',
   'sell_frictions': '預定賣出日整天鎖跌停（還原最高價/前收−1 <= −(漲跌幅−0.6%)）或無開盤價 → 順延到之後第一個可賣開盤；下市 → 最後收盤賣。',
   'dedupe': '持有中同檔新訊號忽略；賣出日當天收盤起可再觸發（狀態型版本賣出後若仍 K<=3 會再進場）。',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；另報毛報酬與壓力（買 0.5%＋賣 0.5%，來回 1%）。教學說 0.5%，以 exlib 為準（較保守）。',
 },
 'portfolio': {
   'slots': '20 槽，每筆進場金額＝當日開盤 NAV×5%（現金不足用剩餘現金，不足 1.25% NAV 不買）；最多 20 檔；不加碼、不再平衡。',
   'order': '每日開盤先賣後買；候選依訊號日 60 日報酬（adj_t/adj_{t-60}−1）由低到高，同值或缺值時 60 日均成交值大者優先；當天剛賣出的股票當天不買回。',
   'cash': '現金 0 報酬。', 'start': '2004-07 起模擬，統計從 2005-01。',
 },
 'index_version': {
   'signal': '加權指數（taiex_twii.csv 收盤，價格指數）算 K（同上定義、平盤不算漲）。monex/rakuten/moomoo 三種進出場、同樣 20 日時間出場與 0050 收盤 <= 進場價×0.9 停損。',
   'execution': '0050 只有含息收盤：t 收盤訊號 → t+1 收盤以 0050 含息價成交（無開盤價，延後一天的保守近似）。不在市場時現金 0 報酬。成本同 exlib（買 0.1855%、賣 0.4855%；壓力各 0.5%）。',
   'compare': '對 0050 買進持有：年化、日頻最大回撤、月超額 t（月窗）、在場時間比例、逐年、每年進出次數。指數版只作參考，不影響主判定。',
 },
 'variants': {'PRIMARY': 'monex 個股版（20 槽）', 'all_reported': '個股：monex、rakuten、moomoo；指數：monex、rakuten、moomoo；全部報告，只用 PRIMARY 判定。'},
 'event_stats': {
   'raw': '賣出價/進場價−1（還原）；net＝(1+raw)×(1−賣)/(1+買)−1',
   'abnormal_EW': 'net − 同期間（進場開盤→賣出開盤）同池等權報酬（同池＝當月股池，每日等權、開盤對開盤）',
   'vs_0050': 'net − 0050 含息（t 收盤→賣出前一日收盤）',
   't': '事件按進場月份平均後對月平均做 t（btlib.tstat）；t_date 只供參考',
 },
 'portfolio_stats': '20 槽組合日頻 NAV；年化、日頻最大回撤、對 0050 含息與同池等權（每月第一個 >= 11 日交易日起的月窗；同池＝流動性股池隔日開盤等權月換股，exlib.sim_portfolio 無成本）的月超額 t、逐年、平均持股數、每年訊號數／進場數。',
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'valid_survivor_2019-05_2023-12': '另報', 'valid_officialPx_only': '驗證期剔除 src 3（2024-02-15~2025-09-23）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期各自：事件淨異常（對同池等權，滑價 0.1%）> 0 且 t_month >= 2，且 20 槽組合淨年化 >= 0050 同期年化 → 該段通過；兩段都過 → 可用；一段 → 待觀察；都沒過 → 不可用。'
                 '組合對同池等權月超額 t 一併報告；事件層過但組合層 t < 2 時在結論明示（不改判定）。',
 'diagnostics_preplanned': ['每年訊號數（股池內 stock-days）、事件數、組合進場數', 'K 值在股池內的分布', '壓力成本與毛報酬', 't 收盤進場（不可成交）的隔夜跳空',
                            '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常', '候選隨機排序 10 種子', '出場原因分布', '極端報酬與 flag 3/4/5 檢查', '驗證期剔除 src 3'],
}
p = HERE / 'results.json'
assert not p.exists(), 'PREREG already written; do not overwrite'
p.write_text(json.dumps({'PREREG': PREREG}, ensure_ascii=False, indent=1), encoding='utf-8')
print('written', PREREG['written_at'])

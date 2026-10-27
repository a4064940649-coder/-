#!/usr/bin/env python3
"""事前設定：在寫回測程式、看任何結果之前寫入 results.json 的 PREREG。之後不得修改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '接近 52 週低點（Chen-Yu 2016 錨定低點 NL）＋ 52 週低點反彈（TradingView 52-Week Low Bounce）',
 'sources': ['https://acfr.aut.ac.nz/__data/assets/pdf_file/0011/56468/657406-H-Yu-Nearness-to-the-52-week-high-and-low-prices,-past-returns,-and-average-stock-returns.pdf',
             'https://vvv.quantpedia.com/?p=685', 'https://www.cxoadvisory.com/technical-trading/implications-of-52-week-highs-and-lows-for-stock-returns/',
             'https://il.tradingview.com/script/rbJpA0fU-52-Week-Low-Bounce-Strategy（本環境無法連線 tradingview，規則依任務文字）',
             'https://thesis.lib.nccu.edu.tw/detail/0957d1de0ad5560009085a56034e5d50/', 'https://international.vlex.com/vid/buy-low-and-sell-1049437958'],
 'type': '主版本＝選股型（月換股組合，以組合報告與判定）；變體 b＝事件型（事件報酬＋日曆時間組合）',
 'panel': ('scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者 → 驗證期偏樂觀）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling，'
           '模擬也只在其餘交易日進行；flag 5 的價格視為缺值；排除 91xx 存託憑證。'),
 'approximation': '面板沒有最低價：52 週低點＝過去 250 個交易日（剔除缺漏日後的列，含當日）adj 收盤最低值，至少 240 個有值；比真正盤中最低價高 → NL 偏小、比論文更早／更常判定「接近低點」（相對於真實低點，接近程度被高估）。',
 'main': {
   'calendar': '訊號日 d＝每月最後一個交易日（剔除缺漏日後）；d 收盤算訊號，d+1 開盤（還原開盤）成交，持有到下個訊號日的隔日開盤換股（月換股）。首個訊號日 2004-12 月底（2005-01 首個交易日成交）。',
   'signal': 'NL_d = adj_d / min(adj_{d-249..d})（250 列、min_periods 240）。NL 越接近 1 越接近 52 週低點。',
   'universe_d': ['上市普通股（排除 91xx）',
                  '60 日均成交值（剔除缺漏日後滾動、上市後停牌日記 0、min_periods 60）>= NT$1,000 萬；d 收盤原始價 >= 10 元；d 有 adj 與 NL',
                  '[d-20, d] 內任一天 flag ∈ {3,4,5} 剔除（全市場同日 flag 4 >= 50 檔的面板缺口日不算）',
                  'd+1 有開盤價與收盤（可成交）；股池 < 100 檔的月不交易'],
   'deciles': 'NL 由小到大排名（同值依欄位順序），十等分：D1＝最小 10%（最接近低點）、D10＝最大 10%（離低點最遠）。',
   'legs': 'D1、D10 的目標檔數＝當月該組檔數；d+1 開盤漲停（含息跳空 >= 漲停 − 0.6%，2015-06-01 前 7%、之後 10%）且目前未持有者視為買不到，由同方向下一名遞補（D1 往 NL 較大者、D10 往 NL 較小者）。D2～D9 只作十分組表（毛報酬、買不到者剔除、不遞補）。',
   'direction_rule': '方向由挑選期決定（事前寫死）：挑選期 D1 − D10 月報酬差（毛、月份）平均 > 0 → 可交易腿＝D1（論文方向）；否則＝D10。驗證期不得改方向。',
   'weights': '等權；換股時漂移後權重調回等權（exlib.sim_portfolio）。持有中停牌／下市以最後收盤 ffill，下次換股賣出。',
 },
 'variants': {
   'a_within_momentum': ('12-1 動能 MOM_d = adj_{d-21}/adj_{d-252} − 1（列數、剔除缺漏日後；須有值）。主股池再要求 MOM 有值；依 MOM 五分組（Q1 輸家～Q5 贏家），'
                         '每個動能組內依 NL 十等分；a_D1＝五個動能組各自 NL 最小 10%（round(0.1×組內檔數)）的聯集、a_D10＝各組 NL 最大 10% 的聯集；同樣遞補規則。'
                         '另報各動能組內 D1−D10 毛月差與 t，以及 NL 價差與動能價差（MOM Q5−Q1 毛）的月報酬相關係數，看是否互補（對沖動能）。'),
   'b_low_bounce': {
     'signal_t_close': ['股池：上市普通股、60 日均成交值 >= NT$1,000 萬、收盤原始價 >= 10 元、t 的 flag ∈ {0,1,2}、[t-20,t] 無 flag 3/4/5（同上例外）',
                        'LOW250_t＝adj 過去 250 列（含 t、min_periods 240）最低值；adj_t <= 1.05 × LOW250_t（收盤在 250 日最低收盤 5% 以內）',
                        'MA20＝adj 20 日簡單均線（20 個都要有值）；adj_{t-1} <= MA20_{t-1} 且 adj_t > MA20_t（當日由下往上穿越）'],
     'entry': 't+1 開盤（還原開盤）買；t+1 無開盤價或開盤漲停 → 該訊號放棄（不遞補）。同檔持有中（尚未到出場日）的新訊號忽略；出場日收盤起可再觸發。',
     'exit_first_of': ['停損：s >= e（含進場當日）任一收盤 adj_s < LOW250_t（進場前 250 日最低收盤）→ s+1 開盤賣',
                       '停利：adj_s >= 1.15 × 進場價 → s+1 開盤賣（收盤判斷；教學若為盤中限價單，本版較保守）',
                       '時間：持有 20 個交易日 → 第 e+20 列開盤賣',
                       '出場日無開盤價（停牌）→ 之後第一個開盤；已下市 → 最後收盤'],
     'diag_time_only': '同訊號、只用 20 日時間出場（無停損停利），僅作參考',
     'event_stats': 'net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 同期間（進場開盤→出場開盤）同池等權報酬（同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤）；對 0050＝net − 0050（t 收盤→出場前一日收盤）；先按進場月份平均再做 t（月份分群）；另報 t_date 僅供參考。',
     'portfolio': '日曆時間組合：每日持有所有在場事件等權（每日再平衡）、進出按 1/n 權重扣成本、空倉日現金 0；報年化、日頻最大回撤、對 0050 與同池等權（收盤對收盤）月超額 t、逐年、曝險比例。',
   },
 },
 'engine': 'exec_rebuild/exlib.py 的 sim_portfolio 與 cost_rates（不重寫）；價格陣列 AO/AC 用 exlib 的（AC_prev 已 ffill 跨過 2018-01-13）。限制：不模擬跌停賣不掉。',
 'costs': {'normal': 'cost_rates(0.001)：買 0.0855%+0.1%、賣 0.0855%+0.3% 稅+0.1%（來回約 0.67%）', 'stress': "cost_rates('stress')：買 0.5% + 賣 0.5%（來回 1%）", 'gross': '0'},
 'benchmarks': {'EW_pool': '主股池（每月）全部等權、同一換股時點（d+1 開盤）、不扣成本', '0050': '0050 含息指數（ffill 到面板日曆）'},
 'stats': '日頻 NAV：年化（實際天數）、年化波動、日頻最大回撤、逐年。t：以日曆月（月底收盤→月底收盤）報酬差做 t（每月一個觀察值＝月份分群），對 0050 與對同池等權。',
 'windows': {'select': ['2005-01-01', '2019-04-30'], 'valid': ['2019-05-01', '2026-10-31'], 'y2024_2025': ['2024-01-01', '2025-12-31'],
             'valid_survivor_only_2019-05_2023-12': '驗證期中只有存活者的一段（偏樂觀）',
             'valid_ex_src3': '驗證期剔除 2024-02-15～2025-09-23（src=3 的 13:20 近似收盤）'},
 'verdict_rule': ('以主版本「方向腿」、一般成本（滑價 0.1%）判定。某期間通過＝對同池等權月 t >= 2 且年化 >= 0050 同期年化。'
                  '挑選期與驗證期都通過 → 可用；只有一段 → 待觀察；都不通過 → 不可用。變體 a、b 各自報通過與否，只作參考，不改判（多重檢定）。'),
 'no_tuning': '250 日、十分組、5%、MA20、20 日、+15%、12-1、五分組皆為論文／教學預設；看結果後不改。',
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    f.write_text(json.dumps({'PREREG': PREREG, **d}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

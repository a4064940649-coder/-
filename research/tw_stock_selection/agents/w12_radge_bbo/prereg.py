#!/usr/bin/env python3
"""Nick Radge 布林突破（BBO：100 日、上軌 +3σ 買、下軌 −1σ 賣）事前設定：看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Nick Radge 布林突破（BBO）：100 日布林上軌（+3 標準差）收盤突破買、收盤跌破下軌（−1 標準差）賣；最多 20 檔等權；有／無大盤 200 日均線濾網',
 'sources': ['https://www.thechartist.com.au/?p=15473', 'https://www.tradingview.com/script/p06VAnFw', 'https://il.tradingview.com/script/p06VAnFw'],
 'type': '選股型（組合，最多 20 檔等權、滿倉不換股）為主報告與判定；另報事件型（不受 20 檔限制的全部訊號）供參考',
 'diff_vs_w3_bollinger_squeeze': 'w3 是 20 日、2σ、需頻寬創 6 個月新低（壓縮）後突破；本條無壓縮條件，100 日長週期、不對稱 +3σ／−1σ 的趨勢跟隨，下軌當移動停損',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市，2019–2023 只有存活者）；有值檔數 < 300 的日子（2018-01-13）整列剔除後再做任何 rolling，模擬也只在其餘交易日進行',
   'price': '布林、ROC 一律用含息還原收盤價（adj），避免除息造成假跌破；教學原文為收盤價',
   'universe': 'BRIEF 流動性池：上市普通股（4 碼、排除 91xx）；訊號日 t：60 日均成交值 >= NT$1,000 萬（mlib.V60：剔除缺漏日後滾動）、原始收盤 >= 10 元；t+1 有開盤價（無前 300 名限制）',
 },
 'signal_t_close': {
   'mid': 'SMA100_t = mean(adj_{t-99..t})（100 根須全部有值，含當日）',
   'sd': 'SD100_t = 母體標準差（ddof=0）of adj_{t-99..t}',
   'upper': 'mid + 3×SD100', 'lower': 'mid − 1×SD100',
   'entry': 'adj_t > upper_t 且 t 在流動性池（狀態型：只要當日收盤在上軌之上且未持有、有空位即可進，不要求當日才首次穿越；穿越型另做診斷）',
   'exit': 'adj_t < lower_t（下軌隨時間上移，視為移動停損）；無其他停損、無固定持有期（持有到跌破下軌）',
   'priority': '同日候選多於空位時，依 ROC(100)_t = adj_t/adj_{t-100} − 1 由高到低',
   'market_filter_B': '加權指數（taiex_twii.csv 收盤，價格指數）t 日收盤 > 其 200 日簡單均線（在指數自身日曆上計算，1999 起有資料）才允許新買進；只擋新進場，不強制出場',
   'market_filter_C_diag': '改用 0050 含息 > 其 200 日均線（診斷）',
 },
 'execution': {
   'entry': 't 收盤訊號 → t+1 開盤買（還原開盤）；t+1 開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）→ 買不到、該日放棄（隔日若仍在上軌上可再進）',
   'exit': '出場條件在收盤判斷（含進場當日收盤）→ 隔日開盤賣；隔日無開盤價（停牌）延到下一個有開盤價的日子；已下市以最後還原收盤賣出；跌停賣不掉未模擬',
   'max_positions': '最多同時 20 檔；滿倉時不換股（新訊號放棄）',
   'sizing': '每檔 = 當日開盤 NAV/20，不足用剩餘現金（現金 < 0.25 份則不買）；持有中不再平衡；現金零利率',
   'order_at_open': '先出場 → 再新進場',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%；另報毛報酬',
   'sim_start': '2004-07-01 起模擬（100 日暖身），統計自各期間起點',
 },
 'variants': {
   'A_noFilter': 'PRIMARY：無大盤濾網',
   'B_taiex200': '加權指數 > 200 日均線才新買（Radge 常用；事前寫明的主變體）',
   'diag_C_0050_200': '0050 > 200 日均線濾網（診斷）',
   'diag_D_cross_noFilter': '穿越型進場：adj_{t-1} <= upper_{t-1} 且 adj_t > upper_t（診斷）',
 },
 'reported_extra': ['訊號日收盤鎖漲停比例', '隔日開盤漲停買不到比例（事件與組合兩口徑）', '平均持有天數、出場前最大持有', '逐年', '持股檔數與投入比例'],
 'benchmarks': {
   '0050': '0050 含息（日頻 ffill 到面板日曆）',
   'EW_pool': '同池等權：每月第一個 >= 11 日交易日 d 收盤的同一流動性池全部等權，d+1 開盤換股，不扣成本（exlib.sim_portfolio）',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一觀察值、依月份分群）對 0050 與對同池等權',
 'event_stats': 'A 版全部訊號（不設 20 檔上限；同檔持有中不重複）：net＝(1+raw)(1−賣)/(1+買)−1；異常＝net − 同池（訊號日股池）同窗（進場開盤→出場開盤）等權報酬；先按進場月份平均再做 t',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'verdict_rule': '以滑價 0.1% 的組合判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。PRIMARY（A）挑選期與驗證期都通過 → 可用；只一段通過 → 待觀察；都不通過 → 不可用。B 版另給判定；若 A 不可用而 B 兩段都通過 → 整體最多升為「待觀察」（多重檢定）；若 A 待觀察而 B 可用 → 仍為待觀察。診斷版不參與判定。',
 'adaptations_vs_original': ['原書（ASX／Nasdaq 個股）搭配指數濾網與依波動決定部位大小；這裡改成等權最多 20 檔',
                             '原書可能以穿越定義進場；這裡主版用狀態型（收盤 > 上軌），穿越型為診斷'],
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 視窗跨過此缺口）',
                  'src 3 期間收盤為 13:20 近似', '93 檔 2024-02-15~03-15 缺值 → 該段 100 日布林為 NaN 約 5 個月', '跌停賣不掉未模擬（開盤即成交）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

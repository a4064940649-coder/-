#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Connors TPS（Time, Price, Scale-in）',
 'sources': ['https://wl6.wealth-lab.com/Forum/Posts/Connors-TPS-Strategy-40197',
             'https://www.quantifiedstrategies.com/tps-trading-strategy/',
             'https://tradingstrategyguides.com/tps-trading-strategy/'],
 'rule_text_as_given': '股池：BRIEF 流動性條件。訊號：收盤 > SMA(200)，且 RSI(2) 連 2 天收盤 < 25。進場：隔日開盤買第一批（該檔預算 10%）。'
                       '加碼：之後任一天收盤低於上一筆買價 → 隔日開盤依序加碼 20%、30%、40%（共 4 批 1:2:3:4）。出場：收盤 RSI(2) > 70 → 隔日開盤全賣。'
                       '濾網：持有期間收盤跌破 SMA(200) 就不再加碼（tradingstrategyguides 版）。時間停損 20 個交易日。每檔滿倉預算 10%，最多 10 檔。'
                       '加碼時點另一說為第 3、4、5 天，兩種都報。報酬以實際投入資金計算。',
 'type': '事件觸發＋固定槽位組合，兩種都報：(1) 事件報酬（不受槽位限制；以實際投入資金加權；扣成本；對同池等權異常；t 按第一批進場月份分群）；'
         '(2) 10 槽組合（年化、日頻最大回撤、對 0050 與同池等權月超額 t、逐年）。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值；排除 91xx。',
   'price_series': '指標只用 adj（含息還原收盤），adj ffill（最多 10 日）跨停牌；成交用還原開盤（開盤×當日還原因子）。訊號日、出場判斷日、加碼判斷日都必須有當日真實收盤。',
   'pool': 'BRIEF 流動性股池 LIQ：t 日 60 日均成交值（上市後停牌記 0、min_periods=60）≥ NT$1,000 萬、原始收盤 ≥ 10、flag ∈ {0,1,2}。',
   'flag_exclusion': '訊號股在 [t-4, t] 任何一天 flag ∈ {3,4,5} → 剔除（全市場同日 flag4 ≥ 50 檔的缺口日不算）。',
   'disposal_exclusion': 't 或 t+1 在處置期間 → 剔除（名單 2021-09 起才有）。',
 },
 'indicators': {
   'RSI2': 'Wilder RSI(2) on adj：AG/AL 用 ewm(alpha=1/2, adjust=False)；AL=0 且 AG>0 → 100；皆 0 → 50。（教學未指定平滑法，用最常見的 Wilder）',
   'SMA200': 'adj 200 日簡單均線（min_periods=200）',
 },
 'signal_t_close': 'LIQ_t 且 adj_t > SMA200_t 且 RSI2_t < 25 且 RSI2_{t-1} < 25（t 與 t-1 都有真實收盤），且無 flag/處置剔除。持有中同檔新訊號忽略。',
 'execution': {
   'entry': 't+1 開盤買第一批（預算 10%）。開盤漲停（2015-06-01 前 7%、之後 10%，各減 0.6% 容忍）或無開盤價 → 不買（組合改買下一候選；事件直接放棄）。',
   'scale_P': '價格版（PRIMARY）：從第一批進場日 e 的收盤起，任一天 s 收盤（adj）< 上一筆實際成交價（還原）→ s+1 開盤加下一批（20%→30%→40%）；每日最多加一批。'
              '濾網 F：s 收盤 < SMA200_s 時不加碼（之後收盤回到 SMA200 上方且仍低於上一筆買價可再加）。s 當天若出場訊號成立，出場優先、不加碼。',
   'scale_D': '天數版：第一批進場日 e 為第 1 天；第 3、4、5 個交易日（e+2、e+3、e+4）開盤依序無條件加 20%、30%、40%（前一日收盤若觸發出場則不加）。'
              '濾網 F：前一日收盤 < SMA200 時該批不加（之後批次照排程判斷）。',
   'add_frictions': '加碼日開盤漲停或無開盤 → 該次不加；P 版下次收盤仍低於上一筆買價可再試；D 版該批作廢。',
   'exit': '從 e 收盤起逐日檢查 RSI2_s > 70 → s+1 開盤全部賣出。時間停損 T20：到 e+19 收盤仍未觸發 → e+20 開盤賣（自第一批起算，最多持有 20 個交易日）。'
           '預定賣出日鎖跌停（還原最高/前收−1 ≤ −(漲跌幅−0.6%)）或無開盤 → 順延；下市 → 最後收盤賣。',
   'cost': 'exlib.cost_rates：買 0.0855%＋滑價 0.1%；賣 0.0855%＋稅 0.3%＋滑價 0.1%（每批買各自計成本）；另報毛與壓力（買 0.5%＋賣 0.5%）。',
 },
 'event_return': {
   'net_on_invested': '每筆交易報酬 = Σ賣出淨收入 / Σ各批實際投入（含買進成本）− 1（實際投入資金口徑，主統計）。',
   'on_full_budget_ref': '另報「以滿倉預算為分母」= 損益 / 預算（只作對照，會低估）。',
   'abnormal': '淨報酬 − 同期間同池等權（每批自其進場開盤至賣出開盤的 LIQ 等權開盤對開盤報酬，依各批投入資金加權）；另報對 0050（同法加權）。',
   't': '按第一批進場月份平均後 t（btlib.tstat）；t_date 只供參考。事件以訊號日歸屬區段。',
 },
 'portfolio': {
   'slots': '10 檔。新部位預算 B = 當日開盤 NAV×10%；可用現金 = 現金 − 現有部位尚未投入的預算（保留給加碼）；可用現金 < 0.25B 不開新倉，否則 B = min(B, 可用現金)。'
            '各批金額 = 批次比例×B（不超過現金）。現金 0 報酬。',
   'order': '每日開盤：先賣出、再加碼既有部位、最後開新倉。新倉候選 = 前一日收盤訊號，依 RSI2_t 由低到高（同值成交值大者優先）；開盤漲停跳過遞補。',
   'start': '2004-07 起模擬，統計從 2005-01。',
   'utilization': '另報平均實際投入比例（持股市值/NAV）。',
 },
 'variants': {
   'PRIMARY': 'P_F_T20：價格版加碼＋SMA200 加碼濾網＋20 日時間停損（滑價 0.1%）',
   'co_reported': 'D_F_T20：天數版（第 3/4/5 天）加碼，其他同 PRIMARY；兩種加碼時點都完整報告與判定，但總判定以 PRIMARY 為準（D 版判定並列）。',
   'all_reported': '{P, D} × {F（濾網）, NF（無加碼濾網）} × {T20, NT（無時間停損）} = 8 組，全部報告；非主版本只作穩健性，不得改判。',
   'diagnostics': ['ALL：第一批即 100% 預算、不加碼（同訊號、同出場），衡量分批加碼本身的貢獻', 'ONE：只買 10% 第一批、不加碼',
                   '加碼批數分布、各批貢獻', '以滿倉預算為分母的報酬（說明低估）', 't 收盤進場（不可成交）vs t+1 開盤', '挑選期子段 2005-2011 / 2012-2019-04',
                   '驗證期剔除 src3', '極端報酬與 flag 3/4/5 檢查', '逐年事件異常'],
 },
 'benchmarks': {'EW_liq': '同池＝LIQ。事件層：每日等權開盤對開盤；組合層：每月第一個 ≥ 11 日交易日以收盤池、隔日開盤等權月換股、無成本。', '0050': '含息 0050。'},
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31（另報）',
             'others': '挑選子段 2005-2011 / 2012-2019-04、驗證存活者段 2019-05~2023-12、驗證剔除 src3、2021-2023 / 2024-2026（參考）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期各自：事件淨異常（對 LIQ 等權，滑價 0.1%，實際投入口徑）> 0 且 t_month ≥ 2，且 10 槽組合淨年化 ≥ 0050 同期 → 該段通過；'
                 '兩段都過 → 可用；一段 → 待觀察；都沒過 → 不可用。任一段事件 < 100 筆或有事件月份 < 24 → 該段資料不足；兩段都不足 → 無法測試。組合對同池等權月超額 t 一併報告。',
 'known_limits': ['面板無最低價、無成交量欄（只用開盤與收盤，符合規則）', '2019–2023 面板只有存活者（逢低加碼偏樂觀）', '2024-01-02~2024-02-14 無資料',
                  'src 3 收盤為 13:20 近似', '處置名單 2021-09 起才有', '已測相關：w3_connors_rsi2_sma5（RSI2<10, SMA5 出場）、w3_connors_crsi、w3_connors_double7'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '量先價行：5 日均量黃金交叉 20 日均量、價格 5 日線尚未突破 20 日線',
 'sources': ['https://www.wantgoo.com/blog/67867/post/180', 'https://n.yam.com/Article/20120709613685',
             'https://forum.xq.com.tw/thread/obv-5%e6%97%a5%e5%b9%b3%e5%9d%87%e7%b7%9a%e4%b8%8a%e7%a9%bf20%e6%97%a5%e5%b9%b3%e5%9d%87%e7%b7%9a'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件每筆等額）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '上市普通股（4 碼數字、排除 91xx 存託憑證）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、收盤（原始）>= 10 元、flag ∈ {0,1,2}',
   'volume': '量 ＝ value / close_raw（股數）；VMA5／VMA20 ＝ 量的 5／20 日簡單平均（含當日，min_periods 等於窗長）',
   'price_ma': 'MA5／MA20／MA60 用 adj（含息還原收盤）的簡單平均，避免除息造成假交叉、假跌破',
 },
 'signal_t_close': {
   '1_vma_golden_cross': 'VMA5_{t-1} <= VMA20_{t-1} 且 VMA5_t > VMA20_t',
   '2_ma5_up': 'MA5_t > MA5_{t-1}',
   '3_price_not_crossed': 'MA5_t <= MA20_t（量先於價，5 日線尚未突破月線）',
   '4_trend_filter': 'MA20_t >= MA20_{t-5} × 0.99 且 MA60_t >= MA60_{t-20} × 0.97（教學無數字，研究中心預設值，事前固定）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 該訊號放棄（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）出現的新訊號忽略；出場當日收盤後可再觸發',
   'exit': 's>=e（e＝進場日）第一個收盤 adj_s < MA20_s，或 adj_s < 進場價(還原開盤) × 0.93 → s+1 開盤賣；兩者皆無則第 e+H 日開盤賣（H＝最長持有交易日數）',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'VL_H20：條件 1~4 全部、最長持有 20 日',
   'VL_H40': '條件 1~4、最長持有 40 日',
   'SYNC_H20': '拿掉條件 3（價量同步交叉對照版）、最長持有 20 日',
   'SYNC_H40': '拿掉條件 3、最長持有 40 日',
   'all_reported': '4 組全部報告；判定只看 PRIMARY，其他只作穩健性參考',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，因 0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 視窗跨過此缺口，可能產生缺口後的假交叉）',
                  'src 3 期間收盤為 13:20 近似、成交值亦為近似', '面板無最低價，停損以收盤判斷（教學亦多以收盤判斷）'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

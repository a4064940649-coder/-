#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '寶塔線（n=3 日收盤）翻紅買、第 2 根轉黑賣（莊文宗雙線戰法）',
 'sources': ['https://finance.ettoday.net/news/3197855', 'https://finance.ettoday.net/news/3178985',
             'https://www.mirrormedia.mg/story/20260701money001', 'https://www.moneyweekly.com.tw/_Article?AID=1686',
             'https://histock.tw/blog/38/318', 'https://www.wantgoo.com/blog/37351/post/5224'],
 'source_note': '本環境無法連線來源網站（DNS 失敗），規則以主研究員給的規則文字為準，逐字轉成程式。',
 'type': '事件型（翻紅觸發、轉黑出場）為主：事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；'
         '另報事件的日曆時間組合，以及「每日持有全部紅狀態股票等權」的組合版（年化、日頻最大回撤、對 0050／同池等權月超額 t、逐年）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '流動性池：上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、原始收盤 >= 10 元、flag ∈ {0,1,2}',
   'price': '寶塔線與 MA20 一律用 adj（含息還原收盤），避免除息造成假翻黑；還原開盤＝open×adj/close_raw',
 },
 'tower_state': {
   'definition': 'n 日收盤寶塔線：HHn_t＝max(adj_{t-1..t-n})、LLn_t＝min(adj_{t-1..t-n})（停牌日不計，取該股最近 n 個有收盤的交易日）；'
                 'adj_t > HHn_t → 紅；adj_t < LLn_t → 黑；其他 → 延續前一日狀態；第一次出現紅或黑之前狀態未定（不產生訊號）',
   'n_primary': 3, 'n_sensitivity': [4, 5],
   'black_bar': '「黑棒」＝當日狀態為黑（寶塔線每日畫一根棒，狀態延續的日子也畫同色棒）',
   'new_black_bar': '「新黑棒」＝當日 adj_t < LLn_t（又創前 n 日收盤新低）',
 },
 'execution': {
   'entry_signal': 't 日狀態由黑翻紅（state_{t-1}＝黑、state_t＝紅），t 日在股池內；MA20 濾網版另需 adj_t >= MA20_t（布林中軌＝20 日簡單均線，min_periods 20）',
   'entry': 't+1 開盤（還原開盤）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 放棄該訊號（不遞補、不追）',
   'dedupe': '同一檔持有中（尚未全部出場）出現的新翻紅訊號忽略；全部出場日收盤後可再觸發',
   'exit_ZWZ（莊文宗，主規格）': '進場日 e 起逐日檢查：s1＝第一個狀態為黑的日子（翻紅後第一次轉黑）→ s1+1 開盤賣 50%；'
                          '剩下 50%：u＝s1 之後第一個狀態仍為黑的日子（又一根黑棒：新黑棒或狀態沒有翻回紅都算）→ u+1 開盤賣出；'
                          '若 s1+1 已翻回紅，剩下 50% 續抱到下一個黑棒日',
   'exit_STRICT（敏感度）': '前 50% 同上；剩下 50% 只在 s1 之後出現「新黑棒」（adj_u < LLn_u）時於 u+1 開盤賣，中間翻回紅也續抱',
   'exit_SIMPLE（簡化版）': 's1（第一次轉黑）→ s1+1 開盤全部賣出',
   'time_cap': '安全上限：進場後 250 個交易日仍未出場，第 250 日開盤全賣（預期幾乎不觸發，報告觸發數）',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤；跌停賣不掉不另處理（已知限制）',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%（來回 1%）；另報扣成本前（gross）',
 },
 'variants': {
   'PRIMARY': 'n=3、MA20 濾網、ZWZ 兩段出場（判定只看這組）',
   'NOMA': 'n=3、不加中軌濾網、ZWZ',
   'SIMPLE': 'n=3、MA20 濾網、第一次轉黑全部出場',
   'SIMPLE_NOMA': 'n=3、不加濾網、第一次轉黑全部出場',
   'STRICT': 'n=3、MA20 濾網、第二段只認新黑棒',
   'N4': 'n=4、MA20 濾網、ZWZ', 'N5': 'n=5、MA20 濾網、ZWZ',
   'RED_HOLD': '組合版：t 日收盤狀態為紅的股池股票，t+1 開盤起等權持有（每日開盤調回等權，換手依漂移後權重計成本；新進者 t+1 開盤漲停或無開盤則當日不買、隔日再看）；n=3、不加濾網',
   'RED_HOLD_MA': '組合版同上，另需 adj_t >= MA20_t（參考）',
   'RED_HOLD_N4／N5': '組合版 n=4、5（參考）',
   'all_reported': '全部報告；判定只看 PRIMARY；不做任何事後調參',
 },
 'event_stats': {
   'raw': '兩段出場：raw＝0.5×(xp1/p0)＋0.5×(xp2/p0)−1；net＝[0.5×xp1＋0.5×xp2]×(1−賣)/[p0×(1+買)]−1',
   'abnormal_EW': 'net − 同期間同池等權報酬（兩段各自用進場開盤→出場開盤的同池等權，按 50/50 加權）；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'abnormal_ALL': 'net − 同期間全上市普通股等權（參考）',
   'vs_0050': 'net − 0050 含息（t 收盤→各出場前一日收盤，50/50 加權）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'portfolio_stats': '事件日曆組合：所有在場事件（以剩餘部位比例加權）每日再平衡等權；進出按部位比例扣成本；無持股日現金 0 報酬。'
                    'RED_HOLD：開盤對開盤日報酬，成本＝買率×Σ買進權重＋賣率×Σ賣出權重。對手：0050（收對收）、同池等權（開對開、不扣成本）。報年化、日頻最大回撤、月超額 t、逐年、平均持股數、日換手',
 'windows': {'select': '2005-01-01~2019-04-30（事件以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且事件日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用；某段事件 < 30 筆或事件月份 < 24 → 無法測試。RED_HOLD 組合版另報是否勝同池等權（月 t >= 2）與 0050，僅作參考。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）',
                  'src 3 期間收盤為 13:20 近似', '出場不考慮跌停賣不掉', '0050 只有收盤，與開盤成交的策略有時點差'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '葛蘭碧買點4：負乖離過大買進（BIAS 均值回歸）',
 'sources': ['https://hao.cnyes.com/post/11280',
             'https://www.stockfeel.com.tw/%e4%b9%96%e9%9b%a2%e7%8e%87-%e6%8a%80%e8%a1%93%e5%88%86%e6%9e%90-%e5%9d%87%e7%b7%9a/',
             'https://www.cmoney.tw/notes/note-detail.aspx?nid=173319', 'https://ctyeh.com/stock/knowledge/11145',
             'https://www.wantgoo.com/blog/98845/post/161'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件等權）的年化／日頻最大回撤／對 0050 與同池等權月超額 t／逐年',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市）；有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '同 w1_granville_pullback_ma60：上市普通股（排除 91xx 存託憑證）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、收盤（原始）>= 10 元、t 日 flag ∈ {0,1,2}',
   'flag_exclusion': '訊號股在 [t-n+1, t]（n＝該訊號均線天數）內任何一天 flag ∈ {3,4,5} → 剔除該訊號（避免公司行動／資料缺口／錯誤造成假乖離）；但「全市場同日 flag 4 檔數 >= 50」的日子（2024-02-15、2024-03-18 這類面板缺口）不算，否則 2024 年整段沒訊號。MA240 濾網不另做 flag 剔除。',
   'disposal_exclusion': '若 t 或 t+1 在處置期間（disposal_5y.csv.gz 的 start~end；t 晚上已公告）→ 剔除。名單只有 2021-09-24 起，之前無法剔除（限制）。',
   'price_for_ma': '均線、乖離、收紅判斷一律用 adj（含息還原），避免除權息造成假負乖離',
 },
 'signal_t_close': {
   'BIAS': 'BIAS_n,t = adj_t / MA_n,t − 1，MA_n＝adj 的 n 日簡單均線（min_periods=n）',
   'sets': {'a_B20': 'BIAS20 <= −10%', 'b_B13': 'BIAS13 <= −6%', 'c_B60': 'BIAS60 <= −7%'},
   'trigger_versions': {'touch': 'BIAS_t <= 門檻（每天只要在門檻下都是訊號；持有中不重複進場）',
                        'red': '首次收紅版：BIAS_t <= 門檻 且 adj_t > adj_{t-1}'},
   'filters': {'none': '無濾網', 'ma240_up': 'MA240_t >= MA240_{t-20}（長期均線未向下）'},
   'params_fixed': '門檻與天數照任務指定的教學常見值；MA240 濾網回看 20 日；沒有其他參數。看結果後不調參。',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 該訊號放棄（不遞補）',
   'dedupe': '同一檔在持有中（尚未到出場開盤日）出現的新訊號忽略；出場日當天收盤後可再觸發（出場日 x 的收盤訊號 → x+1 開盤進場）',
   'exit_A': '乖離歸零：s∈[e, e+19] 第一個 adj_s >= MA_n,s → s+1 開盤賣；否則 e+20 開盤賣（最長 20 日）',
   'exit_B': '固定持有 H∈{5,10,20} 個交易日：e+H 開盤賣',
   'exit_lockdown': '預定出場日若整天鎖跌停（還原最高價/前一日還原收盤 −1 <= −(漲跌幅−0.6%)）→ 賣不掉，順延到下一個非鎖跌停日開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'a_B20（BIAS20 <= −10%）、touch、無濾網、出場 A（乖離歸零即出、最長 20 日）＝教學最直接的字面版本',
   'all_reported': '3 組乖離 × 2 觸發版本 × 2 濾網 × 4 出場（A、B5、B10、B20）＝ 48 組，全部報告',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，因 0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收盤對收盤）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。其他 47 組只作穩健性參考，不得用來改判（報告幾組兩段都過）。',
 'diagnostics_preplanned': ['t 收盤進場（不可成交）vs t+1 開盤的差距（隔夜跳空）', '壓力成本', '排除持有期間下市的事件（事後資訊，僅診斷）', '挑選期分 2005-2011 / 2012-2019-04', '逐年事件異常報酬'],
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀；對抄底策略尤其有利，因為下市前崩跌股不在樣本內）', '2024-01-02~2024-02-14 無資料（rolling 跨缺口）', 'src 3 期間收盤為 13:20 近似', '處置名單 2021-09 起才有', '面板無最低價：鎖跌停用最高價判斷'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

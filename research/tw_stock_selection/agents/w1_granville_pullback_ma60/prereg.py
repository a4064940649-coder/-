#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '葛蘭碧買點2/3：上揚季線量縮回測不破',
 'sources': ['https://www.moneydj.com/z/analyst/ma.htm', 'https://www.wantgoo.com/blog/98845/post/161',
             'https://www.tmgm.com/zh-hant/academy/trading-academy/granville-eight-rules', 'https://histock.tw/blog/histock1688/198',
             'https://www.cmoney.tw/notes/note-detail.aspx?nid=87780'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬；另報日曆時間組合（所有在場訊號等權）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市）；2018-01-13（有值檔數 < 300）整列剔除後才做任何 rolling；flag 5 的價格視為缺值',
   'universe': '上市普通股（4 碼、排除 91xx 存託憑證）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、收盤（原始）>= 10 元、flag ∈ {0,1,2}',
   'price_for_ma': '均線、收盤比較一律用 adj（含息還原）以免除息假跌破；量＝value/close_raw',
 },
 'signal_t_close': {
   '1_ma_up': 'MA_t > MA_{t-20}（20 個交易日前）',
   '2_prior_extension': 'max(adj_{t-19..t}) / MA_t >= 1.10（字面：過去 20 日最高收盤 / 今日 MA）',
   '3_pullback': '0 <= adj_t/MA_t - 1 <= band，且 t-4..t 共 5 日每日 adj >= 當日 MA（不破）',
   '4_volume_dry': 'vol_t < 0.7 × mean(vol_{t-19..t})（20 日均量含當日，與看盤軟體均量線一致）',
   '5_up_day': 'adj_t > adj_{t-1}',
   'author_params_fixed': '1.10、0.7、band、5 日、20 日均為作者設定（教學未給數字），挑選期前固定，不調參',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 該訊號放棄（不遞補）',
   'dedupe': '同一檔在持有中（尚未到出場開盤日）出現的新訊號忽略；出場日當天收盤後可再觸發',
   'exit_A': '固定持有 20 個交易日：第 e+20 日開盤賣（e＝進場日）',
   'exit_B': 's∈[e, e+59] 第一個 adj_s < MA_s×0.97 → s+1 開盤賣；否則 e+60 開盤賣',
   'exit_C': 's>=e 第一個 adj_s < MA_s → s+1 開盤賣；安全上限 e+250 開盤賣（事前加上，避免永不出場）',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'MA60、band 0~3%、出場 A',
   'all_reported': 'MA60 band3 / MA60 band2 / MA60 band5 / MA20 band3（MA20 版的 MA 上揚、漲離、回測、B/C 出場都改用 MA20）× 出場 A/B/C ＝ 12 組，全部報告',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，因 0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。其他 11 組只作穩健性參考，不得用來改判（若 PRIMARY 過但多數變體不過，需註明）。',
 'known_limits': ['面板無最低價：「低點碰均線不破」以收盤距 MA 0~band 且 5 日收盤不破近似', '2019–2023 面板只有存活者（驗證期偏樂觀）',
                  '2024-01-02~2024-02-14 無資料（rolling 視窗跨過此缺口）', 'src 3 期間收盤為 13:20 近似'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

#!/usr/bin/env python3
"""事前設定（在看任何結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '均線扣抵：月線剛翻揚＋接下來扣抵值低（扣抵低價區上揚）＋站上月線',
 'sources': ['https://www.businesstoday.com.tw/article/category/80402/post/201805290018/', 'https://www.stockfeel.com.tw/?p=148371',
             'https://hao.cnyes.com/post/28040', 'https://histock.tw/blog/NINCYS/1199', 'https://www.nstock.tw/author/article?id=226'],
 'type': '事件型（訊號觸發）：主報告＝事件報酬（扣成本、對同池等權異常報酬、t 按進場月份分群）；另報日曆時間組合（所有在場事件等權）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）；「有值檔數 < 300」的日子（2018-01-13）整列剔除後才做任何 rolling／shift；flag 5 的價格視為缺值',
   'universe': '同前幾輪：上市普通股（4 碼數字、排除 91xx）；t 日 60 日均成交值（上市後停牌日記 0）>= NT$1,000 萬、收盤（原始）>= 10 元、flag ∈ {0,1,2}',
   'price': '收盤一律用 adj（含息還原收盤）；MA20／MA60 ＝ adj 的 20／60 日簡單平均（含當日，min_periods 等於窗長）；扣抵值用同一條 adj 序列（避免除息造成假扣抵）',
 },
 'signal_t_close': {
   'C1_ma20_turn_up': 'MA20_{t-1} <= MA20_{t-2} 且 MA20_t > MA20_{t-1}',
   'C2_deduction_low_K5': 'max(adj_{t-19..t-15}) < adj_t（未來 5 天要扣掉的 5 個舊價都低於今日收盤）',
   'C2_deduction_low_K10': '版本 K10：max(adj_{t-19..t-10}) < adj_t',
   'C3_above_ma20': 'adj_t > MA20_t',
   'F_ma60_deduction_low': '濾網版：max(adj_{t-59..t-55}) < adj_t（需 MA60 有值）',
 },
 'execution': {
   'entry': 't+1 開盤（還原開盤＝open×adj/close_raw）；t+1 無開盤價或開盤漲停（開盤/前收−1 >= 漲跌幅−0.6%，2015-06-01 前 7%、之後 10%）→ 該訊號放棄（不遞補）',
   'dedupe': '同一檔持有中（尚未到出場開盤日）出現的新訊號忽略；出場當日收盤後可再觸發',
   'exit_MA20': '主出場：s>=e 第一個收盤 adj_s < MA20_s → s+1 開盤賣；最長持有 60 個交易日（第 e+60 日開盤賣）。教學無停損，不加停損',
   'exit_H10/H20': '固定持有：第 e+10／e+20 日開盤賣',
   'exit_price': '出場日有開盤價用開盤；停牌則用之後第一個開盤；已下市用最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants': {
   'PRIMARY': 'K5_none_MA20：C1＋C2(K5)＋C3、無濾網、MA20 出場（最長 60 日）',
   'grid': 'K ∈ {5,10} × 濾網 ∈ {none, MA60低} × 出場 ∈ {MA20(≤60), H10, H20}，共 12 組全部報告；判定只看 PRIMARY，其他作穩健性參考（另報 12 組各自是否過門檻）',
   'note': '教學沒給的參數用最常見預設：K＝5（題目給定主版）、出場跌破月線、最長 60 日；看結果後不調參',
 },
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）同池等權報酬；同池＝前一日符合股池條件的上市普通股，每日等權、開盤對開盤',
   'vs_0050': 'net − 0050 含息（t 收盤→出場前一日收盤，因 0050 只有收盤）',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；另報 t_date 僅供參考',
 },
 'portfolio_stats': '日曆時間組合：每日持有所有在場事件、等權（每日再平衡），進場日報酬＝收/開，出場日＝出場價/前收，進出各按 1/n 權重扣成本；無持股日現金 0 報酬。報年化、日頻最大回撤、對 0050 與同池等權（收對收）的月超額 t、逐年、曝險天數比例',
 'windows': {'select': '2005-01-01~2019-04-30（以訊號日歸屬）', 'valid': '2019-05-01~2026-10-22', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層）',
             'valid_ex_gap': '驗證期剔除訊號日 60 列回看窗跨過 2024-01-02~02-14 缺口的事件（僅事件層）'},
 'verdict_rule': 'PRIMARY 在挑選期與驗證期都要：事件淨異常報酬（對同池等權）> 0 且 t_month >= 2，且日曆組合淨年化 >= 0050 同期年化 → 可用；只有一段過 → 待觀察；否則不可用。',
 'known_limits': ['2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 視窗跨過此缺口）',
                  'src 3 期間收盤為 13:20 近似', '條件 C2 隱含 adj_t 高於 15~19 日前價格，與 20 日動能高度重疊'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

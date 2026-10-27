#!/usr/bin/env python3
"""事前設定（在看任何回測結果前寫入 results.json 的 PREREG；之後不改）。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '土洋合作：外資與投信同步連買 3 日（含 CMoney 外資 20 日＋投信當日＋營收年增版）',
 'sources': ['https://blog.wistock.ai/practical-cases-experience-sharing/institutional-investors-buying-ai-model-recommendation/',
             'https://blog.wistock.ai/wistalk/institutional-investors-buy-stocks-strategies/',
             'https://www.cmoney.tw/notes/note-detail.aspx?nid=106067', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=12986',
             'https://wantrich.chinatimes.com/news/20211112900673-420101'],
 'type': {'TY（土洋連 3 買）': '事件型：事件報酬（扣成本、對上市流動性池等權異常報酬、t 按進場月份分群）＋20 槽等額組合（對 0050）',
          'CM（CMoney 版）': '選股型：月換股等權組合（年化、日頻最大回撤、對 0050 t、對同池等權 t、逐年）'},
 'data': {
   'inst': 'extras/inst_5y.csv.gz（2021-10-08~2026-10-07，單位張）foreign、trust 欄；某日某股無列＝0；訊號在 inst 交易日曆上計算',
   'prices_main': 'scratchpad/panel 上市日線（2021 起）；「有值檔數 < 300」的日子先剔除再滾動（2018-01-13 類；本期間預期無）；flag 5 價格視為缺值；成交量（股）＝value/close_raw；2024-01-02~2024-02-14 面板無資料 → 該段訊號不成立（缺收盤）、無法進場；出場遇缺口 → 之後第一個開盤',
   'market_scope_main': '教學股池為上市＋上櫃，但上櫃股價只有 2025-09-24 起 → 主檢定只測上市股（重大限制）',
   'supplement_TWSE_TPEX': 'history.csv.gz（上市＋上櫃官方日 K，2025-09-24~2026-10-22）＋ exdiv_5y 的 ref_price/prev_close 做除權息還原，跑同一套 TY 規則（需 60 日回看 → 實際訊號約 2025-12 起），僅作參考，不入判定',
 },
 'universe_t': '普通股（4 碼數字、非 0 開頭、非 91 開頭）；t 日 60 日均成交值（停牌日記 0，自上市首個有值日起滾動，min_periods=60）>= NT$1,000 萬；原始收盤 >= 10 元；flag ∈ {0,1,2}',
 'signal_TY_t_close': {
   '1_streak': 't-2、t-1、t 三個 inst 交易日 foreign > 0 且 trust > 0（每天都買超）',
   '2_increasing': 'A_d =（foreign_d + trust_d）× 原始收盤_d（張×元，與股數單位一致即可），A_{t-2} < A_{t-1} < A_t（嚴格遞增）',
   '3_size': 'sum_{t-2..t}(foreign+trust)（張）>= 10% × sum_{t-2..t} 成交量（張＝value/close_raw/1000）',
   '4_not_extended': '還原收盤_t / min(還原收盤 t-59..t，面板日曆 60 個交易日，min_periods=40) − 1 < 30%',
   '5_volume': 'mean(成交量 t-2..t) > mean(成交量 t-22..t-3)（面板日曆，前 20 日 min_periods=15）',
   'three_days_must_be_panel_days': 't-2..t 必須都有面板收盤與成交值（面板缺口內不出訊號）',
   'priority': '「金額大的優先」：同日多檔時，依 3 日累計合計買超金額 sum(A) 由大到小（只影響 20 槽組合）',
   'signal_start': '訊號日 >= 2021-10-12（inst 第 3 個交易日）',
 },
 'execution_TY': {
   'entry': 't+1（inst 日曆下一交易日，必須是面板日且有開盤價）開盤買（還原開盤＝open×adj/close_raw）；開盤漲停（還原開盤/前一還原收盤 −1 >= 9.4%）→ 放棄不遞補；無開盤 → 放棄',
   'hold': 'H ∈ {5, 20} 個交易日（inst 日曆）：e+H 日開盤賣（開盤對開盤；兩種都報告）；inst 日曆結束後用面板日曆續數；面板結束（2026-10-22）仍未到期 → 最後一日收盤平倉標 censored',
   'dedupe': '同一檔持有中（出場日前）的新訊號忽略（H5、H20 各自獨立）',
   'exit_price': '出場日開盤；停牌/缺口 → 之後第一個開盤；已下市 → 最後收盤',
   'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%；壓力情境 買 0.5%＋賣 0.5%',
 },
 'variants_TY': {
   'PRIMARY': 'TY_H20（全部 5 條件，持有 20 日）',
   'co_reported_own_verdict': 'TY_H5（全部 5 條件，持有 5 日）— 單獨判定並報告，但不改整體判定',
   'reported_not_for_verdict': ['TY_BASE_H5/H20：只用條件 1（外資投信同步連 3 買）＋股池，看其他條件的貢獻',
                                'TY_NOINC_H5/H20：拿掉條件 2（逐日遞增），其餘不變'],
 },
 'portfolio_20': '每日開盤先處理出場（賣出收現金）再處理進場；同日候選依 priority 由大到小；空槽才買，每筆金額＝當日開盤 NAV/20（不超過現金）；不再平衡；未投入現金 0 報酬',
 'event_stats': {
   'raw': '出場價/進場價 − 1（還原）；net＝(1+raw)×(1−賣)/(1+買) − 1',
   'abnormal_EW': 'net − 同期間（進場開盤→出場開盤）上市流動性池等權（前一日符合股池條件，開盤對開盤，單日報酬截 [-50%,100%]）→ 判定用',
   'vs_0050': 'net − 0050 含息（t 收盤 → 出場前一日收盤）參考',
   't': '事件先按進場月份平均，再對月平均做 t（btlib.tstat）；t_date 僅參考',
 },
 'CM_rule': {
   'rebalance': '每月第一個「日 >= 11」的交易日 d（＝exlib.DATES；上月營收 10 日前公告），d 收盤算訊號，d+1 開盤成交（exlib M2 慣例）',
   'signal': 'sum(foreign, d 往回 20 個 inst 交易日含 d) > 1,000 張 且 trust_d > 200 張 且 上月（d 所在月−1）月營收年增率 yoy > 0（rev_5y）；股池同上（上市流動性池）',
   'weights': '入選全部等權；開盤漲停且未持有者不買、不遞補；入選 0 檔 → 空手（現金 0 報酬）',
   'engine': 'exlib.sim_portfolio（漂移後權重計周轉、成本 exlib.cost_rates 滑價 0.1% 與壓力情境）',
   'benchmarks': '同池等權＝同日上市流動性池全部等權（同執行、不扣成本，同 exec_rebuild）；0050 含息',
   'stats': 'exlib.daily_stats：年化、日頻 MDD、月窗（換股日收盤→下一換股日收盤）對 0050 t、對同池等權 t、逐年',
   'start': '第一次換股 2021-11（外資需 20 個 inst 日）',
 },
 'windows': {'select': '訊號／換股 2021-10-12~2023-12-31（2021 起資料的挑選期）',
             'valid': '2024-01-01~2026-10（inst 到 2026-10-07）', 'valid_2024_2025': '2024-01-01~2025-12-31',
             'valid_ex_src3': '驗證期剔除持有期碰到 src 3（2024-02-15~2025-09-23）的事件（僅事件層參考）'},
 'verdict_rule': {
   'TY': '挑選期與驗證期都要：事件淨異常報酬（對上市等權）> 0 且 t_month >= 2，且 20 槽組合扣成本年化 >= 0050 同期年化 → 該段過；兩段過＝可用、一段＝待觀察、零段＝不可用；事件數 < 30 或月份 < 12 → 無法測試',
   'CM': '兩段都要：月窗對同池等權超額 > 0 且 t_vs_EW_m >= 2，且扣成本年化 >= 0050 年化；同上分級；月份 < 12 → 無法測試',
   'overall': '整體判定＝PRIMARY（TY_H20）的判定；TY_H5、CM 各自判定並列報告，不得用來升級整體判定；變體只作穩健性參考',
 },
 'known_limits': ['主檢定只測上市（上櫃無 2025-09 前股價）', '2019–2023 面板只有存活者（偏樂觀）', '2024-01-02~2024-02-14 面板無資料',
                  'src 3 期間（2024-02-15~2025-09-23）開盤/收盤為 5 分 K 近似', '挑選期實際只有 2021-10~2023-12 約 2.2 年',
                  '10% 門檻與 60 日／30% 為研究中心或常見預設，非教學原文精確數字'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

#!/usr/bin/env python3
"""Clenow《Stocks on the Move》回歸斜率×R² 動能：事前設定。在看任何結果前寫入 results.json 的 PREREG；之後不改。
執行：python3 -I prereg.py"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': 'Clenow Stocks on the Move：90 日 ln 價格回歸年化斜率 × R² 排名；SMA100／15% 跳空排除；大盤 > SMA200 才新買；每週三訊號、隔日開盤成交；ATR 風險平價部位、每兩週調整；上限 30 檔',
 'sources': ['https://www.quantconnect.com/forum/discussion/16578/quot-stocks-on-the-move-quot-momentum-strategy-by-cabedovestment/',
             'Andreas Clenow (2015) Stocks on the Move: Beating the Market with Hedge Fund Momentum Strategies（書中原規則：90 日指數回歸、250 年化、×R²、SMA100、15% gap、指數 SMA200、週三交易、兩週調整部位、0.1% 風險因子、前 20%）'],
 'type': '選股型（組合）。以組合報告與判定（年化、日頻最大回撤、對 0050 t、對同池等權 t、逐年）。',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling，模擬也只在其餘交易日進行。',
   'price': '回歸、SMA100、ATR、漲跌幅一律用含息還原收盤 adj（避免除息假跌）；成交用還原開盤（exlib.AO）。',
   'index': '大盤濾網用加權指數 TAIEX 收盤（taiex_twii.csv，價格指數，與 Clenow 用 S&P 500 價格指數一致）。',
 },
 'universe_signal_day_d': {
   'liquidity': '60 日均成交值（mlib.V60：剔除缺漏日後滾動）>= NT$1,000 萬、原始收盤 >= 10 元、d 有 adj 值',
   'common': '排除 91xx（存託憑證）',
   'disposal': 'd 不在處置期間（名單 2021-09 起才有；沿用 mlib/exlib 慣例，因處置股分盤撮合、開盤成交不實際）',
   'top_n': '上述合格者中 60 日均成交值前 200 名＝股池（每個週訊號日重算）',
 },
 'signal_on_d_close': {
   'regression': 'y = ln(adj) 最近 90 個交易日（含 d，須 90 個全有值），對 x = 0..89 做 OLS；slope 為每日對數斜率',
   'annualized_slope': 'exp(slope)^250 − 1',
   'R2': 'OLS 判定係數',
   'score': '年化斜率 × R²',
   'rank': '股池內有分數者依分數由大到小排名；「前 20%」＝ 名次 <= ceil(0.2 × 有分數檔數)（股池 200 檔時約 40 名）',
   'filter_sma100': 'adj_d < SMA100(adj)（100 日，須全有值）→ 排除（不可買；持有中則賣出）',
   'filter_gap15': '最近 90 日內任一日 |adj_t/adj_{t−1} − 1| > 15% → 排除（不可買；持有中則賣出）。台股有 7%/10% 漲跌幅，此條主要抓停牌復牌、新上市、減資等異常；原樣保留',
   'index_filter': 'TAIEX_d > SMA200(TAIEX) 才允許新買；否則只賣不買（既有部位兩週調整仍照做，依書中原規則）',
 },
 'schedule': {
   'signal_day': '每週週三收盤；週三休市則改用同週之後第一個交易日（週四／週五）；該週三之後全休則該週跳過',
   'execution': '訊號日 d 的下一個交易日開盤（還原開盤 AO）成交',
   'rebalance_positions': '每兩週（訊號週序號為偶數的那一週，自 2004-07 起第 0 週算起）把現有持股調回目標股數',
 },
 'sell_rules_at_d': '持股若 (a) 不在前 20%、(b) adj < SMA100、(c) 90 日內有 >15% 單日變動、(d) 當日無分數（停牌、資料不足、跌出 200 檔股池而無排名）→ d+1 開盤賣出全部。d+1 無開盤價延到下一個有開盤價的日子；已下市（之後再無 adj）以最後還原收盤賣出。',
 'position_sizing': {
   'ATR20': 'ATR 以收盤近似：mean(|adj_t − adj_{t−1}|, 最近 20 日，須全有值)，取訊號日 d 的值',
   'target': '目標股數 = 帳戶淨值 × 0.1% / ATR20（以 d+1 開盤時的淨值計；等同目標權重 = 0.001 × adj_d / ATR20 × AO/adj_d 的比例）',
   'max_positions': '同時最多 30 檔',
   'no_weight_cap': '教學未設單檔上限 → 不另設（只受現金與 30 檔限制）',
   'buy_order': '新買：前 20%、通過兩個排除、未持有，依分數由高到低；每檔買目標金額，現金不足時買剩餘現金（剩餘現金 < 目標的 25% 則停止）；不用槓桿、現金零利率',
   'rebalance': '兩週調整日：把每檔現有持股調到目標（全額調整，不設門檻）；加碼受現金限制',
   'order_at_open': '先賣出 → 再調整部位（先減後加）→ 再新買',
   'limit_up': 'd+1 開盤漲停（開盤/前收 − 1 >= 漲跌幅 − 0.6%，2015-06-01 前 7%、之後 10%，mlib.GAP_UP）→ 新買／加碼買不到，新買由下一名遞補',
 },
 'cost': 'exlib.cost_rates：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋稅 0.3%＋滑價 0.1%（來回約 0.67%）；壓力情境 買 0.5%＋賣 0.5%；另報毛報酬。判定用滑價 0.1%。',
 'sim_start': '2004-07-01 起模擬（訊號需 100 日暖身，實際約 2004-07 起才有分數），統計自各期間起點',
 'benchmarks': {
   '0050': '0050 含息（ffill 到面板日曆）',
   'EW_pool_primary': '同池等權（主對手）：每月第一個 >= 11 日交易日 d 收盤的同一 200 檔股池全部等權，d+1 開盤換股，不扣成本（exlib.sim_portfolio；與前幾輪口徑一致）',
   'EW_pool_weekly_diag': '診斷：同 200 檔股池、每週訊號日換股的等權，不扣成本',
 },
 'portfolio_stats': '日頻 NAV：年化、日頻最大回撤、逐年；t＝月窗（每月第一個 >= 11 日交易日收盤 → 下一個）超額報酬 t（每月一觀察值）對 0050 與對同池等權；另報平均持股數、平均投入比例、年周轉',
 'windows': {'select': '2005-01-01~2019-04-30', 'valid': '2019-05-01~2026-10-22', 'y2024_2025': '2024-01-01~2025-12-31',
             'extra': ['2021-01~2023-12 / 2024-01~2026-10（參考）', '驗證期存活者段 2019-05~2023-12', '驗證期官方價格段（剔除 src3 2024-02-15~2025-09-23）']},
 'primary': 'PRIMARY＝上述原規則、滑價 0.1%',
 'diagnostics_reported_not_for_verdict': {
   'D1_score_52wHigh': '同引擎、同排除與部位規則，分數換成 adj/252 日最高 adj（WEB_REPORT_w3 要求報告相對 52 週高點的增量）',
   'D2_score_ret90': '同引擎，分數換成 90 日簡單報酬（檢查 ×R² 是否有增量）',
   'D3_no_index_filter': '原規則但不看大盤 SMA200（檢查大盤濾網貢獻）',
   'D4_equal_weight': '原規則但新買每檔 NAV/30 等權（檢查 ATR 部位貢獻）',
 },
 'verdict_rule': '以 PRIMARY 滑價 0.1% 判定。某期間「通過」＝ 對同池等權月窗 t >= 2 且 年化 >= 0050 同期年化。挑選期與驗證期都通過 → 可用；只一段通過 → 待觀察；都不通過 → 不可用。診斷變體不影響判定。',
 'known_limits': ['ATR 以收盤絕對變動近似（無最低價）', '2019–2023 面板只有存活者（驗證期偏樂觀）', '2024-01-02~2024-02-14 無資料（rolling 視窗跨過此缺口）',
                  'src 3 期間收盤為 13:20 近似', '跌停賣不掉未模擬（開盤即成交）', '處置名單 2021-09 起才有', '股池只有上市，無上櫃'],
}
if __name__ == '__main__':
    f = HERE / 'results.json'
    d = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    assert 'PREREG' not in d, 'PREREG 已存在，不覆寫'
    d = {'PREREG': PREREG, **d}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written')

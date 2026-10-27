#!/usr/bin/env python3
"""林則行《飆股的長相》鉛筆型平穩期後創新高＋量增＋營收成長：事前設定。看任何結果前寫入 results.json 的 PREREG；之後不改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '林則行 鉛筆型平穩期後創 1～2 年新高（非史上新高）＋量增 ≥ 20%＋近 3 月營收年增 ≥ 10%＋K 棒落差 ≤ 4% → 隔日開盤買；-8% 停損／26 日 Spearman < 0／持有 120 日；每檔 NAV 5%、最多 20 檔',
 'sources': ['https://www.eslite.com/product/1001191562682001692002', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=35181',
             'https://www.businesstoday.com.tw/article/category/183008/post/201409190016/', 'https://www.cmoney.tw/notes/note-detail.aspx?nid=12228'],
 'type': '選股型組合（每日訊號、最多 20 檔、每檔開盤 NAV 的 1/20）為主報告與判定；另報事件型（全部訊號、不受 20 檔限制，扣成本、對同池等權的異常報酬、t 按進場月份分群）',
 'data': {
   'panel': 'scratchpad/panel 上市日線（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後再做任何 rolling，模擬也只在其餘交易日進行',
   'price_basis': '新高、史上高、平穩期、停損、Spearman 一律用 adj（含息還原收盤）先 ffill(limit=10)；訊號日本身須有 adj、open、high、close_raw 原始值。股價門檻與 K 棒落差用原始價',
   'revenue': '月營收水準：2020-12 以前用長期檔 NiklavsD/bottleneck-signals tw_revenue.csv（千元，sii 優先去重），2021-01 起用 tw-stock-screener rev_5y.csv.gz（同單位，已核對 99.7% 相同），缺值以另一檔補。營收 ≤ 0 視為缺值',
   'revenue_timing': '無公告日：假設 m 月營收在 m+1 月 10 日收盤後已知。訊號日 t 若日曆日 ≥ 11 → 最近已公告月 = t 月 −1，否則 = t 月 −2',
 },
 'universe_t': '上市普通股（排除 91xx）；訊號日 t：60 日均成交值（剔除缺漏日後，mlib.V60）≥ NT$1,000 萬、close_raw_t ≥ 10 元',
 'entry_signal_t_close (all must hold)': {
   '(1) new_high_1_2y': 'adj_t > max(adj_{t-250..t-1})（前 250 個有效交易日，min_periods 240）',
   '(2) not_all_time_high': 'adj_t < max(adj_{t-2500..t-1})（前 2,500 日，資料不足用全部歷史，min_periods 1）。2004 年起才有面板，2005～2006 年可用歷史短 → 訊號自然少',
   '(3) pencil_base': 'max(adj_{t-120..t-1}) / min(adj_{t-120..t-1}) − 1 ≤ 30%（min_periods 115）',
   '(4) volume_up': 'vol_t = value_t / close_raw_t ≥ 1.2 × mean(vol_{t-20..t-1})（min_periods 18）',
   '(5) revenue_growth': '最近已公告 3 個月營收合計 / 去年同 3 個月合計 − 1 ≥ 10%（6 個月都要有值）',
   '(6) bar_range': '(high_t − min(open_t, close_raw_t)) / close_raw_{t-1} ≤ 4%（原始價；面板無最低價的近似）',
   'trigger_once': '同一檔持有中不重複進場；出場日之後才可再觸發',
 },
 'execution': {
   'entry': 't+1（下一個有效交易日）開盤買，還原開盤價 AO；t+1 無開盤價 → 放棄；t+1 開盤漲停（相對 t 還原收盤跳空 ≥ 漲跌幅 − 0.6%，漲跌幅 2015-06-01 前 7%、之後 10%）→ 買不到、放棄',
   'exit': '從進場日 e 的收盤起每日檢查：(a) adj_c < 進場價 × 0.92 → 隔日開盤賣；(b) 近 26 日 adj 收盤（c-25..c，平均名次處理同值）對時間序的 Spearman 相關 < 0 → 隔日開盤賣；(c) 持有 120 個交易日 → 第 e+120 日開盤賣。賣出日無開盤價 → 順延到下一個有開盤價的日子；下市 → 以最後收盤價出場',
   'portfolio': '每日開盤：先出場，再以當時開盤 NAV/20 為一單位買進；同日候選超過空位時依量比（vol_t / 前 20 日均量）由大到小；現金 < 0.25 單位不買。現金不計息',
   'costs': 'exlib.cost_rates：買 = 手續費 0.1425%×0.6 + 滑價；賣 = 手續費 + 證交稅 0.3% + 滑價；滑價 0 / 0.1%（主）/ 壓力（買賣各 0.5%）；另報毛報酬',
 },
 'benchmarks': {
   'EW_pool': '同池等權：每月第一個 ≥ 11 日交易日 d 收盤的股池（V60 ≥ 1,000 萬、股價 ≥ 10、非 91xx、有 adj）於 d+1 開盤等權重設，無成本',
   '0050': '0050 含息（finlab-txd etf_0050.csv）',
   'event_abnormal': '事件報酬（扣成本）− 同一進場開盤到出場價格的同池等權報酬（訊號日池中 t+1 有開盤價者）',
 },
 'windows': {
   'select': '2005-01～2019-04（價格類挑選期）', 'valid': '2019-05～2026-10（價格類驗證期）', 'y2024_2025': '2024-01～2025-12',
   'r21_select': '2021-01～2023-12', 'r21_valid': '2024-01～2026-10',
   'diag': '驗證期存活者段 2019-05～2023-12、驗證期只用官方價格段（排除 src=3 的 2024-02-15～2025-09-23）、全期',
 },
 'statistics': '組合：日頻 NAV 的 CAGR、波動、最大回撤、逐年；月窗（每月第一個 ≥ 11 日交易日收盤到下一個）超額報酬對 0050 與同池等權的 t。事件：扣成本異常報酬，按進場月份分群取平均後的 t',
 'judgement': '主規則 BASE、滑價 0.1%：挑選期與驗證期各自「組合 t_vs_EW_m ≥ 2 且 CAGR ≥ 0050 CAGR」→ 兩段都過＝可用；一段過＝待觀察；都不過＝不可用。任一段平均持股 < 2 檔或交易 < 30 筆 → 無法測試',
 'variants_all_reported (BASE 才是判定對象；其他只是拆解，不改判定)': {
   'BASE': '(1)～(6) 全部',
   'NO_REV': '拿掉 (5) 營收（看營收條件貢獻；也不受長期營收 2013 起只剩存活公司的偏誤影響）',
   'NO_PENCIL': '拿掉 (3) 平穩期',
   'ALLOW_ATH': '拿掉 (2)，史上新高也買',
   'NO_RANGE': '拿掉 (6) K 棒落差',
   'NH250_ONLY': '只有 (1) 新高（＋股池），同出場：對照組＝「普通 52 週新高」',
   'BASE_RAWPX': '(1)(2)(3) 改用原始收盤（台股看盤軟體常見口徑；含除權息與減資跳動）',
   'BASE_STOPONLY': 'BASE 進場，出場只用 -8% 停損 + 120 日（拿掉 Spearman），看順位相關出場的作用',
 },
 'known_biases': ['長期月營收 2013 年起只剩存活公司 → 2013～2020 已下市公司無營收、無法入選 → 挑選期後段有存活者偏誤（NO_REV 不受影響）',
                  '面板 2019–2023 只有存活者（驗證期前段偏多）', 'src=3（2024-02-15～2025-09-23）收盤為 13:20 近似', '2024-01-02～2024-02-14 面板無資料',
                  '無最低價：平穩期用收盤區間、K 棒落差用 high − min(open, close) 近似；無稅後盈餘資料，省略 >20% 條件；順位相關係數公式未公開，26 日 Spearman < 0 是研究員預設'],
}
if __name__ == '__main__':
    p = HERE / 'results.json'
    if p.exists():
        raise SystemExit('results.json 已存在，PREREG 不可覆寫')
    p.write_text(json.dumps({'PREREG': PREREG}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG written', PREREG['written_at'])

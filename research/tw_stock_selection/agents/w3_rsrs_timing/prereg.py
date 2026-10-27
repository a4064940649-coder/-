#!/usr/bin/env python3
"""事前設定：寫回測程式、看任何結果之前寫入 results.json 的 PREREG（附 sha256）；run.py 開跑前核對雜湊。之後不得修改。"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
PREREG = {
 'rule_name': 'RSRS 阻力支撐相對強度擇時（光大證券 2017）：加權指數 18 日 high~low 回歸斜率的 600 日標準分 ×R²（修正標準分）> 0.7 隔日開盤買 0050，< −0.7 隔日開盤賣出持現金',
 'type': '擇時（狀態型，持有 0050 或現金）：以日頻擇時組合判定；另報每筆交易的事件報酬',
 'sources': ['https://www.luxalgo.com/library/indicator/JOxeFQeX-rsrs-resistance-support-relative-strength',
             'https://cn.tradingview.com/scripts/rsrs', 'https://tw.tradingview.com/scripts/rsrs', 'https://jp.tradingview.com/scripts/rsrs'],
 'rule_verbatim': ('每日用過去 N=18 日的 (low, high) 做 OLS：high = α + β·low，β 就是 RSRS。標準分 z＝(β − 過去 M=600 日 β 的平均) ÷ 過去 600 日 β 的標準差。'
                   '修正版：z_adj＝z × R²（同一個 18 日回歸的 R²），另測 z × R² × β。進場：z（或 z_adj）> 0.7，收盤算、隔日開盤買 0050；'
                   '出場：z < −0.7 隔日開盤賣出、持有現金；介於兩者之間就維持原部位。敏感度：N＝16／18／20，M＝300／600，門檻 ±0.7／±1.0。'
                   '濾網：只在 20 日均線 > 3 日前 20 日均線時才允許進場。'),
 'signal': {
   'data': 'taiex_twii.csv（1999-01-05～2026-10-22，加權價格指數 open/high/low/close），在 TAIEX 自己的交易日曆上計算；訊號只用 t 日收盤以前（含 t）的資料。',
   'beta': 'β_t、R²_t：最近 N 個 TAIEX 日（含 t）的 high 對 low 做普通最小平方（含截距，原始點位不取對數）。',
   'zscore': 'z_t＝(β_t − mean(β_{t−M+1..t})) ÷ std(β_{t−M+1..t})，視窗含當日 β，std 用母體標準差 ddof=0（同聚寬原始程式 np.std），min_periods＝M。',
   'scores': {'z': '標準分 z', 'zR2': '修正標準分 z×R²（主）', 'zR2b': '右偏標準分 z×R²×β'},
   'state': ('狀態機：起始為空手；score_t > +thr → 目標持有；score_t < −thr → 目標空手；其餘維持前一日目標。'
             '濾網版：只有在 MA20_t > MA20_{t−3}（TAIEX 收盤 20 日均線，t−3 為往前 3 個 TAIEX 日）時，空手→持有的轉換才被允許；出場不受濾網影響。'
             '目標狀態從 score 第一個有值日起連續演化（1999 起暖身，M=600 約 2001 年中開始），各視窗開始時沿用當時已有的部位。'),
 },
 'variants': {
   'grid': 'score∈{z, zR2, zR2b} × N∈{16,18,20} × M∈{300,600} × thr∈{0.7,1.0} × filter∈{none, ma20up} = 72 組，全部報告',
   'main': 'zR2_N18_M600_T0.7_none（修正標準分、教學預設參數、不加濾網；判定只用這一組）',
   'baselines_report_only': ['MA200：TAIEX 收盤 > 200 日均線持有、否則空手（同一成交規則），作為常見擇時對照',
                             '曝險相同的隨機擇時：把主規格每日持有 0/1 序列在視窗內循環平移（offset 均勻抽 1000 次、種子 20261027），保留曝險與切換次數，報主規格 CAGR 在分布中的百分位'],
 },
 'execution': {
   'calendar': '交易在 0050 日曆（etf_0050.csv 有列的日子；2026-09-20 壞列剔除；2025-06-11～17 分割停牌無列＝不可交易）。',
   'timing': '0050 交易日 d 的開盤：目標部位＝d 之前最後一個 TAIEX 收盤日的目標狀態；與現有部位不同就在 d 開盤成交（訊號 t 收盤算、最早 t+1 開盤）。',
   'open_proxy': ('0050 CSV 沒有開盤價 → 開盤代理 O50_d＝0050 含息收盤_{前一 0050 日} × TAIEX 開盤_d ÷ TAIEX 收盤_{前一 0050 日}（TAIEX 收盤取當日或之前最後一筆）。'
                  'd 當日沒有 TAIEX 開盤 → 該日不可交易、順延。'),
   'limit': '開盤代理跳空 ≥ +9.4% → 視為鎖漲停買不到、順延；≤ −9.4% → 視為鎖跌停賣不掉、順延。',
   'nav': '日頻 NAV（0050 日曆）：持有時進場日 開盤→收盤、中間 收盤→收盤、出場日 前收→開盤；空手現金報酬 0；成本在成交時扣。',
   'sens_close_exec': '敏感度：改在 t 之後第一個 0050 日的收盤成交（不需開盤代理），只報告。',
 },
 'costs': {
   'main': '來回 0.1%（買 0.05%、賣 0.05%；任務指定 ETF 成本）— 判定用',
   'stress': '來回 0.3%（買 0.15%、賣 0.15%）',
   'exlib_etf': 'exlib 費率（ETF）：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋ETF 證交稅 0.1%＋滑價 0.1%（來回約 0.47%）',
   'exlib_stress': 'exlib 壓力：買 0.5%＋賣 0.5%',
   'gross': '不扣成本（參考）',
 },
 'benchmarks': {
   '0050': '0050 含息買進持有（日收盤對收盤，未扣成本）',
   'EW_pool': ('同池等權：上市面板（exlib.RET），交易日曆＝ret 有值檔數 ≥ 300 的日子（剔除 2004-01-02、2018-01-13）；股池＝前一日 60 日均成交值 ≥ NT$1,000 萬（rolling 60 在此日曆上）且前一日原始收盤 ≥ 10 元；'
               '每日等權平均 ret（未扣成本）。2019–2023 面板只有存活者，等權偏高。'),
   'monthly_keys': '月 t：策略與對手都複利到日曆月；面板 2024-01-02～02-14 無資料，所以 2024-01 與 2024-02 合併成一期（兩邊都合併）。',
 },
 'portfolio_stats': '年化（日曆年數）、日頻最大回撤、曝險、切換次數、逐年報酬；對 0050、對同池等權的日曆月超額平均與 t（btlib.tstat）；另報對 0050 的日超額 t（參考）。',
 'event_stats': ('每筆交易（進場開盤→出場開盤，未平倉以末日收盤結算）：net＝(1+raw)(1−賣)/(1+買)−1；'
                 '異常＝net − 同期流動性股池等權買進持有（訊號日股池、exlib.AO 進場日開盤→出場日開盤；面板日期以 ≥ 該日的第一個面板日對齊）；'
                 '對 0050 無條件＝net − 同視窗所有 0050 可交易日進場、持有同樣 0050 日數的開盤對開盤平均；t 先按進場月份平均再算。交易依進場日歸屬視窗。只報告。'),
 'periods': {'select_price': ['2005-01-01', '2019-04-30'], 'valid_price': ['2019-05-01', '2026-10-31'],
             'select_2021': ['2021-01-01', '2023-12-31'], 'valid_2024': ['2024-01-01', '2026-10-31'], 'y2024_2025': ['2024-01-01', '2025-12-31'],
             'full': ['2005-01-01', '2026-10-31']},
 'verdict_rule': ('只看主規格 zR2_N18_M600_T0.7_none、主成本（來回 0.1%）的擇時組合。挑選期 2005-01～2019-04 與驗證期 2019-05～2026-10 各自 PASS ⇔ '
                  '(i) 勝同池等權：日曆月超額（策略淨值 − 同池等權）平均 > 0 且 t ≥ 2；且 (ii) 至少不輸 0050：策略扣成本年化 ≥ 0050 買進持有年化。'
                  '兩段都過 → 可用；只有一段過 → 待觀察；都不過 → 不可用；某段主規格 0 次進場或少於 12 個月 → 該段資料不足，兩段都不足 → 無法測試。'
                  '其他 71 組、其他成本、事件統計、MA200 與隨機擇時對照一律報告但不改判定；若 exlib_etf 成本下判定不同，在 FINDINGS 標註。'),
 'defaults_not_in_tutorial': ['OLS 用原始點位、含截距', 'std ddof=0、視窗含當日', '起始空手', '濾網只限制進場', '0050 開盤代理（同 w3_adr25_breadth_timing）',
                              '跌停賣不掉順延（對稱於漲停）', '空手報酬 0（不計利息）'],
}
s = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(s.encode()).hexdigest(),
       'PREREG_time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(out['PREREG_sha256'], out['PREREG_time'])

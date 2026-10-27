#!/usr/bin/env python3
"""concentration_equity：事前登記。寫入 results.json 的 PREREG（附 sha256），run.py 開跑前比對雜湊。
執行：python3 -I -B prereg.py && python3 -I -B run.py"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
PREREG = {
 'engine': 'agents/revenue_longhistory/lhlib.py（直接 import）：長期月營收、前 300 名流動性池、0.5×52 週高點百分位＋0.5×近 3 月營收年增率百分位、'
           '每月第一個 ≥11 日交易日收盤訊號、隔日開盤成交、開盤漲停且未持有者跳過遞補（2015-06 前 7%、後 10%）、面板缺漏修補。'
           '本資料夾只把 lhlib 的選股迴圈與 sim_portfolio 改寫成「可給權重、可空手」的版本；改寫版在等權、不空手時必須與 lhlib 原版逐日一致（run.py 會檢查）。',
 'costs': {'main': 'lhlib cost_rates(0.001)：買 0.0855%+0.1%、賣 0.0855%+0.3% 稅+0.1%', 'stress': 'lhlib cost_rates("stress")：買賣各 0.5%（另報）'},
 'periods': {'P1_selection': ['first_signal(2006-04-11)', '2013-12-31'], 'P2_validation': ['2014-01-01', '2020-12-31'],
             'P3_holdout': ['2021-03-01', '2026-10-31'], 'context_only': {'OOS_2006_2020': ['first', '2020-12-31'], 'all': ['first', '2026-10-31']}},
 'A_concentration': {
   'variants': ['EW N ∈ {3,5,10,15,20,30,50}', 'SW10、SW30：分數加權，第 k 名（依分數排序、跳過開盤漲停後的實際持股順序）權重 ∝ N+1−k'],
   'report': ['各期：年化、日頻最大回撤、Calmar、對 0050 月超額 t、對同池等權 t（月份＝換股日到下一換股日，扣成本後）、年化波動、逐年、空頭期報酬與期內回撤',
              '個股風險：最差換股期報酬、最差日曆月報酬、最差單一持股貢獻（權重×該期報酬，毛額）、最差 1% 持股貢獻、單一持股貢獻 ≤ −5% 的次數',
              '換手率、壓力成本版年化',
              '劑量反應：EW 7 個 N 對 年化、MDD、Calmar、t_0050 的 Spearman 相關（每期）',
              '配對差：N=3、5、10 減 N=30 的月報酬差 t（每期），僅報告'],
   'verdict': '「集中有利」成立 ⇔ EW N=3、5、10 每一個在 P1、P2、P3 三段的年化與對 0050 t 都高於 EW N=30。另列單看 N=10 的結果（資訊用）。這不是挑參數，不據此改變主策略的 N。'},
 'B_equity_curve_filter': {
   'targets': ['EW n30', 'EW n10'],
   'shadow_nav': '不切換版主策略（同 N、等權、主成本）的日收盤淨值。開關訊號只用影子淨值（t 日收盤以前），場外期間影子淨值照算。',
   'rules': {
     'E1': '換股訊號日 d 收盤：影子淨值 < 自身 200 日簡單均線 → d+1 開盤起持現金（賣光）；之後的換股訊號日影子淨值 ≥ MA200 → d+1 開盤買當月新選股。MA200 未滿 200 日前視為場內。',
     'E2': '每日狀態機：場內時，影子淨值相對「上次回場（或起點）以來最高點」回撤 > 20% → 狀態轉場外；場外時，影子淨值創 60 日新高（收盤 ≥ 最近 60 個交易日最高收盤，含當日）或 ≥ MA200 → 狀態轉場內，峰值重設為當日影子淨值。'
           '執行：換股訊號日 d 依當日狀態決定 d+1 開盤持股或現金（每月一次）。',
     'E3': '每月最後一個交易日：影子淨值 < 最近 10 個月底淨值平均（含當月）→ 場外；未滿 10 個月底前視為場內。換股訊號日 d 依最近一個 ≤ d 的月底判斷，d+1 開盤執行。',
     'S1_same_engine': '大盤對照：加權指數收盤 < 自身 MA200（TAIEX 自己的日曆計算，asof 對齊），換股訊號日判斷、d+1 開盤執行、回場買新選股（等同 combo_search 引擎開關版）。另引用 timing_daily/results.json 的 S1 疊加版（t+1 收盤成交）。'},
   'cash': '場外現金報酬 0；成本照實扣（出場全賣、回場全買）；場外期間 held＝空，回場時開盤漲停跳過遞補照常',
   'selection_rule': 'n30、n10 各自在 P1 選日頻 Calmar（年化÷|日頻最大回撤|）最高的 E1/E2/E3；S1 不參加挑選，只作對照',
   'verdict': '選中者在 P2 與 P3 都滿足（Calmar > 不切換 且 年化 ≥ 不切換 − 3pp）→ 可用；只有一期 → 待觀察；都不 → 不可用（同 timing_daily）。另以同規則判 S1（同引擎）作直接比較。',
   'report': ['各期：年化、日頻 MDD、Calmar、對 0050 t、對等權 t、切換次數（出場＋回場）、場外月數比例', '逐年', '空頭期 2008、2011、2015、2018、2020-03（2020-02-01～04-30）、2022：報酬與期內回撤',
              '場外段落：起訖、影子淨值在場外期間的報酬（錯過的漲幅／避開的跌幅）', '選中者壓力成本版'],
   'sensitivity_only（不參加挑選與判定）': ['E1d：每日檢查影子淨值 vs MA200，狀態改變隔日開盤執行，回場買最近一次換股日的選股名單',
                                       'E2d：E2 狀態機狀態改變隔日開盤執行（同上）', 'E3m：月底判斷、次月第一個交易日開盤執行（同上）']},
 'known_caveats': ['2013-02～2014-03 訊號的營收腿有 IFRS 口徑斷點（落在 P1 尾與 P2 頭）', '2013 起營收只剩存活公司、2019–2023 價格面板只有存活者（P2、P3 有存活者偏誤）',
                   '面板 2024-01-02～02-14 無資料（淨值一步跳）', '處置濾網 2021-09 起才有資料'],
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob.encode()).hexdigest(),
       'PREREG_written_at': datetime.datetime.now().isoformat(timespec='seconds')}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG written', out['PREREG_sha256'][:12], out['PREREG_written_at'])

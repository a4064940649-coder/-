#!/usr/bin/env python3
"""事前設定：在寫回測程式、看任何結果之前寫入 results.json 的 PREREG。之後不得修改。"""
import json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREREG = {
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
 'rule_name': '一週短期反轉（Lehmann 1990 週反轉；產業調整、殘差、量縮輸家變體）',
 'sources': ['https://quantpedia.com/strategies/short-term-reversal-in-stocks',
             'https://www.quantconnect.com/learning/articles/investment-strategy-library/short-term-reversal',
             'https://www.robeco.com/en-int/insights/2011/11/short-term-residual-reversal',
             'https://www.cxoadvisory.com/momentum-investing/robustness-of-pure-stock-momentum-and-reversal/',
             'https://cris.fau.de/publications/110022484/',
             'https://ideas.repec.org/a/bla/jbfnac/v25y1998i3-4p469-483.html',
             'https://www.atlantis-press.com/article/190.pdf'],
 'panel': '上市日線面板（含已下市；2019–2023 只有存活者）。有值檔數 < 300 的日子（2018-01-13）整列剔除後才做任何 rolling／5 日報酬；flag 5 的價格視為缺值。排除 91xx 存託憑證。',
 'calendar': '週＝週一到週日（W-SUN）；訊號日 d＝每週最後一個交易日（剔除殘缺日後）。2004 年資料只用來暖機，第一個訊號日為 2005 年第一週最後交易日。',
 'signal_R5': 'R5_d = adj_d / adj_{d-5 個交易日} - 1（含息還原）。',
 'universe': ['d 收盤可知：60 日均成交值（剔除殘缺日後滾動，停牌日記 0）>= NT$1,000 萬；d 收盤原始價 >= 10 元；d 有收盤價與 R5',
              '剔除 d 收盤鎖漲停（日報酬 >= 漲停 - 0.6% 且收盤 >= 最高價×0.999）與收盤跌停（日報酬 <= -(跌停 - 0.6%)；面板無最低價，以收在跌停近似鎖跌停）。漲跌幅 2015-06-01 前 7%、之後 10%',
              '剔除 [d-5, d] 內任一天 flag ∈ {3,4,5} 的股票（全市場同日 flag 4 >= 50 檔的面板缺口日不算）',
              'd+1（下週第一個交易日）必須有開盤價（可成交）；股池 < 100 檔的週不交易'],
 'main': '依 R5 由低到高排序，取最低 10%（目標檔數 = round(0.1 × 股池檔數)）等權；d+1 開盤買進，持有到下一個訊號日的隔日開盤換股（週換股）。d+1 開盤漲停（含息跳空 >= 漲停 - 0.6%）且目前未持有者視為買不到，由下一名遞補。',
 'variants': {
   'a_industry_adj': 'R5 減去同產業（stock_list.csv industry；不在名單者依代碼前兩碼回推傳產，電子類回推不了的併「未分類」）股池內等權 R5 後再排序取最低 10%；股池內 < 5 檔的產業併成一組「小產業合併」再算平均（Da-Liu-Schaumburg 2014）。',
   'b_residual': '週報酬＝相鄰訊號日收盤之間的 adj 報酬；市場＝當週股池（同主版本股池條件，不含 flag/鎖漲跌停以外的限制）等權週報酬。每檔用最近 52 週（含當週）對市場做 OLS（含截距），至少 40 週有效；分數＝當週殘差 / 52 週殘差標準差（ddof=2）；取分數最低 10%（Blitz-Huij-Lansdorp-Verbeek 2013／Robeco 2011）。',
   'c_low_volume_losers': '主版本輸家（R5 最低 10%）中只保留「最近 5 個交易日平均日成交量 < 前 50 個交易日（約 10 週）平均日成交量」者；成交量＝value / close_raw（股數）。不遞補到 10%，有幾檔買幾檔；< 5 檔的週改持有全部輸家（避免空倉）——此 fallback 一併報告發生週數（Conrad-Hameed-Niden 1994）。',
   'd_hold2w': '主版本訊號，只在每隔一個訊號日換股（持有 2 週、隔週換股）；主相位從第 1 個訊號日起算，另一相位一併報告。'},
 'engine': 'exec_rebuild/exlib.py 的 sim_portfolio 與 cost_rates（不重寫）：換股時漂移後權重調回等權，買賣金額分別乘費率；持有中停牌／下市以最後收盤價 ffill。限制：引擎不模擬跌停賣不掉。',
 'costs': {'normal': 'cost_rates(0.001)：買 0.0855%+0.1%，賣 0.0855%+0.3% 稅+0.1%（來回約 0.57%）', 'stress': "cost_rates('stress')：買 0.5% + 賣 0.5%（來回 1%）", 'gross': '0'},
 'benchmarks': {'EW_pool': '同一股池（主版本股池，每週）等權、同一換股時點（d+1 開盤）、不扣成本', '0050': '0050 含息指數'},
 'stats': '日頻 NAV：年化（實際天數）、年化波動、日頻最大回撤、逐年。t：以日曆月（月底收盤→月底收盤）報酬差做 t（月份分群），對 0050 與對同池等權；另報週超額的平均。',
 'windows': {'select': ['2005-01-01', '2019-04-30'], 'valid': ['2019-05-01', '2026-10-31'], 'y2024_2025': ['2024-01-01', '2025-12-31'],
             'valid_ex_src3': '驗證期剔除 2024-02-15～2025-09-23（src=3 的 13:20 近似收盤）期間的日報酬與月份'},
 'verdict_rule': '以主版本、一般成本判定：挑選期與驗證期扣成本後對同池等權月 t >= 2 且年化 >= 0050 年化 → 可用；只有一段過 → 待觀察；否則不可用。變體只做參考（5 個版本同測，需注意多重檢定）。',
 'no_tuning': '10%、5 日、52 週、10 週、2 週皆為教學／原論文預設；看結果後不改。',
}
out = HERE / 'results.json'
json.dump({'PREREG': PREREG}, open(out, 'w'), ensure_ascii=False, indent=1)
print('wrote', out)

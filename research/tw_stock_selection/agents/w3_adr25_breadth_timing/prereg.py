#!/usr/bin/env python3
"""事前設定：寫回測程式、看任何結果之前寫入 results.json 的 PREREG（附 sha256）；run.py 開跑前核對雜湊。之後不得修改。"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
PREREG = {
 'rule_name': '騰落レシオ（25 日漲跌家數比 ADR25）逆勢擇時：ADR 由 ≤70 回升到 >70 隔日開盤買 0050，ADR ≥120 或持有 60 日賣出',
 'type': '擇時（事件型）：以每筆交易的事件報酬判定；另報日頻擇時組合（0050／現金）',
 'sources': ['https://media.monex.co.jp/articles/-/27603', 'https://media.monex.co.jp/articles/-/11532',
             'https://media.moneyforward.com/articles/8495', 'https://stock-marketdata.com/advance-decline-growth-market-tse'],
 'rule_verbatim': ('股池：面板全部上市股（每日只計有收盤值者；有值檔數 <300 的日子整天剔除再滾動）。每日 up_t＝ret>0 檔數、dn_t＝ret<0 檔數（平盤不計）。'
                   'ADR25＝Σ25日 up ÷ Σ25日 dn ×100。進場：ADR25 由 ≤70 回升到 >70 的那天，收盤算訊號、隔日開盤買 0050。'
                   '出場：ADR25 ≥120 的隔日開盤賣出，或持有滿 60 個交易日，兩者取先到。空手時持現金（報酬 0）。'),
 'breadth': {
   'panel': 'scratchpad/panel/ret.parquet（含息日報酬；上市 1,249 檔，含已下市；2019–2023 只有存活者）',
   'valid_days': ('交易日曆 T＝面板日期中 ret 有值檔數 ≥ 300 者（剔除 2004-01-02 與 2018-01-13）；2024-01-02～02-14 面板本來就沒有列。'
                  '廣度日 V＝T 去掉 2024-02-15（該日 ret 跨 2023-12-29→2024-02-15 六週，不是日漲跌）。'),
   'counts': 'up_t＝#(ret>0)、dn_t＝#(ret<0)、ret==0 或缺值不計；不另排除 flag（漲跌只看正負號）',
   'ADR': 'ADR_N(t)＝Σ_{最近 N 個 V 日} up ÷ Σ dn ×100（rolling 在 V 上做，min_periods＝N；跨 2024 缺口的視窗照算）。主 N＝25，敏感度 N＝10。',
   'known_noise': ['src 3（2024-02-15～2025-09-23）收盤為 13:20 近似，漲跌家數有誤差', '2025-01-21 close_raw 為中午價，影響 01-21、01-22 的漲跌',
                   '2019–2023 只有存活股 → 下跌家數偏少、ADR 偏高（低門檻較難觸發）；故事前加入分位數版本'],
 },
 'variants': {
   'grid': 'N∈{25,10} × 門檻(L,H)∈{(70,120),(75,115)} × 模式∈{cross（≤L 回升到 >L 當天）, immediate（≤L 當天就買、不等回升）} = 8 組',
   'quantile': ('Q 版（N＝25）：L_t＝ADR25 在最近 1,250 個 V 日（含 t、min_periods 750）的第 5 百分位，H_t＝第 95 百分位；'
                'cross＝ADR_{t-1} ≤ L_{t-1} 且 ADR_t > L_t；immediate＝ADR_t ≤ L_t；出場 ADR_t ≥ H_t 或 60 日。2 組。'),
   'main': 'N25_L70_H120_cross（教學原版；判定只用這一組）',
   'all_reported': '10 組全部報告，不挑最好',
 },
 'execution': {
   'signal': '訊號日 t＝V 日收盤後算 ADR；只在空手時（含當天開盤剛賣出）接受進場訊號；持有中的新進場訊號忽略（不加碼）。',
   'cross': 'cross 模式：ADR_{t-1} ≤ L 且 ADR_t > L（t-1 為前一個 V 日）',
   'immediate': 'immediate 模式：ADR_t ≤ L 且空手（持有到期出場後若仍 ≤ L，隔日可再進場）',
   'entry': ('進場日 e＝t 之後第一個 0050 可交易日（T 且 0050 CSV 有列），以 0050 開盤代理價買。0050 CSV 沒有開盤價 → '
             '開盤代理 O50_d＝0050 含息收盤_{前一 0050 交易日} × TAIEX 開盤_d ÷ TAIEX 收盤_{前一 TAIEX 交易日}（在 0050 自己的日曆上算）。'
             '開盤跳空 ≥ 9.4%（10% 漲停、視為鎖漲停買不到）→ 順延到下一個可交易日開盤。'),
   'exit': ('出場取先到：(a) 從 e 當天收盤起任一 V 日 s 的 ADR_s ≥ H → s 之後第一個 0050 可交易日開盤賣；'
            '(b) 時間：e 之後第 60 個 T 日（持有滿 60 個交易日）開盤賣（該日 0050 不可交易則順延）。'
            '資料末日（2026-10-22）仍持有 → 以末日收盤結算、標記 truncated、照樣計入。'),
   'data_fix': '0050 CSV 2026-09-20（週日）壞列設缺值並剔除（沿用 w4_0050_drawdown_add）；0050 2025-06-11～17 分割停牌 → 不可交易、價格 ffill。',
   'pre_2009_proxy': '0050 含息資料自 2003-07 起即有，故 2009 前不需代理，全期直接用 0050；另報「交易標的改為流動性股池等權」敏感度（見 sens_EW_target）。',
 },
 'costs': {
   'main': '來回 0.1%（買 0.05%、賣 0.05%；任務指定 ETF 成本）',
   'stress': '來回 0.3%（買 0.15%、賣 0.15%）',
   'exlib_etf': 'exlib 費率結構（ETF）：買＝0.1425%×0.6＋滑價 0.1%；賣＝0.1425%×0.6＋ETF 證交稅 0.1%＋滑價 0.1%（來回約 0.47%），另報',
   'exlib_stress': 'exlib 壓力：買 0.5%＋賣 0.5%，另報',
   'net': 'net＝(1+raw)×(1−賣)/(1+買)−1',
 },
 'event_stats': {
   'raw': '0050：O50[x]/O50[e]−1（x＝出場日；truncated 用末日收盤）',
   'abn_EW': ('異常報酬＝net − 同窗上市等權（主＝流動性股池：t 日 60 日均成交值 ≥ NT$1,000 萬（rolling 60 在 T 上、min_periods 60）且 t 收盤原始價 ≥ 10 元、e 有開盤；'
              '每檔 exlib.AO[x]/AO[e]−1 的買進持有等權平均，未扣成本）；另報 EW_all（全部 e 有開盤的上市股）'),
   'vs_0050_uncond': ('對 0050 無條件：net − base_h，base_h＝同一視窗內所有 0050 可交易日 d 為進場日、持有 h 個 T 日（h＝該筆實際持有 T 日數）的 O50 開盤對開盤平均報酬（不扣成本）。'
                      '衡量「依訊號進場」是否勝過「隨便哪天進場持有同樣天數」。'),
   'cluster_t': '同一進場月份的交易先平均，t＝月份平均序列的 t（btlib.tstat）；另報 t_trade（僅參考）、勝率、平均持有日數、ADR 出場比例',
   'window_assignment': '交易依進場日 e 歸屬視窗',
 },
 'portfolio_stats': ('擇時組合日頻 NAV（T 日曆）：持有時 0050（進場日 開盤→收盤、中間 收盤→收盤、出場日 前收→開盤），空手現金 0；成本在成交時扣。'
                     '報年化、日頻最大回撤、曝險比例、逐年、交易數；對 0050 買進持有與流動性股池等權（每日等權、收盤對收盤、未扣成本；股池＝前一日條件）的日曆月超額 t。'),
 'sens_EW_target': '敏感度：同訊號但交易標的改為流動性股池等權（e 開盤→x 開盤，成本用 exlib_etf 以下的 main），報對「無條件同持有天數 EW」的超額與 t；不參與判定。',
 'periods': {'select_price': ['2005-01-01', '2019-04-30'], 'valid_price': ['2019-05-01', '2026-10-31'],
             'select_2021': ['2021-01-01', '2023-12-31'], 'valid_2024': ['2024-01-01', '2026-10-31'], 'y2024_2025': ['2024-01-01', '2025-12-31'],
             'full': ['2005-01-01', '2026-10-31']},
 'verdict_rule': ('只看主規格 N25_L70_H120_cross、主成本（來回 0.1%）。對挑選期 2005-01～2019-04 與驗證期 2019-05～2026-10 各自：'
                  'PASS ⇔ (i) 勝同池等權：異常報酬（對流動性股池 EW）平均 > 0 且月份分群 t ≥ 2（進場月份 < 3 視為不過）；'
                  '(ii) 至少不輸 0050：對 0050 無條件同持有天數的平均超額 ≥ 0。兩段都過 → 可用；只有一段過 → 待觀察；都不過 → 不可用；'
                  '某段主規格交易 < 3 筆 → 該段視為資料不足，兩段都不足 → 無法測試。其他 9 組、其他成本、組合版（含「擇時組合年化 ≥ 0050 買進持有」）一律報告但不改判定。'),
}
s = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(s.encode()).hexdigest(),
       'PREREG_time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(out['PREREG_sha256'], out['PREREG_time'])

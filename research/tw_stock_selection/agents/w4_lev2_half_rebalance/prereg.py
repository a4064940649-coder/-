#!/usr/bin/env python3
"""事前設定（看結果前固定）。寫入 results.json 的 PREREG，附 sha256 與時間戳；run.py 開跑前核對雜湊。"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
PREREG = {
  'rule_source': '「正 2（00631L）50％＋現金 50％ 再平衡」：商周 2025-03-06、2026-07-07 文章；Dcard money 261327021；PTT Stock 1fHZvH9A、hRm1',
  'rule_verbatim': '資產：元大台灣50正2（00631L）與現金。初始 50/50。(A) 每年 1 月第一個交易日回到 50/50；(B) 每半年（1、7 月）檢查一次；(C) 正2 占比偏離 50% 超過 ±10 個百分點，隔日開盤調回 50/50。',
  'assets': {'L2': '00631L 還原（含息）日序列 finlab-txd/data/raw/etf_00631L.csv（2014-10-31 起）；之前用合成',
             'cash': '年利率 1%（主）／0%（敏感度），依日曆日計息'},
  'synthetic_L2': {'formula': 'r_L2 = 2*r_0050TR - (1.0% + 1.0*short_rate)/252，每日重設槓桿',
                   'short_rate_main': '金融業隔夜拆款利率年均值（近似，依記憶硬編碼，見 run.py RATE_ON）',
                   'short_rate_sens': '央行重貼現率（較保守，見 run.py RATE_CBC）',
                   'hybrid_main': '2014-10-31 以前合成、之後用實際 00631L',
                   'check': '2014-11～2026-10 實際日報酬與合成比較（追蹤誤差、年化落差）；2020-08 起與 etf_tr_m 月報酬核對（2026-10 為不完整月，排除）',
                   'note': '00631L 追蹤臺灣50「價格」指數 2 倍，期貨持有 + 現金擔保 ≈ 2×TR − 1×短利 − 費用'},
  'rebalance_specs': {
    'A_annual': '每年第一個交易日把 L2 調回 50%（前一日收盤已知權重）',
    'B_semiannual': '每年 1 月、7 月第一個交易日無條件調回 50/50',
    'C_band10': '每日收盤算 L2 占比；若 |w−50%| > 10pp，下一交易日成交調回 50/50'},
  'execution': 'ETF 沒有開盤價資料 → 以「訊號日的下一個交易日收盤」代替隔日開盤（A/B 的訊號日為前一年底／6 月底最後交易日收盤，成交在 1／7 月第一個交易日收盤）。若成交日 L2 收盤漲幅 ≥ 19.4%（20% 漲停，視為鎖漲停）則該日不能買、順延；跌幅 ≤ −19.4% 不能賣、順延。',
  'costs': {'main_slip0.001': '買 = 0.1425%×0.6 + 0.1%；賣 = 0.1425%×0.6 + ETF 證交稅 0.1% + 0.1%',
            'exlib_stocktax': '同上但證交稅用 exlib 的 0.3%（保守）',
            'stress': '買 0.5%、賣 0.5%（exlib 壓力情境）',
            'gross': '無成本'},
  'benchmarks': {'0050': '100% 0050 含息買進持有（初始買進成本同算）',
                 'L2_100': '100% 正2 買進持有',
                 'EW': '同池等權：momentum_family/long_nav.parquet 的 EW（流動性前 300 名等權、月換股、gross；2019–2023 面板只有存活者，偏高）',
                 'MA200_0050': '對照（防守效果）：每月第一個 ≥11 日交易日收盤，加權指數 < 200 日均線 → 下一交易日收盤賣 0050 改持現金 1%；站回則買回（同主策略開關的檢查頻率）。另引用 timing_switch 主策略 MA200 開關結果。'},
  'windows': {'select_price': ['2005-01-01', '2019-04-30'], 'valid_price': ['2019-05-01', '2026-10-31'],
              'select_2021': ['2021-01-01', '2023-12-31'], 'valid_2024': ['2024-01-01', '2026-10-31'], 'y2024_2025': ['2024-01-01', '2025-12-31'],
              'full_2005': ['2005-01-01', '2026-10-31'], 'long_synth_2003-07': ['2003-07-01', '2026-10-31'],
              'bear_2008': ['2008-01-01', '2008-12-31'], 'bear_2022': ['2022-01-01', '2022-12-31']},
  'simulation': '主：2005-01-03 起連續模擬、各期間切片（狀態延續）；穩健性：驗證期 2019-05-02 重新 50/50 起跑；長歷史：2003-07-01 起（2014-10 前合成）。',
  'stats': '日頻 NAV：年化（實際日數）、日頻最大回撤、波動；t = 月報酬（月底→月底）超額的 t（月份分群，扣成本後）對 0050、對 EW、對 100% 正2；逐年報酬。',
  'data_cleaning': '0050 的 2026-09-20（週日）收盤 −7.2% 隔日 +9.2%，為壞列：該日 0050 設為缺值後 ffill。共同日曆 = 0050 日期 ∪ 00631L 日期，水準 ffill。',
  'verdict_rule': '主規格 = (A) 年度再平衡、現金 1%、成本 slip0.001。挑選期與驗證期扣成本後都勝 EW（月 t ≥ 2）且年化不低於 0050 → 可用；只有一段過 → 待觀察；否則不可用。B、C 與所有敏感度一律報告，不用來改判定。',
}
s = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(s.encode()).hexdigest(), 'PREREG_time': datetime.datetime.now().isoformat(timespec='seconds')}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(out['PREREG_sha256'], out['PREREG_time'])

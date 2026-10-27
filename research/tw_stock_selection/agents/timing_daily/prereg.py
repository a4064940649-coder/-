#!/usr/bin/env python3
"""timing_daily：事前登記。先寫入 results.json 的 PREREG，run.py 會檢查它存在且未被改動（雜湊）後才跑。"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
PREREG = {
 'question': '擇時開關（加權指數均線）以日頻淨值套在主策略 n30／n10 與 0050，長期是否改善 Calmar 且年化損失 ≤ 3pp',
 'data': {'strategy_nav': 'agents/revenue_longhistory/oos_nav.parquet 欄 M2_n30_slip0.001、M2_n10_slip0.001（日收盤淨值，已含選股成本）',
          '0050': '/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv（含息 adj），對齊策略淨值的交易日（ffill）',
          'taiex': '/home/user/jonatw/finlab-txd/data/raw/taiex_twii.csv close，均線在 TAIEX 自己的完整日曆上計算後 asof 對齊'},
 'signals': {
  'S1': '加權指數收盤 < MA200（200 日簡單均線）→ 場外；只在換股日（每月第一個日期 ≥ 11 的交易日）收盤檢查，其餘日子沿用（現版）',
  'S2': '同 S1 但每日收盤檢查',
  'S3': '每日檢查、2% 緩衝帶：場內時收盤 < MA200×0.98 才出場；場外時收盤 > MA200×1.02 才回場；起始狀態＝收盤 ≥ MA200',
  'S4': 'Faber 10 個月均線：每月最後一個交易日收盤 < 最近 10 個月底收盤平均 → 場外，月底檢查，次月沿用',
  'S5': '收盤 < MA120 → 場外，每日檢查'},
 'execution': {'main': '訊號在 t 日收盤後才知道（只用 t 日收盤以前的資料），t+1 日收盤成交；新部位從 t+2 日的日報酬開始適用（pos_t = sig_{t-2}）。因為只有收盤淨值，不用開盤價。',
               'sensitivity_only': '理想版：t 日收盤成交（pos_t = sig_{t-1}），只做敏感度，不用於挑選或判定'},
 'targets': ['M2_n30_slip0.001', 'M2_n10_slip0.001', '0050'],
 'off_market': {'main': '現金報酬 0', 'alt': '現金年利率 1%（依日曆日複利），另報'},
 'costs': {'strategy': {'exit_sell': 0.00486, 'reentry_buy': 0.00186}, '0050': {'exit_sell': 0.001855, 'reentry_buy': 0.000855},
           'reentry_assumption': '場外期間策略淨值照算；回場時以當時策略淨值報酬接續（等於買回同一組股票），扣回場成本。成本在成交日收盤扣。'},
 'periods': {'selection': ['2006-04-01', '2013-12-31'], 'validation': ['2014-01-01', '2020-12-31'], 'holdout': ['2021-03-01', '2026-10-31'],
             'full_context_only': ['2006-04-01', '2026-10-31']},
 'state_carry': '整段連續模擬，期間邊界不重設狀態；各期統計只取該期日報酬',
 'selection_rule': '每個對象各自在挑選期（現金 0、主成交時點、含成本）選日頻 Calmar（年化 ÷ |日頻最大回撤|）最高的開關；驗證期、保留期只看不調。主策略的判定以 n30 為主，n10、0050 各自另判。',
 'verdict_rule': '選中開關在驗證期與保留期都滿足（Calmar > 不切換）且（年化 ≥ 不切換年化 − 3pp）→ 可用；只有一期滿足 → 待觀察；都不滿足 → 不可用',
 'report': ['每個開關×對象×期間：年化、日頻最大回撤、Calmar、對 0050 月超額 t（日曆月）、切換次數（出場＋回場）、場外日數比例、成本拖累（零成本年化 − 含成本年化）',
            '逐年報酬（全期）', '空頭期：2008、2011、2015、2018 全年、2020-03（2020-02-01～2020-04-30）、2022 全年的報酬與期內最大回撤',
            '錯過反彈：場外日策略（或 0050）本身日報酬的加總與連乘；並依場外段落分正段（錯過的漲幅）與負段（避開的跌幅）'],
 'known_caveats': ['面板 2024-01-02～2024-02-14 無資料，策略淨值在此是一步跳；訊號在該期間以 TAIEX 計算但只能在 2024-02-15 起執行',
                   '2018-01-13 為面板異常日（TAIEX 無此日），訊號 asof 沿用前一日',
                   '0050 在 2025-06-11～17 分割停牌，無價格 → ffill'],
}
blob = json.dumps(PREREG, ensure_ascii=False, sort_keys=True)
out = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(blob.encode()).hexdigest(),
       'PREREG_written_at': datetime.datetime.now().isoformat(timespec='seconds')}
(H / 'results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print('PREREG written', out['PREREG_sha256'][:12], out['PREREG_written_at'])

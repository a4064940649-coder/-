#!/usr/bin/env python3
"""strategy_ensemble 事前設定：在看任何組合結果（及驗證期、保留期的相關矩陣）之前寫入 results.json 的 PREREG，附 sha256。
執行：python3 -I -B prereg.py（只寫一次；run.py 開頭比對雜湊）"""
import json, hashlib, datetime
from pathlib import Path
H = Path(__file__).resolve().parent
S_LONG = ['N30', 'N10', 'H52', 'H52K90', 'TURN_C', 'MINV', 'DISP_PX', 'E0050', 'E0050_MA']
S_RECENT = S_LONG + ['DISP_REAL', 'SURGE']
PREREG = {
 'question': '多個獨立策略各自運作、在投組層級組合（不是合成單一分數），能否在樣本外提高風險調整後報酬與統計顯著性；是否可取代現行建議（n10 80%＋處置 20%）',
 'strategies': {
  'N30': '主策略 n30（revenue_longhistory M2_n30_slip0.001）', 'N10': '主策略 n10（M2_n10_slip0.001）',
  'H52': '52 週高點單獨 n30（momentum_family P1_52wHigh_n30_0.001）', 'H52K90': '52 週高點 n30＋K90 緩衝（turnover_buffer A_P1only_n30_K90_0.001）',
  'TURN_C': '營收轉機版本 C、持有 1 月（w1_rev_yoy_turnaround C_H1_0.001）', 'MINV': 'Minervini 趨勢模板主設定（w1_minervini_trend_template BASE_ALL_ME，滑價 0.1%）',
  'DISP_PX': '近似處置股 10 槽（disposal_bear_longhistory 近似規則＋dlib.Sleeve，本資料夾延伸到 2026-10；閒置持 0050；100 萬；0.8% 來回）',
  'E0050': '0050 含息', 'E0050_MA': '0050＋S1 MA200 開關（timing_daily 主版本，本資料夾重建，與原數字核對一致）',
  'DISP_REAL': '真實處置股 10 槽（disposal_reconcile sleeve_nav A_0050，上市正式版、閒置持 0050；2021-10 前以 0050 報酬代替＝閒置資金）',
  'SURGE': '營收驚喜 SURGE 主版本（w3_revenue_sue S_main_0.001）'},
 'idle_choice_note': '處置部位閒置資金一律持 0050（與現行建議 run_alloc2 相同）；閒置現金版只存檔參考，不參與',
 'sets': {'S_long': S_LONG, 'S_recent': S_RECENT},
 'calendar': '共同日曆＝主策略淨值的面板交易日（2006-04-11～2026-10-12）；各策略淨值 ffill 對齊（2024-01-02～02-14 面板缺口一步跳）',
 'simulation': {
  'rebalance': '每個日曆月最後一個交易日收盤；權重只用到前一交易日（t−1）為止的資料計算，於 t 收盤成交；月內權重隨淨值漂移',
  'start': {'S_long': '2006-05-11 收盤首次配置（TURN_C 最早有淨值的日子），之後連續模擬，期間邊界不重設', 'S_recent': '2021-02 最後交易日收盤首次配置'},
  'switch_cost': '每次再平衡成本 = 0.5% × Σ_i |w_target_i − w_drift_i|（保守：每一邊 0.5%）；首次配置 Σ|w|=1 → 0.5%；成本在成交日收盤自淨值扣除',
  'switch_cost_stress': '另報 1%（只報 E1～E6，不用於挑選或判定）',
  'warmup': '某次再平衡時，集合內任何策略的歷史不足該方法所需視窗（E2 60 日、E3 252 日、E4 126 日），該次改用 E1 等權'},
 'methods': {
  'E1': '等權，每月再平衡',
  'E2': '風險平價（簡化）：w ∝ 1/σ_i，σ_i＝過去 60 個交易日日報酬標準差；σ 下限＝年化 5%（避免 E0050_MA 場外期間 σ=0）',
  'E3': '最小變異：過去 252 個交易日日報酬樣本共變異，min w\'Σw，0 ≤ w ≤ 0.5，Σw=1（SLSQP）；兩檔時即 50/50',
  'E4': '策略動能輪動：持有過去 126 個交易日報酬最高的 2 個策略，各 50%',
  'E5': '主策略 N10 60%＋集合內其他策略等分 40%（其他包含 N30）',
  'E6': '現行建議（基準）：N10 80%＋處置 20%（S_long 用 DISP_PX，S_recent 用 DISP_REAL），每月再平衡；不參加挑選'},
 'periods': {
  'S_long': {'selection': ['2006-05-12', '2013-12-31'], 'validation': ['2014-01-01', '2020-12-31'], 'holdout': ['2021-03-01', '2026-10-12']},
  'S_recent': {'selection': ['2021-03-01', '2023-12-31'], 'validation': ['2024-01-01', '2026-10-12']},
  'note': 'S_long 起點因 TURN_C 2006-05-11 才有淨值，比 2006-04 晚一個月；主策略規則本身是在 2021～2026 設計的，所以 S_long 的「保留期」對 N10/N30 不是真正樣本外，只用來比較組合方法'},
 'metrics': {
  'CAGR/vol/maxDD/Calmar': '視窗內日報酬；年化以日曆天數；vol＝日報酬標準差×√252；Calmar＝年化/|日頻最大回撤|',
  't_vs_0050': '日曆月報酬差（組合 − 0050）的 t', 'paired_t_vs_N30_N10': '日曆月報酬差（組合 − N30／N10）的 t',
  'yearly': '逐年報酬', 'bear': {'2008': ['2008-01-01', '2008-12-31'], '2011': ['2011-01-01', '2011-12-31'], '2015': ['2015-01-01', '2015-12-31'],
                                  '2018': ['2018-01-01', '2018-12-31'], '2020-03': ['2020-02-01', '2020-04-30'], '2022': ['2022-01-01', '2022-12-31']}},
 'correlation': '各策略日曆月報酬 Pearson 相關矩陣，分 S_long 挑選／驗證／保留／全期、S_recent 挑選／驗證報告（只是描述，不用於挑選）',
 'selection_rule': '每個集合各自：只看挑選期，在 E1～E5 中選對 0050 月 t 最高者；與第二名 t 差 ≤ 0.05 視為平手，取挑選期 Calmar 高者',
 'verdict_rule': {
  'pass_condition': '某期：選中方法 Calmar > N10 同期 Calmar，且對 0050 t ≥ N10 同期對 0050 t',
  'S_long': '驗證期與保留期都過 → 可用（可取代現行建議）；只過一期 → 待觀察；都不過 → 不可用',
  'S_recent': '只有一段樣本外（驗證期）：過 → 待觀察（短樣本，最高到待觀察）；不過 → 不可用',
  'E6_compare': '另報選中方法對 E6 的各期 Calmar、t 與月配對 t（不改判定）'},
 'EXHAUSTIVE (補充任務，同樣事前固定)': {
  'universe': 'S_long 所有大小 ≥2 的子集（502 個）、S_recent 所有大小 ≥2 的子集（2,036 個），各配 E1、E2 → 1,004／4,072 個組合',
  'pick': '只看挑選期，對 0050 月 t 最高的前 3 名（同 t 取 Calmar 高者）；第 1 名套用同一判定規則（S_long 兩期、S_recent 一期），前 3 名全部報告',
  'distribution': '全部組合在驗證期（S_long 另報保留期）的分佈：年化、Calmar、對 0050 t 的中位數與四分位；Calmar > N10 的比例、t ≥ N10 的比例、兩者都過的比例',
  'multiple_testing': {
   'loss': '日曆月報酬差 d_k = r_k − r_N10（基準為主策略 N10）',
   'SPA': 'Hansen (2005) SPA，studentized，consistent 版（SPA_c），stationary bootstrap 平均區塊 6 個月，B=2000，seed 20261027；另報 RC（White）p 值',
   'max_t': '同一 bootstrap 下 max-t（studentized 平均差，中心化）的族系 p 值，並報第 1 名的調整後 p',
   'windows': 'S_long：挑選、驗證、保留各做一次；S_recent：挑選、驗證',
   'secondary_diagnostics': ['同樣檢定改用基準 0050（d=r_k − r_0050）', '風險調整版：r_k 依該期 σ_N10/σ_k 縮放後再減 r_N10（事後縮放，只當診斷）']},
  'marginal_contribution': '對每個策略 s、每種加權：所有不含 s 的子集 S（|S|≥1，單一策略即其本身淨值、同一模擬器）計算 metric(S∪{s}) − metric(S)，報平均、中位數與 >0 比例；metric＝Calmar 與對 0050 t；分期報告'},
 'written_at': datetime.datetime.now().isoformat(timespec='seconds'),
}
res = {'PREREG': PREREG, 'PREREG_sha256': hashlib.sha256(json.dumps(PREREG, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
(H / 'results.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(res['PREREG_sha256'], PREREG['written_at'])

#!/usr/bin/env python3
"""步驟 4：事前登錄（PREREG）。必須早於任何 2019 年後的分組輸出（斷言 RESULTS_OOS、RESULTS_R2、alpha_all_v1.parquet 都不存在）。"""
import sys, datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fa_common as C

r = C.load_res()
assert 'PREREG' not in r, 'PREREG 已存在，不可重登'
assert 'RESULTS_OOS' not in r and 'RESULTS_R2' not in r and not (C.DATA / 'alpha_all_v1.parquet').exists(), '已有 2019 年後輸出，不能事前登錄'
AG = C.AG; V7 = C.V7DIR
files = [C.HERE / f for f in ('fa_common.py', 'fa_build.py', 'fa_stats.py', 's1_gate.py', 's2_explore.py', 's3_iterate.py', 's4_prereg.py',
                              's5_oos.py', 's6_r2.py', 'forward_track.py', 'explore.json', 'iterations.json')]
files += [V7 / 'v7env.py', V7 / 'ecore7.py', V7 / 'data' / 'model_spec7.json', V7 / 'data' / 'plans_j7.csv', V7 / 'data' / 'jscan_end_days.json',
          V7 / 'navs_v7.parquet', AG / 'independent_audit' / 'ia_core.py', AG / 'main_v2' / 'v2_core.py', AG / 'o_disposal_rejudge' / 'drj_core.py',
          AG / 'o_infra_v3_rebuild' / 'o3_core.py', AG / 'o_v4_revenue_patch_official' / 'v4lib.py', AG / 'o_stagger_tranche' / 'st_core.py',
          AG / 'o_sbl_foreign_level' / 'fo_panel.parquet', AG / 'o_capmatched_selection' / 'raw' / 'agents' / 'tw50_members.json',
          AG / 'o_capmatched_selection' / 'data' / 'membership_pre2015.json', AG.parent / 'main_picks_today.py']
files += sorted((V7 / 'data').glob('*.parquet'))[:0] + [V7 / 'data' / f'{k}.parquet' for k in ('adj', 'close_raw', 'open', 'high', 'low', 'ret', 'flag', 'src')]
files += [V7 / 'data_freeze_v7.json']
PRE = {
 'title': '營收訊號新鮮度：關注度 × 換股時點 j 的異質衰減（o_rev_freshness_attention）',
 'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'selected_version': 'v1（iterations.json；選版規則：建構檢查通過即選 v1，不依數值）',
 'n_versions_tried': 5,
 'data': 'V7 凍結面板（o_v7_listing_exec_engine/data）＋V4 引擎（R_FIX／Y5_FIX 營收、官方處置成交日口徑、P 下市價）；2019～2023 價格面板只有存活者（加註）',
 'universe': 'AB_V7 合格池 el（前 300 流動性池、排除金融與 KY／DR、成交日仍處置者、次日無開盤、缺營收或 52 週比）',
 'j_definition': 'o_stagger_tranche.st_core.day_for(m, j)，j＝0..12；錨定月＝V7 官方訊號月（2006-04～2026-09）',
 'AB': 'st_core.unavail_for（期限日當天依產業延後機率 p 抽樣，seed 0～4 平均；2020-02 延長、2026 INS15）',
 'attention_primary': 'A＝el 內訊號日 V60（gapfix）的池內百分位三分位：≤1/3 低、(1/3,2/3] 中、>2/3 高',
 'attention_secondary': {'M50': '0050 成分（成交日開盤 PIT；2015-09-18 前為 o_capmatched 規則名單）對非成分',
                         'FO': '外資持股比率三分位（o_sbl fo_panel，2021-10-08 起回推值；只當敏感度、無前期）',
                         'MCAP': 'PIT 市值三分位：第 2 題 PIT 股數未過閘門／未產出 → 不用，計為「無法使用」'},
 'portfolio': '各組內依主策略分數（官方同分序）取前 n_g＝max(1, round(0.2×N_g)) 檔等權；開盤漲停且上期未持有者跳過遞補；'
              '成交日（訊號日次一交易日）開盤 → 下一有效錨定月同一 j 的成交日開盤；還原開盤比；停牌下市持平',
 'alpha': 'α_g(m,j)＝組合報酬 − 一般成本（買 0.1855%×新進比例＋賣 0.4855%×賣出比例）− 同組 el 等權報酬（零成本）',
 'test_statistics': {
   'b_g': '每月 α_g(m,j) 對 j 的 OLS 斜率（≥10 個 j），月平均；t＝月份分群（月斜率平均 ÷ 標準誤）',
   'I': 'I(m)＝b_高(m) − b_低(m)，月平均與 t',
   'DDD': 'I_後（2019-05～2026-10 錨定月）− I_前（2006-04～2018-12），Welch t',
   'units': '每多 1 個交易日，月 α 變動（pp）'},
 'predictions': ['I_後 < 0', 'b_低,後 ≈ 0 且 |b_低| < 0.5|b_高|', '前期 |I_前| 的 t < 1.5', 'DDD < 0',
                 '子樣本：衰減在 0050 成分、外資持股前 1/3 較強；在「V60 後 1/3 且非 0050」最弱或沒有',
                 '營收腿（只用營收百分位選股）的 I 比價格腿（只用 52 週比）更負'],
 'falsification': {'A': 'b_低,後 與 b_高,後 都 t ≤ −2 且兩者差的 t > −1 → 否定關注度機制（全面加速定價，只剩準時換股）',
                   'B': '前期 I 的 t ≤ −2 → 固有大型股特性，不是 2019 年後的結構變化'},
 'mechanism_rule': {'支持': '全部成立：I_後 t ≤ −2.0；DDD t ≤ −1.65；|I_前 t| < 1.5；b_低,後 95% 區間含 0 且 |b_低| < 0.5|b_高|；'
                          '3 個副指標中至少 2 個 I_後 < 0（市值無法使用 → M50 與 FO 兩個都 < 0）；逐年剔除後 I_後 符號不變；單一年份占 I_後 ≤ 40%；'
                          '股票分群 bootstrap 單一股票占比 ≤ 30%',
                    '部分支持': '未被否證、未全部成立，但 I_後 t ≤ −1.65 且 DDD t ≤ −1.65',
                    '不支持': '其餘情況或任一否證條件成立'},
 'verdict_cap': '待觀察（正式結論等 2026-11 起的 j0／j1／j6 前瞻名單）',
 'timestamp_gate': 'GATE_TIMESTAMP：覆蓋上界 33.6% < 60% → 不通過；訊號年齡分析不做；R_roll 不回測，只做 2026-11 起前瞻（forward_track.py 規則凍結）',
 'R2_candidate': 's6_r2.py 檔頭（j＝1～12、seed 0～4、ecore7 K5T 100 萬 FIX 主欄；條件 (1)～(4) 全成立才待觀察）',
 'R_roll_rule': 'forward_track.py 檔頭（公告窗＝錨定月第一個交易日～期限日；已公告者更新營收；跌出前 15 才賣；前 10 未持有者補到 10 檔；隔日開盤）',
 'robustness_reported': ['VAL／HOLD 分段', '毛 α 與壓力成本 α', '逐年剔除', '年份占比', '股票分群 bootstrap 與最大貢獻股', '分產業', '中三分位'],
 'multiple_testing': '主檢定 1 個（I_後）＋DDD＋副指標 2 個＋腿分解 2 個＋子樣本 3 個；建構版本 5 個；R2 候選 1 個（12 個 j）；全部報告',
 'hashes': {str(p): C.sha(p) for p in files},
}
C.save_key('PREREG', PRE)
C.save_key('PREREG_sha256', C.canon_sha(C.load_res()['PREREG']))
C.check_prereg()
print('PREREG', C.load_res()['PREREG_sha256'], PRE['timestamp_utc'], len(PRE['hashes']))

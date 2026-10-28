#!/usr/bin/env python3
"""步驟 0e：PREREG_DESIGN（設計登記）。必須在任何事後分布（含合成驗證、o6 重現、焦點事後）輸出之前寫入（o6 教訓 21）。
寫入 results.json 的 PREREG_DESIGN、PREREG_DESIGN_sha256、PREREG_DESIGN_utc；並斷言目前沒有任何事後分布輸出。
用法：python3 -I -B s0e_prereg_design.py
"""
import sys, json, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T

TIERS = ['30萬', '100萬', '300萬', '1000萬']

DESIGN = {
 'topic': 'o_tier_shrink_sequential：分資金級距、GROW／FIX、群內挑選懲罰的階層收縮配置表＋2026-11 起前瞻盲測的序貫判定規則（方法題）',
 'verdict_ceiling': '可用（方法與登記）；配置表數字只在兩衝擊模型 x* 差 ≤ 30pp 的格子進 SOP，其餘寫範圍；不改選股。',
 'seen_ledger': '2006-04～2026-10 全部已被看過（o1～o6、integrate_o1～o6、SPA、V4～V6）；本題沒有歷史樣本外證據，歷史只當先驗。'
                '新策略 o_capmatched_selection（挑選期 2019～2022、驗證期 2023～2026 已跑）與 o_concentration_regime（同）亦已看過。',
 'units': '對 0050 含息的換股窗超額，pp／窗（1.0＝1%）；窗格點＝AB_V4 訊號日＋2026-10-12，245 窗（2006-04-11～2026-10-12）。',
 'periods': {'W_long': ['2006-04-01', '2026-10-12'], 'SEL': list(T.SEL), 'VAL': list(T.VAL), 'HOLD': list(T.HOLD)},
 'family': {'source': 'raw/spa/family_excess.parquet＋family_members.csv（F_long，2,319 條，一般成本，缺值補 0）',
            'clusters': 'raw/o6shrink/clusters_k161.csv（o6：相關距離 average linkage 161 群，medoid＝群內平均相關最高者；不重新分群）',
            'focus_members': {'AB': T.MAIN_ID, 'B10': T.B10_ID}},
 'hierarchical_model': {
   'form': 'α̂_k＝α_k＋e_k；α_k＝μ＋u_c＋v_k。群間：medoid 的 α 視為群水準 a_c，先驗 g_B；群內：δ_k＝α_k−α_medoid(c)。',
   'error_decomp': 'o6 decomp：e＝族群共同誤差（族群平均序列，HAC L6 變異 vP，事後卷積加回）＋個別誤差（成員減族群平均的 HAC L6 變異）。',
   'within_noise': 's_d²＝HAC L6 Var( mean(x_k − x_medoid) )（差序列）。',
   'sigma_w': '動差法：所有 m_c ≥ 3 群，非 medoid 成員 d_k＝α̂_k−α̂_med 的群內樣本變異（權重 m_c−1）減平均雜訊 s_d²，除以 2（δ＝v_k−v_med）；下限 0。',
   'versions_to_iterate': {
     'V1_o6': 'G2（df 5、c0 0.1、grid 節點、格點 [−8,4] 步 0.02）×MAIN medoids；焦點直接用 α̂（o6 主欄，無群內處理）。',
     'V2_HN': 'G2 群間（同 V1）＋群內常態：δ~N(0, 2σ_w²)，δ|d~N(Bd, B s_d²)，B＝2σ_w²/(2σ_w²+s_d²)；α_k 密度＝medoid 事後 ⊛ δ 事後。',
     'V3_G2MAX': 'V1＋群內最大值懲罰：b＝群內 m_c 個成員（各自去均值）以平穩 bootstrap L6 同列重抽、取重抽平均最大值的分布（B 400），'
                 'α̂′＝α̂−b，事後＝對 b 的 200 個分位點的 G2 事後混合。',
     'V4_HNMAX': 'V2＋群內最大值懲罰：b＝群內差序列（成員減 medoid、去均值）重抽平均最大值的分布，d′＝d−b，δ 事後對 b 混合。',
     'V5_HT': 'G2 群間＋群內 t(ν＝3)：δ 先驗 t3、尺度使 sd＝√2σ_w；δ 事後在格點上數值計算。',
     'V6_HNG3': 'G3（NPMLE）群間＋群內常態（同 V2）。',
     'V7_HNC': 'V2，但 m_c ≥ 10 的群用該群自己的動差法 σ_w（下限 0），其他群用合併值。'},
   'max_versions': 8},
 'step1_validation': {
   'o6_reproduction': 'V1 以 raw/o6shrink/focus_windows.parquet、sd_cost 0.0362 重算 G2×MAIN 的 AB／B10／EXEC／EXEC_J 事後均值與 P(α>0)，'
                      '與 o6 results.json 差 ≤ 0.01 才通過（程式一致性）。',
   'synthetic': {
     'residuals': '真實族群 W_long 245×2,319 每成員去均值；每次以平穩 bootstrap（L6，同列）重抽 245 列（保留成員間相關與共同成分）。',
     'truth': 'a_c~g_true（每群 iid）；焦點群（第 114 群結構，228 成員）的 a_114 取 g_true 的右尾 10%；v_k~群內分布；α_k＝a_c＋v_k（medoid 也有 v）。',
     'focus': '焦點＝第 114 群內合成 α̂ 最大的成員（群內最大值挑選）。另報：同群隨機成員、各群 medoid（一般校準）。',
     'scenarios': {'P_primary': 'g_true＝真實 medoid 擬合的 V1 G2 先驗；群內 N(0, 0.10²)（探索 E2 的 SEL 動差估計）',
                   'S_T2': 'g_true＝0.75·N(−0.6,0.7²)＋0.25·N(0.4,0.25²)；群內 N(0,0.10²)',
                   'S_w0': '主情境但 σ_w＝0（變體全部同 α，純雜訊挑選）',
                   'S_w25': '主情境但 σ_w＝0.25',
                   'S_t3': '主情境但群內 t3（sd 0.10）'},
     'replicates': {'P_primary': 200, 'others': 100},
     'gate': '主情境：焦點事後均值偏誤 |mean(事後均值−真值)| ≤ 0.05pp／窗，且 80% 區間涵蓋率 ∈ [0.75, 0.85]。其他情境只報告。'},
   'cv_predictive_loglik': {
     'design': '真實資料時間切半（A＝前 122 窗、B＝後 123 窗）；對每個 m_c ≥ 5 的群 c：以「拿掉 c 的其他群」在 A 上估 g_B、σ_w，'
               '在 A 上挑群內 α̂ 最大的成員 k*，以該版本求 α_k* 的事後，預測分布＝事後 ⊛ N(0, s_B²)（s_B²＝k* 在 B 的 HAC L6 變異），'
               '計 B 期實現 α̂ 的對數預測密度；再反向（B 估、A 驗）。加總兩方向。',
     'also_report': '各群 medoid 的同法預測對數似然（一般校準）；選中成員的「實現÷事後均值」斜率。',
     'note': 'B 期含 2019～2026（已看過），且成員超額在後期更分散（o6 8.1）：所有版本同受影響。'},
   'selection_rule': '只看合成驗證與 CV：在主情境閘門通過的版本中取 CV 預測對數似然最高者；全部不通過時，取主情境 |偏誤| 最小且涵蓋率 ∈ [0.70,0.90] 者，'
                     '並標「驗收未過」；仍無則取 |偏誤| 最小。不得看主策略的 x*。'},
 'cells': {
   'AB': 'AB_V4 官方日（族群成員）；AB_J：AB 事後 ⊛ 13 點 j 平移（AB_V6|normal|Jj − AB_V6|normal 的窗平均）',
   'B10': 'AB_V4_B10 官方日；B10_J 同理（AB_V6_B10|normal|Jj − AB_V6_B10|normal）',
   'stress_ref': 'AB_stress、B10_stress（官方日、J 混合）＝事後 ⊛ 平均(壓力序列 − 一般序列) ⊛ 13 點 j 平移（壓力 jscan）',
   'exec': {'tiers': TIERS, 'modes': ['FIX', 'GROW'], 'costs': ['normal', 'stress'], 'impact_models': ['KAPPA', 'TICK'],
            'formula': 'α_cell＝α_B10（方法事後）＋S，S 為 13 點等權混合：S_j＝mean_w[cell_j(w) − B10(w)]；'
                       '目前輸入 cell_j(w)＝B10(w)＋drag_cell(w)＋[B10_Jj(w) − B10_official(w)]（V6 jscan；j 窗缺值補 0）；'
                       '一般成本 drag＝KAPPA：navs_abavg main|級距|模式|A − 零衝擊 AB_V4；TICK：data/navs_tick_abavg tick|級距|模式|A − 零衝擊 AB_V4；'
                       '壓力成本＝(B10_stress − B10)＋壓力拖累（KAPPA：stress|級距|模式|B；TICK：tickstress|級距|模式|B（手續費 B、零股壓力、1 跳））＋壓力 j 平移。',
            'limitation': '拖累是官方日的拖累，沒有對每個 j 實際模擬換股衝擊（o6 教訓 13）；V7（o_v7_listing_exec_engine）交件後只換輸入重跑。',
            'no_cost_param_uncertainty': '兩衝擊模型本身就是 κ 未辨識的上下界，不再加 o6 的 sd_cost。'},
   'residuals_for_allocation': '各格的窗超額序列（官方日 cell 序列；AB/B10 用自身）去均值，與 0050 窗報酬同列重抽。'},
 'allocation': {
   'x_grid': [i / 10 for i in range(11)],
   'portfolio': '主策略 x＋0050 (1−x)，每窗再平衡（o6 sc_lib.simulate，含再平衡成本）',
   'bootstrap': '平穩 bootstrap L6（敏感度 L12），20,000 條路徑，持有 10 年（120 窗）與 20 年（240 窗）；α 每條路徑由該格事後分布抽一次（逆 CDF，共同隨機數）。',
   'utility': '對數財富 E[log W_p]（與 E[log(W_p/W_0050)] 同最適點）；x*＝argmax（格點 10%）。主欄 20 年，10 年並列；不同時兩者都報。',
   'win': 'P(W_p(H) > W_0050(H))；另報 P(財富比 < 0.75)、財富比 p10／p50、組合 MDD 中位數（窗頻）。',
   'output_rule': '同一格兩衝擊模型 x* 差 ≤ 30pp → 寫成數字＝兩者較小值；否則只寫範圍 [較小, 較大]。',
   'sop_conditional': '若所有可執行 GROW 格（4 級距 × 一般／壓力 × 兩模型）的 P(α>0) 都 < 0.5 → SOP 加註「大資金、獲利滾入時，預設配置應以 0050 為主」。'},
 'sensitivity': ['G2 c0 ∈ {0.01, 0.05, 0.2, 1}', '格點 [−6,6]、[−10,6]', 'j 改取 13 個 j 的平均數或中位數（點值）', '拿掉 2021 年的窗（族群與焦點同時）',
                 '拿掉焦點（B10）最好 10 窗（族群與焦點同時）', 'σ_w 改為 0.5×、2× 動差估計', '區塊長度 L12'],
 'oos_step': {'family_level': '以 SEL 估 g_B、σ_w（選定版本與 V1 對照），對每個 m_c ≥ 5 群在 SEL 內挑最大成員，報 VAL、HOLD 的實現 α̂ 落在事後預測分布的百分位、'
                              '80% 涵蓋與對數預測密度；兩期都已看過，只當描述。',
              'focus_level': 'SEL 估、看 VAL／HOLD：B10 與 EXEC 格的實現值百分位。',
              'comparisons': '各格 x* 組合、主策略 100%、0050 在 SEL／VAL／HOLD／ALL 的年化、MDD、對 0050 t（歷史，事後描述）。'},
 'predictions': {
   'P1': '群內懲罰後事後÷原始 0.45～0.65（AB／B10 官方日）；B10 可執行 GROW 的 P(α>0) 0.35～0.60。',
   'P2': 'x* 隨級距與 GROW 單調下降：30 萬 FIX 40～90%，1,000 萬 GROW 0～20%；300 萬以上兩模型 x* 差 > 30pp（若所有級距兩模型差都 ≤ 30pp → 否定）。',
   'P3': 'x＝100% 的 10 年勝率：最有利格 ≤ 0.55、最不利格 ≤ 0.35。',
   'P4': '序貫規則模擬：α 0.2／0.5 時 36 個月內升級 < 10%／< 25%；α −1 時 24 個月內下修 ≥ 70%；α 0 時任何時點誤升級 ≤ 5%。',
   'P5': '新策略（凍結版）的事後 P(α>0) 低於各自原始 t 推算的 Φ(t)。'},
 'new_strategies': {
   'capmatched': '凍結版 F1_K5_N20_sqrt|normal（raw/newstrat/capmatched_navs_oos.parquet）；本輪 11 個版本序列（F1／F2／F3 normal）當成一個新群加入族群，'
                 '以選定方法求事後（群 medoid 由群內平均相關決定）；核心衛星 C50＝0.5α、C70＝0.3α。',
   'regime': '凍結版 V5（S2，L6，xC 1，h 0）：以 raw/newstrat/regime_W_full.parquet 的 top5 重建 8 個版本的窗層級輪動（狀態 C→0050、D→AB_V4；'
             '換狀態時扣單邊成本：主腿 0.4855%／0.1855%、0050 0.2355%／0.1355%），8 版當一個新群；核心衛星 CORE50＝0.5α。'
             '登記焦點 d＝輪動 − 靜態混合（w̄ 0.4792）：先驗 N(0, 0.5²)，以全期 d̂ 減去 8 版最大值偏誤（bootstrap）更新（常態-常態）。',
   'approximation': '窗層級重建與原研究的日頻引擎（T+1 收盤移轉）不同，只用於先驗，誤差另報。'},
 'forward': {
   'start': '2026-11 起第一個換股窗（訊號日 2026-11-11 左右）；只用前瞻資料。',
   'lines': {
     'L1_MAIN_B10_EXEC': '主策略 B10 可執行（帳戶級距／模式取 live 設定，預設 100 萬 FIX）；先驗＝該格 KAPPA 與 TICK 事後的 50/50 混合（一般成本）',
     'L2_MAIN80_DISP20': '主策略 80%＋處置 20%：α＝0.8α_L1＋0.2α_disp，α_disp~群間先驗 g_B（研究中心典型候選；處置長歷史 t −0.89）',
     'L3_MAIN70_E50_30': '0050 30% 混合：α＝0.7α_L1',
     'L4_MAIN70_2330_30': '2330 30%：α＝0.7α_L1＋0.3α_2330，α_2330~N(0, se²)，se＝2330 對 0050 窗超額的 HAC L6 標準誤（單一事後贏家，歷史不得當證據）',
     'L5_CAPM_F': '市值匹配選股凍結版；L6_CAPM_C50、L7_CAPM_C70：0.5α、0.3α',
     'L8_REGIME': '權值集中輪動凍結版 V5（對 0050）；L9_REGIME_CORE50：0.5α；L10_REGIME_D：輪動 − 靜態混合（登記焦點）'},
   'monthly_stats': {'post_P': '以本題事後分布為先驗、前瞻月超額的常態概似（σ＝該線歷史窗超額的長期 sd：HAC L6）更新後的 P(α>0)',
                     'E_up': 'H0：α ≤ 0；E＝∫ Π_t φ(x_t;θ,σ)/φ(x_t;0,σ) dπ⁺(θ)，π⁺＝先驗限制在 θ>0 後正規化',
                     'E_down': 'H0：α ≥ α_min，α_min＝0.5×max(先驗均值, 0)；E＝∫ LR(θ vs α_min) dπ⁻(θ)，π⁻＝先驗限制在 θ<α_min 後正規化',
                     'wealth_ratio': '前瞻累積財富比（線 ÷ 0050）'},
   'thresholds': {'upgrade': 'E_up ≥ 20 且前瞻 ≥ 24 個月（主策略：第 3 級回第 2 級；新策略：待觀察→可用的必要條件）',
                  'down_alloc': 'E_down ≥ 10，或（更新後 P(α>0) < 0.25 且前瞻 ≥ 18 個月）→ 該線配置降到配置表不利格',
                  'review': '財富比 < 0.75 或 E_down ≥ 20 → 只觸發總監審查，不自動停用',
                  'change_constraints': '任何配置變更：前瞻 ≥ 12 個月、±20pp 死區、每年最多一次'},
   'threshold_adjustment': '步驟 5 模擬中 α＝0 的誤升級率 > 5% 時，可在 PREREG_FORWARD 凍結前提高 E_up 門檻（20→40→80），重新登記；其他門檻不調。',
   'freeze_rule': '凍結後不得引用 2026-10 以前的報酬調整任何參數；歷史只透過先驗進入。'},
 'step5_simulation': {'alphas_pp': [-1.0, 0.0, 0.2, 0.5, 0.76], 'horizon_months': 120, 'paths': 4000,
                      'residuals': '各線歷史窗超額去均值，平穩 bootstrap L6（保留自相關與厚尾）', 'report': '觸發時間分布、12／18／24／36／60／120 個月內的升級、下修、審查機率'},
 'v7_rerun': '待 o_v7_listing_exec_engine 交件：只把輸入換成 V7（AB/B10 官方淨值、jscan、各 j 的兩模型可執行淨值），以同一程式 run_formal.py --inputs v7 重跑一次，'
             '方法、先驗族、門檻全部不變；結果另存 RESULTS_V7，配置表以 V7 版取代（若兩版 x* 差 > 30pp，該格改寫範圍）。',
 'replication': '複製者自 raw/ 重算：x* 差 ≤ 10pp、P(α>0) 差 ≤ 0.02；同一份假前瞻序列跑 forward_monitor.py 逐位一致；超過時取不利值（o6 教訓 22）。',
 'multiple_testing_disclosure': '版本數 ≤ 8（方法族），敏感度 7 組；判定欄只有一個（選定版本）。',
}

def main():
    res = T.load_res()
    assert not any(k.startswith(('RESULTS', 'PREREG')) for k in res), '已有結果或 PREREG，不得重寫設計登記'
    assert not glob.glob(str(T.DATA / 'post*')), '已有事後分布檔'
    files = sorted([p.name for p in HERE.glob('*.py')]) + ['raw/manifest.json', 'explore.json', 'data/gaps.json']
    hashes = {f: T.sha(HERE / f) for f in files}
    D = dict(DESIGN); D['hashes'] = hashes; D['utc'] = T.now_utc()
    T.save_key('PREREG_DESIGN', D)
    T.save_key('PREREG_DESIGN_sha256', T.canon_sha(D))
    print('PREREG_DESIGN', T.canon_sha(D), D['utc'], len(hashes), 'files')

if __name__ == '__main__':
    main()

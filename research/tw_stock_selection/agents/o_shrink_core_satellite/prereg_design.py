#!/usr/bin/env python3
"""步驟 0 的設計登記（PREREG_DESIGN）：在任何收縮（先驗估計、事後分布）計算之前寫入 results.json。
誠實揭露：explore.py（只有描述統計與群集，不含任何先驗或事後計算）比本登記早跑一次；見 time_utc 與 explore.json。"""
import sys, json, datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

DESIGN = {
    'topic': 'o_shrink_core_satellite：主策略優勢的經驗貝氏收縮與核心衛星比例（方法題，不立策略判定）',
    'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    'disclosure': 'explore.py 在本登記前執行過一次（描述統計、群集、挑選期 α̂ 分布、動差法 τ² 粗估）；沒有計算任何先驗擬合或事後分布。',
    'loss': '扣成本後對 0050 含息的窗超額（pp／窗；窗＝AB_V4 訊號日到下一訊號日，245 窗，約 1 個月）。族群成員用各研究自己的一般成本版，缺值以 0 填補（SPA 題規則）。',
    'window_main': 'W_long：2006-04-11～2026-10-12（245 窗）；SEL 2006-04～2019-04（157 窗）只用於探索與合成驗證；VAL 2019-05～2023-12、HOLD 2024-01～2026-10 只用於樣本外檢查。',
    'families': {
        'MAIN': 'F_long（2,319 條）以 W_long 相關係數群集化：距離 1−ρ、average linkage、切成 Li-Ji 有效數 161 群，每群取 medoid（群內平均相關最高者），共 161 條。',
        'ALL': '全部 2,319 條（敏感度）。',
        'LOCO': 'MAIN 去掉主策略 AB_V4 所屬群集的 medoid（留一群集，敏感度；B10、可執行口徑同一群集）。',
        'THEORY': 'F_main（主策略改版鏈）∪ 學術／制度機制資料夾（explore.THEORY_FOLDERS 共 19 個）共 521 條，以同法群集成其 Li-Ji 有效數（52）群取 medoid（敏感度）。',
        'EXT_MP': '外部先驗（B 級，只當敏感度）：McLean-Pontiff（2016）發表後衰退 58%：α ~ N(0.42·α̂_f, s_f²)。',
        'EXT_FUND': '外部先驗（B 級，只當敏感度）：頂尖主動基金扣費後 +3～6pp／年 → 先驗 N(0.375, 0.125²) pp／月，與焦點概似結合（常態共軛）。'},
    'focus': {
        'AB': 'AB_V4|normal（o_v4_revenue_patch_official/navs_v4.parquet）',
        'B10': 'AB_V4_B10|normal（同檔；V5 的 AB_V5_B10 只做核對）',
        'EXEC': 'B10 ＋ (o_exec_capacity_tiers main|100萬|FIX|A − 零衝擊 AB_V4) 逐窗拖累（o5 成本審查移植公式）；標準誤加入成本參數不確定 sd_cost＝(1.30−0.20)|I|/√12（I＝100萬 FIX 衝擊成分，區間 [0.2I,1.3I] 視為均勻）',
        'EXEC_J': 'EXEC ＋ δ_j（j＝0～12 版 AB_V5 對 0050 平均窗超額的中位數 − 官方日 AB_V5；o5 教訓 10）＝「可執行保守口徑」，SOP 推薦比例與 P(α>0) 觸發條件的判定欄。',
        'stress_and_refs': 'AB_stress、B10_stress、EXEC_stress（B10_stress＋stress|100萬|FIX|B 拖累）、B10_V5、EXEC_30F／300F／100G、EW（同池等權零成本，參考）只做敏感度。',
        'input_file': 'data/focus_windows.parquet（build_inputs.py 產生）；所有正式程式以 --focus 參數吃焦點檔，總監整合時可換 V6、B10R、校準後可執行版。'},
    'se': '各成員平均窗超額的 HAC（Newey-West Bartlett，L＝6）標準誤。',
    'correlation_adjustment': '誤差拆成族群共同部分（族群平均序列的平均誤差，變異 v_P 以 HAC 計）與個別部分 u_k＝ē_k−ē_P；去卷積用 u_k 的 HAC 變異；焦點事後分布＝β 的事後分布與 N(0, v_P) 卷積（平移不確定、均值不變）。等價有效數 K_eff＝平均個別變異÷v_P 一併報告。若合成驗證不通過，迭代只能改這一步（最多 3 版）。',
    'models': {
        'G1': '常態先驗 N(μ, τ²)，REML；事後變異加 (1−B)²Var(μ̂)＋v_P。',
        'G2': 'Efron（2016）log-spline g-modeling：θ 格點 [−6, 6] pp、間距 0.02；自然三次樣條 df＝5（中心化、SVD 正交、單位長度欄）；懲罰 c0‖β‖，c0＝1（deconvolveR 預設）；BFGS。',
        'G3': 'NPMLE（Kiefer-Wolfowitz／Koenker-Mizera 格點版），同格點，EM 求解（最多 6,000 次、相對收斂 1e-10）。',
        'main_column': 'G2 × MAIN 族群 × 焦點 EXEC_J（SOP 判定）；AB 與 B10、EXEC 同一先驗。'},
    'validation': {
        'base': 'SEL 157 窗的 MAIN medoid 殘差（去均值）＋焦點 AB 殘差；每次以平穩 bootstrap（L＝6）重抽列，保留成員間相關。',
        'truths_pp': {'T1': 'N(−0.3, 0.6²)', 'T2': '0.75·N(−0.6, 0.7²)＋0.25·N(0.4, 0.25²)', 'T3': '0.5·N(0, 0.05²)＋0.5·N(−0.2, 0.8²)'},
        'reps': 'G2：每個真值 150 次 × 3 個種子；G1、G3：每個真值 100 次 × 1 個種子',
        'pass': '(1) 80% 事後區間對族群成員的平均涵蓋率在 75～85%（每個真值）；(2) Tweedie：族群 α̂ 最大者，事後平均−真值的平均偏誤 ≤ 原始偏誤的 25%，且偏誤平均的 t < 2；(3) 換種子：3 個種子的涵蓋率差 ≤ 3pp、最大者偏誤差 ≤ 0.05pp。',
        'versions': 'v1＝誤差獨立（各成員完整 HAC 變異、v_P＝0）；v2＝共同／個別拆解（上述）；v3＝依 v1、v2 結果事後定義（最多 3 版）。依序取第一個 G2 全部通過的版本；都不通過則取涵蓋率最接近 80% 者並標記未通過。',
        'no_peeking': '不得根據主策略的事後結果改先驗或族群；驗證只用合成族群。'},
    'allocation': {
        'x_grid': '0, 10, …, 100%（另加 56%＝「選股部位 30% 改放 0050」在選股 80% 下的主策略比重、80%＝現行選股 80%＋處置 20%（處置以 0050 代替））',
        'rebalance': '每窗再平衡（主）；每 12 窗再平衡（敏感度）。再平衡成本：主策略賣出 手續費 0.0855%＋滑價 0.1%＋稅 0.3%，買進 0.1855%；0050 賣出 0.0855%＋稅 0.1%＋滑價 0.05%，買進 0.1355%。',
        'horizons': '10 年（120 窗）與 20 年（240 窗）。',
        'paths': '對 (0050 窗報酬, 主策略超額去均值殘差) 做聯合平穩區塊 bootstrap，L＝6（主）／12；20,000 條路徑；每條路徑先由事後分布抽 α 再加回；固定種子 20261029（L6）／20261030（L12），α 抽樣種子 7。所有 x 與口徑共用同一組索引（共同隨機數）。',
        'outputs': 'P(期末贏 0050)；財富比中位數、p10、p90；P(財富比<0.75)、P(<0.5)；組合最大回撤中位數與 p90（嚴重端）；相對 0050 最大落後中位數與 p90；P(任一 60 窗滾動期輸 0050)（主）與 P(財富比連續 ≥60 窗低於 1)（副）；E[log 財富比]／年。',
        'controls': 'NOSHRINK＝α 固定為原始 α̂（純歷史 bootstrap）；FLAT＝α ~ N(α̂, s²)（不收縮但有不確定）。',
        'kelly': '相對 Kelly：x* ＝ argmax_x E[log(W_p/W_0050)]（x 格點 10pp；另報 1pp 細格點）；主欄 10 年、L6、每窗再平衡、G2×MAIN×EXEC_J。'},
    'decision': {
        'sop_ratio': '可執行保守口徑 EXEC_J 下的 x*（G2×MAIN×L6×10 年）；AB 的 x* 只當上緣參考。',
        'conditional': 'G1／G2／G3（MAIN 族群）三種先驗或 L6／L12 兩種區塊長度下，x* 最大−最小 > 40pp：SOP 只寫範圍並註明「由先驗決定」。',
        'vs_current': '現行配置＝x 80%（選股 80%＋處置 20%，處置以 0050 代替；x 100% 為敏感度）；0050 30% 選項＝x 56%。新比例相對現行：財富比 p10 改善 ≥ 0.05，或（組合最大回撤中位數改善 ≥ 5pp 且勝率損失 ≤ 2pp）——10 年與 20 年都成立才改寫配置建議；只有一個成立寫成條件式；都不成立維持現行配置，只把勝率表放進級別說明。',
        'p_pos_trigger': 'EXEC_J 的 P(α>0)（G2×MAIN）< 0.5：SOP 改寫為「以 0050 為主，主策略只能當信念押注」，由總監整合確認（EXEC 一併報告）。',
        'director_rerun': '總監整合時須把焦點檔換成 V6、B10R 與校準後的可執行版重跑；判定以重跑版為準。'},
    'oos': {
        'family_holdout': '以 SEL 的 α̂ 估 MAIN 先驗，算 F_long 全部成員的事後平均；與 VAL、HOLD 各自實現的平均窗超額比較：MSE（原始 vs 收縮）、實現值對原始 α̂ 的斜率、SEL 前 100 名的實現均值。各跑一次。',
        'focus_predictive': 'AB、B10、EXEC 的 SEL 事後分布加上 VAL／HOLD 抽樣誤差（HAC）形成預測分布，報實現值的百分位。'},
    'robustness': '判定欄 EXEC_J 與 AB：拿掉每一年、起點 2008、拿掉最好 10 窗、壓力成本焦點、V5 B10、其他級距；分產業與去掉最大貢獻股不適用（焦點為固定淨值序列，族群層級檢定），沿用 o4／o5 數字。',
    'predictions': '見研究計畫：先驗中心 ≤ 0、τ 0.2～0.5pp；AB 事後 0.25～0.60、P 0.70～0.92；B10 0.15～0.45、P 0.60～0.85；可執行 0.05～0.35、P 0.55～0.80；可證偽：任一口徑事後平均 ≥ 原始 80%；x＝100% 可執行 10 年勝率 0.52～0.66、20 年 0.54～0.72；x 20～100% 勝率差 ≤ 6pp；P(財富比<0.75) x＝100% 15～30%、x＝30% 0～5%；20 年 MDD 中位數 x＝100% −60～−70%、x＝50% −55～−62%；相對落後中位數 x＝100% −35～−50%；x* AB 70～100、B10 40～90、可執行 20～70。',
}

def main():
    res = S.load_res()
    assert 'PREREG' not in res, '正式 PREREG 已存在，不可再改設計'
    res['PREREG_DESIGN'] = DESIGN
    res['PREREG_DESIGN_sha256'] = S.canon_sha(DESIGN)
    S.RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('PREREG_DESIGN written', res['PREREG_DESIGN_sha256'], DESIGN['time_utc'])

if __name__ == '__main__':
    main()

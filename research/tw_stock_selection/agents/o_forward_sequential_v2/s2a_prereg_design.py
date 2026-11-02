#!/usr/bin/env python3
"""步驟 2a：設計登記 PREREG_DESIGN（在任何 CV、先驗、OC 輸出之前）。登記選版規則、校準規則、規則族、DGP、凍結條件與迭代範圍。
用法：python3 -I -B s2a_prereg_design.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fslib as F

DESIGN = {
    'topic': 'o_forward_sequential_v2：可證偽的前瞻序貫規則 2.0',
    'decision_line': '主策略 B10 可執行 K5T（V7 navs_exec7；V7X 交件後換輸入、方法不變）一般成本，13 個 j 的錨定月窗超額（對 0050 含息，pp）逐月取 j 中位數；主決策格 100 萬 FIX；其他 7 格同法另報',
    'periods': {'SEL': '錨定月 ≤ 2019-04（探索、迭代、挑選期 CV）', 'VAL': '2019-05～2023-12', 'HOLD': '2024-01～2026-10'},
    'cv': {
        'family': 'o_main_spa_breaks F_long 2,319 條 × V4 格點 245 窗（vendor_o7/raw/spa），161 群（clusters_k161），m_c ≥ 5 的 31 群',
        'versions': ['V1_o6', 'V2_HN', 'V3_G2MAX', 'V4_HNMAX', 'V5_HT', 'V6_HNG3', 'V7_HNC'],
        'schemes_sel_phase': '窗 0～156（SEL）：時間切半 78／79 兩方向；滾動起點（擴張估計窗）T0 ∈ {60, 84, 108}，驗證 48 窗',
        'schemes_final': '全 245 窗：時間切半 122／123 兩方向（＝o7 口徑）；滾動起點 T0 ∈ {84, 120, 157, 192}，驗證 48 窗（157＝2019-05 起）',
        'targets': 'tag=sel（估計期群內 α̂ 最大者）為主；medoid 只報告',
        'selection_rule': '合併所有切分的 sel 預測對數似然總和最大者；與最佳差 < 1.0 nat 視為同分，同分取「主策略 B10 在估計期的事後均值」較低者（較保守）',
        'calib_slope_s': '選定版本在 half 與 roll 兩方案各自合併的「實現 y 對事後均值」OLS 斜率，取較小者（不利端），截在 [0, 1]',
        'calib_k': 'k 從 1.0 以 0.1 遞增（上限 6.0），取第一個讓 sel 合併 80% 預測區間涵蓋率 ≥ 0.80 的值；報告該 k 的涵蓋率（目標帶 0.75～0.85）；只放大不縮小',
        'transform': "θ' ＝ s·μ ＋ k·(θ − μ)，μ＝事後均值",
    },
    'prior': '選定版本 V* 的階層事後：族群在估計窗擬合（全族群，不排除），焦點＝決策線序列對齊族群窗（缺窗補 0）、群 114；再做 (s, k) 變換',
    'sigma_design': '決策線 HAC（Bartlett L6）長期 sd；挑選期用 SEL；最終＝max(全期, 2019-05 起)',
    'rules': {
        'common': '第 12 個月前不動作；先觸發者吸收；前瞻月＝錨定月（2026-11 起）',
        'R1': '貝氏後驗門檻，6 個月一次：升級 P(α>α_min)≥0.975；下修 P(α<0)≥0.80；α_min＝0.25pp／月',
        'R2': "群組序貫，6 個月一次：O'Brien-Fleming 型升級 z_n ≥ c_u·√(H_c/n)；Pocock 型下修 z_n ≤ −c_d",
        'R3': '單邊 CUSUM 下修（k_ref −0.5pp，門檻 h，t ≥ 12）＋R1 的後驗升級',
        'R4': '只在 12／24／36（之後每 12 個月）檢視：升級 z ≥ c_u4、下修 z ≤ −c_d4',
        'calibration': '設計 DGP＝iid 常態（σ_design）200,000 條路徑、seed 20261102，以分位數精確解：α＝0 時 H_c＝36 個月內誤升級＝target_up；α＝−1 時 36 個月內下修＝target_pow',
    },
    'dgps': {
        'D1': 'iid 常態，sd＝σ_design',
        'D2': '參考期殘差（決策線減均值）6 個月區塊 bootstrap（環狀、不縮放）；挑選期＝SEL 殘差；最終＝2019-05 起殘差（題目指定）',
        'D3': 't(4) 單位變異 × 波動狀態（乘數 0.7／1.4、高波動穩態 0.347、高→低 0.06、低→高 0.0319）× σ_design',
    },
    'oc': 'α ∈ {−1, −0.5, 0, +0.5, +1}pp／月；12～60 個月每 6 個月；每 DGP×α ≥ 20,000 條路徑（最終 40,000）',
    'freeze_conditions': 'α＝−1：36 個月內下修 ≥ 50%；α＝0：36 個月內誤升級 ≤ 5%（含所有中間檢視）；三種 DGP 都要成立；誤下修照實報告',
    'rule_choice': '通過的規則中，選 α＝0 時 36 個月誤下修（三 DGP 最大值）最低者；同分選檢視次數少者',
    'fallback': '若選定規則誤下修 > 30%，或沒有規則通過：把 H_c 改為 42、48、54、60、72、84、96、120 依序重校準，回報第一個「三 DGP 都過、誤下修 ≤ 30%」的 H_c＝最短可行期限',
    'refit_clause': '前瞻 ≥ 12 個月起，前瞻月超額 sd > 1.5 × σ_design 時，σ_design ← 前瞻 sd，重算 R2～R4 門檻（同校準程序）；升級門檻（pp 尺度的 S_n 邊界）只能變嚴、不能變鬆。這是唯一允許的 refit',
    'iteration_knobs': '迭代只在挑選期，只改：設計目標緩衝（target_up 0.05／0.04；target_pow 0.50／0.55）、σ_design 安全係數（1.00／1.15）、決策線（j 中位數／官方日）；另跑 o7 原規則（E7）當基準。版本數 3～8',
    'allocation': 'K5T 單一模型；30 萬／100 萬／300 萬／1,000 萬 × FIX／GROW；x 0～100% 每 10%；E[log W]／年的平坦區間＝與最大值差 ≤ 0.25%／年的 x 集合（o7 成本審查的辨識力門檻）',
    'utc': F.now_utc(),
}

if __name__ == '__main__':
    h = F.register('PREREG_DESIGN', DESIGN)
    print('PREREG_DESIGN', h)

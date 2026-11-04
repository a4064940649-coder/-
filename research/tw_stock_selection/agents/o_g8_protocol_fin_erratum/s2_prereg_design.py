#!/usr/bin/env python3
"""步驟 2：事前登記設計（PREREG_DESIGN）——在跑任何候選規則 OC 之前寫死：
候選規則族（最多 4 版）、各版界限的計算法、共同下修模組、refit、DGP、α 格點、模擬次數與 seed、合格條件、選版規則、不合格時的退路。
用法：python3 -I -B s2_prereg_design.py"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import g8lib as G

E = G.load_res()['EXPLORE']
c2 = E['inflation_c2_candidates']
# 膨脹係數 c（事前公式）：c² ＝ max(1, 殘差樣本變異/σ_d², HAC L3、L6 長期變異/σ_d²)；L12 因 89 個月下估計不穩（且偏小）不列入
C2 = max(1.0, c2['plain_sd'], c2['hac_L3'], c2['hac_L6'])
C = math.sqrt(C2)

o8 = np.load(G.AG / 'o_forward_sequential_v2/data/forward_prior_main.npz')
th, pr = o8['theta'], o8['prior']
m = (th > 0) & (pr > 1e-10)
th_pos, w_pos = np.round(th[m], 4), pr[m] / pr[m].sum()
w_pos = np.round(w_pos, 12); w_pos = w_pos / w_pos.sum()

design = dict(
    utc=G.now_utc(),
    scope='主策略（52 週高點百分位 0.5＋近 3 月營收年增率百分位 0.5、前 300 流動性池、月換股 10 檔、排除金融）的前瞻 G8 序貫檢定協定。只用前瞻資料判定；歷史資料只給設計 σ 與殘差形狀。',
    decision_line=dict(line='forward_returns.csv 的 line＝JMED、status＝complete、n_j＝13（或作廢規則下 ≥ 7 個有效 J 窗）的 excess_pp',
                       engine='V7X K5T GROW 100 萬（一般成本）', unit='pp／錨定月', first_anchor='2026-11'),
    sigma_design=G.SIGMA_D,
    inflation_c=dict(formula='c＝sqrt(max(1, s²/σ_d², LRV_L3/σ_d², LRV_L6/σ_d²))，s、LRV 取自 2019-05 起決策線去均值殘差（89 個月）', c2_inputs=c2, value=C),
    alphas=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], horizons=[12, 18, 24],
    dgps_primary=dict(D1='iid 常態 N(α, σ_d²)', D2='t(4) 厚尾：α＋σ_d·t4/√2', D3='定態區塊 bootstrap（平均區塊 6 個月、環狀）2019-05～2026-09 殘差，不縮放'),
    dgps_sensitivity=dict(D3_2024='同 D3，只用 2024-01 起殘差', D3_L3='區塊 3', D3_L12='區塊 12', D4_o8='o8 的 t(4)×波動狀態（乘數 0.7／1.4）',
                          STRESS='D1～D3 以壓力成本線：α 平移 −0.577pp／月（挑選期 stress−normal 平均，explore.cost_shifts_pp）'),
    common_down=dict(name='DN', stat='z_n＝S_n／(σ_eff·√n)，σ_eff＝c·σ_d', looks='n＝12、18、24（之後每 6 個月）', rule='z_n ≤ −c_d 即下修',
                     calibration='c_d：D1（σ_d）、α＝−1、P＝400,000、seed 20261106，使 24 個月內下修機率＝0.55'),
    refit='協定 (b)：檢視點 n ≥ 12 時前 n 個月樣本 sd ＞ 1.5σ_d → 自該月起 σ_eff＝max(c·σ_d, 樣本 sd)（下修只會放寬；升級只會收緊）',
    candidates={
        'v1': dict(rule='R-FIX', up_looks=[12, 24], boundary='z_n ≥ b·√(24/n)，b：單位布朗運動、looks {12,24}、穿越機率 0.05（P＝2,000,000、seed 20261105）', inflation='c'),
        'v2': dict(rule='R-OBF', up_looks=[6, 12, 18, 24], boundary='z_n ≥ b·√(24/n)，b：同法、looks {6,12,18,24}', inflation='c'),
        'v3': dict(rule='R-E', looks='每季末（月 3、6、…、24）', boundary='季和 e-process E_k ≥ 20（Ville：任意停時誤升級 ≤ 5%，iid 常態已知變異下）',
                   mixture='o8 校準放寬後先驗（forward_prior_main.npz，prior_cal）θ>0 部分重新正規化', inflation='無（σ_d）'),
        'v4': dict(rule='R-E+', looks='同 v3', boundary='同 v3，但季和除以 c·σ_d·√3', mixture='同 v3', inflation='c'),
    },
    upgrade_semantics='升級＝拒絕 H0: α ≤ 0；24 個月內任何檢視點觸發都記錄，但 S1 翻案只在第 24 個有效月期滿時生效；字面 G8 未過則升級路徑關閉（合併程序另報）',
    sims=dict(iteration=dict(P=20_000, seed_base=20261110), final=dict(P=40_000, seed_base=20261111),
              seed_rule='每格 seed＝[seed_base, round((α+3)×100), DGP 序號]；同一格所有規則共用同一批路徑'),
    pass_criteria=dict(primary_dgps=['D1', 'D2', 'D3'], false_up_24_max=0.05, down_power_24_alpha_m1_min=0.50, false_down_24_max=0.30,
                       note='點估計判定；另報 +2 MC 標準誤'),
    selection='合格版本中取 min_{D1,D2,D3} P(24 個月內升級｜α＝+1) 最高者；差距 < 2×MC 標準誤（合併）視為同分，取較簡單者（v1＜v2＜v3＜v4）',
    fallback=dict(no_upgrade_rule='沒有任何版本在 D1～D3 誤升級 ≤ 5% → 判「24 個月內 S1 翻案路徑不可行」，只凍結字面 G8＋下修規則',
                  down_infeasible='DN 在任一主 DGP 不能同時滿足（α＝−1 下修 ≥ 50%、α＝0 誤下修 ≤ 30%）→ 依 H∈{30,36,42,48,60} 重算 c_d（同法，目標期限 H），報最短可行期限，協定的下修期限改用它',
                  final_check='凍結後以 seed 集 B（P＝40,000）重跑；選定版若在任一主 DGP 違反條件超過 2 個 MC 標準誤 → 依退路處理並照實報告'),
    mixture_prior=dict(theta=th_pos.tolist(), w=w_pos.tolist(), P_pos_of_full_prior=float(pr[th > 0].sum()),
                       mean_pos=float((th_pos * w_pos).sum()), source_sha256=G.sha(G.AG / 'o_forward_sequential_v2/data/forward_prior_main.npz')),
    max_versions=4,
    file_sha256={'g8lib.py': G.sha(G.HERE / 'g8lib.py'), 's1_explore.py': G.sha(G.HERE / 's1_explore.py'), 's2_prereg_design.py': G.sha(G.HERE / 's2_prereg_design.py')},
    predictions=dict(
        P1='字面 G8（iid 常態 σ＝6.14）α＝0 通過率 0.50±0.02，α＝+0.5 約 0.61；與 o9 差 > 0.03 表示程式錯',
        P2='相依性校正序貫規則在 D1～D3 都把 24 個月內誤升級控制在 ≤ 5%；α＝+1 升級 15～30%、α＝+2 升級 45～70%',
        P3='下修：α＝−1 24 個月內 ≥ 50%；α＝0 誤下修 ≤ 30%；否則最短可行期限約 36～48 個月',
        P4='只用 2024～2026 殘差時升級檢定力低 5～10pp；壓力成本 α 平移 −0.1～−0.2pp／月、字面 G8 通過率再低 2～4pp',
        P5='協定每條線都能從 forward_returns.csv 現有欄位機械計算，否則屬缺口'),
)
sh = G.register('PREREG_DESIGN', design)
print('PREREG_DESIGN', sh, 'c', C, 'n_mix', len(th_pos), 'mean_pos', design['mixture_prior']['mean_pos'])

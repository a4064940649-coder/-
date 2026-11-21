#!/usr/bin/env python3
"""S91 規則表（看完訓練期 2016-02～2021-12 的分組後定案；PREREG 登記後不得改）。F＝s91lib.features() 的輸出（長度 N 的陣列）。
回傳 True＝不買。任何用到的數字缺值 → 比較式為 False → 保留。門檻的來源（訓練期分組）寫在 PREREG.json 的 rules_text。"""
import numpy as np

RULES = {
    # A. 散戶槓桿過熱（原假說方向）
    'R01_MUSE8': lambda F: F['m_use'] > 0.08,                       # 融資餘額 > 發行股數 8%
    'R02_MD20_3': lambda F: F['m_d20s'] > 0.03,                     # 近 20 日融資增加 > 發行股數 3%
    'R03_MG5_P90': lambda F: F['m_g5_pct'] >= 0.90,                 # 近 5 日融資增幅在股池最高 10%
    'R04_MG20_P90': lambda F: F['m_g20_pct'] >= 0.90,               # 近 20 日融資增幅在股池最高 10%
    # B. 沒有人放空＝沒有人氣（訓練期分組看到的方向；與「券資比高不買」相反）
    'R05_SM_P50': lambda F: F['sm_ratio_pct'] < 0.50,               # 券資比在股池後半
    'R06_SM_ABS2': lambda F: F['sm_ratio'] < 0.02,                  # 券資比 < 2%
    'R07_SM_P67': lambda F: F['sm_ratio_pct'] < 2 / 3,              # 券資比不在股池最高三分之一
    'R08_SS_P67': lambda F: F['s_sh_pct'] < 2 / 3,                  # 融券餘額占股本不在股池最高三分之一
    # C. 借券／放空餘額偏高（原假說方向）與籌碼沉澱
    'R09_SBLD20_P80': lambda F: F['sbl_d20s_pct'] >= 0.80,          # 借券賣出餘額近 20 日增加（占股本）在股池最高 20%
    'R10_SBL_P80': lambda F: F['sbl_sh_pct'] >= 0.80,               # 借券賣出餘額占股本在股池最高 20%
    'R11_SHORTDV_P33': lambda F: F['short_dv_pct'] >= 1 / 3,        # （融券＋借券賣出餘額）÷ 20 日均量 不在股池最低三分之一
    'R12_MDV_P33': lambda F: F['m_dv_pct'] >= 1 / 3,                # 融資餘額 ÷ 20 日均量 不在股池最低三分之一
    # D. 組合
    'R13_HOT_NOSHORT': lambda F: (F['m_g5_pct'] >= 0.90) & (F['sm_ratio_pct'] < 2 / 3),   # 散戶 5 日內搶進、又沒什麼人放空
    'R14_OVERHEAT': lambda F: (F['m_use'] > 0.08) | (F['m_d20s'] > 0.03),                 # R01 或 R02
}

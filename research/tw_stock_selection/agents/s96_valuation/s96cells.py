#!/usr/bin/env python3
"""S96 的 10 格（PREREG 固定）。每條規則＝函式(f)→布林陣列（True＝不買）；f 是 s96lib.Val.feat 的訊號日特徵。
方向依訓練期描述（step0.json）：虧損股、淨值比偏低、本益比在自己 3 年低檔的交易較弱；高本益比／高淨值比反而是大贏家所在（後三格是傳統「避開貴的」方向，當對照）。"""
import numpy as np
T3 = 1 / 3
RULES = {
    'NOPER': lambda f: ~f['per_ok'],                                        # 官方本益比為空（虧損或無法計算）→ 不買
    'PBLOW': lambda f: f['p_pbr'] <= T3,                                    # 股價淨值比落在同日股池最低 1/3 → 不買（無淨值比者照買）
    'NOPER_PBLOW': lambda f: (~f['per_ok']) | (f['p_pbr'] <= T3),           # 上兩條任一成立 → 不買
    'P3LOW': lambda f: f['per3y'] <= T3,                                    # 本益比在自己過去 3 年最低 1/3 → 不買（虧損或歷史不足者照買）
    'PE30': lambda f: f['per_ok'] & (f['per'] > 30),                        # 對照（傳統方向）：本益比 > 30 → 不買
    'PBHI': lambda f: f['p_pbr'] >= 0.8,                                    # 對照（傳統方向）：淨值比在同日股池最高 20% → 不買
    'YLD0': lambda f: f['yld'] == 0,                                        # 殖利率為 0（沒配息）→ 不買
}
CELLS = ('NOPER|Z', 'NOPER|F', 'PBLOW|Z', 'PBLOW|F', 'NOPER_PBLOW|Z', 'NOPER_PBLOW|F', 'P3LOW|F', 'PE30|F', 'PBHI|F', 'YLD0|F')
assert len(CELLS) == 10 and all(c.split('|')[0] in RULES for c in CELLS)

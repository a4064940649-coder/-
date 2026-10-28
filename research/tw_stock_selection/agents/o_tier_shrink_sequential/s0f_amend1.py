#!/usr/bin/env python3
"""PREREG_DESIGN_AMEND1（全量合成驗證之前）：σ_w 動差估計式修正。
原登記「(群內 d_k 變異 − 平均差序列雜訊變異)/2」有兩個錯：(1) medoid 的 v_med 對同群所有 d_k 是共同平移，橫斷面變異只含 σ_w²，不應除以 2；
(2) 同群成員雜訊高度相關，橫斷面雜訊離散遠小於單一差序列的雜訊變異。修正為：σ_w²＝Σ_c (m_c−1)[Var_k(d_k) − mean_k HACVar(D_k − D̄_c)·m/(m−1)] / Σ_c (m_c−1)。
發現經過：s2b_synth.py 冒煙 1 次（P_primary r＝0）看到 σ_w 估計 0.047（真值 0.10），再以 8 次只含 σ_w 的模擬（不含任何焦點事後）確認新式近似不偏（0／0.10／0.25 → 0.03／0.095／0.25）。
冒煙那一次的焦點事後輸出已看過（1 次合成，非真實焦點），揭露於 iterations.json。δ 先驗仍為 N(0, 2σ_w²)（δ＝v_k − v_med）。
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T
A = {'reason': __doc__, 'changed_files': ['hmethods.py'], 'hashes': {f: T.sha(HERE / f) for f in ('hmethods.py', 's2b_synth.py', 's2a_repro_o6.py', 'tlib.py')},
     'utc': T.now_utc(), 'real_sigma_w_new_formula': {'W_long': 0.1906, 'SEL': 0.1560}}
res = T.load_res(); assert 'STEP1_SYNTH' not in res and 'PREREG' not in res
T.save_key('PREREG_DESIGN_AMEND1', A); T.save_key('PREREG_DESIGN_AMEND1_sha256', T.canon_sha(A)); print('amend1', T.canon_sha(A), A['utc'])

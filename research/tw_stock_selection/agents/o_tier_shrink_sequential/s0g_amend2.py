#!/usr/bin/env python3
"""PREREG_DESIGN_AMEND2（PREREG 之前）：
(1) tlib.convolve_dens：np.convolve(mode='same') 在核長 > 格點長時回傳核的長度（s4 冒煙在 VAL／HOLD 大變異預測分布時出錯）；
    改為 full 後置中裁切。核長 ≤ 格點長時兩者逐位相同，故已完成的 STEP1_O6_REPRO／STEP1_SYNTH／STEP1_CV 不受影響（那些計算若觸發此情形會直接報錯）。
(2) tlib.check_block 新增 files 參數：設計登記的檔案雜湊是當時快照，之後以 files＝False 只檢查登記內容未被改；PREREG（凍結）仍檢查全部檔案。
(3) s2d_synth_posthoc.py（事後診斷）已在看過 STEP1_SYNTH 之後設計、執行，不改變選版規則。
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tlib as T
A = {'reason': __doc__, 'hashes': {f: T.sha(HERE / f) for f in ('tlib.py', 's4_formal.py', 's2d_synth_posthoc.py')}, 'utc': T.now_utc()}
res = T.load_res(); assert 'PREREG' not in res
T.save_key('PREREG_DESIGN_AMEND2', A); T.save_key('PREREG_DESIGN_AMEND2_sha256', T.canon_sha(A)); print('amend2', T.canon_sha(A), A['utc'])

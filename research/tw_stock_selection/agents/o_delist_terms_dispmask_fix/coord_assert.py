#!/usr/bin/env python3
"""與題 1 協調用的斷言：題 1 的官方處置 C 濾網必須等於本題 patch 的聯集版（DISP_OFFICIAL='union'＝dispmask_official 'union'＝integrate_o17 C_DSP）。
用法（題 1 程式內）：
    import coord_assert as CA
    CA.assert_c_filter_equals_union(my_c_filter, K.IDX, K.CI, K.N, K._disposal_mask_5y 或 K.disposal_mask(原版), sig_days)
逐訊號日比對布林遮罩，不同即 AssertionError。"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import dispmask_official as DO; sys.path.pop(0)

def assert_c_filter_equals_union(c_filter, IDX, CI, N, orig_fn, days):
    ref = DO.make_mask(orig_fn, IDX, CI, N, 'union')
    bad = [str(d)[:10] for d in days if not np.array_equal(np.asarray(c_filter(d), bool), np.asarray(ref(d), bool))]
    assert not bad, f'題 1 C 濾網與本題聯集版不同：{bad[:10]}'
    return True

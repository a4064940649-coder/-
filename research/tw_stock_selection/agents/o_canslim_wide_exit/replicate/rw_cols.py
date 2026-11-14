# -*- coding: utf-8 -*-
"""審查用：重建研究員「精簡面板」的欄集合（曾經通過 N、S、L、M 的股票；規格 2.7 與 PREREG implementation_note），
以便用相同形狀的亂數矩陣逐條重現路徑擾動與安慰劑。用法：python3 -I -B rw_cols.py <scratch_dir>
"""
import sys, os
import numpy as np
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_ext_canslim_xs/replicate")
import rep_lib as L
S = sys.argv[1]
P = L.load_panel(S)
K = L.tech(P)
base = K["condN"] & K["condS"] & K["condL"] & K["mok"][:, None]
out = {}
for nm, m in (("any_mkt_full", base & (P.M >= 1)), ("any_mkt_inbar", base & (P.M >= 1) & P.inbar & (P.Cf > 0)), ("raw", base),
              ("from_ext", (base & (P.M >= 1))[P.dstr.index("2015-12-25"):]), ("from_t0", (base & (P.M >= 1))[P.t0:])):
    out[nm] = m.any(axis=0)
    print(nm, int(out[nm].sum()), "rows", int(m.sum()), "rows from t0", int(m[-(P.T - P.t0):].sum()) if m.shape[0] == P.T else "")
np.savez_compressed(os.path.join(S, "cols.npz"), **out)

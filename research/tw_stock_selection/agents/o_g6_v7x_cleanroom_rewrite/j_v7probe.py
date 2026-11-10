# J 診斷（解封後，只讀 import 原程式）：V7 B10 名單在指定訊號日的逐檔狀態
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v7_listing_exec_engine')
import v7env
L = v7env.load('v7'); K = L.K
XR = L.setup_px(); YFX, Y5F = L.load_rfix(); GAP = L.gap_dict(); DMT = L.disp_mask_fn('trade')
S = L.V4(YFX, Y5F, DMT, gap=GAP, pfix=1.0)
COLS = list(K.COLS) if hasattr(K, 'COLS') else list(L.COLS)
out = {}
for d, codes in [('2013-06-11', ['2311', '2106']), ('2014-12-11', ['2311', '2387', '3048']), ('2012-04-11', ['6286', '2473', '1476']), ('2017-10-11', ['2311', '2325']), ('2024-01-11', ['1519'])]:
    try:
        cs = S.cross_section(pd.Timestamp(d))
    except Exception as e:
        out[d] = 'ERR ' + repr(e); continue
    r = {}
    for c in codes:
        j = COLS.index(c)
        r[c] = {k: (bool(cs[k][j]) if cs[k].dtype == bool else float(cs[k][j])) for k in ('el', 'sc', 's52', 'y3', 'gm', 'top', 'ok_other')}
    r['in_gap'] = {c: str(GAP.get(c)) for c in codes}
    out[d] = r
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

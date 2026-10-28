#!/usr/bin/env python3
"""冒煙測試（PREREG 之前）：只重現已發布的 o3 數字 R_DEL|E1（官方名單、訊號日口徑、P＋X、不明文排除 KY、fin2 關）＝26.95%／t 2.55，確認引擎組裝無誤。不跑任何 V4 版本。"""
import sys, time, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
t0 = time.time()
XR = L.setup_px()
S = L.V4(L.load_rdel(), L.y5_orig(), L.disp_mask_fn('signal'), exclude_foreign=False, fin2=False)
nav, ew, sd, pl, lg = L.run_nav(S, xr=XR)
st = L.seg(nav, ew, sd, '2006-04-01', '2026-10-12')
ref = json.loads((L.AG / 'o_disposal_rejudge' / 'v3_results.json').read_text())['runs']['R_DEL|E1']['stats']['ALL']
print('mine', round(st['CAGR'], 4), round(st['t_0050'], 4), 'o3', round(ref['CAGR'], 4), round(ref['vs0050']['t'], 4), 'sec', round(time.time() - t0, 1))
assert abs(st['CAGR'] - ref['CAGR']) < 1e-6 and abs(st['t_0050'] - ref['vs0050']['t']) < 1e-6
(HERE / 'smoke_o3.json').write_text(json.dumps({'CAGR': st['CAGR'], 't': st['t_0050'], 'o3_CAGR': ref['CAGR'], 'o3_t': ref['vs0050']['t'], 'match': True}, indent=1))

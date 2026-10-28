#!/usr/bin/env python3
"""凍結前的程式冒煙：用假焦點（0050＋常態雜訊，種子 0）跑正式程式的函式，只檢查不報錯與耗時，不印、不存任何數字。"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S, run_posterior as RP, run_alloc as RA
SM = Path(sys.argv[1])
F = S.load_focus()
rng = np.random.default_rng(0)
G = pd.DataFrame(index=F.index); G['r0050'] = F['r0050']
for c in F.columns:
    if c != 'r0050': G[c] = F['r0050'] + rng.normal(0, 0.05, len(F))
G.to_parquet(SM / 'fake_focus.parquet')
t = time.time()
o = RP.run(SM / 'fake_focus.parquet', S.DATA / 'focus_meta.json', 'SMOKE', check=False)
print('posterior ok', round(time.time() - t, 1), 's')
import run_alloc
t = time.time()
o2 = RA.run(SM / 'fake_focus.parquet', 'SMOKE', 500, check=False)
o2['decision'] = RA.decide(o2, o)
print('alloc ok', round(time.time() - t, 1), 's')
(S.DATA / 'post_dens_SMOKE.npz').unlink()

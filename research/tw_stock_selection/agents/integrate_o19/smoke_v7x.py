#!/usr/bin/env python3
"""integrate_o19 V7X 程式路徑煙霧測試（不寫 results.json）：REAL_opt j0 seed0 單格 100 萬一般，檢查 pstats／vsmain／blend 路徑與恆等性（自比相關 1、月差 0、混合＝本身）。
python3 -I -B smoke_v7x.py"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import run_v7x as RV; sys.path.pop(0)
PL = pd.read_csv('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/integrate_o19/plans_strict.csv', dtype={'picks': str, 'inserted': str})
g = PL[(PL.variant == 'REAL_opt') & (PL.j == 0) & (PL.seed == 0)]
cfg = dict(tier=1e6, mode='GROW', model='K5T', cost='normal'); cfg.update(RV.BASE)
nv = RV.CV.CSim(RV.to_plan(g), dict(cfg), settle=RV.ST, cmode='consideration').run()
nv = RV.S.avg_nav([nv])
ps = RV.pstats(nv, 0); vm = RV.vsmain(nv, nv, 0); bn = RV.blend(nv, nv, 0, 0.8)
e = RV.S.end_day('AB_V7_B10', 0)
out = dict(periods_keys=list(ps), self_corr=vm['ALL']['corr'], self_diff=vm['ALL']['diff_pp'], self_t=vm['ALL']['t'], daily=vm['daily_corr'],
           blend_maxabs=float((bn / nv.loc[:e].dropna().reindex(bn.index) - 1).abs().max()), n_blend=len(bn), n_nav=len(nv.loc[:e].dropna()))
(HERE / 'smoke_v7x.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str)); print(out)

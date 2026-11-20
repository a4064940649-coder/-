#!/usr/bin/env python3
"""S78 驗收（只跑現行格，不看任何新規則的成績）：
A1 make_signal(現行格) 與 s10lib.make_signal_s10(B2 SURGE) 的 sig／X／info 逐格相同；一般化算出的 STD24 與 S71.rev_rule('B2') 的差 < 1e-9。
A2 現行格淨值與 S75 存檔 nav_2008_2026.csv.gz 的 V_BRIDGE 逐日相同（四捨五入到 0.01 元）。
A3 單次模擬耗時。
python3 -I -B accept.py → accept.json"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s78lib as Q
from s78lib import K, S71, s10lib
HERE = Path(__file__).resolve().parent
t0 = time.time()
P, U = K.load_cont(); Q.prep_panel(P)
F = Q.scores(P)
F0 = S71.rev_rule(P, 'B2', '2008-01')
a, b = F['S']['STD24'].to_numpy(), F0['S']['STD'].to_numpy()
nan_same = bool((np.isfinite(a) == np.isfinite(b)).all()); md = float(np.nanmax(np.abs(a - b)))
sg0 = s10lib.make_signal_s10(P, U, dict(S=F0['S']['STD'], av=F0['av']), shift=0)
F1 = dict(S=dict(F['S'], STD24=F0['S']['STD']), av=F['av'])          # 現行格分數直接用 S71 原函式（逐位相同）
sg = Q.make_signal(P, U, F1, 'STD24', 60, 'T10M10', 'X20', 'NONE')
sg_gen = Q.make_signal(P, U, F, 'STD24', 60, 'T10M10', 'X20', 'NONE')
a1 = dict(std24_nan_pattern_same=nan_same, std24_max_abs_diff=md, sig_same=bool((sg['sig'] == sg0['sig']).all()), X_same=bool((sg['X'] == sg0['XV']['E0']).all()),
          info_same=[(x['k'], x['targets']) for x in sg['info']] == [(x['k'], x['targets']) for x in sg0['info']],
          gen_sig_same=bool((sg_gen['sig'] == sg0['sig']).all()), gen_X_same=bool((sg_gen['X'] == sg0['XV']['E0']).all()), n_months=len(sg['info']))
t1 = time.time()
nav, lg, st = Q.simulate(P, sg, U, 'W70')
t2 = time.time()
ref = pd.read_csv(K.HERE / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
d = (nav.round(2) - ref.reindex(nav.index)).abs()
a2 = dict(max_abs_diff_twd=float(d.max()), final_nav=float(nav.iloc[-1]), cagr=K.cagr(nav), n_closed=int((lg['why'] != 'open_end').sum()))
out = dict(A1=a1, A2=a2, A3_sim_seconds=round(t2 - t1, 2), passed=bool(all(v for k, v in a1.items() if k.endswith('same')) and md < 1e-9 and d.max() < 0.011), runtime_s=round(time.time() - t0, 1))
(HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

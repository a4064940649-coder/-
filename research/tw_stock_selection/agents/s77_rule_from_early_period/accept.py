#!/usr/bin/env python3
"""S77 驗收（只用現行規則，數字都是 S75 已知的；不看任何新候選）：
A1 本題訊號 Z24＝s71lib.rev_rule('B2') 逐格相同；一般化訊號產生器的現行格＝s10lib.make_signal_s10 名單逐月相同；
A2 現行格經本題包裝（simulate_s47）跑出的淨值＝S75 存檔 V_BRIDGE 逐日相同（四捨五入到分）；
A3 截斷面板（到 2015-12-31）跑出的淨值＝完整面板同一段逐日相同（引擎與訊號不偷看未來）。
python3 -I -B accept.py → accept.json"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import s77lib as M
from s77lib import K

P, U = M.load()
ZS, av = M.signals(P)
F, RT = K.surge(P, 'B2')
a, b = ZS['Z24'].to_numpy(), F['S']['STD'].reindex(ZS['Z24'].index).to_numpy()
A1 = dict(z24_nan_pattern_same=bool((np.isfinite(a) == np.isfinite(b)).all()), z24_max_abs_diff=float(np.nanmax(np.abs(a - b))), n_cells=int(np.isfinite(a).sum()))
sg0 = K.signal(P, U, RT, 0)
sg, nav, lg, st = M.run_cfg(P, U, ZS, av, M.CURRENT)
A1['n_signal_months'] = [len(sg0['info']), len(sg['info'])]
A1['targets_same'] = bool(len(sg0['info']) == len(sg['info']) and all(x['sig_date'] == y['sig_date'] and x['targets'] == y['targets'] for x, y in zip(sg0['info'], sg['info'])))
A1['X_same'] = bool((sg0['X'] == sg['X']).all())
ref = pd.read_csv(M.AG / 's75_continuous_2008_2026' / 'nav_2008_2026.csv.gz', index_col=0, parse_dates=True)['V_BRIDGE']
A2 = dict(max_abs_diff_vs_S75_nav=float((nav.round(2) - ref.reindex(nav.index)).abs().max()), final_nav=float(nav.iloc[-1]), n_trades=int(len(lg)))
Pt, Ut = M.load(M.TRAIN_END)
ZSt, avt = M.signals(Pt, last_month='2015-12')
_, navt, lgt, _ = M.run_cfg(Pt, Ut, ZSt, avt, M.CURRENT)
A3 = dict(trunc_last=str(navt.index[-1].date()), max_rel_diff_trunc_vs_full=float((navt / nav.reindex(navt.index) - 1).abs().max()),
          all_signals_same_to_2015_11={s: bool(np.array_equal(ZSt[s].loc[:'2015-11'].to_numpy(), ZS[s].loc[:'2015-11'].to_numpy(), equal_nan=True)) for s in M.SIGNALS})
ok = A1['z24_nan_pattern_same'] and A1['z24_max_abs_diff'] < 1e-9 and A1['targets_same'] and A1['X_same'] and A2['max_abs_diff_vs_S75_nav'] < 0.011 and A3['max_rel_diff_trunc_vs_full'] < 1e-12 and all(A3['all_signals_same_to_2015_11'].values())
out = dict(A1=A1, A2=A2, A3=A3, reused_codes_dropped=P.reused_codes, signal_coverage_2010_06={s: int(ZS[s].loc['2010-06'].notna().sum()) for s in M.SIGNALS}, **{'pass': bool(ok)})
(M.HERE / 'accept.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(out, ensure_ascii=False, indent=1))

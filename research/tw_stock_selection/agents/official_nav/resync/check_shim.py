#!/usr/bin/env python3
"""墊片驗收（正式開關全開）：
 (1) harness 包過的 s4lib.make_signal（S7／S9 用，RT＝s2lib.rev_tables）＝ s10lib.make_signal_s10 E0（RT＝s19lib.rev_std＋a10_rows，正式管線用）：
     位移 0～3 的 sig、X、訊號列逐格相同，rank 差 < 1e-9；
 (2) harness 包過的 s9lib.signal_rows(conv='A', D=10) ＝ cslib.rev_avail_days（開 DEADLINE_FIX）逐月相同；
 (3) s42lib.av_family(D=10) 同 (2)。
用 harness 的同一套墊片：python3 -I -B harness.py official_nav/resync check_shim.py <official|exright> check_shim_<模式>
rank（SURGE 值）兩條路徑（s2lib.rev_tables 與 s19lib.rev_std）有 ≤ 4e-10 的浮點差（只影響 rank 欄存的分數，名單、出場、排序都相同），所以比到 1e-9（exright 模式同樣如此＝原本就有，非墊片造成）。"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import s19lib, s42lib                                     # noqa: F401（harness 會先 import 並裝墊片）
from s19lib import s10lib, s9lib, s4lib, s2lib, stlib, cslib

OFFICIAL = bool(cslib.DEADLINE_FIX and cslib.INS15_MASK)
assert cslib.EXRIGHT_FIX
P = s19lib.load()
s19lib.e50_fix_suspension(P)
INNOV, _ = s4lib.innov_panel(P)
U = stlib.universe(P, 'twse') & ~INNOV
S, _ = s19lib.rev_std(P)
months = list(S.index)
avA = s19lib.a10_rows(P, months)
RT = s2lib.rev_tables(P)
out = dict(av_rev_tables_eq_a10=bool(all(RT['av'][m] == avA[m] for m in months)))
for s in range(4):
    a = s4lib.make_signal(P, U, RT, shift=s, depth=10, mom_lo=0.10, exit='top20', keep_info=True)
    b = s10lib.make_signal_s10(P, U, dict(S=S, av=avA), shift=s)
    out[f'shift{s}'] = dict(sig=bool(np.array_equal(a['sig'], b['sig'])), X=bool(np.array_equal(a['X'], b['X'])),
                            rank=bool(np.array_equal(np.isnan(a['rank']), np.isnan(b['rank'])) and np.nanmax(np.abs(a['rank'] - b['rank'])) < 1e-9),
                            rank_bitwise=bool(np.array_equal(np.nan_to_num(a['rank'], nan=-9e9), np.nan_to_num(b['rank'], nan=-9e9))),
                            rank_max_abs_diff=float(np.nanmax(np.abs(a['rank'] - b['rank']))),
                            rows=[x['k'] for x in a['info']] == [x['k'] for x in b['info']])
sr, _ = s9lib.signal_rows(P, months, 'A', D=10)
out['s9_signal_rows_A10_eq_rev_avail_days'] = bool(all(sr[m] == avA[m] for m in months))
af = s42lib.av_family(P, months, 10)
out['s42_av_family_10_eq_rev_avail_days'] = bool(all(af[m] == avA[m] for m in months))
cslib.DEADLINE_FIX = False
sr0, _ = s9lib.signal_rows(P, months, 'A', D=10)
out['deadline_months_moved'] = {str(m): [str(P.IDX[sr0[m]].date()), str(P.IDX[sr[m]].date())] for m in months if sr0[m] != sr[m] and sr[m] < P.T}
cslib.DEADLINE_FIX = OFFICIAL
out['pass'] = bool(out['av_rev_tables_eq_a10'] and all(all(v for k, v in out[f'shift{s}'].items() if k in ('sig', 'X', 'rank', 'rows')) for s in range(4))
                   and out['s9_signal_rows_A10_eq_rev_avail_days'] and out['s42_av_family_10_eq_rev_avail_days'])
json.dump(out, open(Path('/home/user/-/research/tw_stock_selection/agents/official_nav/resync/out') / f"check_shim_{'official' if OFFICIAL else 'exright'}.json", 'w'), ensure_ascii=False, indent=1)
print(out)

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rcore as C, numpy as np, pandas as pd
C.REVRULE['mode'] = 'anchor'
T = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
for j in [int(a) for a in sys.argv[1:]]:
    days = [(m, C.jday(m, j)) for m in C.ANCHORS]
    pl = C.make_plan(days, 10, C.AB(0))
    wd = [(m, s) for m, s in days if s in C.POS]
    nx = C.jday(pd.Period('2026-10', 'M'), j); e_ = nx if (nx is not None and nx in C.POS) else C.IDX[-1]
    nav, lg = C.tsim([pl], *C.NORMAL, C.POS[wd[0][1]], C.POS[e_])
    t = T[f'J{j:+d}'].dropna(); t = t / t.loc[nav.index[0]]
    r1 = nav.pct_change(); r2 = t.reindex(nav.index).pct_change()
    d = (r1 - r2).abs(); bad = d[d > 1e-10]
    print(j, 'n bad days', len(bad), 'first', bad.index[:3].tolist(), 'their start', T[f'J{j:+d}'].first_valid_index().date(), 'end', T[f'J{j:+d}'].last_valid_index().date())
    if len(bad):
        b0 = bad.index[0]; pre = [p for p in pl if p[3] <= b0][-1]
        print('  last plan before', pre[3].date(), str(pre[2]), [C.COLS[x] for x in pre[1]])

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ind_core as C, numpy as np, pandas as pd
TT = pd.read_parquet(pathlib.Path(__file__).resolve().parents[1] / 'navs_oos.parquet')
nav = TT['OFF_AB0'].dropna(); print('researcher OFF_AB0', C.cagr(nav), C.tst(C.exw(nav, C.OFFSD)), nav.index[0], nav.index[-1])
p = C.ANCH[5]; s = C.OFF[p]; print(p, s, C.Dm(p), C.ab_kind(p, s))
# first differing window 2006-09-11: what kind?
for p in C.ANCH[:12]: print(p, C.OFF[p].date(), C.Dm(p).date(), C.ab_kind(p, C.OFF[p]))

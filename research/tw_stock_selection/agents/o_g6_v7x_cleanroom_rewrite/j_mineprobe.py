# J 診斷：本題名單層在指定日的逐檔狀態
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_g6_v7x_cleanroom_rewrite/')
import xlists
L = xlists.Lists(); X = L.X
for c in ['2311', '2325', '6286', '2473']:
    j = L.ci[c]; print(c, 'FOR', L.FOR[j], 'FIN', L.FIN[j], 'term', L.term.get(j), 'names', L.names_all.get(c))
for d in ['2017-10-11', '2012-04-11', '2024-01-11']:
    si = int(np.searchsorted(L.dates.values, np.datetime64(d))); print(d, L.dates[si], 'next', L.dates[si + 1], 'GOOD', X['GOOD'][si], X['GOOD'][si + 1])
    for c in ['2311', '2325', '6286', '2473', '1476', '1519']:
        j = L.ci[c]; V60 = X['V60'][si]
        print('  ', c, 'V60', V60[j], 'C', X['C'][si, j], 'r52', L.R52[si, j], 'Oe', X['O'][si + 1, j], 'disp', L.disp_mask(L.dates[si], L.dates[si + 1])[j], 'y3', L.y3(L.dates[si], (1, 2, 3))[L.ci[c]])
od = L.official_days(); print([ (rm, str(L.dates[si].date())) for rm, si in od if '2023-11' <= rm <= '2024-02'])

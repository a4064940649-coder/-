import sys, pickle
sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/main_v2')
import numpy as np, pandas as pd
import v2_core as V
YL, Y5, Y5f = V.load_revenue()
out = {}
for latest in (False, True):
    Y = Y5f if latest else Y5
    m = V.V2('gapfix', YL, Y, exfin=True, latest=latest)
    pl = m.plan(10, 'official')
    rows = []
    for e, pick, elidx, d in pl:
        for r, j in enumerate(pick):
            rows.append(dict(sig=d.date().isoformat(), exec=V.IDX[e].date().isoformat(), rank=r+1, code=V.COLS[j], case=V.deadline_case(d), npool=len(elidx)))
    out[latest] = pd.DataFrame(rows)
    print(latest, len(pl), len(rows), pl[0][3], pl[-1][3])
pickle.dump(out, open('c0.pkl','wb'))

#!/usr/bin/env python3
"""複製員：R_DEL 相對 ORIG 的新進股，只算「ORIG 當月缺營收、靠修補才合格」者，持有期報酬減同月其他入選股平均。"""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r_lib import *
YO = load_yoy('ORIG'); _, Y5, _ = V.load_revenue()
S0 = V.V2('gapfix', YO, Y5)
def plans(tag): return json.loads((H / f'my_plans_{tag}.json').read_text())
P0, P1 = plans('ORIG'), plans('R_DEL')
out = {}
for key in ('EXFIN|n10', 'ALL|n10', 'EXFIN|n30'):
    a = dict((d, set(c)) for d, c in P0[key])
    pl = [(IDX.get_loc(pd.Timestamp(d)) + 1, np.array([K.CI[c] for c in cs], int), None, pd.Timestamp(d)) for d, cs in P1[key]]
    _, per = V.stock_contrib(pl, IDX.get_loc(V.END1012))
    rows = []
    for d, pk, r in per:
        codes = [COLS[j] for j in pk]; hp = dict(zip(codes, r))
        new = [c for c in codes if c not in a.get(str(d.date()), set())]
        if not new: continue
        y0 = S0.y3(d)
        for c in new:
            patched = not np.isfinite(y0[K.CI[c]])
            oth = [hp[x] for x in codes if x not in new]
            rows.append((str(d.date()), c, patched, (hp[c] - np.mean(oth)) * 100))
    D = pd.DataFrame(rows, columns=['d', 'code', 'patched', 'ex_pp'])
    def st(x):
        x = x.to_numpy(); return {'n': len(x), 'mean_pp': round(float(x.mean()), 2), 't': round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 else None}
    out[key] = {'all_new': st(D.ex_pp), 'patched_only': st(D[D.patched].ex_pp), 'patched_months': int(D[D.patched].d.nunique()),
                'patched_list': D[D.patched].round(2).values.tolist()}
print(json.dumps(out, ensure_ascii=False, indent=1))
(H / 'my_entrants.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

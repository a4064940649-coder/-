import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sys, json
import numpy as np, pandas as pd
from r_lib import *
import r_src as S

MINP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
EXCL = int(sys.argv[2]) if len(sys.argv) > 2 else 130

D, first, last, calall = denominators()
idx, cols = D.index, D.columns
# trading-day position in full calendar
pos = pd.Series(np.arange(len(calall)), index=calall)
fpos = first.map(pos).reindex(cols)
P = np.broadcast_to(pos.reindex(idx).values[:, None], D.shape)
age = P - fpos.values[None, :]
Dm = D.values & ~(age < EXCL)
Dall = D.values
delisted = (last.reindex(cols) < pd.Timestamp('2026-10-01')).values
W = value_panel(calall)
pool = pool300(pd.DataFrame(Dm, idx, cols), W, minp=MINP).values
pool_all = pool300(D, W, minp=MINP).values


def raw_v1():
    t = S.tej(); b = S.bb(); h = S.mhist(); sm = S.snaps('margin-twse')
    mats = {'tej': S.wide(t[t.mbal.notna()], 'mbal'),
            'bb': S.wide(b, 'Margin_Balance'),
            'mh': S.wide(h[h.margin_bal.notna()], 'margin_bal'),
            'snap': S.wide(sm, 'margin_lots')}
    return {k: v.reindex(index=idx, columns=cols) for k, v in mats.items()}


def report(have, tag):
    out = {}
    for s in SEG:
        sm = seg_mask(idx, s)[:, None]
        r = {}
        for nm, den in [('all', Dm), ('pool', pool), ('delisted', Dm & delisted[None, :]),
                        ('surv', Dm & ~delisted[None, :]), ('all_noexcl', Dall)]:
            c, n, miss = cov(have, den & sm)
            r[nm] = round(100 * c, 2); r[nm + '_miss'] = miss
        dmiss = Dm & sm & ~have & delisted[None, :]
        r['delisted_miss_codes'] = int(dmiss.any(axis=0).sum())
        out[s] = r
    yr = {}
    for y in range(2006, 2027):
        ym = (idx.year == y)[:, None]
        yr[y] = round(100 * cov(have, Dm & ym)[0], 2)
    out['year'] = yr
    gm = ((idx >= '2008-09-01') & (idx <= '2009-03-31'))[:, None]
    out['2008-09~2009-03'] = round(100 * cov(have, Dm & gm)[0], 2)
    return out


if __name__ == '__main__':
    M = raw_v1()
    have = np.zeros(D.shape, bool)
    for k, v in M.items():
        have |= v.notna().values
    res = report(have, 'V1')
    print(json.dumps(res, ensure_ascii=False, indent=0))
    print('n codes', len(cols), 'delisted', delisted.sum())

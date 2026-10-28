import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylib import *
RN = leg_rates('main'); RF = leg_rates('etf'); RNs = leg_rates('main', True); RFs = leg_rates('etf', True)
def R(w, r, a, b, st=False):
    return run(w, r, a, b, GM['stress3' if st else 'normal'], GF, RNs if st else RN, RFs if st else RF)[0]
def wins(n, stub):
    pts = SD[(SD >= n.index[0]) & (SD <= n.index[-1])]
    if stub and n.index[0] not in pts: pts = pts.insert(0, n.index[0])
    if stub and n.index[-1] not in pts: pts = pts.append(pd.DatetimeIndex([n.index[-1]]))
    x = n.reindex(pts); return (x.shift(-1) / x - 1).iloc[:-1]
for seg, a, b, cend in (('VAL', '2018-12-28', '2024-02-15', '2023-12-29'), ('VAL', '2018-12-28', '2023-12-29', '2023-12-29'), ('POST', '2018-12-28', '2026-10-12', '2026-10-12'), ('HOLD', '2023-12-29', '2026-10-12', '2026-10-12')):
    for st in (False, True):
        n1 = R(0.3, 'NONE', a, b, st); n2 = R(0.3, 'M', a, b, st)
        c1 = n1[n1.index <= cend]; c2 = n2[n2.index <= cend]
        for stub in (False, True):
            d = wins(n1, stub) - wins(n2, stub)
            print(seg, a, b, 'S' if st else 'N', 'stub' if stub else '', f'dCAGR {cagr(c1)-cagr(c2):.3f} t {tstat(d):.2f} n {len(d)}')
print('---')
def wins2(n, front, back):
    pts = SD[(SD >= n.index[0]) & (SD <= n.index[-1])]
    if front and n.index[0] not in pts: pts = pts.insert(0, n.index[0])
    if back and n.index[-1] not in pts: pts = pts.append(pd.DatetimeIndex([n.index[-1]]))
    x = n.reindex(pts); return (x.shift(-1) / x - 1).iloc[:-1]
def c252(s): return ((s.iloc[-1] / s.iloc[0]) ** (252 / (len(s) - 1)) - 1) * 100
for seg, a, b in (('VAL', '2018-12-28', '2023-12-29'), ('VAL', '2019-01-02', '2023-12-29'), ('POST', '2018-12-28', '2026-10-12'), ('POST', '2019-01-02', '2026-10-12'), ('HOLD', '2023-12-29', '2026-10-12'), ('SEL', '2006-04-11', '2019-01-11'), ('SEL', '2006-04-11', '2018-12-28')):
    for st in (False, True):
        n1 = R(0.3, 'NONE', a, b, st); n2 = R(0.3, 'M', a, b, st)
        for f, bk in ((0, 1), (1, 0)):
            d = wins2(n1, f, bk) - wins2(n2, f, bk)
            print(seg, a, b, 'S' if st else 'N', f, bk, f'dCAGR {cagr(n1)-cagr(n2):.3f} d252 {c252(n1)-c252(n2):.3f} t {tstat(d):.2f} n {len(d)}')

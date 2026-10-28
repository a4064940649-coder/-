import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylib import *
RN = leg_rates('main'); RF = leg_rates('etf'); RNs = leg_rates('main', True); RFs = leg_rates('etf', True); RS = leg_rates('stock')
def R(w, r, a, b, st=False, gs=None):
    return run(w, r, a, b, GM['stress3' if st else 'normal'], GF if gs is None else gs, RNs if st else RN, (RFs if st else RF) if gs is None else RS)[0]
def wr_filter(n, a, b):
    """所有 sd 的窗報酬（窗起點在 [a,b]）"""
    x = n.reindex(SD[(SD >= n.index[0]) & (SD <= n.index[-1])])
    r = (x.shift(-1) / x - 1).iloc[:-1]
    return r[(r.index >= a) & (r.index <= b)]
VARS = {'HOLD': [('2024-02-15', '2026-10-12'), ('2023-12-29', '2026-10-12'), ('2023-12-11', '2026-10-12')],
        'VAL': [('2019-01-11', '2024-02-15'), ('2018-12-28', '2023-12-29'), ('2019-01-11', '2023-12-29'), ('2019-01-11', '2023-12-11'), ('2018-12-28', '2024-02-15')],
        'POST': [('2019-01-11', '2026-10-12'), ('2018-12-28', '2026-10-12'), ('2018-12-11', '2026-10-12')]}
WIN = {'HOLD': ('2024-01-01', '2026-10-12'), 'VAL': ('2019-01-01', '2023-12-31'), 'POST': ('2019-01-01', '2026-10-12')}
for seg, vs in VARS.items():
    for a, b in vs:
        for st in (False, True):
            n1 = R(0.3, 'NONE', a, b, st); n2 = R(0.3, 'M', a, b, st)
            wa, wb = WIN[seg]
            d = wr_filter(n1, wa, wb) - wr_filter(n2, wa, wb)
            line = f'{seg} {a}..{b} {"S" if st else "N"} dCAGR {cagr(n1)-cagr(n2):.3f} t {tstat(d):.2f} n {len(d)}'
            if not st: line += f' | M {cagr(n2):.2f} NONE {cagr(n1):.2f} 2330M {cagr(R(0.3,"M",a,b,gs=GS)):.2f}'
            print(line)

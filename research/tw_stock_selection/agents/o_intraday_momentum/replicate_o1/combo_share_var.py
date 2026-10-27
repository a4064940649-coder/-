import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = ['x', 'neutral']
src = open(Path(__file__).resolve().parent / 'combo.py').read().split('t0, t1, sd = S.bounds()')[0]
exec(src)
import numpy as np
t0, t1, sd = S.bounds(); tS = IDX.get_loc(R.SEL_END)
base = R.sim(pl0, t0, tS); ew = R.ew_nav(pl0, t0, tS)
A, B, WB = '2006-04-01', '2019-05-13', '2019-04-30'
def var_share(I, O_, kind):
    tot = I + O_
    if kind == 'abs': return I / (np.abs(I) + np.abs(O_))
    if kind == 'low': return np.where(tot > 0, I / np.where(tot > 0, tot, 1), np.where(np.isfinite(tot), -1e9, np.nan))
    if kind == 'raw': return I / tot
for kind in ('low', 'abs', 'raw'):
    SIG['sh126'] = var_share(I126, O126, kind); SIG['sh25221'] = var_share(I25221, O25221, kind)
    for v in ('v2', 'v4', 'v6', 'v7'):
        sig, how = VERS[v]
        nav = R.sim(mkplan(sig, how), t0, tS)
        o = R.seg(nav, sd, A, B, ew, A, WB); p = R.paired(nav, base, sd, A, WB)
        print(kind, v, o['CAGR'], o['maxDD'], o['t_0050'], o['t_EW'], p['mean_pct'], p['t'], flush=True)

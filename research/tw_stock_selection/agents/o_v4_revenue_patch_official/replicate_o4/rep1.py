import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
for n, (pls, out) in R.items():
    row = []
    for cost in ['normal', 'stress', 'feefull']:
        nav, st = out[cost]; o = K.summary(nav, pls[0])
        row.append(f"{cost}: {o['cagr']*100:.3f}%/{o['t']:.3f}")
    nav = out['normal'][0]; w = K.win_ret(nav, pls[0]).values
    s4 = K.summary(nav, pls[0], full=False)
    seg4 = (nav[K.END] / nav[nav.index < '2024-01-01'].iloc[-1]) ** (365.25 / (K.END - nav[nav.index < '2024-01-01'].index[-1]).days) - 1
    db = np.sort(w)[::-1]; dbt = K.tstat(np.delete(w, np.argmax(w)))
    print(n.ljust(22), ' | '.join(row), f"S4 {seg4*100:.1f} dropbest1 {dbt:.3f} nx {out['normal'][1]['nx']} frz {out['normal'][1]['nfrz']}")

"""Independent replication o6: priors and focus posteriors."""
import json
import sys
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6")
import numpy as np
import pandas as pd
from r_lib import TH, hac, decompose, fit_g1, fit_g2, fit_g3, post_grid, summ, post_g1
from scipy.stats import norm

OUT = '/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6/'
A = '/home/user/-/research/tw_stock_selection/agents/'
foc = pd.read_parquet(OUT + 'focus.parquet')
X = pd.read_parquet(OUT + 'flong.parquet') * 100
fam = json.load(open(OUT + 'fam.json'))
ref = json.load(open(A + 'o_shrink_core_satellite/data/families.json'))
SDC = 0.03624025193680153
FOC = ['AB', 'B10', 'EXEC', 'EXEC_J', 'AB_S', 'B10_S', 'EXEC_S']
COST = {'EXEC', 'EXEC_J', 'EXEC_S'}


def theory_matrix():
    fe = pd.read_parquet(A + 'o_main_spa_breaks/data/family_excess.parquet')
    return fe[ref['THEORY']].fillna(0.0).values * 100


def run(Amat, foc, rows=slice(None), L=6, verbose=True, priors=('G1', 'G2', 'G3'), ret_dens=False):
    Am = Amat[rows]
    x, s2, m, vP = decompose(Am, L)
    out = {'vP': vP, 'prior': {}}
    gs = {}
    if 'G2' in priors:
        g2, r = fit_g2(x, s2)
        gs['G2'] = g2
        c = np.cumsum(g2)
        out['prior']['G2'] = dict(mean=g2 @ TH, sd=np.sqrt(g2 @ TH ** 2 - (g2 @ TH) ** 2), P=g2[TH > 0].sum(),
                                  q95=TH[np.searchsorted(c, .95)], q99=TH[np.searchsorted(c, .99)], conv=bool(r.success))
    if 'G3' in priors:
        g3 = fit_g3(x, s2)
        gs['G3'] = g3
        c = np.cumsum(g3)
        out['prior']['G3'] = dict(mean=g3 @ TH, q95=TH[np.searchsorted(c, .95)], q99=TH[np.searchsorted(c, .99)],
                                  Keff=1 / np.sum(g3 ** 2))
    if 'G1' in priors:
        mu, t2, vmu = fit_g1(x, s2)
        out['prior']['G1'] = dict(mu=mu, tau=np.sqrt(t2))
    dens = {}
    for f in FOC:
        e = foc[f].values[rows]
        xf = e.mean()
        s2f = hac(e - m, L) + (SDC ** 2 if f in COST else 0)
        res = {'raw': xf, 't': xf / np.sqrt(s2f + vP)}
        for G, g in gs.items():
            p = post_grid(g, xf, s2f, vP)
            res[G] = summ(p)
            dens[(f, G)] = p
        if 'G1' in priors:
            res['G1'] = post_g1(mu, t2, vmu, xf, s2f, vP)
        res['s2f'] = s2f
        out[f] = res
    if ret_dens:
        return out, dens
    return out


if __name__ == '__main__':
    cols = list(X.columns)
    fams = {'MAIN': X[fam['MAIN']].values, 'MAIN_ref': X[ref['MAIN']].values, 'LOCO': X[fam['LOCO']].values,
            'LOCO_ref': X[ref['LOCO']].values, 'ALL': X.values, 'THEORY': theory_matrix()}
    allres = {}
    for nm, Am in fams.items():
        o, d = run(Am, foc, ret_dens=True)
        allres[nm] = o
        print('==', nm, 'K', Am.shape[1], 'sd common', np.sqrt(o['vP']))
        for k, v in o['prior'].items():
            print('  prior', k, {a: round(float(b), 3) for a, b in v.items()})
        for f in FOC:
            r = o[f]
            print('  %-7s raw %.3f t %.2f | G1 %.3f P %.3f | G2 %.3f P %.3f [%.2f,%.2f] ratio %.2f | G3 %.3f P %.3f' % (
                f, r['raw'], r['t'], r['G1']['mean'], r['G1']['P'], r['G2']['mean'], r['G2']['P'], r['G2']['q10'],
                r['G2']['q90'], r['G2']['mean'] / r['raw'], r['G3']['mean'], r['G3']['P']))
        if nm == 'MAIN':
            np.savez(OUT + 'post_main.npz', **{f'{f}|{G}': d[(f, G)] for (f, G) in d})
            np.savez(OUT + 'post_mainref.npz', **{})
        if nm == 'MAIN_ref':
            np.savez(OUT + 'post_mainref.npz', **{f'{f}|{G}': d[(f, G)] for (f, G) in d})
    json.dump(allres, open(OUT + 'post.json', 'w'), default=float, indent=1)

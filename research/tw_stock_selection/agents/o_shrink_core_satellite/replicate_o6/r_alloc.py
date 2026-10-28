"""Independent replication o6: core-satellite allocation simulation."""
import json
import sys
sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6")
import numpy as np
import pandas as pd
from r_lib import TH, hac, decompose, fit_g1, fit_g2, fit_g3, post_grid, summ, g1_grid

OUT = '/home/user/-/research/tw_stock_selection/agents/o_shrink_core_satellite/replicate_o6/'
A = '/home/user/-/research/tw_stock_selection/agents/'
SDC = 0.03624025193680153
COST = {'EXEC', 'EXEC_J', 'EXEC_S'}
XG = [0, .1, .2, .3, .4, .5, .56, .6, .7, .8, .9, 1.0]
C_OVER = 0.004855 + 0.001355   # main overweight: sell main + buy 0050
C_UNDER = 0.002355 + 0.001855  # main underweight: sell 0050 + buy main


def sb_index(n, H, L, B, seed):
    rng = np.random.default_rng(seed)
    idx = np.empty((B, H), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, B)
    new = rng.random((B, H)) < 1.0 / L
    jump = rng.integers(0, n, (B, H))
    for t in range(1, H):
        idx[:, t] = np.where(new[:, t], jump[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def draw_alpha(p, B, seed=7):
    rng = np.random.default_rng(seed)
    a = rng.choice(TH, size=B, p=p) + rng.uniform(-0.01, 0.01, B)
    return a / 100


def simulate(r50, rm, x, reb_every=1):
    """r50, rm: B x H decimal window returns. returns wealth paths of portfolio and 0050."""
    B, H = r50.shape
    wm = np.full(B, x)
    w5 = np.full(B, 1 - x)
    Wp = np.empty((B, H))
    for t in range(H):
        wm = wm * (1 + rm[:, t])
        w5 = w5 * (1 + r50[:, t])
        tot = wm + w5
        if 0 < x < 1 and ((t + 1) % reb_every == 0):
            dev = wm - x * tot
            cost = np.where(dev > 0, C_OVER, C_UNDER) * np.abs(dev)
            tot = tot - cost
            wm = x * tot
            w5 = (1 - x) * tot
        Wp[:, t] = tot
    W5 = np.cumprod(1 + r50, axis=1)
    return Wp, W5


def mdd(W):
    W = np.concatenate([np.ones((W.shape[0], 1)), W], 1)
    return (W / np.maximum.accumulate(W, 1) - 1).min(1)


def stats(Wp, W5):
    R = Wp / W5
    Rf = R[:, -1]
    H = R.shape[1]
    R0 = np.concatenate([np.ones((R.shape[0], 1)), R], 1)
    roll = R0[:, 60:] / R0[:, :-60]
    below = R < 1
    # longest run below 1
    run = np.zeros(R.shape[0])
    cur = np.zeros(R.shape[0])
    for t in range(H):
        cur = np.where(below[:, t], cur + 1, 0)
        run = np.maximum(run, cur)
    md = mdd(Wp)
    return dict(win=float((Rf > 1).mean()), p10=float(np.quantile(Rf, .1)), p50=float(np.median(Rf)),
                p90=float(np.quantile(Rf, .9)), lt75=float((Rf < .75).mean()), lt50=float((Rf < .5).mean()),
                mdd_med=float(np.median(md)), mdd_p10=float(np.quantile(md, .1)), lag_med=float(np.median(mdd(R))),
                any5=float((roll < 1).any(1).mean()), run60=float((run >= 60).mean()),
                elog=float(np.mean(np.log(Rf)) / (H / 12)))


def posteriors(Amat, foc, L=6, rows=slice(None)):
    Am = Amat[rows]
    x, s2, m, vP = decompose(Am, L)
    g2, _ = fit_g2(x, s2)
    g3 = fit_g3(x, s2)
    mu, t2, vmu = fit_g1(x, s2)
    out = {}
    for f in ['AB', 'B10', 'EXEC', 'EXEC_J']:
        e = foc[f].values[rows]
        xf = e.mean()
        s2f = hac(e - m, L) + (SDC ** 2 if f in COST else 0)
        out[(f, 'G2')] = post_grid(g2, xf, s2f, vP)
        out[(f, 'G3')] = post_grid(g3, xf, s2f, vP)
        out[(f, 'G1')] = g1_grid(mu, t2, vmu, xf, s2f, vP)
        out[(f, 'NOSHRINK')] = np.eye(len(TH))[np.argmin(np.abs(TH - xf))] if False else None
        out[(f, 'xhat')] = xf
        out[(f, 'FLAT')] = post_grid(np.ones(len(TH)) / len(TH), xf, s2f, vP)
    return out


def run_focus(foc, f, dens, idx, label, prior='G2', xs=XG, H=None, noshrink=False):
    e = foc[f].values / 100
    res = e - e.mean()
    r50 = foc['r0050'].values[idx]
    B = idx.shape[0]
    if noshrink:
        alpha = np.full(B, foc[f].values.mean() / 100)
    else:
        alpha = draw_alpha(dens[(f, prior)], B)
    rm = r50 + alpha[:, None] + res[idx]
    out = {}
    for x in xs:
        Wp, W5 = simulate(r50, rm, x)
        out[x] = {'10y': stats(Wp[:, :120], W5[:, :120]), '20y': stats(Wp, W5)}
    return out


if __name__ == '__main__':
    foc = pd.read_parquet(OUT + 'focus.parquet')
    X = pd.read_parquet(OUT + 'flong.parquet') * 100
    ref = json.load(open(A + 'o_shrink_core_satellite/data/families.json'))
    Amat = X[ref['MAIN']].values
    dens = posteriors(Amat, foc)
    idx6 = sb_index(245, 240, 6, 20000, 20261029)
    idx12 = sb_index(245, 240, 12, 20000, 20261030)
    R = {}
    R['EXEC_J|G2|L6'] = run_focus(foc, 'EXEC_J', dens, idx6, '')
    for f in ['AB', 'B10', 'EXEC']:
        R[f + '|G2|L6'] = run_focus(foc, f, dens, idx6, '', xs=[0, .5, .6, .7, .8, .9, 1.0])
    for G in ['G1', 'G3']:
        for f in ['AB', 'B10', 'EXEC', 'EXEC_J']:
            R[f'{f}|{G}|L6'] = run_focus(foc, f, dens, idx6, '', prior=G, xs=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    R['EXEC_J|G2|L12'] = run_focus(foc, 'EXEC_J', dens, idx12, '', xs=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    for f in ['AB', 'EXEC_J']:
        R[f + '|NOSHRINK|L6'] = run_focus(foc, f, dens, idx6, '', xs=[1.0], noshrink=True)
    # fine grid x* for EXEC_J G2
    fine = run_focus(foc, 'EXEC_J', dens, idx6, '', xs=[i / 100 for i in range(80, 101)])
    R['EXEC_J|G2|L6|fine'] = fine
    json.dump({k: {str(x): v for x, v in d.items()} for k, d in R.items()}, open(OUT + 'alloc.json', 'w'), indent=1)

    def xstar(d, h='10y'):
        ks = [x for x in d if abs(round(x * 10) - x * 10) < 1e-9]
        return max(ks, key=lambda x: d[x][h]['elog'])

    E = R['EXEC_J|G2|L6']
    print('EXEC_J G2 L6')
    for x in XG:
        a, b = E[x]['10y'], E[x]['20y']
        print(' x %.2f win %.3f/%.3f p10 %.2f/%.2f p50 %.2f/%.2f lt75 %.3f/%.3f lt50 %.3f/%.3f mdd %.3f/%.3f mddp10 %.3f/%.3f lag %.3f/%.3f any5 %.2f/%.2f run60 %.3f/%.3f elog %.5f' % (
            x, a['win'], b['win'], a['p10'], b['p10'], a['p50'], b['p50'], a['lt75'], b['lt75'], a['lt50'], b['lt50'],
            a['mdd_med'], b['mdd_med'], a['mdd_p10'], b['mdd_p10'], a['lag_med'], b['lag_med'], a['any5'], b['any5'],
            a['run60'], b['run60'], a['elog']))
    for k, d in R.items():
        xs = xstar(d) if 'fine' not in k else max(d, key=lambda x: d[x]['10y']['elog'])
        print(k, 'x*', xs, 'x=1 win %.3f/%.3f p10 %.2f/%.2f' % (d[1.0]['10y']['win'], d[1.0]['20y']['win'], d[1.0]['10y']['p10'], d[1.0]['20y']['p10']))

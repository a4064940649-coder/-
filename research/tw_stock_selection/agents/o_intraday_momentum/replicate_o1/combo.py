import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np, pandas as pd
import rlib as R
K, V = R.K, R.V
IDX, N = R.IDX, R.N
S = R.main_strategy(); pl0 = R.plan(10)
ci, co = R.daily_split()
def sums(look, skip):
    I, mI = R.roll_sum(ci, look, skip); O_, mO = R.roll_sum(co, look, skip)
    ok = mI & mO
    return np.where(ok, I, np.nan), np.where(ok, O_, np.nan)
def share(I, O_):
    tot = I + O_
    return np.where(tot > 0, I / np.where(tot > 0, tot, 1), np.nan)
I126, O126 = sums(126, 0); I252, O252 = sums(252, 0); I25221, O25221 = sums(273, 21)
SIG = {'ci126': I126, 'sh126': share(I126, O126), 'imo252': I252 - O252, 'sh25221': share(I25221, O25221), 'ci25221': I25221}
VERS = {'v1': ('ci126', 'leg3'), 'v2': ('sh126', 'leg3'), 'v3': ('imo252', 'leg3'), 'v4': ('sh126', 'mul52'), 'v5': ('ci126', 'mul52'),
        'v6': ('sh126', 'drop5'), 'v7': ('sh25221', 'drop5'), 'v8': ('ci25221', 'leg3')}
FILL = sys.argv[1] if len(sys.argv) > 1 else 'neutral'   # neutral：缺訊號給 0.5 百分位；drop：剔除
def pr(x, el):
    return pd.Series(np.where(el, x, np.nan)).rank(pct=True).to_numpy()
def mkplan(sig, how):
    out = []; held = set()
    for d in S.sig_days:
        cs = S.cross_section(d); el = cs['el'].copy()
        if el.sum() < 50: continue
        t, e = cs['t'], cs['e']
        x = SIG[sig][t]
        if FILL == 'drop' and how != 'drop5': el &= np.isfinite(x)
        r52, rrv = pr(cs['s52'], el), pr(cs['y3'], el)
        rx = pr(x, el & np.isfinite(x)); rx = np.where(el & ~np.isfinite(rx), 0.5, rx)
        if how == 'leg3': sc = (r52 + rrv + rx) / 3
        elif how == 'mul52': sc = 0.5 * r52 * rx + 0.5 * rrv
        else: sc = 0.5 * r52 + 0.5 * rrv
        sc = np.where(el, sc, np.nan)
        cs2 = dict(cs, sc=sc, el=el)
        gu = K.gap_up_open(e)
        cand = V.V2.order(cs2, 'official')[:int(el.sum())]
        lim = 20 if how == 'drop5' else 10
        pick = []
        for j in cand:
            if gu[j] and j not in held: continue
            pick.append(j)
            if len(pick) == lim: break
        if how == 'drop5':
            xx = np.array([x[j] if np.isfinite(x[j]) else np.inf for j in pick])   # 缺訊號不剔
            drop = set(np.array(pick)[np.argsort(xx, kind='stable')[:5]])
            pick = [j for j in pick if j not in drop][:10]
        out.append((e, np.array(pick, int), np.where(el)[0], d))
        held = set(pick)
    return out
t0, t1, sd = S.bounds()
tS = IDX.get_loc(R.SEL_END)
base = R.sim(pl0, t0, tS); ew = R.ew_nav(pl0, t0, tS)
A, B, WB = '2006-04-01', '2019-05-13', '2019-04-30'
res = {'base': {k: v for k, v in R.seg(base, sd, A, B, ew, A, WB).items() if k != 'yearly'}, 'fill': FILL}
print('base', res['base'])
for v, (sig, how) in VERS.items():
    p = mkplan(sig, how)
    nav = R.sim(p, t0, tS)
    o = {k: x for k, x in R.seg(nav, sd, A, B, ew, A, WB).items() if k != 'yearly'}
    o['paired'] = R.paired(nav, base, sd, A, WB)
    res[v] = o; print(v, sig, how, o, flush=True)
json.dump(res, open(R.HERE / f'combo_{FILL}.json', 'w'), ensure_ascii=False, indent=1)

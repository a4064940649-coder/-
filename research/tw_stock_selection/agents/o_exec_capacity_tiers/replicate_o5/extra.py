import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import *
res = pickle.load(open(os.path.join(OUT, 'A.pkl'), 'rb'))
ts = np.array([tvs(z[c], e50, sig) for c in z.columns])
print('zero-impact seed t median %.3f range %.3f-%.3f lt2.0 %d' % (np.median(ts), ts.min(), ts.max(), (ts < 2).sum()))
# falsification 30 vs 300
for mode in ('FIX', 'GROW'):
    a, _ = ab_nav(res, 'A', 3e5, mode); c, _ = ab_nav(res, 'A', 3e6, mode)
    d = win_rets(a, sig) - win_rets(c, sig); se = d.std(ddof=1) / np.sqrt(len(d))
    print(mode, '30-300 cagr diff %.3f pp, pair %.4f pp/m t %.3f MDE80 %.3f' % ((cagr(a) - cagr(c)) * 100, d.mean() * 100, d.mean() / se, 2.8 * se * 1200))
# breakeven
def t_after(tier, mode, x):
    R = []
    for s in range(40):
        r = res[('A', s, tier, mode)]; n = r['nav']
        rr = np.r_[0, n[1:] / n[:-1] - 1] - x * r['comp'][:, 5]
        R.append(rr)
    nav = pd.Series(np.cumprod(1 + np.mean(R, 0)), index=idx)
    return tvs(nav, e50, sig)
for mode in ('FIX', 'GROW'):
    for tier in TN:
        out = []
        for tgt in (2.3, 2.0):
            if t_after(tier, mode, 0) < tgt: out.append(0.0); continue
            lo, hi = 0.0, 0.01
            for _ in range(40):
                m = (lo + hi) / 2
                if t_after(tier, mode, m) >= tgt: lo = m
                else: hi = m
            out.append(lo * 100)
        print('breakeven', mode, TN[tier], ['%.4f%%' % v for v in out])
# V4 main plan
for mode in ('GROW', 'FIX'):
    for tier in TN:
        n = pd.Series(res[('A', -1, tier, mode)]['nav'], index=idx)
        print('V4', mode, TN[tier], 'cagr %.2f t %.3f' % (cagr(n) * 100, tvs(n, e50, sig)))

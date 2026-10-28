import sys, math; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylib import *
RN = leg_rates('main'); RF = leg_rates('etf')
s = NAV['normal']
for a in ('2018-12-28', '2019-01-02', '2019-01-11'):
    x = s[(s.index >= a) & (s.index <= '2026-10-12')]; print('main POST from', a, cagr(x))
for a in ('2018-12-28', '2019-01-02', '2019-01-11'):
    print('50|B15 POST from', a, cagr(run(0.5, 'B15', a, '2026-10-12', GM['normal'], GF, RN, RF)[0]))
m = seg_nav('normal', *SEG['SEL']); f = seg_nav('e50', *SEG['SEL'])
rm, rf = winret(m), winret(f)
x = (np.log1p(rm) - np.log1p(rf)).to_numpy(); xc = x - x.mean()
print('log acf', {L: round(float((xc[L:] * xc[:-L]).sum() / (xc ** 2).sum()), 3) for L in (1, 6, 12, 36)})
for w in (0.2, 0.3, 0.4):
    nM = run(w, 'M', *SEG['SEL'], GM['normal'], GF, RN, RF)[0]; rM = winret(nM)
    print(w, 'log-window geo diff', 12 * (np.log1p(rM).mean() - w * np.log1p(rf).mean() - (1 - w) * np.log1p(rm).mean()) * 100,
          'simple-window arith diff', 12 * (rM.mean() - w * rf.mean() - (1 - w) * rm.mean()) * 100)
_, wN, _ = run(0.3, 'NONE', '2006-04-11', '2026-10-12', GM['normal'], GF, RN, RF)
print({str(d.date()): round(wN.loc[d], 4) for d in SD if d.year == 2026})
def vr(x, q):
    xc = x - x.mean(); s = np.convolve(xc, np.ones(q), 'valid'); return float((s ** 2).mean() / (q * (xc ** 2).mean()))
def vr_lm(x, q):
    n = len(x); mu = x.mean(); s1 = ((x - mu) ** 2).sum() / (n - 1)
    sq = np.convolve(x, np.ones(q), 'valid'); m = q * (n - q + 1) * (1 - q / n)
    return float(((sq - q * mu) ** 2).sum() / m / s1)
rng = np.random.default_rng(1)
print('VR log', {q: (round(vr(x, q), 3), round(vr_lm(x, q), 3), np.round(np.percentile([vr_lm(rng.permutation(x), q) for _ in range(2000)], [5, 95]), 2).tolist()) for q in (12, 36, 60)})

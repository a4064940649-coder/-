import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from mylib import *
O = {}
def P(*a): print(*a, flush=True)
RN = leg_rates('main'); RF = leg_rates('etf'); RNs = leg_rates('main', True); RFs = leg_rates('etf', True); RS = leg_rates('stock'); RSs = leg_rates('stock', True)
def R(w, rule, seg, stress=False, leg='0050', gm=None, **kw):
    a, b = SEG[seg] if isinstance(seg, str) else seg
    gm_ = GM['stress3' if stress else 'normal'] if gm is None else gm
    if leg == '0050': return run(w, rule, a, b, gm_, GF, RNs if stress else RN, RFs if stress else RF, **kw)
    return run(w, rule, a, b, gm_, GS, RNs if stress else RN, RSs if stress else RS, **kw)
a, b = SEG['SEL']
main = seg_nav('normal', a, b); e50 = seg_nav('e50', a, b)
rel = winret(main) - winret(e50)
P('SEL rel mean', rel.mean() * 100, 'sd', rel.std(ddof=1) * 100, 'TE', rel.std(ddof=1) * math.sqrt(12) * 100, 'n', len(rel))
x = rel.to_numpy(); xc = x - x.mean()
ac = {L: float((xc[L:] * xc[:-L]).sum() / (xc ** 2).sum()) for L in (1, 6, 12, 36)}
P('autocorr', ac, 'ci', 1.645 / math.sqrt(len(x)))
def vr(x, q):
    xc = x - x.mean(); s = np.convolve(xc, np.ones(q), 'valid')
    return float((s ** 2).mean() / (q * (xc ** 2).mean()))
rng = np.random.default_rng(0)
for q in (12, 36, 60):
    null = [vr(rng.permutation(x), q) for _ in range(2000)]
    P('VR', q, vr(x, q), np.percentile(null, [5, 95]))
dm = pd.Series(GM['normal'], index=IDX) - 1; df_ = pd.Series(GF, index=IDX) - 1
sel = (IDX > pd.Timestamp(a)) & (IDX <= pd.Timestamp(b))
yc = pd.DataFrame({'m': dm[sel], 'f': df_[sel]}).groupby(lambda d: d.year).apply(lambda g: g.m.corr(g.f))
P('daily corr by year', yc.round(2).to_dict())
P('monthly corr', np.corrcoef(winret(main), winret(e50))[0, 1])
var_ann = rel.var(ddof=1) * 12
for w in (0.2, 0.3, 0.4):
    nM, _, tM = R(w, 'M', 'SEL'); nN, wN, _ = R(w, 'NONE', 'SEL')
    geo = (1 - w) * cagr(main) + w * cagr(e50)
    # 曝險對齊：用 NONE 的平均 0050 權重做每月再平衡
    wbar_d = wN.mean(); wbar_m = wN.reindex(SD[(SD >= a) & (SD < b)]).mean()
    al_d = R(wbar_d, 'M', 'SEL')[0]; al_m = R(wbar_m, 'M', 'SEL')[0]
    P(f'w={w} theory divret', 0.5 * w * (1 - w) * var_ann * 100, 'M-geo', cagr(nM) - geo, 'M-NONE', cagr(nM) - cagr(nN),
      'NONE-aligned(daily wbar %.3f)' % wbar_d, cagr(nN) - cagr(al_d), 'NONE-aligned(win wbar %.3f)' % wbar_m, cagr(nN) - cagr(al_m))
nNall, wNall, _ = R(0.3, 'NONE', 'ALL')
P('NONE drift w at 2019-01-11', wNall.loc['2019-01-11'], 'end', wNall.iloc[-1], 'w at first 2019 day', wNall[wNall.index.year == 2019].iloc[0])
# 規則表
P('--- SEL w=30 rules')
RULES = ['M', 'Q', 'H', 'A', 'NONE', 'B5', 'B10', 'B15', 'MOM12']
tab = {}
for st in (False, True):
    nM = R(0.3, 'M', 'SEL', st)[0]
    for r in RULES:
        n, _, tr = R(0.3, r, 'SEL', st)
        p = paired(n, nM); s = stats(n, seg_nav('e50', a, b))
        tab[(st, r)] = (p, s, len(tr))
        P('stress' if st else 'normal', r, 'd=%.3f t=%.2f' % (p['dCAGR'], p['t']), 'CAGR=%.2f TE=%.1f MDD=%.1f n_tr=%d' % (s['CAGR'], s['TE'], s['MDD'], len(tr)))
nM = R(0.3, 'M', 'SEL')[0]
P('A anchors', {m: round(paired(R(0.3, f'A{m}', 'SEL')[0], nM)['dCAGR'], 3) for m in range(1, 13)})
# 拿掉最大年
def drop_year(n1, n2, yr):
    out = []
    for n in (n1, n2):
        g = n / n.shift(1); g = g[g.index.year != yr].dropna()
        yrs = ((n.index[-1] - n.index[0]).days - (pd.Timestamp(f'{yr}-12-31') - pd.Timestamp(f'{yr}-01-01')).days) / 365.25
        out.append(g.prod() ** (1 / yrs) * 100 - 100)
    return out[0] - out[1]
for st in (False, True):
    n1 = R(0.3, 'NONE', 'SEL', st)[0]; n2 = R(0.3, 'M', 'SEL', st)[0]
    d = winret(n1) - winret(n2)
    yd = {}
    for yr in range(2006, 2019):
        g1 = n1[n1.index.year == yr]; g2 = n2[n2.index.year == yr]
        yd[yr] = (g1.iloc[-1] / n1[n1.index < g1.index[0]].iloc[-1] if yr > 2006 else g1.iloc[-1]) - (g2.iloc[-1] / n2[n2.index < g2.index[0]].iloc[-1] if yr > 2006 else g2.iloc[-1])
    ymax = max(yd, key=yd.get)
    dw = d[d.index.year != ymax]
    P('stress' if st else 'normal', 'max year', ymax, 'drop-year dCAGR(daily)', drop_year(n1, n2, ymax), 'mean win diff ex-year*12', dw.mean() * 1200, 't', tstat(dw))
# 樣本外
P('--- OOS w=30 NONE vs M')
for seg in ('VAL', 'HOLD', 'POST'):
    for st in (False, True):
        p = paired(R(0.3, 'NONE', seg, st)[0], R(0.3, 'M', seg, st)[0]); P(seg, 'stress' if st else 'normal', p)
nNa = R(0.3, 'NONE', 'ALL')[0]; nMa = R(0.3, 'M', 'ALL')[0]
for seg in ('POST', 'HOLD'):
    a2, b2 = SEG[seg]
    s1 = nNa[(nNa.index >= a2) & (nNa.index <= b2)]; s2 = nMa[(nMa.index >= a2) & (nMa.index <= b2)]
    P('continuous', seg, paired(s1, s2))
# 並列
P('--- side by side')
def row(lab, segs, f):
    res = {}
    for sg in segs: res[sg] = f(sg)
    return res
for st in (False, True):
    for lab, f in (('main', lambda sg: seg_nav('stress3' if st else 'normal', *SEG[sg])), ('70/30 M', lambda sg: R(0.3, 'M', sg, st)[0]),
                   ('70/30 NONE', lambda sg: R(0.3, 'NONE', sg, st)[0]), ('2330 30 M', lambda sg: R(0.3, 'M', sg, st, leg='2330')[0]),
                   ('0050', lambda sg: seg_nav('e50', *SEG[sg])), ('EW', lambda sg: seg_nav('ew', *SEG[sg]))):
        for sg in ('ALL', 'HOLD'):
            n = f(sg); s = stats(n, seg_nav('e50', *SEG[sg]))
            P('stress' if st else 'normal', lab, sg, 'CAGR %.2f t %.2f MDD %.1f TE %.1f' % (s['CAGR'], s['t'], s['MDD'], s['TE']))
# HOLD with base 2023-12-29 for main
s = NAV['normal']; P('main HOLD base 2023-12-29', cagr(s[(s.index >= '2023-12-29') & (s.index <= '2026-10-12')]))
s = NAV['e50']; P('0050 HOLD base 2023-12-29', cagr(s[(s.index >= '2023-12-29') & (s.index <= '2026-10-12')]))
s = NAV['nodisc']; sn = s[s.index <= '2026-10-12']; P('nodisc ALL', cagr(sn), stats(sn / sn.iloc[0], seg_nav('e50', *SEG['ALL']))['t'])
# 每月再平衡成本
for sg in ('SEL', 'ALL'):
    n, _, tr = R(0.3, 'M', sg)
    def nocost(): return run(0.3, 'M', *SEG[sg], GM['normal'], GF, (0, 0), (0, 0))[0]
    P(sg, 'M cost pp/yr', cagr(nocost()) - cagr(n), 'median transfer %', np.median([t[1] for t in tr]) * 100)
# TE 前緣
P('--- TE frontier')
for sg in ('SEL', 'ALL', 'POST', 'HOLD'):
    mn = seg_nav('normal', *SEG[sg]); e = seg_nav('e50', *SEG[sg]); sm = stats(mn, e)
    vals = []
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        s = stats(R(w, 'M', sg)[0], e); vals.append((w, round(s['TE'] / sm['TE'], 4), round((sm['CAGR'] - s['CAGR']) / (sm['TE'] - s['TE']), 3)))
    P(sg, vals)
# 放棄風險
P('--- abandonment')
def longest_under(n, e):
    r = (winret(n) + 1).cumprod() / (winret(e) + 1).cumprod(); r = pd.concat([pd.Series([1.0]), r.reset_index(drop=True)])
    pk = r.cummax(); below = (r < pk - 1e-12).to_numpy()
    best = cur = 0
    for x in below:
        cur = cur + 1 if x else 0; best = max(best, cur)
    return best, float((r / pk - 1).min() * 100)
def roll36(n, e):
    a1 = (winret(n) + 1); b1 = (winret(e) + 1)
    ra = a1.rolling(36).apply(np.prod, raw=True); rb = b1.rolling(36).apply(np.prod, raw=True)
    d = (ra - rb).dropna()
    return float((d < 0).mean()), float(d.min() * 100)
for sg in ('ALL', 'HOLD'):
    e = seg_nav('e50', *SEG[sg])
    for w in (0, 0.2, 0.3, 0.4):
        n = seg_nav('normal', *SEG[sg]) if w == 0 else R(w, 'M', sg)[0]
        P(sg, w, longest_under(n, e), roll36(n, e) if sg == 'ALL' else '')
# TE 預算表
P('--- TE budget')
grid = {}
for w in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
    for r in (RULES if w > 0 else ['M']):
        sS = stats(R(w, r, 'SEL')[0] if w > 0 else seg_nav('normal', *SEG['SEL']), seg_nav('e50', *SEG['SEL']))
        grid[(w, r)] = sS
for tgt in (10, 12, 15, 18):
    ok = {k: v for k, v in grid.items() if v['TE'] <= tgt}
    k = max(ok, key=lambda k: ok[k]['CAGR'])
    w, r = k
    sP = stats(R(w, r, 'POST')[0] if w > 0 else seg_nav('normal', *SEG['POST']), seg_nav('e50', *SEG['POST']))
    P(tgt, k, 'SEL TE %.1f' % grid[k]['TE'], 'POST TE %.1f CAGR %.1f MDD %.1f' % (sP['TE'], sP['CAGR'], sP['MDD']))

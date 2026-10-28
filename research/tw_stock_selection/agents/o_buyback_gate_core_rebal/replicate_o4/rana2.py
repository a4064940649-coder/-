import sys, math; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
from mylib import *
import mylib
SEG2 = {'SEL': ('2006-04-11', '2019-01-11'), 'VAL': ('2018-12-28', '2023-12-29'), 'HOLD': ('2023-12-29', '2026-10-12'), 'POST': ('2018-12-28', '2026-10-12'), 'ALL': ('2006-04-11', '2026-10-12')}
def wins(n):
    pts = SD[(SD >= n.index[0]) & (SD <= n.index[-1])]
    if n.index[0] not in pts: pts = pts.insert(0, n.index[0])
    x = n.reindex(pts); return (x.shift(-1) / x - 1).iloc[:-1]
def st2(n, e):
    d = wins(n) - wins(e); return dict(CAGR=cagr(n), MDD=mdd(n), t=tstat(d), TE=float(d.std(ddof=1) * math.sqrt(12) * 100))
RN = leg_rates('main'); RF = leg_rates('etf')
def R(w, r, sg, gm=None, rm=RN, rf=RF, gf=GF):
    return run(w, r, *SEG2[sg], GM['normal'] if gm is None else gm, gf, rm, rf)[0]
def snav(col, sg):
    s = NAV[col]; a, b = SEG2[sg]; s = s[(s.index >= a) & (s.index <= b)]; return s / s.iloc[0]
for sg in ('HOLD', 'POST', 'VAL'):
    e = snav('e50', sg)
    for lab, n in (('main', snav('normal', sg)), ('70/30M', R(0.3, 'M', sg)), ('NONE', R(0.3, 'NONE', sg)), ('2330M', run(0.3, 'M', *SEG2[sg], GM['normal'], GS, RN, leg_rates('stock'))[0]), ('0050', e)):
        print(sg, lab, {k: round(v, 3) for k, v in st2(n, e).items()})
print('--- TE frontier (HOLD/POST base day)')
for sg in ('POST', 'HOLD'):
    e = snav('e50', sg); sm = st2(snav('normal', sg), e)
    print(sg, [(w, round((sm['CAGR'] - s['CAGR']) / (sm['TE'] - s['TE']), 3)) for w in (0.1, 0.2, 0.3, 0.4, 0.5) for s in [st2(R(w, 'M', sg), e)]])
def longest_under(n, e):
    r = (wins(n) + 1).cumprod() / (wins(e) + 1).cumprod(); r = pd.concat([pd.Series([1.0]), r.reset_index(drop=True)])
    pk = r.cummax(); best = cur = 0
    for x in (r < pk - 1e-12): cur = cur + 1 if x else 0; best = max(best, cur)
    return best
e = snav('e50', 'HOLD'); print('HOLD longest under', [longest_under(snav('normal', 'HOLD') if w == 0 else R(w, 'M', 'HOLD'), e) for w in (0, 0.2, 0.3, 0.4)])
print('--- TE budget POST realized (base 2018-12-28)')
eP = snav('e50', 'POST')
for k in ((0.5, 'B15'), (0.4, 'B15'), (0.2, 'B10'), (0, 'M')):
    n = snav('normal', 'POST') if k[0] == 0 else R(k[0], k[1], 'POST'); print(k, {kk: round(v, 2) for kk, v in st2(n, eP).items()})
# 連續路徑
nN = R(0.3, 'NONE', 'ALL'); nM = R(0.3, 'M', 'ALL')
for a, b in (('2018-12-28', '2026-10-12'), ('2019-01-11', '2026-10-12')):
    s1 = nN[(nN.index >= a) & (nN.index <= b)]; s2 = nM[(nM.index >= a) & (nM.index <= b)]
    print('continuous POST', a, cagr(s1) - cagr(s2), tstat(wins(s1) - wins(s2)))
_, wN, _ = run(0.3, 'NONE', *SEG2['ALL'], GM['normal'], GF, RN, RF)
print('drift w: 2019-01-11', wN.loc['2019-01-11'], '2026 first', wN[wN.index.year == 2026].iloc[0], '2026 mean', wN[wN.index.year == 2026].mean(), 'end', wN.iloc[-1])
# 幾何加權
a, b = SEG2['SEL']; m = snav('normal', 'SEL'); f = snav('e50', 'SEL'); cm, cf = cagr(m), cagr(f)
for w in (0.2, 0.3, 0.4):
    nMw = R(w, 'M', 'SEL'); g1 = (1 - w) * cm + w * cf; g2 = ((1 + cm / 100) ** (1 - w) * (1 + cf / 100) ** w - 1) * 100
    lm = np.log(m.iloc[-1]); lf = np.log(f.iloc[-1]); yrs = (m.index[-1] - m.index[0]).days / 365.25
    print('w', w, 'M-arith', cagr(nMw) - g1, 'M-geo', cagr(nMw) - g2)
# 自相關另一口徑
x = (winret(m) - winret(f)).to_numpy()
print('ac corrcoef', {L: round(np.corrcoef(x[L:], x[:-L])[0, 1], 3) for L in (1, 6, 12, 36)})
# 損益兩平滑價
SL = pd.read_parquet(HERE / 'my_navs_slip.parquet'); sl = np.array([float(c) for c in SL.columns])
def g_(s): g = (s / s.shift(1)).to_numpy().copy(); g[0] = 1; return np.where(np.isfinite(g), g, 1.0)
def interp(vals, target):
    vals = np.array(vals)
    for i in range(len(sl) - 1):
        if (vals[i] - target) * (vals[i + 1] - target) <= 0:
            return sl[i] + (target - vals[i]) * (sl[i + 1] - sl[i]) / (vals[i + 1] - vals[i])
    return float('nan')
res = {}
for mode in ('etf_fixed', 'etf_same'):
    T = {'main': [], 'M': [], 'NONE': []}; H = {'main': [], 'M': []}
    for c, s_ in zip(SL.columns, sl):
        gm = g_(SL[c]); rm = leg_rates('main', slip=s_); rf = leg_rates('etf', slip=(None if mode == 'etf_fixed' else s_))
        eA = snav('e50', 'ALL'); eH = snav('e50', 'HOLD')
        mA = SL[c] / SL[c].iloc[0]; mA = mA[mA.index <= '2026-10-12']
        T['main'].append(st2(mA, eA)['t'])
        T['M'].append(st2(R(0.3, 'M', 'ALL', gm, rm, rf), eA)['t']); T['NONE'].append(st2(R(0.3, 'NONE', 'ALL', gm, rm, rf), eA)['t'])
        mH = mA[mA.index >= '2023-12-29']; mH = mH / mH.iloc[0]
        H['main'].append(cagr(mH) - cagr(eH)); H['M'].append(cagr(R(0.3, 'M', 'HOLD', gm, rm, rf)) - cagr(eH))
    print(mode, 't=2.3', {k: round(interp(v, 2.3) * 100, 3) for k, v in T.items()}, 't=2.0', {k: round(interp(v, 2.0) * 100, 3) for k, v in T.items()},
          'HOLD ex=0', {k: round(interp(v, 0) * 100, 3) for k, v in H.items()})
# 資金可行性
C = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/close_raw.parquet')
Op = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/open.parquet')
for cap in (1e6, 3e6):
    slot = cap * 0.7 / 10
    for which in ('open_exec', 'close_sig'):
        rows = []
        for ex, sg, picks in META['plan']:
            for c in picks:
                p = Op.at[pd.Timestamp(ex), c] if which == 'open_exec' else C.at[pd.Timestamp(sg), c]
                if np.isfinite(p): rows.append((ex, p * 1000))
        df = pd.DataFrame(rows, columns=['ex', 'lot']); df['ex'] = pd.to_datetime(df.ex)
        df['nolot'] = df.lot > slot; df['inv'] = np.floor(slot / df.lot) * df.lot / slot
        h = df[df.ex >= '2024-01-01']
        print(int(cap), which, 'nolot %.3f HOLD %.3f' % (df.nolot.mean(), h.nolot.mean()), 'inv %.3f HOLD %.3f' % (df.inv.mean(), h.inv.mean()), len(df))
print('2330 lot', C['2330'].dropna().iloc[-1] * 1000, C['2330'].dropna().index[-1])
# 多重檢定
from math import erf
p = lambda t: 2 * (1 - 0.5 * (1 + erf(abs(t) / math.sqrt(2))))
print('p(2.62)', p(2.62), 'bonf63', min(1, 63 * p(2.62)), 'p(1.74)', p(1.74), 'bonf', min(1, 63 * p(1.74)))

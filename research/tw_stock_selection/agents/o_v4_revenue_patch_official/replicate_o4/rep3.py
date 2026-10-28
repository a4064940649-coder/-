import sys, pickle; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/o_v4_revenue_patch_official/replicate_o4')
import rcore as K, pandas as pd, numpy as np
SP = '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/'
R = pickle.load(open(SP + 'rep_o4_batch.pkl', 'rb'))
PL = R['V4'][0][0]
LG = np.log(K.GROW); CL = np.vstack([np.zeros(K.N), np.cumsum(LG, 0)])
def sret(i0, i1, j, mode):
    if mode == 'cc': return np.exp(CL[i1 + 1, j] - CL[i0 + 1, j]) - 1
    # 開盤到開盤：e 開盤 → e' 開盤
    a = K.On[i0 + 1, j]; c = K.Cn[i0 + 1, j]; b = K.On[i1 + 1, j] if i1 + 1 < len(K.IDX) else np.nan; c1 = K.Cn[i1 + 1, j] if i1 + 1 < len(K.IDX) else np.nan
    g = np.exp(CL[i1 + 1, j] - CL[i0 + 2, j]) if i1 + 1 < len(K.IDX) else np.exp(CL[i1+1, j] - CL[i0+2, j])
    f0 = c / a if np.isfinite(a) and np.isfinite(c) else 1.0
    f1 = (b / c1) if np.isfinite(b) and np.isfinite(c1) else 1.0
    return f0 * g * f1 - 1 if mode == 'oo' else np.nan
def falsify(name, isg, mode='cc'):
    pl = R[name][0][0]; ds = [p['i_d'] for p in pl]; de = ds[1:] + [K.TPOS[K.END]]
    out = []
    for p, i1 in zip(pl, de):
        g = [j for j in p['picks'] if isg(p, j)]; o = [j for j in p['picks'] if not isg(p, j)]
        if g and o:
            out.append(np.mean([sret(p['i_d'], i1, j, mode) for j in g]) - np.mean([sret(p['i_d'], i1, j, mode) for j in o]))
    out = np.array(out) * 100
    return round(out.mean(), 2), round(K.tstat(out), 2) if len(out) > 1 else None, len(out)
for mode in ['cc', 'oo']:
    print(mode, 'B10', falsify('B10', lambda p, j: j in p['gap'], mode))
    def isq(p, j):
        d = K.IDX[p['i_d']]; m = pd.Period(d, 'M')
        return K.COLS[j] in K.GAP and not np.isfinite(K.y3(K.YF if d < K.SWITCH else K.Y5, m)[j])
    print(mode, 'BQ', falsify('BQ', isq, mode))
    def isr(p, j):
        d = K.IDX[p['i_d']]; m = pd.Period(d, 'M')
        return d < K.SWITCH and not np.isfinite(K.y3(K.YD, m)[j])
    print(mode, 'RFIX', falsify('V4', isr, mode))
# 處置口徑不同的名額
n = 0; r0 = 0; an = 0; top10 = 0
for p in PL:
    i_d, i_e = p['i_d'], p['i_e']; d, e = K.IDX[i_d], K.IDX[i_e]; m = pd.Period(d, 'M')
    Y = K.YF if d < K.SWITCH else K.Y5; yv = K.y3(Y, m)
    v = K.V60[i_d]; c = K.Cn[i_d]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(c) & (c >= 10)
    bi = np.where(base)[0]; bi = bi[np.argsort(-v[bi], kind='stable')][:300]
    top = np.zeros(K.N, bool); top[bi] = True
    ok = top & np.isfinite(K.S52[i_d]) & np.isfinite(K.On[i_e]) & np.isfinite(K.Cn[i_e]) & ~K.FINMASK & ~K.FIN2MASK & ~K.FORMASK & np.isfinite(yv)
    a = K.disposal_mask(d, e); b = K.disposal_mask(d, d)
    dif = np.where(ok & (a != b))[0]
    for j in dif:
        n += 1
        sel = (K.DISP_J == j) & (K.DISP_A <= np.datetime64(d))
        st = K.DISP_E[sel & (K.DISP_S <= np.datetime64(d)) & (np.datetime64(d) <= K.DISP_E)]
        if b[j] and not a[j]: r0 += 1
        elif a[j] and not b[j] and (K.DISP_A[sel & (K.DISP_S <= np.datetime64(e)) & (np.datetime64(e) <= K.DISP_E)] == np.datetime64(d)).any(): an += 1
    # 前 10：在兩口徑的聯集池中計分
    pool = ok & ~(a & b)
    idx = np.where(pool)[0]
    s = pd.Series(K.S52[i_d, idx]).rank(pct=True).to_numpy(); q = pd.Series(yv[idx]).rank(pct=True).to_numpy()
    sc = np.round(0.5 * s + 0.5 * q, 10); order = idx[np.lexsort((idx, -yv[idx], -sc))]
    top10 += len(set(order[:10]) & set(dif))
print('disposal diff slots', n, 'rem0', r0, 'ann=d', an, 'in top10', top10)
# 池內缺口
rows = []
PLD = {p['i_d']: p for p in R['RDEL'][0][0]}
for p in PL:
    i_d = p['i_d']; d = K.IDX[i_d]
    add = len(set(p['pool']) - set(PLD[i_d]['pool'])) if i_d in PLD else np.nan
    rows.append((d, add))
B = {p['i_d']: p for p in R['B10'][0][0]}
BQp = R['BQ'][0][0]
for i, p in enumerate(PL):
    b = B.get(p['i_d']); d = K.IDX[p['i_d']]; m = pd.Period(d, 'M')
    ng = len(b['gap']) if b else np.nan
    q = K.y3(K.YQ, m); nq = int(np.isfinite(q[b['gap']]).sum()) if b and len(b['gap']) else 0
    rows[i] = rows[i] + (ng, nq)
df = pd.DataFrame(rows, columns=['d', 'rfix_add', 'gap_missing', 'q_cover']).set_index('d')
for a, b in [('2006', '2012'), ('2013', '2018'), ('2019', '2026')]:
    x = df.loc[a:b]; print(a, b, x.mean().round(3).to_dict())
print('rfix pick months', [(K.IDX[p['i_d']].date(), K.COLS[j]) for p in PL for j in p['picks'] if K.IDX[p['i_d']] < K.SWITCH and not np.isfinite(K.y3(K.YD, pd.Period(K.IDX[p['i_d']], 'M'))[j])])
# fin2 / 2008-04
p3 = {p['i_d']: p for p in R['O3+forex'][0][0]}
print('o3 picks fin2', [(K.IDX[k].date(), K.COLS[j]) for k, p in p3.items() for j in p['picks'] if K.FIN2MASK[j]])
pS = {p['i_d']: set(p['picks']) for p in R['O3+forex+fin2'][0][0]}; pT = {p['i_d']: set(p['picks']) for p in R['O3+forex+fin2+trade'][0][0]}
dm = [K.IDX[k].date() for k in pS if pS[k] != pT.get(k)]; print('trade vs sig months differ', len(dm), dm)

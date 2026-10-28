import sys; sys.path.insert(0, "/home/user/-/research/tw_stock_selection/agents/o_rev_event_trigger/replicate_o5")
import json, pickle, math
import rcore as C
import pandas as pd, numpy as np
out, meta = pickle.load(open('runs.pkl', 'rb'))
off = pd.read_parquet('nav_off.parquet')
R = {}
def r3(x): return None if x is None else round(float(x), 3)
def full(name, nav, ew):
    m = meta[name]; sd = m['sd']; am = m['am']; end = m['end']
    o = {k: {kk: (r3(vv) if not isinstance(vv, str) else vv) for kk, vv in C.seg_stats(nav, ew, sd, am, a, b, end).items()} for k, (a, b) in C.SEGS.items()}
    o['cal_t'] = r3(C.tstat(C.cal_x(nav)))
    o['SOP_2024'] = {'CAGR': r3(C.cagr(nav, '2023-12-29', end) * 100), 'E50': r3(C.cagr(C.E50, '2023-12-29', end) * 100)}
    return o
for k in ('Jp1_AB', 'Jp1_N', 'J0_AB', 'J0_N', 'J0_W', 'J0N_AB'):
    R[k] = full(k, out[k]['normal'], out[k]['EW'])
    print(k, 'ALL', R[k]['ALL']['CAGR'], R[k]['ALL']['t'], 'SEL t', R[k]['SEL']['t'], 'cal', R[k]['cal_t'], 'SOP24', R[k]['SOP_2024'])
J = 'Jp1_AB'; nv = out[J]['normal']; ew = out[J]['EW']; m = meta[J]; sd = m['sd']; am = m['am']; end = m['end']
print(json.dumps(R[J], ensure_ascii=False))
st = full(J, out[J]['stress'], ew); print('stress ALL', st['ALL']['CAGR'], st['ALL']['t'], 'HOLD', st['HOLD']['CAGR'], st['HOLD']['E50'])
# slips
be = []
for k in sorted(x for x in out[J] if x.startswith('slip')):
    be.append((float(k[4:]), C.seg_stats(out[J][k], ew, sd, am, '2006-04', '2026-10', end)['t']))
print('slips', [(s, round(t, 3)) for s, t in be])
for i in range(len(be) - 1):
    if be[i][1] >= 2.0 > be[i + 1][1]:
        s0, t0 = be[i]; s1, t1 = be[i + 1]; print('t=2 slip', s0 + (t0 - 2) / (t0 - t1) * (s1 - s0))
# per seed
ts = []
for nvs in m['seeds']:
    x = C.winx(nvs, sd); ts.append(C.tstat(x.values))
ts = np.array(ts); print('seed t', ts.min(), np.median(ts), ts.max(), (ts < 2.3).sum(), (ts < 2.0).sum())
d1 = [C.tstat(np.sort(C.winx(nvs, sd).values)[:-1]) for nvs in m['seeds']]; print('seed drop best1 median', np.median(d1))
# endpoints (window excess)
x = C.winx(nv, sd); amx = pd.PeriodIndex([am[d] for d in x.index], freq='M')
print('mean pp', x.mean() * 100)
print('drop2006', C.tstat(x[amx.year != 2006])); print('from2008', C.tstat(x[amx.year >= 2008])); print('drop last', C.tstat(x.iloc[:-1]))
xs = np.sort(x.values)
for k in (1, 3, 5, 10): print('drop best', k, C.tstat(xs[:-k]))
yr = x.groupby(amx.year).sum(); 
cy = (nv.groupby(nv.index.year).last() / nv.groupby(nv.index.year).last().shift(1) - 1)
by = yr.sort_values(ascending=False).index[:3].tolist(); print('best years by window-sum', by)
print('drop best year', C.tstat(x[amx.year != by[0]]), 'drop best3', C.tstat(x[~amx.year.isin(by)]))
p = 0.5 * math.erfc(R[J]['ALL']['t'] / math.sqrt(2)); print('bonf10', min(1, 10 * p))
# yearly vs 0050
e = C.E50.reindex(nv.index)
yy = pd.DataFrame({'s': nv.groupby(nv.index.year).last(), 'e': e.groupby(e.index.year).last()})
yy = yy / yy.shift(1) - 1; yy.loc[2006] = [nv.loc['2006'].iloc[-1] / nv.iloc[0] - 1, e.loc['2006'].iloc[-1] / e.loc['2006-04-11'] - 1]
print('lose years', [int(y) for y in yy.index if yy.loc[y, 's'] < yy.loc[y, 'e']])
# lookahead + paired (calendar month pp/month)
def cal(n): return C.cal_x(n)
def pair(a, b, a0='2006-05', b0='2026-09'):
    d = (C.cal_x(a, a0=None) if False else None)
    xa = C.cal_x(a, a=a0, b=b0); xb = C.cal_x(b, a=a0, b=b0); dd = (xa - xb).dropna()
    return round(dd.mean() * 100, 4), round(C.tstat(dd), 3), round(2.8 * dd.std(ddof=1) / math.sqrt(len(dd)) * 100, 3)
for nm, a, b in [('look J+1', 'Jp1_N', 'Jp1_AB'), ('look J0', 'J0_N', 'J0_AB')]:
    print(nm, 'ALL', pair(out[a]['normal'], out[b]['normal']), 'SEL', pair(out[a]['normal'], out[b]['normal'], '2006-05', '2018-12'))
AB_V4 = off['AB_V4']
print('J+1 - off', pair(nv, AB_V4)); print('J0 - J+1', pair(out['J0_AB']['normal'], nv)); print('J0N - J+1', pair(out['J0N_AB']['normal'], nv), 'SEL', pair(out['J0N_AB']['normal'], nv, '2006-05', '2018-12'))
print('J0N - J0', pair(out['J0N_AB']['normal'], out['J0_AB']['normal']))
print('J0 worst SEL t', R['J0_W']['SEL']['t'])
print('expected live: 0050+0.5x', R[J]['ALL']['E50'] + 0.5 * (R[J]['ALL']['CAGR'] - R[J]['ALL']['E50']))
json.dump(R, open('stats.json', 'w'), ensure_ascii=False, indent=1)

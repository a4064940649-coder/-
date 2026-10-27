#!/usr/bin/env python3
"""抽樣手算核對主規格事件＋診斷（同池等權開對開年化、事件期間市場狀態、訊號日漲幅分組、離 60 日高距離分組）。"""
import sys; sys.path.insert(0, '/home/user/-/research/tw_stock_selection/agents/w4_obv_leads_new_high')
import run as R, numpy as np, pandas as pd
ew_oo = pd.Series(np.diff(R.LEW), index=R.IDX)
lew = pd.Series(R.LEW[:-1], index=R.IDX); f20 = np.expm1(lew.shift(-20) - lew)
for a, b in [('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')]:
    m = (R.IDX >= a) & (R.IDX <= b); yrs = (R.IDX[m][-1] - R.IDX[m][0]).days / 365.25
    print(a, 'EW open-open CAGR %.4f' % (np.exp(ew_oo[m].sum() / yrs) - 1), 'EW close-close CAGR %.4f' % ((1 + R.ew_cc[m]).prod() ** (1 / yrs) - 1),
          'uncond EW 20d mean %.4f' % f20[m].mean())
df = pd.read_csv(R.L.P.parent / 'w4_obv_MAIN_FIX20_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
sd, dist = [], []
for _, r in df.iterrows():
    j = R.cols.index(r.code); t = R.IDX.get_loc(r.sig)
    sd.append(R.ACn[t, j] / R.ACn[t - 1, j] - 1); dist.append(R.ACn[t, j] / R.HI60.iat[t, j] - 1)
df['sdr'] = sd; df['dist'] = dist
for a, b in [('2005-01-01', '2019-04-30'), ('2019-05-01', '2026-10-31')]:
    s = df[(df.sig >= a) & (df.sig <= b)]
    print(a, 'MAIN_FIX20 n', len(s), 'mean EW over event %.4f' % s.ew.mean(), 'signal-day ret %.4f' % s.sdr.mean(), 'gap %.4f' % s.gap.mean(), 'dist to HI60 %.4f' % s.dist.mean())
    for q, g in s.groupby(pd.qcut(s.sdr, 5), observed=True):
        gm = g.groupby(g.entry.dt.to_period('M'))['ab_gross'].mean(); print('  sdr', q, len(g), 'ab_gross %.3f%%' % (g.ab_gross.mean() * 100), 't', R.tstat(gm))
    for q, g in s.groupby(pd.qcut(s.dist, 4), observed=True):
        gm = g.groupby(g.entry.dt.to_period('M'))['ab_gross'].mean(); print('  dist', q, len(g), 'ab_gross %.3f%%' % (g.ab_gross.mean() * 100), 't', R.tstat(gm))
# 抽樣手算
ev = pd.read_csv(R.L.P.parent / 'w4_obv_MAIN_events.csv.gz', parse_dates=['sig', 'entry'], dtype={'code': str})
row = ev.sample(1, random_state=11).iloc[0]; print(row.to_dict())
c = row.code; j = R.cols.index(c); t = R.IDX.get_loc(row.sig)
a = R.ACf[c]; v = (R.V[c] / R.CR[c])
seg0 = R.IDX < R.GAP_END if row.sig < R.GAP_END else R.IDX >= R.GAP_END
aa = a[seg0]; vv = v[seg0].where(R.ADJ[c][seg0].notna())
obv = (np.sign(aa.diff()).fillna(0) * vv).fillna(0).cumsum()
tt = aa.index.get_loc(row.sig)
print('manual OBV_t', obv.iloc[tt], 'prog', R.OBV[c].iloc[t], '| manual prev59 max', obv.iloc[tt - 59:tt].max(), 'prog', R.OBV_prev59max[c].iloc[t])
print('adj_t', a.iloc[t], 'HI60 manual', aa.iloc[tt - 59:tt + 1].max(), 'ratio', a.iloc[t] / aa.iloc[tt - 59:tt + 1].max(), 'MA60 manual', aa.iloc[tt - 59:tt + 1].mean(), 'prog', R.MA60[c].iloc[t])
print('V60', R.V60[c].iloc[t], 'close_raw', R.CR[c].iloc[t])
e = t + 1; x = e + int(row.hold)
seg = pd.DataFrame({'adj': R.ACn[e:x, j], 'ma20': R.MA20n[e:x, j], 'obv': R.OBVn[e:x, j], 'oma20': R.OMA20n[e:x, j]}, index=R.IDX[e:x]); print(seg)
print('entry', R.AOn[e, j], 'exit px', R.PX[x, j], R.IDX[x], 'raw', R.PX[x, j] / R.AOn[e, j] - 1, 'csv raw', row.raw)

#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀 import ocore，不寫任何研究員檔案；結果只印到 stdout。
A. 台積電相對 0050 的月超額本身是否顯著（t 提升的來源）
B. 把台積電對 0050 的漂移歸零（只留避險效果）後，30% 版的 t
C. 安慰劑：30% 改放其他固定大型股
D. Δt 的 bootstrap 信賴區間
E. 容量主檢定的檢定力（MDE）
F. 驗證期 1～50 組 +4.73pp 的集中度"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import ocore as O
K = O.K
S = O.V2R(1, 300)
pl = S.plan(10); t0, t1, sd = S.bounds(); P = O.periods(sd)
ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)
KEYS = ('SEL', 'VAL', 'HOLD', 'S1', 'S2', 'S3', 'ALL')
def wr(x): return O.window_returns(x, sd)
E50 = wr(K.E50); TS = wr(O.TSMC_IDX)
def seg(s, k):
    a, b = P[k][:2]; return s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].dropna()

print('== A. 台積電−0050 月窗超額 ==')
for k in KEYS:
    x = seg(TS - E50, k); print(k, f'mean {x.mean()*100:.2f}pp t {K.tstat(x):.2f} n {len(x)}')
x = seg(TS - E50, 'ALL'); yr = x.groupby(x.index.year).sum().sort_values(ascending=False)
print('逐年加總前 6 年 pp', (yr.head(6) * 100).round(1).to_dict(), '全部加總', round(yr.sum() * 100, 1))
for drop in (3, 5):
    xx = x[~x.index.year.isin(yr.index[:drop])]; print(f'去掉最好 {drop} 年 t', round(K.tstat(xx), 2), 'mean', round(xx.mean() * 100, 2))

nav0, _ = O.simulate_w(O.tsmc_plan(pl, 0.0), *O.NORMAL, t0, t1)
nav30, _ = O.simulate_w(O.tsmc_plan(pl, 0.3), *O.NORMAL, t0, t1)
def st(nav): return {k: O.segstat(nav, ew, sd, *P[k]) for k in KEYS}
s0, s30 = st(nav0), st(nav30)
print('w0 ', {k: (s0[k]['CAGR'], s0[k]['t_0050'], s0[k]['mean_ex_0050_m']) for k in KEYS})
print('w30', {k: (s30[k]['CAGR'], s30[k]['t_0050'], s30[k]['mean_ex_0050_m']) for k in KEYS})

print('== B. 台積電漂移歸零（日報酬減去全期平均日超額 vs 0050），只留避險 ==')
g = pd.Series(K.GROW[:, O.J2330], index=O.IDX); e = (K.E50.pct_change() + 1).reindex(O.IDX)
m = (O.IDX >= O.IDX[t0]) & (O.IDX <= O.IDX[t1]) & np.isfinite(e.to_numpy()) & np.isfinite(g.to_numpy())
for lab, mask in (('ALL 均值', m), ('只用 SEL 均值', m & (O.IDX < O.SPLIT))):
    drift = float(np.nanmean((g - e).to_numpy()[mask]))
    G2 = K.GROW.copy(); G2[m, O.J2330] = G2[m, O.J2330] - drift
    nv, _ = O.simulate_w(O.tsmc_plan(pl, 0.3), *O.NORMAL, t0, t1, G=G2)
    sb = st(nv)
    print(lab, f'drift {drift*1e4:.2f}bp/日', {k: (sb[k]['CAGR'], sb[k]['t_0050'], sb[k]['maxDD'], sb[k]['TE_ann']) for k in ('SEL', 'VAL', 'HOLD', 'ALL')})

print('== C. 安慰劑：30% 改放其他固定大型股（2006 年就是權值股者）==')
cands = ['2317', '2454', '2412', '1301', '1303', '2002', '2303', '2882', '2308', '3045', '1326', '2357', '2886', '2891', '2382', '2311', '2409', '3008']
res = {}
for c in cands:
    if c not in K.CI: continue
    j = K.CI[c]
    plan = [(p[0], np.concatenate([np.asarray(p[1], int), [j]]), np.concatenate([np.full(len(p[1]), 0.7 / len(p[1])), [0.3]])) for p in pl]
    try:
        nv, _ = O.simulate_w(plan, *O.NORMAL, t0, t1)
        sc = st(nv); res[c] = {k: (sc[k]['CAGR'], sc[k]['t_0050'], sc[k]['maxDD']) for k in ('SEL', 'VAL', 'HOLD', 'ALL')}
        print(c, res[c], flush=True)
    except Exception as ex:
        print(c, 'err', ex)
tall = [v['ALL'][1] for v in res.values()]; tval = [v['VAL'][1] for v in res.values()]
print('安慰劑 ALL t 中位數', np.median(tall), '範圍', min(tall), max(tall), '高於 w0 2.643 的檔數', sum(t > 2.643 for t in tall), '/', len(tall))
print('安慰劑 VAL t 中位數', np.median(tval), '高於 w0 1.344 的檔數', sum(t > 1.344 for t in tval), '/', len(tval))
print('安慰劑 ALL CAGR', sorted([v['ALL'][0] for v in res.values()]))

print('== D. Δt（w30 − w0，對 0050）月份 bootstrap ==')
y0, y30 = wr(nav0) - E50, wr(nav30) - E50
rng = np.random.default_rng(1)
for k in ('ALL', 'VAL', 'HOLD'):
    a, b = seg(y0, k), seg(y30, k); idx = a.index.intersection(b.index); a, b = a[idx].to_numpy(), b[idx].to_numpy()
    T = len(a); d = []
    for _ in range(4000):
        # 區塊 bootstrap，區塊長度 6 個月
        st_ = rng.integers(0, T, int(math.ceil(T / 6))); ii = np.concatenate([np.arange(s, s + 6) % T for s in st_])[:T]
        d.append(K.tstat(b[ii]) - K.tstat(a[ii]))
    d = np.array(d)
    print(k, 'Δt obs', round(K.tstat(b) - K.tstat(a), 2), 'CI95', np.round(np.quantile(d, [0.025, 0.975]), 2),
          'Δmean pp', round((b.mean() - a.mean()) * 100, 3), 'P(t30>2)', round(float(np.mean([K.tstat(b[rng.integers(0, T, T)]) > 2 for _ in range(2000)])), 2))

print('== E. 容量主檢定檢定力 ==')
EV = O.event_table(S, pl)
for lab, lo, hi in (('SEL', '2006-01-01', '2019-04-10'), ('VAL', '2019-04-11', '2026-12-31'), ('ALL', '2006-01-01', '2026-12-31')):
    s = O.mid_minus_large(EV, 'ex_pool', lo, hi); se = s.std(ddof=1) / math.sqrt(len(s))
    print(lab, f'mean {s.mean()*100:.2f} se {se*100:.2f}pp n {len(s)} MDE(80%,雙尾5%) {2.8*se*100:.2f}pp/月')
    per = EV[(EV['d'] >= pd.Timestamp(lo)) & (EV['d'] <= pd.Timestamp(hi))].groupby(['d', EV['bucket'] >= 2]).size().unstack()
    print('   每月檔數中位數 大/中', per[False].median(), per[True].median())

print('== F. 驗證期 1～50 組超額集中度 ==')
v = EV[(EV['d'] >= O.SPLIT) & (EV['bucket'] == 0)]
tot = v['ex_pool'].sum(); bycode = v.groupby('code')['ex_pool'].sum().sort_values(ascending=False)
print('事件數', len(v), '總和 pp', round(tot * 100, 1), '前 5 檔貢獻', (bycode.head(5) * 100).round(1).to_dict(), '占比', round(bycode.head(5).sum() / tot, 2))
byy = v.groupby(v['d'].dt.year)['ex_pool'].sum(); print('逐年', (byy * 100).round(1).to_dict())
mm = v.groupby('d')['ex_pool'].mean()
for drop in (1, 3, 5):
    mm2 = v[~v['code'].isin(bycode.index[:drop])].groupby('d')['ex_pool'].mean()
    print(f'去前 {drop} 檔 月均 {mm2.mean()*100:.2f} t {K.tstat(mm2):.2f}（原 {mm.mean()*100:.2f} t {K.tstat(mm):.2f}）')

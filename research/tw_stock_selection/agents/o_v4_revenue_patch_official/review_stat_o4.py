#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員的檔案，不寫入任何研究員檔案。輸出只印到 stdout。
用法：python3 -I -B review_stat_o4.py [--seeds]"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

NV = pd.read_parquet(HERE / 'navs_v4.parquet')
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
import v4lib as L
K = L.K
A_, B_ = '2006-04-01', '2026-10-12'
plan = pd.read_csv(HERE / 'plan_V4.csv')
sd = pd.DatetimeIndex(pd.to_datetime(plan.signal))
# 訊號窗日期：研究員的 sd 來自 S.bounds()；這裡用 plan 的訊號日加上最後一日
sd = sd.append(pd.DatetimeIndex([NV.index[-1]]))

def xs(col, a=A_, b=B_):
    return L.excess(NV[col], K.E50, sd, a, b)
def t(x): return L.tstat(np.asarray(x))
def p1(tv): return 0.5 * math.erfc(tv / math.sqrt(2))

out = {}
for col in ('V4|normal', 'AB_V4|normal', 'AB_V4_B10|normal', 'AB_V4_BQ|normal', 'V4_SIG|normal', 'AB_V4|stress'):
    x = xs(col)
    out[col] = dict(n=len(x), t=round(t(x), 3), mean_pp=round(x.mean() * 100, 3))
print('[1] 重算 t（與 results.json 比對）', out)
print('    results.json AB t =', RES['RESULTS']['table']['AB_V4|normal']['ALL']['t_0050'])

x = xs('AB_V4|normal')
s = x.sort_values(ascending=False)
print('[2] 右尾集中度（AB_V4）：')
tot = x.sum()
for k in (1, 2, 3, 5, 8, 10, 15, 20):
    print(f'    拿掉最好 {k:2d} 個月 t={t(x.drop(s.index[:k])):.3f}  這 {k} 個月占全期超額總和 {s.iloc[:k].sum() / tot * 100:.1f}%')
kk = next(k for k in range(1, 100) if t(x.drop(s.index[:k])) < 2.0)
print(f'    t 跌破 2.0 需拿掉 {kk} 個月（占 {kk / len(x) * 100:.1f}% 樣本）')
kk3 = next(k for k in range(1, 100) if t(x.drop(s.index[:k])) < 2.3)
print(f'    t 跌破 2.3 需拿掉 {kk3} 個月')
print('    最好 10 個月：', [(str(d.date()), round(v * 100, 1)) for d, v in s.iloc[:10].items()])
# 對稱檢查：同時拿掉最好與最差 k 個月（修剪均值）
for k in (3, 5, 10):
    print(f'    雙側修剪 {k} 個月 t={t(x.drop(s.index[:k]).drop(s.index[-k:])):.3f}')

print('[3] 分年：拿掉單一年份後的 t（AB_V4）')
yrs = sorted(set(x.index.year))
loyo = {y: round(t(x[x.index.year != y]), 3) for y in yrs}
print('   ', loyo)
yr = x.groupby(x.index.year).sum().sort_values(ascending=False)
print('    年度超額合計（pp）前 5：', [(int(y), round(v * 100, 1)) for y, v in yr.iloc[:5].items()])
print('    同時拿掉最好 2 年', list(yr.index[:2]), 't=', round(t(x[~x.index.year.isin(yr.index[:2])]), 3))
print('    同時拿掉 2006、2021', 't=', round(t(x[~x.index.year.isin([2006, 2021])]), 3))

print('[4] 起點／期間敏感度（AB_V4）')
for a in ('2007-01-01', '2008-01-01', '2009-01-01', '2010-01-01', '2014-01-01', '2019-01-01'):
    xx = x[x.index >= pd.Timestamp(a)]
    print(f'    起點 {a}: n={len(xx)} t={t(xx):.3f} 月均 {xx.mean() * 100:.2f}pp')
for a, b in (('2006-04-01', '2013-12-31'), ('2014-01-01', '2026-10-12'), ('2006-04-01', '2018-12-31'), ('2019-01-01', '2026-10-12')):
    xx = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
    print(f'    {a}～{b}: n={len(xx)} t={t(xx):.3f}')
print('    最後一個窗：', str(x.index[-1].date()), round(x.iloc[-1] * 100, 2), 'pp；拿掉 2026 年 t=', round(t(x[x.index.year != 2026]), 3))

print('[5] 多重檢定（AB t=2.512）：')
tab = RES['RESULTS']['table']['AB_V4|normal']['ALL']['t_0050']
for m in (1, 10, 15, 20, 30, 50, 100):
    print(f'    m={m:3d} Bonferroni p={min(1, m * p1(tab)):.3f}；需要 t={-np.sqrt(2) * __import__("scipy.special", fromlist=["erfcinv"]).erfcinv(2 * 0.05 / m) * -1:.2f}')
ag = sorted(p.name for p in (HERE.parent).iterdir() if p.is_dir())
print('    agents/ 資料夾數：', len(ag), '；名稱含 main／o_ 者：', sum(1 for a in ag if a.startswith('main') or a.startswith('o_')))

print('[6] 萬海（2021-06-11）成交日口徑效應：')
for col in ('V4|normal', 'V4_SIG|normal', 'AB_V4|normal'):
    xx = xs(col); print('   ', col, '2021-06 窗超額 pp =', round(xx.loc[pd.Timestamp('2021-06-11')] * 100, 2))
xsig = xs('V4_SIG|normal'); ss = xsig.sort_values(ascending=False)
print('    V4_SIG 最好月', str(ss.index[0].date()), round(ss.iloc[0] * 100, 1), '；V4_SIG t=', round(t(xsig), 3), '拿掉最好 1 個月', round(t(xsig.drop(ss.index[:1])), 3))

print('[7] 最大回撤：')
for col in ('V4|normal', 'AB_V4|normal'):
    nv = NV[col][NV.index >= pd.Timestamp(A_)]; dd = (nv / nv.cummax() - 1).min() * 100
    print('   ', col, round(dd, 2))

print('[8] 不利組合（研究員已揭露數字的組合，非新跑）：')
dp = json.loads((HERE / 'diag_post.json').read_text(encoding='utf-8'))
print('    嚴格 PIT（AB_V4_RDEL）t=', round(dp['AB_V4_RDEL_normal']['ALL']['t_0050'], 3), '拿掉最好 1 月', round(dp['AB_V4_RDEL_drop_best1_t'], 3))

if '--seeds' in sys.argv:
    print('[9] AB 逐 seed t 分佈（V4 規格、R_FIX、成交日口徑；與 run_v4 的 ab() 相同但不平均）')
    XR = L.setup_px(); YFX, Y5F = L.load_rfix(); DMT = L.disp_mask_fn('trade')
    YRD = L.load_rdel(); Y5o = L.y5_orig()
    for lab, YL, Y5 in (('V4', YFX, Y5F), ('V4_RDEL(嚴格PIT)', YRD, Y5o)):
        ts, tsd, cg = [], [], []
        rets = []
        for seed in range(40):
            S = L.V4(YL, Y5, DMT, unavail=L.V.make_unavail('mc', seed))
            ta, tb, sdd = S.bounds(); pl = S.plan(10, 'official')
            nav, _ = L.sim_plan(pl, ta, tb, 'normal', XR)
            rets.append(nav.pct_change())
            xx = L.excess(nav, K.E50, sdd, A_, B_); ss_ = xx.sort_values(ascending=False)
            ts.append(t(xx)); tsd.append(t(xx.drop(ss_.index[:1])))
            cg.append(L.seg(nav, nav, sdd, A_, B_)['CAGR'])
        ts, tsd = np.array(ts), np.array(tsd)
        mn = (1 + pd.concat(rets, axis=1).mean(axis=1).fillna(0)).cumprod()
        print(f'    {lab}: 平均淨值 t={t(L.excess(mn, K.E50, sdd, A_, B_)):.3f}；逐 seed t 中位數 {np.median(ts):.3f} 範圍 {ts.min():.3f}～{ts.max():.3f}；'
              f't<2.30 的 seed {int((ts < 2.3).sum())}/40；拿掉最好 1 月後 <2.30 的 seed {int((tsd < 2.3).sum())}/40；CAGR 範圍 {min(cg):.2f}～{max(cg):.2f}')

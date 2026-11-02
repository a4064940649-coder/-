#!/usr/bin/env python3
"""反方審查（成本與可成交，o8）：只讀研究員的 data/alpha_all_v1.parquet、results.json、forward/firstseen.csv，不改任何檔案。
E1 各關注組（A、main 腿）周轉、成本拖累（一般／壓力）與 α 水準（前期／後期／VAL／HOLD）；低組在壓力成本下是否還有 α
E2 j0 vs j≥6 在壓力成本下的差（「照官方日換股」提醒的成本後幅度）
E3 R2 by-j：零衝擊 extra15 的配對差中位數（核對 FINDINGS 的 +0.011pp）；表內 R2/B10 中位數年化 vs 逐 j 差中位數的不一致
E4 R2 前期配對差（所有格）符號與 t
E5 forward/firstseen.csv 的 ann_date 覆蓋（前瞻追蹤是否依賴每日實跑）
"""
import json
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
A = pd.read_parquet(HERE / 'data' / 'alpha_all_v1.parquet')
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
P = lambda s: pd.Period(s, 'M')
PER = {'PRE': (P('2006-04'), P('2018-12')), 'POST': (P('2019-05'), P('2026-10')), 'VAL': (P('2019-05'), P('2023-12')), 'HOLD': (P('2024-01'), P('2026-10'))}
A['mp'] = pd.PeriodIndex(A.m, freq='M')

def t(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else np.nan

print('E1 A/main 各組：每月平均（先對 j 平均再對月平均）')
X = A[(A.scheme == 'A') & (A.leg == 'main')]
for pn, (a, b) in PER.items():
    s = X[(X.mp >= a) & (X.mp <= b)]
    for g in (0, 1, 2):
        q = s[s.g == g].groupby('m')[['alpha_gross', 'alpha', 'alpha_stress', 'newf', 'soldf', 'Ng', 'n']].mean()
        print(f' {pn} g{g}: gross {q.alpha_gross.mean()*100:6.3f}pp (t {t(q.alpha_gross):5.2f})  normal {q.alpha.mean()*100:6.3f} (t {t(q.alpha):5.2f})  '
              f'stress {q.alpha_stress.mean()*100:6.3f} (t {t(q.alpha_stress):5.2f})  newf {q.newf.mean():.2f} soldf {q.soldf.mean():.2f}  '
              f'drag_n {(q.alpha_gross-q.alpha).mean()*100:.3f} drag_s {(q.alpha_gross-q.alpha_stress).mean()*100:.3f}  Ng {q.Ng.mean():.0f} n {q.n.mean():.1f}')

print('\nE2 低組／高組：j0 − mean(j6..12) 的月差（一般與壓力成本）')
for pn in ('PRE', 'POST', 'VAL', 'HOLD'):
    a, b = PER[pn]; s = X[(X.mp >= a) & (X.mp <= b)]
    for g in (0, 2):
        for col in ('alpha', 'alpha_stress'):
            p = s[s.g == g].pivot_table(index='m', columns='j', values=col)
            d = (p[0] - p[[j for j in range(6, 13) if j in p]].mean(axis=1)).dropna()
            print(f' {pn} g{g} {col:12s}: {d.mean()*100:6.3f}pp  t {t(d):5.2f}  n {len(d)}')

print('\nE3／E4 R2')
C = R['RESULTS_R2']['cells']
for ck in ('zero|normal', 'zero|extra15', 'zero|stress', 'K5T|normal|100萬|FIX', 'K5T|stress|100萬|FIX', 'K5T|normal|30萬|FIX', 'K5T|normal|1000萬|FIX'):
    bj = C[ck]['by_j']
    js = sorted(bj, key=int)
    dm = [bj[j]['diff_mean_pp'] for j in js]; cd = [bj[j]['CAGR_diff'] for j in js]
    pre = [bj[j]['diff_pre_mean_pp'] for j in js]; pret = [bj[j]['diff_pre_t'] for j in js]
    post = [bj[j]['diff_post_mean_pp'] for j in js]; postt = [bj[j]['diff_post_t'] for j in js]
    r2 = np.median([bj[j]['CAGR_R2'] for j in js]); b10 = np.median([bj[j]['CAGR_B10'] for j in js])
    print(f' {ck}: med CAGR R2 {r2:.2f} B10 {b10:.2f} (差 {r2-b10:+.2f}) vs med CAGR_diff {np.median(cd):+.2f}; n_j CAGR_diff<0 {sum(x<0 for x in cd)}/12; '
          f'med diff_mean_pp {np.median(dm):+.3f}; pre med {np.median(pre):+.3f} (t med {np.median(pret):+.2f}, n_j pre<0 {sum(x<0 for x in pre)}/12); '
          f'post med {np.median(post):+.3f} (t med {np.median(postt):+.2f})')

print('\nE5 firstseen')
fs = pd.read_csv(HERE / 'forward' / 'firstseen.csv', dtype=str)
fs['has_ann'] = fs.ann_date.notna() & (fs.ann_date.fillna('') != '')
print(fs.groupby('ym').agg(n=('code', 'size'), ann=('has_ann', 'mean'), fs_min=('first_seen', 'min'), fs_max=('first_seen', 'max')))

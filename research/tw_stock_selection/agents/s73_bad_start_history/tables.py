#!/usr/bin/env python3
"""S73 呈現用：只讀 dense_b.csv.gz／starts_a.csv.gz 列清單（不產生新統計口徑）。python3 -I -B tables.py"""
import pandas as pd, numpy as np
from pathlib import Path
H = Path(__file__).resolve().parent
B = pd.read_csv(H / 'dense_b.csv.gz'); A = pd.read_csv(H / 'starts_a.csv.gz')
f = lambda x: 'NA' if pd.isna(x) else f'{x*100:.2f}'
print('樣本 B 全期 r6 ≤ −10% 逐日')
for _, r in B[B.r6 <= -0.10].iterrows():
    print(r['date'], r['day6_date'], f(r.r6), f(r.r6_b), '| 3m', f(r.a6_s3), '6m', f(r.a6_s6), '12m', f(r.a6_s12), 'x12', f(r.a6_x12), 'fall', f(r.a6_fall), 'rec_mo', None if pd.isna(r.a6_rec_mo) else round(r.a6_rec_mo, 2))
print('策略 ≤ −13.27% 天數', (B.r6 <= -0.1327).sum(), sorted(B[B.r6 <= -0.1327].date), '0050 ≤ −12.85%', sorted(B[B.r6_b <= -0.1285].date))
BM = B[(B.date >= '2016-02-01') & (B.date <= '2025-10-31')]
for nm, d in (('B主', BM), ('B全', B)):
    print(nm, '百分位排名：策略 −13.27%', f((d.r6 <= -0.1327).mean()), '0050 −12.85%', f((d.r6_b <= -0.1285).mean()))
A0 = A[A['shift'] == 0]
print('樣本 A 位移 0：r6 最差 5', [(r.exec_date, f(r.r6), f(r.r6_b)) for _, r in A0.nsmallest(5, 'r6').iterrows()])
print('樣本 A 合併：r6 最差 8', [(int(r['shift']), r.exec_date, f(r.r6), f(r.r6_b)) for _, r in A.nsmallest(8, 'r6').iterrows()])
print('樣本 A 合併：首月 ≤ −10%', [(int(r['shift']), r.exec_date, r.month_end, f(r.rM), f(r.rM_b), f(r.aM_s12), f(r.aM_x12)) for _, r in A[A.rM <= -0.10].sort_values('exec_date').iterrows()])
w = A0[A0.r6 <= np.percentile(A0.r6, 10)]
print('A0 最差 10%：12 個月繼續−停手 < 0 的起點', [(r.exec_date, f(r.a6_x12)) for _, r in w[w.a6_x12 < 0].iterrows()], '；回本最長', w.loc[w.a6_rec_mo.idxmax(), ['exec_date', 'a6_rec_mo', 'a6_rec_date']].tolist())
wp = A[A.r6 <= np.percentile(A.r6, 10)]
print('A 合併最差 10%：回本最長', wp.loc[wp.a6_rec_mo.idxmax(), ['shift', 'exec_date', 'a6_rec_mo', 'a6_rec_date']].tolist(), '再跌最深', wp.loc[wp.a6_fall.idxmin(), ['shift', 'exec_date', 'a6_fall']].tolist())
print('A0 最差10% 首 6 日：策略−0050 中位數', f((w.r6 - w.r6_b).median()), '；0050 也跌的比例', f((w.r6_b < 0).mean()))

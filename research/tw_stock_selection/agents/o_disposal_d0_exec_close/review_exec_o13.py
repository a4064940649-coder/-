#!/usr/bin/env python3
"""反方審查（成本與可成交角度）o13：只讀研究員產出，不改任何研究員檔案。
python3 -I -B review_exec_o13.py"""
import sys, json, math
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import dlib as D

OUT = {}
tr = pd.read_csv(HERE / 'trades_all.csv.gz', parse_dates=['date', 'exit_date', 'announce'])
print('trades by timing/cost', tr.groupby(['timing', 'cost']).size().to_dict())


def tt(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 else None


def desc(df, col='net_x'):
    x = df[col].dropna()
    if not len(x): return {}
    m = df.assign(ym=df['date'].dt.to_period('M')).groupby('ym')[col].mean()
    srt = np.sort(x.values)[::-1]
    k = max(1, int(round(0.05 * len(srt))))
    return dict(n=int(len(x)), mean=round(x.mean() * 100, 2), median=round(x.median() * 100, 2), hit=round((x > 0).mean(), 3),
                t_month=tt(m), mean_ex_top5pct=round(srt[k:].mean() * 100, 2), mean_ex_top10=round(srt[10:].mean() * 100, 2) if len(srt) > 20 else None)


PER = {'VAL': ('2024-01-02', '2026-10-22'), 'FULL': ('2006-04-01', '2026-10-22'), 'SEL': ('2021-10-01', '2023-12-31'),
       '2019+': ('2019-01-01', '2026-10-22'), 'NEWREG': ('2026-08-10', '2026-10-22')}
# 1) 逐筆分布：均值 vs 中位數、去頭部
R1 = {}
for tim in sorted(tr['timing'].unique()):
    for cost in ('normal', 'stress'):
        d = tr[(tr.timing == tim) & (tr.cost == cost)]
        for pn, (a, b) in PER.items():
            s = d[(d.date >= a) & (d.date <= b)]
            R1[f'{tim}|{cost}|{pn}|net_x'] = desc(s, 'net_x')
            R1[f'{tim}|{cost}|{pn}|net50_x'] = desc(s, 'net50_x')
OUT['per_trade'] = R1
for pn in PER:
    for col in ('net_x', 'net50_x'):
        for cost in ('normal', 'stress'):
            ks = [k for k in R1 if k.endswith(f'|{cost}|{pn}|{col}')]
            med_mean = np.median([R1[k]['mean'] for k in ks if R1[k]])
            med_median = np.median([R1[k]['median'] for k in ks if R1[k]])
            med_ex = np.median([R1[k]['mean_ex_top5pct'] for k in ks if R1[k]])
            med_hit = np.median([R1[k]['hit'] for k in ks if R1[k]])
            print(f'{pn:6s} {cost:6s} {col:8s} 9時點中位: mean {med_mean:+.2f}% median-trade {med_median:+.2f}% ex-top5% {med_ex:+.2f}% hit {med_hit:.3f}')

# 2) 成本拆解：每筆 gross（理想）- 可執行 net
d = tr[tr.cost == 'normal']
d = d.assign(exec_drag=(d['gross'] - d['net_x']))
print('執行拖累（理想毛額−可執行淨，對EW）9時點中位 FULL:', round(d.groupby('timing')['exec_drag'].mean().median() * 100, 2), '%')
# 零股限定 vs 有整張
d2 = d[d.timing == 'C1xCL']
for pn, (a, b) in PER.items():
    s = d2[(d2.date >= a) & (d2.date <= b)]
    oo = s[s.nl == 0]; ll = s[s.nl > 0]
    print(pn, 'odd-only share', round(len(oo) / max(len(s), 1), 3), 'odd-only net_x', desc(oo).get('mean'), 'lot net_x', desc(ll).get('mean'))

# 3) 零股價差敏感度（以 10 萬槽 C1xCL 逐筆近似重算：零股部分每邊再加 Δ）
for extra in (0.004, 0.008, 0.012):
    # 零股占比：no*px / (nl*1000*px + no*px) ≈ no/(nl*1000+no)
    w = d2['no'] / (d2['nl'] * 1000 + d2['no'])
    adj = d2['net_x'] - 2 * extra * w
    for pn in ('VAL', 'FULL'):
        a, b = PER[pn]; m = (d2.date >= a) & (d2.date <= b)
        print(f'零股每邊再加 {extra:.3f}（主欄0.4%→{0.004+extra:.3f}） {pn} C1xCL net_x mean {adj[m].mean()*100:+.2f}%  vs0050 {(d2["net50_x"]-2*extra*w)[m].mean()*100:+.2f}%')

# 4) 年度分布（C1xCL normal）
yr = d2.assign(y=d2.date.dt.year).groupby('y').agg(n=('net_x', 'size'), net_x=('net_x', 'mean'), net50=('net50_x', 'mean'))
print(yr.round(4).tail(10))

# 5) 衛星層：自行重算 9 時點驗證期月 t（對 0050）與現金拖累的量級
nav = pd.read_parquet(HERE / 'navs.parquet')
e50 = pd.Series(D.E50, index=D.IDX)
for grp in ('MAIN', 'STRESS', 'GROW100', 'GROW1000'):
    ts = []; mm = []
    for c in [c for c in nav.columns if c.startswith(grp + '|')]:
        n = nav[c].dropna()
        x = (D.monthly(n) - D.monthly(e50.reindex(n.index))).dropna()
        v = x[(x.index >= '2024-01-01') & (x.index <= '2026-10-31')]
        ts.append(tt(v)); mm.append(v.mean() * 100)
    print(grp, 'VAL t 9時點', ts, 'median', np.median(ts), 'mean pp median', round(float(np.median(mm)), 3))
    OUT[f'sat_{grp}_VAL_t'] = ts

# 6) 需要多少逐筆超額才能讓 VAL t 到 1.0（粗估）：衛星月超額 sd 與現有平均
c = 'MAIN|C1xCL'; n = nav[c].dropna()
x = (D.monthly(n) - D.monthly(e50.reindex(n.index))).dropna(); v = x[x.index >= '2024-01-01']
need = 1.0 * v.std(ddof=1) / math.sqrt(len(v)) - v.mean()
print('C1xCL VAL 月超額 mean %.3fpp sd %.3fpp n %d；要到 t=1 每月需再多 %.3fpp' % (v.mean() * 100, v.std(ddof=1) * 100, len(v), need * 100))
# 現金池若有 1% 年息
print('現金池 26pct x 年息1pct ≈ 每月 %.3fpp' % (0.26 * 0.01 / 12 * 100))

# 7) 交易頻率與年周轉（每年筆數 × 每筆來回成本）
nper = d2[(d2.date >= '2024-01-01')].shape[0] / (2 + 10 / 12)
print('VAL 每年筆數 C1xCL ≈', round(nper, 1))
v26=d2[d2.date>='2026-01-01']; print('2026 C1xCL 逐筆', desc(v26), 'vs0050', desc(v26,'net50_x'))

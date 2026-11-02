#!/usr/bin/env python3
"""反方審查（成本與可成交）：只讀研究員檔案，不改寫任何既有檔案。python3 -I -B review_exec_o10.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

DATA = HERE / 'data'


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


z = np.load(DATA / 'panel.npz', allow_pickle=False)
IDX = pd.DatetimeIndex(z['IDX']); O, C, LIM, V, V60, SIG = z['O'], z['C'], z['LIMUP'], z['V'], z['V60'], z['SIG']
T = len(IDX)
E = pd.read_parquet(DATA / 'events.parquet')
E = E[(E['k'] >= 2019 * 4 + 1) & E['SUEP'].notna()].copy()
E['grp'] = np.where(E['in0050'], '0050', np.where(E['att'] == 1, 'low', np.where(E['att'] == 2, 'mid', 'high')))
nf = ~E['in0050']; top = E['top_SUEP']
for j in range(10):
    E[f'ab{j}'] = E[f'r60_j{j}'] - E.groupby(['k', 'grp'])[f'r60_j{j}'].transform('mean')


def rep(name, d, col):
    s = d.groupby('k')[col].mean(); dd = d.groupby('A')[col].mean()
    print(f'{name:58s} mean {d[col].mean():+.4f} t_season {tstat(s):5.2f} t_date {tstat(dd):5.2f} n {d[col].notna().sum()}')


T0 = E[nf & top].copy()
print('=== 1. 可成交性：j0 漲停／處置／放棄 ===')
a1 = T0['ai'].to_numpy() + 1; jj = T0['j'].to_numpy()
ok = a1 < T
lu = np.zeros(len(T0), bool); lu[ok] = O[a1[ok], jj[ok]] >= LIM[a1[ok], jj[ok]] - 1e-9
T0['lu_j0'] = lu
print('非0050 top 事件', len(T0), ' A+1 開盤漲停比例', round(lu.mean(), 4), ' j0 放棄(e=-1)比例', round((T0['e_j0'] < 0).mean(), 4),
      ' j0 順延(e>A+1)比例', round(((T0['e_j0'] > a1) & (T0['e_j0'] >= 0)).mean(), 4))
post = T0['A'] >= '2021-09-01'
print('  2021-09 後 放棄比例', round((T0.loc[post, 'e_j0'] < 0).mean(), 4), ' 之前', round((T0.loc[~post, 'e_j0'] < 0).mean(), 4))
# 放棄者若能以 A+1 收盤買（事後），報酬？（看「買不到」是否剛好丟掉好事件）
rep('  非0050 top ab60 j0（可成交者）', T0, 'ab0')

print('=== 2. 開盤進場的依賴：j0 開盤→收盤、以 j0 收盤進場 ===')
e0 = T0['e_j0'].to_numpy(); okk = e0 >= 0
oc = np.full(len(T0), np.nan); oc[okk] = C[e0[okk], jj[okk]] / O[e0[okk], jj[okk]] - 1
T0['oc_j0'] = oc
print('  j0 當日 開→收 平均', np.nanmean(oc).round(5), ' 中位數', np.nanmedian(oc).round(5))
# 用 j0 收盤進場的 60 日報酬近似：(1+r60_j0)/(1+oc)-1
T0['r60_close'] = (1 + T0['r60_j0']) / (1 + T0['oc_j0']) - 1
Eall = E.copy(); ea = Eall['e_j0'].to_numpy(); ja = Eall['j'].to_numpy(); oa = ea >= 0
x = np.full(len(Eall), np.nan); x[oa] = C[ea[oa], ja[oa]] / O[ea[oa], ja[oa]] - 1
Eall['r60_close'] = (1 + Eall['r60_j0']) / (1 + x) - 1
Eall['abc'] = Eall['r60_close'] - Eall.groupby(['k', 'grp'])['r60_close'].transform('mean')
rep('  非0050 top ab60 以 j0 收盤進場', Eall[nf & top], 'abc')

print('=== 3. 扣成本後的事件層級（相對同組平均；這仍不是可交易報酬，因為無法放空同組）===')
for c, nm in [(0.006, '一般 0.6%'), (0.014, '壓力 1.4%')]:
    T0[f'net{nm}'] = T0['ab0'] - c
    rep(f'  非0050 top ab60 − {nm}', T0, f'net{nm}')
T0['vs50'] = T0['r60_j0'] - T0['b50'] if 'b50' in T0 else np.nan

print('=== 4. 公告日判定法分組（非 fsq 法需後續 2 個觀測確認）===')
for m_, g in T0.groupby('A_method'):
    rep(f'  method={m_}', g, 'ab0')
nfsq = T0[T0['A_method'] != 'fsq']
for j in [0, 1, 2, 3]:
    rep(f'  非fsq 法 j{j}', nfsq, f'ab{j}')
for j in [0, 2]:
    rep(f'  fsq 法 j{j}', T0[T0['A_method'] == 'fsq'], f'ab{j}')
rep('  上櫃 j2（實盤若需等確認）', T0[T0['mkt'] == 'tpex'], 'ab2')

print('=== 5. 容量：100 萬每槽 5 萬 → 整股可否買 ===')
px = T0['close'].to_numpy()
print('  A 日收盤 >50 元比例（5 萬買不到 1 張）', round((px > 50).mean(), 3), ' >100 元', round((px > 100).mean(), 3))
print('  V60 分位（萬）p10/p50/p90', np.round(np.nanquantile(T0['V60'] / 1e4, [0.1, 0.5, 0.9]), 0))
zc = 0.0001928525621986513
print('  盤中零股每日上限 zcap×V60 分位（元）', np.round(np.nanquantile(zc * T0['V60'], [0.1, 0.5, 0.9]), 0))

print('=== 6. 引擎：實際跑 MAIN j0 100 萬 GROW 一般，看零股占比與成交比例 ===')
import pengine
from s3_iterate import VARIANTS, cand, M, E50
from pengine import run, perf
E1 = pd.read_parquet(DATA / 'events.parquet'); E1 = E1[(E1['k'] >= 2019 * 4 + 1) & E1['SUEP'].notna()]
Cc = cand(E1, VARIANTS['v3'])
ev = pd.DataFrame({'col': Cc['j'], 'e': Cc['e_j0'], 'score': Cc['SUEP']}); ev = ev[ev['e'] >= 0]
for K in [1e6, 1e7]:
    nav, st = run(M, ev, S=20, H=60, K=K, mode='GROW', cost='normal', start='2019-05-01', end='2026-10-30')
    p = perf(nav, E50.loc[nav.index])
    print(f'  K={K:.0e} j0 ex_ann {p["ex_ann"]:+.4f} t {p["t_m"]:.2f}; buys {st["buys"]} skipped_full {st["skipped_full"]}',
          'odd_share med', round(float(np.median(st['odd_share'])), 3), 'fill med', round(float(np.median(st['filled_ratio'])), 3),
          'fill p10', round(float(np.quantile(st['filled_ratio'], 0.1)), 3))
    # 平均曝險：持股市值/淨值 —— 用 0050 槽占比反推不易，改以 fill×槽滿率近似
# 7. 零股入場日開→收漏算（2020-10-26 後 val=odd_amt）：以 oc_j0 量測偏誤方向
post20 = T0['A'] >= '2020-10-26'
print('=== 7. 引擎零股入場日漏算 開→收（2020-10-26 後）: 事件 oc_j0 平均', round(float(T0.loc[post20, 'oc_j0'].mean()), 5))

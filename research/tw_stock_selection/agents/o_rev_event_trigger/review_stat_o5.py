#!/usr/bin/env python3
"""反方審查（資料與統計）— o_rev_event_trigger。唯讀：只讀研究員的 navs_oos.parquet、results.json、explore.json、iterations.json，
輸出只到 stdout。執行：python3 -I -B review_stat_o5.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import fx_core as F

NV = pd.read_parquet(HERE / 'navs_oos.parquet')
RES = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
E50 = F.E50

def days_of(name):
    if name == 'OFF':
        st = F.setup(); return [d for d in st['off_days']], st['off_end']
    j = {'J+1': 1, 'J0': 0, 'J0N': 0}[name]
    dd, e = F.version_days(j); return [d for _, d in dd], e

def p1(t): return 0.5 * math.erfc(t / math.sqrt(2))

out = {}
W = {}
for nm in ('J+1', 'J0', 'J0N', 'OFF'):
    sd, e = days_of(nm); W[nm] = F.windows(sd, e)
    nav = NV[f'{nm}|AB'].dropna()
    b = F.block(nav, W[nm], ('ALL', 'SEL', 'VAL', 'HOLD', 'POST'))
    out[nm] = {k: (round(v['CAGR'], 2), round(v['t_0050'], 3), round(v['t_0050_cal'], 3), v['n']) for k, v in b.items()}
print('== 重現（CAGR, t, t_cal, n）'); print(json.dumps(out, ensure_ascii=False, indent=0))

# 1) 日曆：J+1 與官方日相同的月數
sdJ, _ = days_of('J+1'); sdO, _ = days_of('OFF')
mJ = {pd.Period(d, 'M'): d for d in sdJ}; mO = {pd.Period(d, 'M'): d for d in sdO}
same = sum(mJ.get(m) == d for m, d in mO.items()); print('\n== 官方日＝J+1 的月數', same, '/', len(mO))

# 2) J+1 超額的年份集中度
nav = NV['J+1|AB'].dropna()
ex = F.wret(nav, W['J+1']) - F.wret(E50.reindex(nav.index), W['J+1'])
ex = ex.dropna()
yr = ex.groupby(ex.index.year).sum().sort_values(ascending=False)
tot = ex.sum()
print('\n== 逐年超額（窗超額加總 pp）'); print((yr * 100).round(1).to_dict())
print('總超額 pp', round(tot * 100, 1), '前 3 年占比', round(yr.iloc[:3].sum() / tot, 3), '前 5 年占比', round(yr.iloc[:5].sum() / tot, 3))
print('超額 > 0 的年數', int((yr > 0).sum()), '/', len(yr))
for drop in ([2021], [2006], [2006, 2021], [2006, 2007, 2021]):
    x = ex[~ex.index.year.isin(drop)]; print('拿掉', drop, 't', round(F.tstat(x), 3))
# 只有 2009 年以後（金融海嘯後）
for st_ in ('2009-01', '2010-01', '2011-01'):
    print('起點', st_, 't', round(F.tstat(ex[ex.index >= pd.Period(st_, 'M')]), 3))
# 滾動 5 年 t
r5 = [(str(ex.index[i]), round(F.tstat(ex.iloc[i - 59:i + 1]), 2)) for i in range(59, len(ex), 12)]
print('滾動 60 月 t（每 12 月取樣）', r5)

# 3) J+1 與官方的差異集中在哪些月（J0 月）
navO = NV['OFF|AB'].dropna()
cJ = F.cal_ret(nav); cO = F.cal_ret(navO.reindex(nav.index.intersection(navO.index)))
d = (cJ - cO).dropna()
print('\n== J+1 − 官方 日曆月差: mean pp', round(d.mean() * 100, 3), 't', round(F.tstat(d), 2), 'n', len(d))

# 4) 多重檢定：以不同試驗數 m 做 Bonferroni（單尾）與 Sidak
tA = RES['RESULTS']['J+1']['AB']['ALL']['t_0050']
for m in (1, 6, 10, 23, 29, 40):
    print(f'm={m:3d} Bonferroni p={min(1, m * p1(tA)):.3f}  Sidak p={1 - (1 - p1(tA)) ** m:.3f}')
# j-scan 的分布：J+1 在 J0~J12 中的排名
js = RES['RESULTS']['jscan']['by_j']
tv = sorted([(int(k), v['t_ALL']) for k, v in js.items()], key=lambda x: -x[1])
print('\n== j-scan 排名', [(k, round(t, 2)) for k, t in tv])
t04 = [js[str(j)]['t_ALL'] for j in range(5)]; print('J0~J4 平均 t', round(np.mean(t04), 3), 'sd', round(np.std(t04, ddof=1), 3))
sel = [js[str(j)]['t_SEL'] for j in range(13)]; post = [js[str(j)]['t_POST'] for j in range(13)]
print('SEL t vs POST t 相關', round(np.corrcoef(sel, post)[0, 1], 3))

# 5) 驗證期與保留期：J+1 對 0050 的有效樣本
for k in ('VAL', 'HOLD', 'POST'):
    x = F.selp(ex, k); print(k, 'mean pp', round(x.mean() * 100, 3), 't', round(F.tstat(x), 2), 'n', len(x), '勝率', round((x > 0).mean(), 3))
# 樣本外均值對挑選期均值的縮水
s_ = F.selp(ex, 'SEL'); p_ = F.selp(ex, 'POST')
print('SEL mean pp', round(s_.mean() * 100, 3), 'POST mean pp', round(p_.mean() * 100, 3), 'POST−SEL 差 t（Welch）',
      round((p_.mean() - s_.mean()) / math.sqrt(p_.var(ddof=1) / len(p_) + s_.var(ddof=1) / len(s_)), 2))

# 6) 「實盤預期」公式的不確定性：超額年化的 95% CI（以月超額均值 ±1.96 SE 推）
mu, se = ex.mean(), ex.std(ddof=1) / math.sqrt(len(ex))
print('\n== 月超額均值 pp', round(mu * 100, 3), '95% CI', round((mu - 1.96 * se) * 100, 3), round((mu + 1.96 * se) * 100, 3))
print('對應 0.5×超額年化區間（粗估，pp/年）', round(0.5 * 12 * (mu - 1.96 * se) * 100, 2), round(0.5 * 12 * (mu + 1.96 * se) * 100, 2))

# 7) 前視模型依賴：iterations 中 it6 worst vs it2 random
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
g = {o['name']: o for o in IT['iterations']}
print('\n== it2 J0 random-p AB t', round(g['it2_J0']['AB']['normal']['t_0050'], 3), '| it6 worst AB t', round(g['it6_J0_ABworst']['AB']['normal']['t_0050'], 3),
      '| it2 nonAB t', round(g['it2_J0']['nonAB']['normal']['t_0050'], 3))
print('J+1 lookahead by period', {k: (round(v['mean_pp'], 4), v['n_nonzero']) for k, v in RES['RESULTS']['J+1']['lookahead_nonAB_minus_AB'].items()})

#!/usr/bin/env python3
"""迭代建模（只用挑選期：訊號日 2021-10-12～2023-12-11；評價到 2024-02-15 訊號日收盤為止）。

H3b 版本（事先列好，全部記錄）：
  B1 第二條腿：市值三組內外資持股比率最高各 10 檔（共 30），月換股、等權
  B2 第二條腿：市值三組內外資持股 6 個月增量最高各 10 檔
  B3 第二條腿：池內外資持股比率最高 30 檔（不控市值，對照）
  C1 主策略 n10 80% ＋ 選定第二條腿 20%（每個成交日重設權重）
  C2 主策略第三分數：0.4×52 週＋0.4×營收＋0.2×市值組內外資持股百分位
  C3 同 C2，第三分數改用 6 個月增量
選版規則（寫在跑之前）：第二條腿取「扣一般成本後對同池等權 t」最高者；組合版取「對主策略配對差 t」最高者；
都要報告鄰近版本（每組 5／15 檔）。H3a 在挑選期沒有資料，不迭代。
輸出 iterations.json。
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sbl_lib as L

K, V, IDX = L.K, L.V, L.IDX
S = L.strategy(True)
a, b = L.SEL
nxt = [d for d in S.sig_days if d > b][0]           # 2024-02-15：挑選期最後一個持有期的終點
t1 = IDX.get_loc(nxt)
assert nxt < pd.Timestamp('2024-03-01')
E50 = K.E50

base = L.main_plan(S, a, b)
t0 = base[0][0]
nav_main = {c: L.run_plan(base, c, t1)[0] for c in L.COST}
ew = L.ew_nav(base, t1)
SDw = [p[3] for p in base] + [nxt]

def stats(nav, sd=SDw, lo=a, hi=b, ref=None):
    o = {}
    for nm, bench in [('vs_EW', ew), ('vs_0050', E50), ('vs_main', ref if ref is not None else nav_main['normal'])]:
        x = L.monthly_t(nav, bench, sd, lo, hi)
        o[nm] = L.summ(x)
    o.update(L.cagr_dd(nav, IDX[t0], IDX[t1]))
    return o

def excess_series(nav, bench, sd=SDw, lo=a, hi=b):
    return L.monthly_t(nav, bench, sd, lo, hi)

res = {'window': f'訊號日 {base[0][3].date()}～{base[-1][3].date()}，評價到 {nxt.date()}（{len(base)} 期）', 'versions': {}, 'neighbors': {}}
res['main_n10'] = {c: stats(nav_main[c]) for c in L.COST}
res['EW'] = L.cagr_dd(ew, IDX[t0], IDX[t1])
res['0050'] = L.cagr_dd(E50, IDX[t0], IDX[t1])
mx_main = excess_series(nav_main['normal'], ew)

sleeves = {}
for vid, mode in [('B1', 'lvl_size'), ('B2', 'd6_size'), ('B3', 'lvl_pool')]:
    pl = L.sleeve_plan(S, mode, a, b)
    navs = {c: L.run_plan(pl, c, t1)[0] for c in L.COST}
    # 同起點的等權與主策略（B2 從 2022-04 起）
    lo = pl[0][3]
    o = {c: stats(navs[c], lo=lo) for c in L.COST}
    nv, lg = L.run_plan(pl, 'normal', t1)
    sx = excess_series(navs['normal'], ew, lo=lo)
    mx = mx_main.reindex(sx.index)
    o['corr_with_main_excess_vs_EW'] = round(float(np.corrcoef(sx, mx)[0, 1]), 3)
    o['turnover_per_rebalance'] = round(lg['turnover'], 3)
    o['first_signal'] = str(lo.date()); o['n_periods'] = len(pl)
    # 市值組內是否成立：各組 10 檔對該組等權
    if mode.endswith('_size'):
        sub = {}
        for gg, nm in [(0, 'small'), (1, 'mid'), (2, 'large')]:
            plg, plew = [], []
            for p in pl:
                cs = S.cross_section(p[3]); f = L.fo_features(cs['t']); g = L.terciles(f['mcap'], cs['el'])
                plg.append((p[0], np.array([j for j in p[1] if g[j] == gg], int)))
                plew.append((p[0], np.where(cs['el'] & (g == gg))[0]))
            ng, _ = K.simulate(plg, *L.COST['normal'], pl[0][0], t1)
            ne, _ = K.simulate(plew, 0, 0, pl[0][0], t1)
            sub[nm] = L.summ(L.monthly_t(ng, ne, SDw, lo, b))
        o['within_size_group_vs_group_EW_net'] = sub
    res['versions'][vid] = {'desc': {'B1': '市值三組內外資持股比率最高各 10 檔', 'B2': '市值三組內外資持股 6 月增量最高各 10 檔',
                                     'B3': '池內外資持股比率最高 30 檔（不控市值）'}[vid], **o}
    sleeves[vid] = (pl, navs)
    print(vid, json.dumps({c: o[c]['vs_EW'] for c in L.COST}, ensure_ascii=False), o['corr_with_main_excess_vs_EW'], flush=True)

# 鄰近版本：每組 5／15 檔
for vid, mode in [('B1', 'lvl_size'), ('B2', 'd6_size')]:
    for npg in (5, 15):
        pl = L.sleeve_plan(S, mode, a, b, n_per=npg)
        nv = L.run_plan(pl, 'normal', t1)[0]
        res['neighbors'][f'{vid}_n{npg}x3'] = stats(nv, lo=pl[0][3])['vs_EW']

# 選第二條腿
sel_sleeve = max(['B1', 'B2', 'B3'], key=lambda v: res['versions'][v]['normal']['vs_EW'].get('t', -9))
res['chosen_sleeve'] = sel_sleeve
pl_s, navs_s = sleeves[sel_sleeve]

# C1：80/20（從第二條腿起點開始，主策略同起點重跑）
lo = pl_s[0][3]
basec = [p for p in base if p[3] >= lo]
navm_c = {c: L.run_plan(basec, c, t1)[0] for c in L.COST}
comb = {}
for c in L.COST:
    nm = navm_c[c]; ns = navs_s[c].reindex(nm.index)
    comb[c] = L.combine(nm, ns, basec, 0.8)
o = {c: stats(comb[c], lo=lo, ref=navm_c[c]) for c in L.COST}
for c in L.COST:
    o[c]['main_same_window'] = L.cagr_dd(navm_c[c], IDX[basec[0][0]], IDX[t1])
    # 追蹤誤差（對 0050，月）
    for nm_, nv_ in [('combo', comb[c]), ('main', navm_c[c])]:
        x = L.monthly_t(nv_, E50, SDw, lo, b)
        o[c][f'TE_vs_0050_ann%_{nm_}'] = round(float(x.std(ddof=1) * np.sqrt(12) * 100), 2)
res['versions']['C1'] = {'desc': f'主策略 n10 80% ＋ {sel_sleeve} 20%', **o}
print('C1', o['normal']['vs_main'], flush=True)

for vid, th in [('C2', 'lvl'), ('C3', 'd6')]:
    pl = L.main_plan(S, a, b, third=th)
    navs = {c: L.run_plan(pl, c, t1)[0] for c in L.COST}
    o = {c: stats(navs[c], ref=nav_main[c]) for c in L.COST}
    # 和原版名單的差異
    diff = [len(set(p[1]) - set(q[1])) for p, q in zip(pl, base)]
    o['avg_names_changed_per_month'] = round(float(np.mean(diff)), 2)
    res['versions'][vid] = {'desc': f'主策略第三分數（權重 0.2）＝市值組內外資持股{"比率" if th == "lvl" else " 6 月增量"}百分位', **o}
    print(vid, o['normal']['vs_main'], o['avg_names_changed_per_month'], flush=True)
    # 鄰近：權重 0.1／0.3
    for w3 in (0.1, 0.3):
        pl2 = L.main_plan(S, a, b, third=th, w3=w3)
        nv = L.run_plan(pl2, 'normal', t1)[0]
        res['neighbors'][f'{vid}_w{w3}'] = stats(nv, ref=nav_main['normal'])['vs_main']

comb_choice = max(['C1', 'C2', 'C3'], key=lambda v: res['versions'][v]['normal']['vs_main'].get('t', -9))
res['chosen_combo'] = comb_choice
res['n_versions_tested'] = 6
res['n_neighbors_reported'] = len(res['neighbors'])
(L.HERE / 'iterations.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({'chosen_sleeve': sel_sleeve, 'chosen_combo': comb_choice}, ensure_ascii=False))

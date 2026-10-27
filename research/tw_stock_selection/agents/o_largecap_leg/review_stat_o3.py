#!/usr/bin/env python3
"""反方審查（資料與統計）：只讀研究員檔案，不寫入任何研究員輸出。
檢查：(1) L 腿對池等權的成本不對稱（L 扣成本、池零成本）；(2) Newey-West t；(3) 年份集中（去 2021）；
(4) 組合減主策略配對差的年度區塊 bootstrap；(5) 「權值股領漲年份」子樣本是否有鑑別力；(6) 2330 漂移反事實。"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lcore as L
sys.path.pop(0)
import numpy as np, pandas as pd
K, V, O = L.K, L.V, L.O

def nw_t(x, lag=None):
    x = pd.Series(x).dropna().to_numpy(); n = len(x)
    if lag is None: lag = int(math.floor(4 * (n / 100) ** (2 / 9)))
    m = x.mean(); e = x - m; g0 = e @ e / n; s = g0
    for k in range(1, lag + 1):
        gk = e[k:] @ e[:-k] / n; s += 2 * (1 - k / (lag + 1)) * gk
    return round(float(m / math.sqrt(s / n)), 3), lag

def ms(x):
    x = pd.Series(x).dropna()
    return {'mean_pp': round(float(x.mean()) * 100, 3), 't': round(float(K.tstat(x)), 3), 'nw_t': nw_t(x)[0], 'n': int(len(x))}

def yblock_boot(x, B=5000, seed=7):
    x = pd.Series(x).dropna(); g = [v.to_numpy() for _, v in x.groupby(x.index.year)]
    rng = np.random.default_rng(seed); ny = len(g); out = []
    for _ in range(B):
        idx = rng.integers(0, ny, ny); out.append(np.concatenate([g[i] for i in idx]).mean())
    out = np.array(out)
    return {'p_mean_le0': round(float((out <= 0).mean()), 4), 'ci95_pp': [round(float(np.quantile(out, .025)) * 100, 3), round(float(np.quantile(out, .975)) * 100, 3)]}

t1 = L.T1
mp = L.MAIN_PLAN
leg = L.plan_leg(50, 5, '55', 'ew')
navL, tovL = L.sim(L.wplan_leg(leg), L.NORMAL, t1)
navL0, _ = L.sim(L.wplan_leg(leg), (0.0, 0.0), t1) if False else (None, None)
ewP0 = L.pool_ew(leg, t1)
ewPn, _ = L.sim(L.wplan_pool(leg, 1.0), L.NORMAL, t1)   # 池等權扣一般成本
_, tovP = L.sim(L.wplan_pool(leg, 1.0), L.NORMAL, t1)
navM, _ = L.sim(L.wplan_main(mp), L.NORMAL, t1)
navC, _ = L.sim(L.wplan_mix(mp, leg, 0.3), L.NORMAL, t1)
navT, _ = L.sim(O.tsmc_plan(mp, 0.3), L.NORMAL, t1)
E = L.wr(K.E50)
WL, WP0, WPn, WM, WC, WT = map(L.wr, (navL, ewP0, ewPn, navM, navC, navT))
OUT = {'turnover_L': round(tovL, 4), 'turnover_poolEW': round(tovP, 4)}
for p in ('SEL', 'VAL', 'VAL_A', 'HOLD', 'ALL'):
    o = {}
    o['L_net_vs_pool_zero_cost'] = ms(L.win(WL - WP0, p))
    o['L_net_vs_pool_net_cost'] = ms(L.win(WL - WPn, p))
    o['COMBO_minus_MAIN'] = ms(L.win(WC - WM, p))
    o['COMBO_minus_TSMC30'] = ms(L.win(WC - WT, p))
    o['COMBO_ex0050'] = ms(L.win(WC - E, p)); o['MAIN_ex0050'] = ms(L.win(WM - E, p)); o['TSMC30_ex0050'] = ms(L.win(WT - E, p))
    o['TSMC30_minus_MAIN'] = ms(L.win(WT - WM, p))
    OUT[p] = o
# 年度區塊 bootstrap
OUT['boot_COMBO_minus_MAIN_SEL'] = yblock_boot(L.win(WC - WM, 'SEL'))
OUT['boot_L_vs_poolnet_VAL'] = yblock_boot(L.win(WL - WPn, 'VAL'))
OUT['boot_L_vs_pool0_SEL'] = yblock_boot(L.win(WL - WP0, 'SEL'))
# 去 2021
x = L.win(WL - WP0, 'VAL'); OUT['VAL_L_vs_pool0_ex2021'] = ms(x[x.index.year != 2021])
x = L.win(WC - WM, 'VAL'); OUT['VAL_COMBO_minus_MAIN_ex2021'] = ms(x[x.index.year != 2021])
x = L.win(WL - WP0, 'VAL'); OUT['VAL_L_vs_pool0_share_2021_of_sum'] = round(float(x[x.index.year == 2021].sum() / x.sum()), 3)
# 每年組合減主策略（SEL）負的年數
x = L.win(WC - WM, 'SEL'); yy = x.groupby(x.index.year).sum()
OUT['SEL_COMBO_minus_MAIN_years_negative'] = f'{int((yy < 0).sum())}/{len(yy)}'
# 「權值股領漲」在 SEL 占幾年
OUT['note_lead_years_SEL'] = '11/13 年（explore e_subsamples），子樣本≈全樣本，沒有鑑別力'
# L 腿 SEL 毛（零成本）對池
navLg, _ = L.sim(L.wplan_leg(leg), (0.0, 0.0), t1)
OUT['SEL_L_gross_vs_pool0'] = ms(L.win(L.wr(navLg) - WP0, 'SEL'))
OUT['VAL_L_gross_vs_pool0'] = ms(L.win(L.wr(navLg) - WP0, 'VAL'))
# 多重檢定：12 版中 L 對池 SEL 最佳 t 0.598 → V11 1.18；Bonferroni 無版本達 1.5
IT = json.loads((HERE / 'iterations.json').read_text(encoding='utf-8'))
OUT['iter_best_L_vs_pool_SEL_t'] = max((v['leg']['L_vs_poolEW_net']['t'], k) for k, v in IT['versions'].items())
OUT['iter_best_combo_t'] = max((v['SEL_normal']['t_0050'], k) for k, v in IT['versions'].items())
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))

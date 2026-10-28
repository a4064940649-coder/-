#!/usr/bin/env python3
"""反方審查（資料與統計）o_v5_pit_revenue：只讀研究員檔案，不寫任何研究員檔案。python3 -I -B review_stat_o5.py"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v5lib as L          # 只讀 import（不呼叫 setup，不跑策略）
K, L4, R = L.K, L.L4, L.R
from scipy import stats

NV = pd.read_parquet(HERE / 'navs_v5.parquet')
SN = pd.read_parquet(HERE / 'seed_navs_v5.parquet')
JN = pd.read_parquet(HERE / 'navs_jscan_v5.parquet')
res = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
sd = [pd.Timestamp(x) for x in pd.read_csv(HERE / 'plan_V5.csv')['signal']] + [pd.Timestamp(L.B0)]
E50 = K.E50
def xs(col, a=L.A0, b=L.B0, nav=None):
    nav = NV[col] if nav is None else nav
    return L4.excess(nav, E50.reindex(nav.index), sd, a, b)
def t(x): return float(L.tstat(x))
def cagr(nav, a, b): return float(R.cagr_mdd(nav, a, b)[0])
out = {}

# 1) 重算主欄 t 與分段；2014 以後合併
x = xs('AB_V5|normal')
out['t_all_recomputed'] = t(x); out['n'] = len(x)
for nm, (a, b) in {'pre2014': ('2006-04-01', '2013-12-31'), 'post2014': ('2014-01-01', L.B0), 'post2008': ('2008-01-01', L.B0),
                   'post2010': ('2010-01-01', L.B0), '2014_2023': ('2014-01-01', '2023-12-31'), 'post2021': ('2021-01-01', L.B0)}.items():
    xx = x[(x.index >= pd.Timestamp(a)) & (x.index <= pd.Timestamp(b))]
    out[f't_{nm}'] = {'t': t(xx), 'n': len(xx), 'mean_pp': float(xx.mean() * 100)}

# 2) 集中度：最好 k 個月占總超額比例；逐年超額加總；不含 2021 且起點 2008
srt = x.sort_values(ascending=False)
out['share_of_sum_excess_top_k'] = {k: float(srt.iloc[:k].sum() / x.sum()) for k in (1, 3, 5, 10, 20)}
out['top10_months'] = {str(d.date()): round(float(v) * 100, 2) for d, v in srt.iloc[:10].items()}
yr = x.groupby(x.index.year).sum() * 100
out['excess_sum_by_year_pp'] = {int(k): round(float(v), 2) for k, v in yr.items()}
xx = x[(x.index >= pd.Timestamp('2008-01-01')) & (x.index.year != 2021)]
out['t_post2008_ex2021'] = t(xx)
# 截尾（winsor 2.5%）t 與符號檢定
lo, hi = x.quantile(0.025), x.quantile(0.975)
out['t_winsor2.5'] = t(x.clip(lo, hi))
out['median_excess_pp'] = float(x.median() * 100); out['share_positive'] = float((x > 0).mean())
out['sign_test_p_one_sided'] = float(stats.binomtest(int((x > 0).sum()), len(x), 0.5, alternative='greater').pvalue)

# 3) 正式保守口徑 B10 的三項檢查（研究員只對 AB_V5 做）
xb = xs('AB_V5_B10|normal'); xbs = xs('AB_V5_B10|stress')
out['B10'] = {'t': t(xb), 'stress_t': t(xbs), 'drop_best1': t(xb.drop(xb.sort_values(ascending=False).index[:1])),
              'drop_best3': t(xb.drop(xb.sort_values(ascending=False).index[:3])),
              'start2008_t': t(xb[xb.index >= pd.Timestamp('2008-01-01')]),
              'start2008_CAGR': cagr(NV['AB_V5_B10|normal'], '2008-01-01', L.B0),
              'post2014_t': t(xb[xb.index >= pd.Timestamp('2014-01-01')]),
              'stress_CAGR': cagr(NV['AB_V5_B10|stress'], L.A0, L.B0),
              'stress_start2008_t': t(xbs[xbs.index >= pd.Timestamp('2008-01-01')])}
xw = xs('ABW_V5_B10|normal'); out['ABW_V5_B10_t'] = t(xw); out['ABW_V5_B10_CAGR'] = cagr(NV['ABW_V5_B10|normal'], L.A0, L.B0)
out['E50_CAGR'] = cagr(E50.reindex(NV.index), L.A0, L.B0)

# 4) SOP 實盤預期敏感度
e50 = out['E50_CAGR']
def live(c): return e50 + (c - e50) * 0.5
jc = {j: v['CAGR'] for j, v in res['RESULTS_JSCAN']['by_j'].items()}
jmed = float(np.median(list(jc.values())))
b10_drag = res['RESULTS']['table']['AB_V5|normal']['ALL']['CAGR'] - res['RESULTS']['table']['AB_V5_B10|normal']['ALL']['CAGR']
out['SOP'] = {'official': live(res['RESULTS']['table']['AB_V5_B10|normal']['ALL']['CAGR']),
              'jscan_median_CAGR': jmed, 'jscan_median_minus_B10_drag': jmed - b10_drag, 'live_jscan_median_B10': live(jmed - b10_drag),
              'live_B10_stress': live(out['B10']['stress_CAGR']), 'B10_drag_pp': b10_drag}

# 5) 換股日：官方 vs j 版的 t 差；官方排第幾
jt = {int(j): v['t'] for j, v in res['RESULTS_JSCAN']['by_j'].items()}
out['jscan_t'] = jt; out['official_rank_among_14'] = 1 + sum(1 for v in jt.values() if v > out['t_all_recomputed'])
# 官方 vs j=1（最接近官方日）逐窗的相關：兩條淨值同期年報酬
# 6) 首次值機制否定的統計力：top10/top30 vs all 的比值與 Poisson／Fisher 區間
fv = json.loads((HERE / 'explore.json').read_text(encoding='utf-8'))['first_value']
def ratio_ci(k1, n1, k0, n0):
    r = (k1 / n1) / (k0 / n0)
    # Katz log 法 95% CI
    se = math.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0) if k1 > 0 else float('nan')
    return {'ratio': r, 'lo95': r * math.exp(-1.96 * se), 'hi95': r * math.exp(1.96 * se), 'k1': k1, 'n1': n1}
A = fv['all']
out['fv_ratio'] = {
    'top30_raw': ratio_ci(fv['top30_cur']['n_disagree'], fv['top30_cur']['n_cells'], A['n_disagree'], A['n_cells']),
    'top30_confirmed': ratio_ci(fv['top30_cur']['n_disagree_consensus_or_internal'], fv['top30_cur']['n_cells'], A['n_disagree_consensus_or_internal'], A['n_cells']),
    'top10_raw': ratio_ci(fv['top10_cur']['n_disagree'], fv['top10_cur']['n_cells'], A['n_disagree'], A['n_cells']),
    'top10_confirmed': ratio_ci(fv['top10_cur']['n_disagree_consensus_or_internal'], fv['top10_cur']['n_cells'], A['n_disagree_consensus_or_internal'], A['n_cells']),
}
# top10 已佐證的 Wilson 95% 上界（對照 MC 用的 1.258%）
def wilson_hi(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return (c + s) / d
out['wilson_hi'] = {'top30_confirmed': wilson_hi(5, 926), 'top10_confirmed': wilson_hi(3, 345), 'top10_raw': wilson_hi(5, 345), 'all_raw': wilson_hi(113, 6513)}
# 7) 首次值配對差的 MDE 意義：非零月數
pr = res['RESULTS']['paired']['AB_V5_minus_AB_V4']['ALL']
out['paired_nonzero'] = {'n_nonzero': pr['n_nonzero'], 'n': pr['n'], 'note': 'MDE 只量 36 個替換格的效果，不是更正值前視本身'}
# 8) 逐 seed：拿掉最好 1 個月
ts1 = []
for c in SN.columns:
    xx = xs(None, nav=SN[c]); ts1.append(t(xx.drop(xx.sort_values(ascending=False).index[:1])))
out['seed_drop_best1'] = {'median': float(np.median(ts1)), 'n_below_2.30': int((np.array(ts1) < 2.30).sum()), 'n_below_2.0': int((np.array(ts1) < 2.0).sum())}
# 9) 多重檢定：以 jscan 13 版 + 官方 = 14 個近似等價規格，官方為最大值；計算 max-of-14 的經驗折扣
tj = np.array(list(jt.values()))
out['jscan_mean_t'] = float(tj.mean()); out['official_minus_jscan_median'] = out['t_all_recomputed'] - float(np.median(tj))
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))

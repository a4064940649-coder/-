#!/usr/bin/env python3
"""S52 事後診斷（不在 PREREG 判定內，只幫助解讀）：
(a) 格點敏感度：焦點對 0050 月超額 t 隨「每月第 d 日起第一個交易日」格點（d＝1～28）的分布；廣族群的窗格點（每月 ≥11 日）落在哪裡。
(b) DSR 的 SR0 改用廣族群實際觀察到的最大 SR（經驗版「最佳試驗」門檻），以及只用 SR > 0 成員估 V[SR]。
用法：python3 -I -B posthoc.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AG = HERE.parent
ROOT = AG.parent.parent.parent
EMC = 0.5772156649015329
off = pd.read_csv(ROOT / 'xq_scripts' / 'site' / 'navs_official.csv.gz', index_col=0, parse_dates=True)
res = json.loads((HERE / 'results.json').read_text())
out = {'note': '事後診斷，不改 PREREG 判定'}

# (a) 格點敏感度
idx = off.index
g = {}
for start, key in (('2015-07-01', 'full'), ('2019-01-01', 'since2019')):
    rows = []
    for d in range(1, 29):
        dates = []
        for p in pd.period_range(start[:7], '2026-10', freq='M'):
            c = idx[(idx >= pd.Timestamp(p.year, p.month, d)) & (idx.to_period('M') == p)]
            if len(c):
                dates.append(c[0])
        dates = [x for x in dates if x >= pd.Timestamp(start)]
        nav = off.loc[dates]
        r = nav.pct_change().iloc[1:]
        ex = (r['E0|W10_70'] - r['0050']).to_numpy()
        rows.append(dict(day=d, T=len(ex), SR_ann=float(ex.mean() / ex.std(ddof=1) * math.sqrt(12)), t=float(ex.mean() / ex.std(ddof=1) * math.sqrt(len(ex)))))
    ts = np.array([x['t'] for x in rows])
    g[key] = dict(rows=rows, t_min=float(ts.min()), t_median=float(np.median(ts)), t_max=float(ts.max()),
                  t_day11=rows[10]['t'], calendar_month_end_t=res['focus'][key]['excess_vs_0050']['t_iid'],
                  broad_window_grid_t=res['broad_family'][key]['focus_t_window'])
out['grid_sensitivity'] = g

# (b) DSR 經驗門檻
def psr(sr, s0, T, g3, g4):
    return float(stats.norm.cdf((sr - s0) * math.sqrt(T - 1) / math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))))

FE = pd.read_parquet(AG / 'o_main_spa_breaks' / 'data' / 'family_excess.parquet')
b = {}
for start, key in (('2015-07-01', 'full'), ('2019-01-01', 'since2019')):
    sub = FE[FE.index >= pd.Timestamp(start)]
    sub = sub.loc[:, sub.notna().all()]
    sub = sub.T.round(12).drop_duplicates().T
    sub = sub.loc[:, sub.std(ddof=1) > 1e-10]
    X = sub.to_numpy(); sr = X.mean(0) / X.std(0, ddof=1)
    e = res['focus'][key]['excess_vs_0050']
    smax = float(sr.max())
    pos = sr[sr > 0]
    Vpos = float(np.var(pos, ddof=1))
    iqr_sd = float((np.quantile(sr, 0.75) - np.quantile(sr, 0.25)) / 1.349)
    rowsN = {}
    for N in (3006, 100, 1000):
        f = lambda V: math.sqrt(V) * ((1 - EMC) * stats.norm.ppf(1 - 1 / N) + EMC * stats.norm.ppf(1 - 1 / (N * math.e)))
        rowsN[str(N)] = dict(SR0_ann_Vpos=f(Vpos) * math.sqrt(12), DSR_Vpos=psr(e['SR_m'], f(Vpos), e['T'], e['skew'], e['kurt']),
                             SR0_ann_IQR=f(iqr_sd ** 2) * math.sqrt(12), DSR_IQR=psr(e['SR_m'], f(iqr_sd ** 2), e['T'], e['skew'], e['kurt']))
    b[key] = dict(observed_max_SR_ann=smax * math.sqrt(12), best_member=str(sub.columns[int(np.argmax(sr))]),
                  PSR_vs_observed_max=psr(e['SR_m'], smax, e['T'], e['skew'], e['kurt']),
                  n_pos=int(len(pos)), SD_SR_ann_pos_only=math.sqrt(Vpos) * math.sqrt(12), SD_SR_ann_IQR=iqr_sd * math.sqrt(12),
                  SD_SR_ann_all=res['broad_family'][key]['SD_SR_window'] * math.sqrt(12),
                  SD_SR_ann_pure_noise=math.sqrt(12 / e['T']), by_N=rowsN)
out['dsr_empirical'] = b
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'rows'} for k, v in g.items()}, ensure_ascii=False, indent=1))
print(json.dumps(b, ensure_ascii=False, indent=1))

# (c) 廣族群 SPA_c（L＝6）的族群最大值 bootstrap 分布，換算「若焦點 t 是格點中位數／日曆月底」時的焦點 p（近似：族群其餘不變）
import sys
sys.path.insert(0, str(AG / 'o_main_spa_breaks'))
import spa_lib                                              # noqa: E402
c = {}
for start, key in (('2015-07-01', 'full'), ('2019-01-01', 'since2019')):
    sub = FE[FE.index >= pd.Timestamp(start)]
    sub = sub.loc[:, sub.notna().all()]
    sub = sub.T.round(12).drop_duplicates().T
    sub = sub.loc[:, sub.std(ddof=1) > 1e-10]
    grid = list(sub.index) + [pd.Timestamp('2026-10-12')]
    ns, nb = off['E0|W10_70'].reindex(grid).to_numpy(), off['0050'].reindex(grid).to_numpy()
    fx = ns[1:] / ns[:-1] - nb[1:] / nb[:-1]
    D = np.column_stack([fx, sub.to_numpy()]); n = D.shape[0]; L = 6
    W = spa_lib.sb_counts(n, L, 5000, 5252 + L + (0 if key == 'full' else 100))
    Mb = spa_lib.boot_means(W, D)
    dbar = D.mean(0); om2 = spa_lib.lrv(D, 'pr', L); om = np.sqrt(om2); sn = math.sqrt(n)
    thr = -np.sqrt(om2 / n * 2 * math.log(math.log(n)))
    gk = np.where(dbar >= thr, dbar, 0.0)
    Tb = np.maximum(((Mb - gk[None, :].astype(np.float32)) * (sn / om).astype(np.float32)[None, :]).max(axis=1), 0.0)
    tf = float(sn * dbar[0] / om[0])
    ratio = tf / res['broad_family'][key]['focus_t_window']        # 標準化 t 對 iid t 的比
    gs = out['grid_sensitivity'][key]
    c[key] = dict(focus_t_std_window=tf, p_at_window=float((Tb >= tf).mean()),
                  p_if_grid_median=float((Tb >= gs['t_median'] * ratio).mean()), p_if_grid_min=float((Tb >= gs['t_min'] * ratio).mean()),
                  p_if_calendar_month_end=float((Tb >= gs['calendar_month_end_t'] * ratio).mean()),
                  note='把焦點的 iid t 換成其他格點的 iid t（乘上同一個標準化比），對照同一個族群最大值分布；族群其餘成員的格點無法重建，故為近似')
out['spa_grid_approx'] = c
(HERE / 'results_posthoc.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(json.dumps(c, ensure_ascii=False, indent=1))

"""探索：只用挑選期（生效日 ≤ 2018-12-31）事件的描述統計。輸出 explore.json。"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
ev = L.load_events()
ev = ev[(ev.eff <= L.SEL_END) & (ev.eff >= '2008-01-01') & ev.in_panel].copy()
assert ev.eff.max() <= L.SEL_END
T = L.event_table(ev)
T = T[T.ok == True].copy()
D = L.panel(); A = D['adj'].ffill(limit=30); R = D['ret'].clip(-0.5, 0.5); EW = D['ew']; idx = A.index
out = {'scope': '挑選期 2008-01～2018-12（只有 2010-11 起有 MSCI 名單；臺灣50 自 2015-09）', 'n_events_ok': int(len(T))}
# (1) 事件數
T['year'] = T.eff.dt.year
T['liq_t'] = pd.qcut(T.v20.rank(method='first'), 3, labels=['lowADV', 'midADV', 'highADV'])
out['counts_by_class_year'] = {c: g.groupby('year').size().to_dict() for c, g in T.groupby('cls')}
out['counts_by_class_liq'] = {c: g.groupby('liq_t', observed=False).size().to_dict() for c, g in T.groupby('cls')}
# (2) CAR −10..+20（日異常＝個股日報酬 − 上市等權）
car = {}
for c, g in T.groupby('cls'):
    M = []
    for r in g.itertuples():
        j = A.columns.get_loc(r.code); i = idx.get_loc(r.eff)
        if i - 10 < 0 or i + 20 >= len(idx): continue
        ar = (R.iloc[i - 10:i + 21, j] - EW.iloc[i - 10:i + 21]).to_numpy()
        M.append(ar)
    M = np.array(M); m = np.nanmean(M, 0); car[c] = {'n': int(len(M)), 'AR_by_day': {int(k): round(float(v) * 100, 3) for k, v in zip(range(-10, 21), m)},
              'CAR_-10_-1': round(float(np.nansum(m[:10])) * 100, 2), 'AR_0': round(float(m[10]) * 100, 2),
              'CAR_+1_+5': round(float(np.nansum(m[11:16])) * 100, 2), 'CAR_+1_+20': round(float(np.nansum(m[11:])) * 100, 2),
              'AR_0_t': round(float(np.nanmean(M[:, 10]) / (np.nanstd(M[:, 10], ddof=1) / np.sqrt(np.isfinite(M[:, 10]).sum()))), 2)}
out['CAR'] = car
# (3) 賣壓強度：刪除事件
Dl = T[T.is_del].copy()
Dl['p_t'] = pd.qcut(Dl.pressure.rank(method='first'), 3, labels=['P_low', 'P_mid', 'P_high'])
cut = Dl.pressure.quantile([1 / 3, 2 / 3]).tolist(); cut2 = float(Dl.pressure.median())
out['pressure_def'] = {'formula': 'pressure = K_class / 20 日均成交值（E−1）', 'K_class_NTD': L.K_CLASS,
    'rationale': 'MSCI/臺灣50 的增刪由流通市值排名決定，同類刪除股的市值集中在門檻附近，被動賣出額 ≈ 被動持有比例 × 門檻市值，類內近似常數；所以賣壓 ÷ 成交值主要由流動性決定。K：MSCI 標準 ≈ 流通市值 NT$300～400 億 × 被動約 1.5% ≈ 5 億；小型股 ≈ 60 億 × 0.8% ≈ 0.5 億；臺灣50／高股息 ETF 粗估 3 億。',
    'tercile_cuts_select': [round(x, 3) for x in cut], 'median_cut_select': round(cut2, 3)}
def grp(g, col='x_closeE_20'):
    return L.stats(g[col], g.eff.dt.to_period('M').astype(str))
dose = {}
for p, g in Dl.groupby('p_t', observed=False):
    dose[str(p)] = {'n': int(len(g)), 'pressure_median': round(float(g.pressure.median()), 3), 'vol_ratio_E_median': round(float(g.vol_ratio.median()), 2),
                    'AR_E_mean%': round(float((g.ret_E - g.ew_E).mean()) * 100, 2), 'AR_E_median%': round(float((g.ret_E - g.ew_E).median()) * 100, 2),
                    'lockup_E_share': round(float(g.lockup_E.mean()), 3),
                    **{f'gross_{en}_{h}': grp(g, f'x_{en}_{h}') for en in ('closeE', 'openE1') for h in (5, 10, 20)}}
out['dose_response_deletions_gross'] = dose
by_cls = {}
for c, g in Dl.groupby('cls'):
    by_cls[c] = {'n': int(len(g)), 'vol_ratio_E_median': round(float(g.vol_ratio.median()), 2), 'AR_E_mean%': round(float((g.ret_E - g.ew_E).mean()) * 100, 2),
                 **{f'gross_closeE_{h}': grp(g, f'x_closeE_{h}') for h in (5, 10, 20)}}
out['deletions_by_class_gross'] = by_cls
# 成交值放大倍數與賣壓強度的相關（機制：賣壓越大，E 日成交相對越大、價格壓得越低）
out['spearman_pressure_vs'] = {'vol_ratio_E': round(float(Dl[['pressure', 'vol_ratio']].corr('spearman').iloc[0, 1]), 3),
                              'AR_E': round(float(pd.concat([Dl.pressure, Dl.ret_E - Dl.ew_E], axis=1).corr('spearman').iloc[0, 1]), 3),
                              'x_closeE_10': round(float(Dl[['pressure', 'x_closeE_10']].corr('spearman').iloc[0, 1]), 3)}
# 公告後到生效前的漂移（資訊傳遞速度）：E−10..E−1 由 CAR 表看
# 子樣本：STD 轉小型 vs 小型出局；E 日下跌（AR_E<0）是否才有回補
Dl['ar_neg'] = (Dl.ret_E - Dl.ew_E) < 0
out['by_E_day_move'] = {str(k): grp(g, 'x_closeE_10') for k, g in Dl.groupby('ar_neg')}
# 納入股（給 P4 參考）
Ad = T[T.cls.isin(['MSCI_STD_ADD', 'MSCI_SC_ADD', 'TW50_ADD'])]
out['additions_gross'] = {c: {f'x_closeE_{h}': grp(g, f'x_closeE_{h}') for h in (5, 10, 20)} for c, g in Ad.groupby('cls')}
json.dump(out, open(os.path.join(L.HERE, 'explore.json'), 'w'), ensure_ascii=False, indent=1, default=str)
T.to_csv(os.path.join(L.HERE, 'explore_events_select.csv'), index=False)
print(json.dumps({k: out[k] for k in ['n_events_ok', 'counts_by_class_liq', 'pressure_def', 'spearman_pressure_vs']}, ensure_ascii=False, indent=1))
for c, v in car.items(): print(c, v['n'], 'pre', v['CAR_-10_-1'], 'E', v['AR_0'], v['AR_0_t'], '+1..5', v['CAR_+1_+5'], '+1..20', v['CAR_+1_+20'])
for p, v in dose.items(): print(p, v['n'], v['pressure_median'], 'vol', v['vol_ratio_E_median'], 'ARE', v['AR_E_mean%'], v['AR_E_median%'], 'lock', v['lockup_E_share'], {k: (v[k]['mean'], v[k]['median'], v[k]['t_month']) for k in v if k.startswith('gross')})
for c, v in by_cls.items(): print(c, v)
print(out['by_E_day_move'])

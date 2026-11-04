#!/usr/bin/env python3
"""反方審查（資料與統計）o13：只讀研究員產出，不改任何研究員檔案。用法：python3 -I -B review_stat_o13.py [--sleeve]"""
import sys, math, json
sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

OUT = {}
tr = pd.read_csv(HERE / 'trades_all.csv.gz', parse_dates=['date', 'exit_date', 'announce'], dtype={'code': str})
TN = sorted(tr['timing'].unique())


def tt(x):
    x = pd.Series(x).dropna()
    return round(float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), 2) if len(x) > 2 and x.std(ddof=1) > 0 else None


def per(df, a, b):
    return df[(df['date'] >= a) & (df['date'] <= b)]


FULL = ('2006-04-01', '2026-10-22'); VAL = ('2024-01-02', '2026-10-22')
c = tr[(tr.timing == 'C1xCL') & (tr.cost == 'normal')].copy()
cs = tr[(tr.timing == 'C1xCL') & (tr.cost == 'stress')].copy()
f = per(c, *FULL).dropna(subset=['net_x'])

# 1) 重疊與聚集：同日／同月事件數，事件 t vs 月叢集 t vs 年叢集 t
m = f.groupby(f['date'].dt.to_period('M'))['net_x'].mean()
y = f.groupby(f['date'].dt.year)['net_x'].mean()
OUT['overlap'] = {'n': len(f), 'n_months': int(m.size), 'events_per_month_median': float(f.groupby(f['date'].dt.to_period('M')).size().median()),
                  'mean_pct': round(f.net_x.mean() * 100, 2), 't_event': tt(f.net_x), 't_month': tt(m), 't_year_cluster': tt(y), 'n_years': int(y.size),
                  'same_day_multi_share': round(float((f.groupby('date').size() > 1).reindex(f['date']).mean()), 3)}

# 2) 年份集中：逐年平均與貢獻；去掉最好 k 年
ys = f.groupby(f['date'].dt.year)['net_x'].agg(['size', 'mean', 'sum'])
OUT['yearly'] = {int(k): {'n': int(v['size']), 'mean_pct': round(v['mean'] * 100, 2), 'sum_pct': round(v['sum'] * 100, 1)} for k, v in ys.iterrows()}
order = ys['sum'].sort_values(ascending=False).index
for k in (1, 2, 3, 5):
    g = f[~f['date'].dt.year.isin(order[:k])]
    OUT[f'drop_best{k}_years'] = {'years': [int(x) for x in order[:k]], 'mean_pct': round(g.net_x.mean() * 100, 2), 't_month': tt(g.groupby(g['date'].dt.to_period('M'))['net_x'].mean())}
OUT['share_sum_from_top3_years'] = round(float(ys['sum'][order[:3]].sum() / ys['sum'].sum()), 2)
OUT['n_years_positive_mean'] = int((ys['mean'] > 0).sum()); OUT['n_years'] = int(len(ys))

# 3) 股票／事件集中：前 1%／5% 事件、前 10 檔
xs = f.net_x.sort_values(ascending=False)
for q in (0.01, 0.05, 0.10):
    k = int(math.ceil(len(xs) * q))
    OUT[f'drop_top{int(q*100)}pct_events_mean_pct'] = round(xs.iloc[k:].mean() * 100, 2)
OUT['symmetric_trim5_mean_pct'] = round(xs.iloc[int(len(xs) * .05):int(len(xs) * .95)].mean() * 100, 2)
OUT['median_event_pct'] = round(f.net_x.median() * 100, 2)
bc = f.groupby('code')['net_x'].sum().sort_values(ascending=False)
for k in (1, 5, 10, 20):
    g = f[~f.code.isin(bc.index[:k])]
    OUT[f'drop_top{k}_stocks_mean_pct'] = round(g.net_x.mean() * 100, 2)

# 4) 驗證期逐筆：對 EW 與對 0050，一般／壓力；9 時點
rows = {}
for nm in TN:
    for cost in ('normal', 'stress'):
        d = per(tr[(tr.timing == nm) & (tr.cost == cost)], *VAL).dropna(subset=['net_x'])
        mm = d.groupby(d['date'].dt.to_period('M'))
        rows[f'{nm}|{cost}'] = {'n': len(d), 'net_x': round(d.net_x.mean() * 100, 2), 't_m_x': tt(mm['net_x'].mean()),
                                'net50': round(d.net50_x.mean() * 100, 2), 't_m_50': tt(mm['net50_x'].mean()),
                                'ew_minus_e50_pct': round((d.ew - d.e50).mean() * 100, 2)}
OUT['VAL_by_timing'] = rows
for cost in ('normal', 'stress'):
    v = [rows[f'{nm}|{cost}'] for nm in TN]
    OUT[f'VAL_{cost}_median'] = {k: float(np.median([r[k] for r in v if r[k] is not None])) for k in ('net_x', 't_m_x', 'net50', 't_m_50', 'ew_minus_e50_pct')}

# 5) 9 時點的相關（中位數不是 9 次獨立證據）
piv = per(tr[tr.cost == 'normal'], *FULL).groupby([pd.Grouper(key='date', freq='ME'), 'timing'])['net_x'].mean().unstack()
cc = piv.corr().to_numpy(); OUT['timing_corr_monthly_net_x_offdiag_median'] = round(float(np.median(cc[~np.eye(len(cc), dtype=bool)])), 3)
me = pd.read_csv(HERE / 'monthly_excess.csv', index_col=0, parse_dates=True)
nv = pd.read_parquet(HERE / 'navs.parquet')
mainc = [x for x in nv.columns if x.startswith('MAIN|')]
mr = nv[mainc].resample('ME').last().pct_change()
cm = mr.corr().to_numpy(); OUT['timing_corr_sat_monthly_offdiag_median'] = round(float(np.median(cm[~np.eye(len(cm), dtype=bool)])), 3)

# 6) 衛星：驗證期與全期分年貢獻；FULL 正值依賴哪幾年
s = me['sat_median_excess'].dropna()
OUT['sat_yearly_sum_pp'] = {int(k): round(v * 100, 2) for k, v in s.groupby(s.index.year).sum().items()}
s21 = s[s.index.year != 2021]
OUT['sat_FULL_ex2021'] = {'mean_pp': round(s21.mean() * 100, 3), 't': tt(s21)}
OUT['sat_FULL'] = {'mean_pp': round(s.mean() * 100, 3), 't': tt(s)}
sv = s[(s.index >= '2024-01-01')]
OUT['sat_VAL'] = {'mean_pp': round(sv.mean() * 100, 3), 't': tt(sv), 'n': int(len(sv)), 'pos_months': int((sv > 0).sum())}
# 和主策略：去 2021 後的相關
both = me.dropna()
OUT['corr_main_FULL'] = round(float(both.corr().iloc[0, 1]), 3)
b2 = both[both.index.year != 2021]; OUT['corr_main_ex2021'] = round(float(b2.corr().iloc[0, 1]), 3)

# 7) 下市／出場延遲（需 dlib；只讀）
try:
    import dlib as D
    LV = {c_: int(D.LASTV[D.CI[c_]]) for c_ in f.code.unique() if c_ in D.CI}
    ex_i = D.IDX.searchsorted(f.exit_date.values)
    f2 = f.assign(ex_i=ex_i, lastv=f.code.map(LV))
    dl = f2[(f2.ex_i >= f2.lastv) & (f2.lastv < D.T - 1)]
    OUT['delisted_exit'] = {'n': int(len(dl)), 'mean_net_x_pct': round(dl.net_x.mean() * 100, 2) if len(dl) else None,
                            'mean_ex_delist_pct': round(f2.drop(dl.index).net_x.mean() * 100, 2),
                            'codes': dl.code.tolist()[:20]}
    if '--sleeve' in sys.argv:
        # 8) 鄰域 K5 恰為 0：檢查是不是程式路徑問題
        e, u, st = D.build_x(dict(entry='C1', exit='CL'))
        o = {}
        for K_ in (5, 10):
            nav, stt, _ = D.sleeve(e, u, tier_2026=1e6, mode='FIX', K_=K_, prepay='strict', pool_k=2, t_start='2006-04-01', t_end='2026-10-22')
            ex = D.ex_vs_0050(nav)
            o[K_] = {'mean_pp': round(ex.mean() * 100, 3), 't': tt(ex), 'entered': stt['entered'], 'mean_invested': stt['mean_invested'],
                     'mean_cash_share': stt['mean_cash_share'], 'cash_skip': stt['cash_skip'], 'no_slot': stt['no_slot']}
        OUT['K5_check_C1xCL'] = o
except Exception as ex_:  # noqa
    OUT['dlib_error'] = repr(ex_)

# 9) 結果檔內 NB_K5
R = json.loads((HERE / 'results.json').read_text())['RESULTS']
OUT['NB_K5_in_results'] = {p: R['SAT']['NB_K5'][p]['median_series'] for p in ('FULL', 'VAL')}
OUT['NB_K5_by_timing_VAL'] = R['SAT']['NB_K5']['VAL']['by_timing_t']
OUT['NB_K5_ledger'] = R['SAT']['NB_K5']['ledger']
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str))

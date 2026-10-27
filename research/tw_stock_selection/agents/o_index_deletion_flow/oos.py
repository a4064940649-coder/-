"""樣本外：驗證期 2019～2023、保留期 2025-09-24～2026-10，以及合併判定。必須先有 PREREG。"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
res_path = os.path.join(L.HERE, 'results.json'); RES = json.load(open(res_path))
assert 'PREREG' in RES and RES['PREREG'].get('sha256'), '先執行 prereg.py'
CUT = 2.7478  # 由 iterations.json V6 的切點
it = json.load(open(os.path.join(L.HERE, 'iterations.json'))); CUT = [v for v in it['versions'] if v['v'] == 'V6'][0]['cuts'][0]
H, EN = 20, 'closeE'; X = f'x_{EN}_{H}'
ev = L.load_events()
ev = ev[ev.in_panel & ev.is_del & (ev.eff >= '2008-01-01')].copy()
T = L.event_table(ev)
T = T[(T.ok == True) & ~((T.eff >= L.SRC3[0]) & (T.eff <= L.SRC3[1]))].copy()
T['grp'] = np.where(T.pressure >= CUT, 'high', 'low')
T['m'] = T.eff.dt.to_period('M').astype(str)
T.to_csv(os.path.join(L.HERE, 'event_returns_all.csv'), index=False)
PER = {'select': ('2008-01-01', '2018-12-31'), 'validate': ('2019-01-01', '2023-12-31'), 'holdout': ('2025-09-24', '2026-10-22'),
       'final_2019_2026': ('2019-01-01', '2026-10-22'), 'sub_2019_2020': ('2019-01-01', '2020-12-31'), 'sub_2021plus': ('2021-01-01', '2026-10-22')}
def hl(S):
    hi = S[S.grp == 'high'].groupby('m')[X].mean(); lo = S[S.grp == 'low'].groupby('m')[X].mean(); d = (hi - lo).dropna()
    return {'pooled_diff%': round(float(S[S.grp == 'high'][X].mean() - S[S.grp == 'low'][X].mean()) * 100, 2),
            'paired_month_diff%': round(float(d.mean()) * 100, 2) if len(d) else None, 'n_months': int(len(d)),
            't': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if len(d) > 2 else None}
out = {'cut': CUT, 'by_period': {}}
for k, (a, b) in PER.items():
    S = T[(T.eff >= a) & (T.eff <= b) & T[X].notna()]
    o = {g: L.stats(S[S.grp == g][X], S[S.grp == g].m, cost=0.005) for g in ['high', 'low']}
    o['high_stress'] = L.stats(S[S.grp == 'high'][X], S[S.grp == 'high'].m, cost=0.01)
    o['high_minus_low'] = hl(S)
    o['high_vs_0050'] = None
    o['high_by_class'] = {c: L.stats(g[X], g.m, cost=0.005) for c, g in S[S.grp == 'high'].groupby('cls')}
    o['all_by_class'] = {c: L.stats(g[X], g.m, cost=0.005) for c, g in S.groupby('cls')}
    out['by_period'][k] = o
# 對 0050：事件期間 0050 報酬
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date').adj.sort_index()
D = L.panel(); idx = D['adj'].index
e50 = E50.reindex(idx).ffill()
def r50(E):
    i = idx.get_loc(E); return e50.iloc[i + H] / e50.iloc[i] - 1 if i + H < len(idx) else np.nan
T['r50'] = [r50(e) for e in T.eff]; T['x50'] = T[f'r_{EN}_{H}'] - T['r50']
for k, (a, b) in PER.items():
    S = T[(T.eff >= a) & (T.eff <= b) & (T.grp == 'high') & T.x50.notna()]
    out['by_period'][k]['high_vs_0050'] = L.stats(S.x50, S.m, cost=0.005)
# 逐年（高賣壓組，淨）
S = T[(T.grp == 'high') & T[X].notna()]
out['yearly_high_net'] = {int(y): L.stats(g[X], g.m, cost=0.005) for y, g in S.groupby(S.eff.dt.year)}
# 穩健性（合併判定窗）
F = T[(T.eff >= '2019-01-01') & (T.grp == 'high') & T[X].notna()].copy()
top = F.groupby('code')[X].sum().sort_values(ascending=False)
out['robust_final'] = {
  'drop_top_contrib_stock': {'code': top.index[0], **L.stats(F[F.code != top.index[0]][X], F[F.code != top.index[0]].m, cost=0.005)},
  'drop_top3_stocks': L.stats(F[~F.code.isin(top.index[:3])][X], F[~F.code.isin(top.index[:3])].m, cost=0.005),
  'by_industry': {},
  'msci_only': L.stats(F[F.cls.str.startswith('MSCI')][X], F[F.cls.str.startswith('MSCI')].m, cost=0.005),
  'etf_only': L.stats(F[~F.cls.str.startswith('MSCI')][X], F[~F.cls.str.startswith('MSCI')].m, cost=0.005),
  'openE1_entry': L.stats(F['x_openE1_20'], F.m, cost=0.005),
  'hold5': L.stats(F['x_closeE_5'], F.m, cost=0.005), 'hold10': L.stats(F['x_closeE_10'], F.m, cost=0.005)}
sl = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/stock_list.csv', dtype=str).set_index('code').industry
F['ind'] = F.code.map(sl).fillna('NA')
for ind, g in F.groupby('ind'):
    if len(g) >= 5: out['robust_final']['by_industry'][ind] = L.stats(g[X], g.m, cost=0.005)
T.to_csv(os.path.join(L.HERE, 'event_returns_all.csv'), index=False)
RES = json.load(open(res_path)); RES['OOS_events'] = out
json.dump(RES, open(res_path, 'w'), ensure_ascii=False, indent=1, default=str)
for k, o in out['by_period'].items():
    print(k, 'HIGH', o['high'], '\n   LOW', o['low'], '\n   H-L', o['high_minus_low'], '\n   vs0050', o['high_vs_0050'], '\n   stress', o['high_stress'].get('mean'))
print(json.dumps(out['yearly_high_net'], ensure_ascii=False))
print(json.dumps(out['robust_final'], ensure_ascii=False, indent=0)[:3000])

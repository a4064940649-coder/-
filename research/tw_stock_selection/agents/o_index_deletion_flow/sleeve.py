"""凍結版的 10 槽部位模擬（閒置放 0050），與主策略、處置部位、0050、同池等權比較；配置檢定 (iv)(v)。"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import iflib as L
RES = json.load(open(os.path.join(L.HERE, 'results.json'))); assert 'PREREG' in RES, '先執行 prereg.py'
it = json.load(open(os.path.join(L.HERE, 'iterations.json'))); CUT = [v for v in it['versions'] if v['v'] == 'V6'][0]['cuts'][0]
T = pd.read_csv(os.path.join(L.HERE, 'event_returns_all.csv'), parse_dates=['eff'], dtype={'code': str})
T = T[(T.pressure >= CUT) & T.r_closeE_20.notna()].sort_values(['eff', 'pressure'], ascending=[True, False])
D = L.panel(); A = D['adj'].ffill(limit=30); idx = A.index; An = A.to_numpy()
E50 = pd.read_csv('/home/user/jonatw/finlab-txd/data/raw/etf_0050.csv', parse_dates=['date']).set_index('date').adj.sort_index().reindex(idx).ffill().to_numpy()
FEE = 0.001425 * 0.6
def run(cost_buy=0.001, cost_sell=0.004, start='2010-11-01', K=10, cap=1e6):
    t0 = idx.searchsorted(pd.Timestamp(start)); evs = {}
    for r in T.itertuples():
        i = idx.get_loc(r.eff)
        if i >= t0: evs.setdefault(i, []).append((A.columns.get_loc(r.code), i + 20, 0.05 * r.v20))
    etf = cap / E50[t0 - 1]; pos = []  # (j, exit_i, shares)
    nav = []; used = []
    for t in range(t0, len(idx)):
        # 出場（收盤）
        keep = []
        for j, ex, sh in pos:
            if t >= ex:
                val = sh * An[t, j] * (1 - cost_sell); etf += val * (1 - FEE) / E50[t]
            else: keep.append((j, ex, sh))
        pos = keep
        eq = etf * E50[t] + sum(sh * An[t, j] for j, ex, sh in pos)
        for j, ex, capv in evs.get(t, []):
            if len(pos) >= K or ex >= len(idx): continue
            amt = min(eq / K, capv)
            need = amt * (1 + cost_buy)
            sell_etf = need / (1 - FEE - 0.001)
            if etf * E50[t] < sell_etf: continue
            etf -= sell_etf / E50[t]; pos.append((j, ex, amt / An[t, j]))
        eq = etf * E50[t] + sum(sh * An[t, j] for j, ex, sh in pos)
        nav.append(eq); used.append(len(pos))
    return pd.Series(nav, index=idx[t0:]), np.array(used)
def st(nav, a=None, b=None):
    n = nav.loc[a:b] if a or b else nav
    yrs = (n.index[-1] - n.index[0]).days / 365.25
    b50 = pd.Series(E50, idx).reindex(n.index); b50 = b50 / b50.iloc[0]
    mr = n.resample('ME').last().pct_change().dropna(); mb = b50.resample('ME').last().pct_change().dropna(); ex = (mr - mb).dropna()
    return {'CAGR%': round(((n.iloc[-1] / n.iloc[0]) ** (1 / yrs) - 1) * 100, 2), 'maxDD%': round(float((n / n.cummax() - 1).min()) * 100, 1),
            '0050_CAGR%': round(((b50.iloc[-1]) ** (1 / yrs) - 1) * 100, 2), 't_vs_0050_month': round(float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))), 2) if len(ex) > 2 else None,
            'excess_month_mean%': round(float(ex.mean()) * 100, 3), 'n_months': int(len(ex))}
out = {}
nav, used = run(); navS, _ = run(0.0035, 0.0065)
out['sleeve_normal'] = {k: st(nav, a, b) for k, (a, b) in {'2010-11_2018': ('2010-11-01', '2018-12-31'), '2019_2026': ('2019-01-01', None), 'all': (None, None)}.items()}
out['sleeve_stress'] = {k: st(navS, a, b) for k, (a, b) in {'2019_2026': ('2019-01-01', None), 'all': (None, None)}.items()}
out['avg_slots_used'] = round(float(used.mean()), 2); out['share_days_any_position'] = round(float((used > 0).mean()), 3)
# 主策略、處置部位
MN = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')['EXFIN|n10|normal'].dropna()
EWp = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/main_v2/navs_gapfix.parquet')['EXFIN|EW'].dropna()
DP = pd.read_parquet('/home/user/-/research/tw_stock_selection/agents/disposal_reconcile/sleeve_nav.parquet')['A_0050'].dropna()
m = lambda s: s.resample('ME').last().pct_change()
M = pd.concat({'del': m(nav), 'main': m(MN), 'disp': m(DP), 'e50': m(pd.Series(E50, idx)), 'ew': m(EWp)}, axis=1)
out['corr_monthly'] = {'del_vs_main_2010_2026': round(float(M[['del', 'main']].dropna().corr().iloc[0, 1]), 3),
                       'del_vs_disp_2021_2026': round(float(M[['del', 'disp']].dropna().corr().iloc[0, 1]), 3),
                       'del_excess_vs_main_excess': round(float((M['del'] - M['e50']).corr(M['main'] - M['e50'])), 3)}
def combo(ws, a, b):
    """月再平衡配置（月報酬加權）。"""
    X = M.loc[a:b, list(ws)].dropna(); r = sum(X[k] * w for k, w in ws.items()); n = (1 + r).cumprod()
    ex = r - M.loc[X.index, 'e50']
    return {'CAGR%': round(float(n.iloc[-1] ** (12 / len(n)) - 1) * 100, 2), 'maxDD_month%': round(float((n / n.cummax() - 1).min()) * 100, 1),
            't_vs_0050': round(float(ex.mean() / (ex.std(ddof=1) / math.sqrt(len(ex)))), 2), 'n_months': int(len(n))}
out['alloc'] = {
  '2021-10_2026-10': {'main80_disp20': combo({'main': .8, 'disp': .2}, '2021-10-31', '2026-10-12'),
                      'main70_disp20_del10': combo({'main': .7, 'disp': .2, 'del': .1}, '2021-10-31', '2026-10-12')},
  '2010-12_2026-10': {'main100': combo({'main': 1.0}, '2010-12-31', '2026-10-12'), 'main90_del10': combo({'main': .9, 'del': .1}, '2010-12-31', '2026-10-12')},
  'benchmarks_2010-12_2026-10': {'0050': combo({'e50': 1.0}, '2010-12-31', '2026-10-12'), 'EW_pool': combo({'ew': 1.0}, '2010-12-31', '2026-10-12')}}
nav.to_frame('del_sleeve_normal').join(navS.to_frame('del_sleeve_stress')).to_parquet(os.path.join(L.HERE, 'sleeve_nav.parquet'))
RES = json.load(open(os.path.join(L.HERE, 'results.json'))); RES['SLEEVE'] = out
json.dump(RES, open(os.path.join(L.HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1))

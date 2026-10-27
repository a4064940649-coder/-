"""反方審查（成本與可成交）：o_index_deletion_flow。只讀研究員產出，不改其檔案。python3 -I -B review_exec_o1.py"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import iflib as L

T = pd.read_csv(os.path.join(HERE, 'event_returns_all.csv'), parse_dates=['eff'], dtype={'code': str})
it = json.load(open(os.path.join(HERE, 'iterations.json')))
CUT = [v for v in it['versions'] if v['v'] == 'V6'][0]['cuts'][0]
X = 'x_closeE_20'
D = L.panel(); idx = D['adj'].index; C = D['close_raw']; V = D['value']; O = D['open']
def mt(x, g):
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna().groupby(level=0).mean()
    if len(s) < 3: return (round(float(s.mean()) * 100, 2) if len(s) else None, None, int(len(s)))
    return round(float(s.mean()) * 100, 2), round(float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), 2), int(len(s))
out = {}
T['era'] = np.where(T.eff <= '2018-12-31', 'select', 'oos')
H = T[(T.grp == 'high')].copy()
# 1) 毛報酬與損益兩平成本
for era, g in H[H[X].notna()].groupby('era'):
    out[f'1_gross_{era}'] = {'n': len(g), 'gross_mean%': round(g[X].mean() * 100, 2), 'gross_median%': round(g[X].median() * 100, 2),
                             'month_mean_t': mt(g[X], g.m), 'breakeven_roundtrip_cost%': round(g[X].mean() * 100, 2)}
# 2) 漲跌停與可成交：E 日收盤漲停被排除筆數、E 日跌停（買得到但不代表賣壓價）、E+20 出場日跌停（賣不掉）
rows = []
for r in H.itertuples():
    if r.code not in C.columns or r.eff not in idx: continue
    j = C.columns.get_loc(r.code); i = idx.get_loc(r.eff); lp = L.limit_pct(r.eff)
    ce, cp = C.iat[i, j], C.iat[i - 1, j]
    k = i + 20
    ex_lock_dn = False; ex_noval = False
    if k < len(idx):
        c1, c0 = C.iat[k, j], C.iat[k - 1, j]
        ex_lock_dn = bool(np.isfinite(c1) and np.isfinite(c0) and c1 <= c0 * (1 - lp) + 1e-9)
        ex_noval = bool(not (V.iat[k, j] > 0))
    ordE = V.iat[i, j]
    rows.append(dict(event_id=r.event_id, era=r.era, cls=r.cls, code=r.code, eff=r.eff, v20=r.v20, valE=ordE,
                     E_limdn=bool(np.isfinite(ce) and ce <= cp * (1 - lp) + 1e-9), lockup=bool(r.lockup_E),
                     exit_limdn=ex_lock_dn, exit_noval=ex_noval, net=(getattr(r, X) - 0.005) if np.isfinite(getattr(r, X)) else np.nan))
R = pd.DataFrame(rows)
out['2_tradeability'] = {era: {'n': len(g), 'E_lockup_excluded': int(g.lockup.sum()), 'E_limit_down': int(g.E_limdn.sum()),
                               'exit_limit_down': int(g.exit_limdn.sum()), 'exit_no_trade': int(g.exit_noval.sum())} for era, g in R.groupby('era')}
# 3) 容量：每槽金額 = 資本/10；相對 E 日成交值與 20 日均值
cap = {}
for acct in (3e5, 1e6, 3e6, 1e7):
    slot = acct / 10
    for era, g in R.groupby('era'):
        cap[f'{int(acct/1e4)}萬_{era}'] = {'slot_over_5pct_v20_share': round(float((slot > 0.05 * g.v20).mean()), 3),
                                          'median_slot_vs_E_value%': round(float((slot / g.valE).median() * 100), 3),
                                          'max_slot_vs_E_value%': round(float((slot / g.valE).max() * 100), 3)}
out['3_capacity'] = cap
out['3_v20_high_group_quantiles_NT$M'] = {era: [round(float(x) / 1e6, 1) for x in g.v20.quantile([.1, .5, .9])] for era, g in R.groupby('era')}
# 4) 處置股：2021 後有處置記錄（disposal_5y），E 日或持有期內處置中的筆數
try:
    DP = pd.read_csv('/home/user/stevenhsiehsheng-ctrl/tw-stock-screener/data/extras/disposal_5y.csv.gz', dtype={'code': str}, parse_dates=['date', 'start', 'end'])
    def disp(code, a, b):
        d = DP[(DP.code == code)]
        return bool(((d.start <= b) & (d.end >= a)).any())
    AllT = T[T[X].notna() & (T.eff >= DP.start.min())].copy()
    AllT['dispE'] = [disp(c, e, e) for c, e in zip(AllT.code, AllT.eff)]
    AllT['dispHold'] = [disp(c, e, idx[min(idx.get_loc(e) + 20, len(idx) - 1)]) for c, e in zip(AllT.code, AllT.eff)]
    out['4_disposal_since'] = str(DP.start.min().date())
    out['4_disposal'] = {'all_del_n': len(AllT), 'disp_on_E': int(AllT.dispE.sum()), 'disp_during_hold': int(AllT.dispHold.sum()),
                         'high_n': int((AllT.grp == 'high').sum()), 'high_disp_during_hold': int((AllT.dispHold & (AllT.grp == 'high')).sum())}
except Exception as e:
    out['4_disposal'] = f'unavailable: {e}'
# 5) 成本敏感度：樣本外高組在 0 成本下仍為負？ 挑選期在 1.5% 成本下？
for c in (0.0, 0.005, 0.01, 0.015, 0.02):
    for era, g in H[H[X].notna()].groupby('era'):
        out.setdefault('5_cost_sens', {})[f'{era}_cost{c*100:.1f}%'] = mt(g[X] - c, g.m)
# 6) 挑選期集中度：每事件月平均（淨），以及去掉 2010、2013 兩個月後
g = H[(H.era == 'select') & H[X].notna()].copy(); g['net'] = g[X] - 0.005
mm = g.groupby('m').net.agg(['mean', 'size'])
out['6_select_month_means'] = {k: (round(v['mean'] * 100, 2), int(v['size'])) for k, v in mm.iterrows()}
out['6_select_ex_2013'] = mt(g[g.eff.dt.year != 2013].net, g[g.eff.dt.year != 2013].m)
out['6_select_2014_2018'] = mt(g[g.eff >= '2014-01-01'].net, g[g.eff >= '2014-01-01'].m)
# 7) HD ETF 事件 E 定義：過渡期多日，E 日成交倍數
hd = T[T.cls == 'HDETF_DEL']
out['7_hdetf'] = {era: {'n': len(gg), 'vol_ratio_E_median': round(float(gg.vol_ratio.median()), 2), 'net20_mean%': round(float((gg[X] - 0.005).mean() * 100), 2)} for era, gg in hd.groupby('era')}
# 8) 隔日開盤進場（不需收盤盯盤）的樣本外與挑選期
for era, gg in H.groupby('era'):
    out.setdefault('8_openE1_20_net', {})[era] = mt(gg['x_openE1_20'] - 0.005, gg.m)
# 9) P4：事件逐筆（成本角度：E 日收盤賣出是否漲停鎖住→賣得掉；額外成本）
P4 = pd.read_csv(os.path.join(HERE, 'p4_events.csv'), parse_dates=['eff'], dtype={'code': str})
p4r = []
for r in P4.itertuples():
    j = C.columns.get_loc(r.code); i = idx.get_loc(r.eff); lp = L.limit_pct(r.eff)
    p4r.append(dict(code=r.code, eff=str(r.eff.date()), type=r.type, src3=r.src3, diff=round(r.diff * 100, 2), w=round(r.w, 3),
                    E_ret=round(float(C.iat[i, j] / C.iat[i - 1, j] - 1) * 100, 2), E_limup=bool(C.iat[i, j] >= C.iat[i - 1, j] * (1 + lp) - 1e-9)))
out['9_p4_events'] = p4r
M = P4[~P4.src3].copy(); M['m'] = M.eff.dt.to_period('M').astype(str)
out['9_p4_drop_worst_month'] = mt(M[M.m != M.groupby('m')['diff'].mean().idxmin()]['diff'], M[M.m != M.groupby('m')['diff'].mean().idxmin()].m)
out['9_p4_MSCI_only_2016plus'] = mt(M[(M.type != 'TW50_ADD') & (M.eff >= '2016-01-01')]['diff'], M[(M.type != 'TW50_ADD') & (M.eff >= '2016-01-01')].m)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))

"""反方審查（資料與統計）：o_index_deletion_flow。只讀研究員產出，不改其檔案。python3 -I -B review_stat_o1.py"""
import sys, os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd
import iflib as L

rng = np.random.default_rng(7)
T = pd.read_csv(os.path.join(HERE, 'event_returns_all.csv'), parse_dates=['eff'], dtype={'code': str})
it = json.load(open(os.path.join(HERE, 'iterations.json')))
CUT = [v for v in it['versions'] if v['v'] == 'V6'][0]['cuts'][0]
X = 'x_closeE_20'; C = 0.005
def mt(x, g):
    s = pd.Series(np.asarray(x, float), index=np.asarray(g)).dropna().groupby(level=0).mean()
    if len(s) < 3: return (round(float(s.mean()) * 100, 2) if len(s) else None, None, len(s))
    return round(float(s.mean()) * 100, 2), round(float(s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))), 2), len(s)
out = {}
S = T[(T.eff <= '2018-12-31') & T[X].notna()].copy()
H = S[S.grp == 'high'].copy(); H['net'] = H[X] - C
out['A_select_high_repro'] = {'pooled_mean': round(H.net.mean() * 100, 2), 'month_mean_t_n': mt(H.net, H.m), 'n': len(H)}
# A1: 逐月平均（16 個月）與留一月/留一年
mm = H.groupby('m').net.agg(['mean', 'size'])
out['A1_month_means'] = {k: (round(v['mean'] * 100, 2), int(v['size'])) for k, v in mm.iterrows()}
out['A1_leave_one_year_out'] = {int(y): mt(H[H.eff.dt.year != y].net, H[H.eff.dt.year != y].m) for y in sorted(H.eff.dt.year.unique())}
out['A1_drop_2013'] = mt(H[H.eff.dt.year != 2013].net, H[H.eff.dt.year != 2013].m)
out['A1_drop_2010_2013_2018'] = mt(H[~H.eff.dt.year.isin([2010, 2013, 2018])].net, H[~H.eff.dt.year.isin([2010, 2013, 2018])].m)
top = H.sort_values('net', ascending=False)
out['A1_top5_events'] = top[['eff', 'code', 'name', 'cls', 'net']].head(5).assign(net=lambda d: (d.net * 100).round(2)).astype(str).values.tolist()
for k in (1, 3, 5):
    h = top.iloc[k:]; out[f'A1_drop_top{k}_events'] = {'pooled': round(h.net.mean() * 100, 2), 'month': mt(h.net, h.m)}
out['A1_by_class_select_high'] = {c: (len(g), mt(g.net, g.m)) for c, g in H.groupby('cls')}
# B: 多重檢定：8 版 jointly 月區塊 bootstrap（虛無：各版高組平均＝0），max-t 的家族錯誤率
Ssel = S[S.is_del].copy()
q3 = Ssel.pressure.quantile([1 / 3, 2 / 3]).tolist(); q2 = float(Ssel.pressure.median())
VERS = [(en, h, c) for en, h, c in [('closeE', 5, 'T3'), ('closeE', 10, 'T3'), ('closeE', 20, 'T3'), ('closeE', 5, 'T2'), ('closeE', 10, 'T2'),
        ('closeE', 20, 'T2'), ('openE1', 5, 'T3'), ('openE1', 20, 'T3')]]
months = sorted(Ssel.m.unique()); Mser = []
for en, h, c in VERS:
    hi = Ssel.pressure >= (q3[1] if c == 'T3' else q2)
    s = (Ssel[hi][f'x_{en}_{h}'] - C).groupby(Ssel[hi].m).mean().reindex(months)
    Mser.append(s.values)
Mmat = np.array(Mser).T  # months × versions
def tvec(M):
    o = []
    for k in range(M.shape[1]):
        v = M[:, k]; v = v[np.isfinite(v)]
        o.append(v.mean() / (v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 2 and v.std(ddof=1) > 0 else np.nan)
    return np.array(o)
t_obs = tvec(Mmat); M0 = Mmat - np.nanmean(Mmat, 0)
B = 5000; mx = np.empty(B)
for b in range(B):
    ii = rng.integers(0, len(months), len(months)); mx[b] = np.nanmax(tvec(M0[ii]))
out['B_multiple_testing'] = {'t_obs_versions': np.round(t_obs, 2).tolist(), 'n_months_universe': len(months),
    'V6_t': round(float(t_obs[5]), 2), 'p_single_bootstrap': round(float((np.array([tvec(M0[rng.integers(0, len(months), len(months))])[5] for _ in range(3000)]) >= t_obs[5]).mean()), 4),
    'p_family_maxT_8versions': round(float((mx >= t_obs[5]).mean()), 4), 'maxT_95pct': round(float(np.quantile(mx, 0.95)), 2)}
# B2: K_class 敏感度（挑選期）：K 是事前粗估；改 K 比例後高組（中位數切）結果
D = L.panel()
alt = {'研究員 K（STD5億/SC0.5億/TW50 3億）': {'MSCI_STD_DEL': 5e8, 'MSCI_SC_DEL': 5e7, 'TW50_DEL': 3e8, 'HDETF_DEL': 3e8},
       'K 全部相同（純流動性）': {'MSCI_STD_DEL': 1, 'MSCI_SC_DEL': 1, 'TW50_DEL': 1, 'HDETF_DEL': 1},
       'SC 改 1.5 億': {'MSCI_STD_DEL': 5e8, 'MSCI_SC_DEL': 1.5e8, 'TW50_DEL': 3e8, 'HDETF_DEL': 3e8},
       'SC 改 0.2 億': {'MSCI_STD_DEL': 5e8, 'MSCI_SC_DEL': 2e7, 'TW50_DEL': 3e8, 'HDETF_DEL': 3e8},
       'STD 改 15 億': {'MSCI_STD_DEL': 1.5e9, 'MSCI_SC_DEL': 5e7, 'TW50_DEL': 3e8, 'HDETF_DEL': 3e8}}
out['B2_K_sensitivity_select'] = {}
for nmK, K in alt.items():
    p = Ssel.cls.map(K) / Ssel.v20; cut = p.median(); hi = Ssel[p >= cut]; lo = Ssel[p < cut]
    out['B2_K_sensitivity_select'][nmK] = {'high': mt(hi[X] - C, hi.m), 'low': mt(lo[X] - C, lo.m), 'high_n_by_cls': hi.cls.value_counts().to_dict()}
# C: 樣本外
O = T[(T.eff >= '2019-01-01') & T[X].notna()].copy(); O['net'] = O[X] - C
OH = O[O.grp == 'high']
out['C_oos_high'] = {'n': len(OH), 'pooled': round(OH.net.mean() * 100, 2), 'month': mt(OH.net, OH.m), 'by_cls': OH.cls.value_counts().to_dict(),
                     'months': OH.groupby('m').net.agg(lambda s: round(s.mean() * 100, 2)).to_dict(),
                     'share_oos_events_high': round(float((O.grp == 'high').mean()), 3), 'share_select_events_high': round(float((S.grp == 'high').mean()), 3)}
# 樣本外用研究員自己的挑選期定義中可交易的全部 MSCI 刪除（不依賴 K）
OM = O[O.cls.str.startswith('MSCI')]
out['C_oos_all_msci_del'] = mt(OM.net, OM.m); out['C_oos_all_msci_del_n'] = len(OM)
# D: 生效日對齊：每個審核群組，成交值峰值落在哪一天（相對 E）
V = D['value']; idx = V.index
ev = L.load_events(); ev = ev[ev.in_panel & ev.is_del & (ev.eff >= '2010-01-01')]
rows = []
for r in ev.itertuples():
    j = V.columns.get_loc(r.code); i = idx.get_loc(r.eff)
    if i + 3 >= len(idx): continue
    base = V.iloc[i - 25:i - 5, j].median()
    if not (base > 0): continue
    w = V.iloc[i - 3:i + 4, j].to_numpy() / base
    rows.append(dict(eff=r.eff, cls=r.cls, **{f'd{k}': w[k + 3] for k in range(-3, 4)}))
W = pd.DataFrame(rows)
g = W.groupby(['cls', W.eff.dt.year >= 2019]).median(numeric_only=True).round(1)
out['D_volume_ratio_by_day_median'] = {f'{a}|{"oos" if b else "sel"}': v.to_dict() for (a, b), v in g.iterrows()}
W['peak'] = W[[f'd{k}' for k in range(-3, 4)]].to_numpy().argmax(1) - 3
out['D_peak_day_share_E'] = W.groupby('cls').peak.apply(lambda s: round(float((s == 0).mean()), 2)).to_dict()
# E: 機制 E 日異常：按審核日分群（同日事件相關），挑選 vs 樣本外
R = D['ret'].clip(-.5, .5); EW = D['ew']
T2 = T.copy()
T2['AR0'] = [R.iat[idx.get_loc(e), R.columns.get_loc(c)] - EW.iat[idx.get_loc(e)] for c, e in zip(T2.code, T2.eff)]
T2['era'] = np.where(T2.eff <= '2018-12-31', 'sel', 'oos')
out['E_AR0_clustered_by_review'] = {f'{c}|{e}': (int(len(g)), mt(g.AR0, g.eff.astype(str)))
                                    for (c, e), g in T2.groupby(['cls', 'era']) if c.startswith('MSCI')}
# naive t（研究員 explore 用的是事件獨立 t）
g = T2[(T2.cls == 'MSCI_STD_DEL') & (T2.era == 'sel')].AR0.dropna()
out['E_naive_t_STD_sel'] = round(float(g.mean() / (g.std(ddof=1) / math.sqrt(len(g)))), 2)
out['E_STD_sel_AR0_by_review'] = T2[(T2.cls == 'MSCI_STD_DEL') & (T2.era == 'sel')].groupby('m').AR0.agg(lambda s: (round(s.mean() * 100, 2), len(s))).to_dict()
# F: 存活者：樣本外刪除事件中不在面板／無報酬
evall = L.load_events(); evall = evall[evall.is_del & (evall.eff >= '2010-01-01')]
evall['era'] = np.where(evall.eff <= '2018-12-31', 'sel', np.where((evall.eff >= L.SRC3[0]) & (evall.eff <= L.SRC3[1]), 'src3', 'oos'))
dl = set(json.load(open(L.P + 'delisted_unknown.json')))
out['F_survivorship'] = {e: {'n': int(len(g)), 'not_in_panel': int((~g.in_panel).sum()), 'code_in_delisted_unknown': int(g.code.isin(dl).sum())} for e, g in evall.groupby('era')}
ret_nan = T[(T.ok == True) & T[X].isna()]
out['F_ok_but_no_20d_return'] = ret_nan.groupby(np.where(ret_nan.eff <= '2018-12-31', 'sel', 'oos')).size().to_dict()
# G: P4 事件與穩健性
P4 = pd.read_csv(os.path.join(HERE, 'p4_events.csv'), parse_dates=['eff'], dtype={'code': str})
M4 = P4[~P4.src3].copy(); M4['m'] = M4.eff.dt.to_period('M').astype(str)
out['G_p4_events'] = M4[['eff', 'code', 'name', 'type', 'diff']].assign(diff=lambda d: (d['diff'] * 100).round(2)).astype(str).values.tolist()
mm4 = M4.groupby('m')['diff'].mean()
out['G_p4_month_means'] = (mm4 * 100).round(2).to_dict()
out['G_p4_leave_one_month_out_t'] = {k: mt(M4[M4.m != k]['diff'], M4[M4.m != k].m)[1] for k in mm4.index}
out['G_p4_event_level_t'] = round(float(M4['diff'].mean() / (M4['diff'].std(ddof=1) / math.sqrt(len(M4)))), 2)
# G2: 全部 MSCI 標準納入（非 P4），E+1..+5 超額，按審核日分群
evA = L.load_events(); evA = evA[evA.in_panel & (evA.type == 'MSCI_STD_ADD') & ~((evA.eff >= L.SRC3[0]) & (evA.eff <= L.SRC3[1]))]
TA = L.event_table(evA, holds=(5,), entries=('closeE',)); TA = TA[TA.ok == True]
TA['era'] = np.where(TA.eff <= '2018-12-31', 'sel', 'oos')
out['G2_STD_ADD_x5_clustered'] = {e: (int(g.x_closeE_5.notna().sum()), mt(g.x_closeE_5, g.eff.astype(str))) for e, g in TA.groupby('era')}
json.dump(out, open(os.path.join(os.environ.get('REVIEW_OUT', '/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad'), 'review_stat_o1_out.json'), 'w'), ensure_ascii=False, indent=1, default=str)
for k, v in out.items(): print(k, json.dumps(v, ensure_ascii=False, default=str)[:1800])

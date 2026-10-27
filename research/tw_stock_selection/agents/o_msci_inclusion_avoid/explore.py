#!/usr/bin/env python3
"""探索（只用挑選期：生效日 ≤ 2018-12-31 的事件、訊號日 ≤ 2018-12 的主策略持有期）。輸出 explore.json、explore_events_select.csv、explore_main_hits.csv。
事件數（不含報酬）另報全期，作為檢定力估計。"""
import sys, os, json, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import oa_lib as O
K, V = O.K, O.V
IDX, COLS, CI = O.IDX, O.COLS, O.CI

INC = O.inclusion_events()
out = {'scope': '挑選期：E ≤ 2018-12-31（MSCI 名單 2010-11 起；臺灣50 納入事件表自 2015-09；高股息 ETF 挑選期 0 筆）'}
out['n_events_all_periods_no_returns'] = INC.groupby(['cls', 'period']).size().unstack(fill_value=0).to_dict()

# ---------------- (c) 事件層級（挑選期） ----------------
X = INC[(INC.period == 'select') & ~INC.src3].copy()
assert X.eff.max() <= O.SEL_END
T = O.event_metrics(X)
T = T[T.ok == True].copy()
T['grp'] = T.review
out['n_events_select_ok'] = T.groupby('cls').size().to_dict()

W = ['car_preA10', 'ar_A', 'car_A_E', 'car_pre10', 'ar_E', 'ar_p1', 'car_p1_5', 'car_p1_10', 'car_p1_20', 'bhar_p1_5']
def block(g):
    return {w: O.clus(g[w], g.grp) for w in W if w in g}
out['CAR_by_class'] = {c: block(g) for c, g in T.groupby('cls')}
# 日異常路徑（以 E 為 0；池化平均）
def path(g, lo=-12, hi=20):
    M = []
    for r in g.itertuples():
        j = CI[r.code]; i = r.iE
        M.append(O._RN[i + lo:i + hi + 1, j] - O.EW_R[i + lo:i + hi + 1])
    m = np.mean(np.array(M), 0)
    return {int(k): round(float(v) * 100, 3) for k, v in zip(range(lo, hi + 1), m)}
out['AR_path_E'] = {c: path(g) for c, g in T.groupby('cls')}
# 以公告日 A 為 0 的路徑（只 MSCI）
def pathA(g, lo=-5, hi=15):
    M = []
    for r in g.itertuples():
        j = CI[r.code]; i = r.iA
        M.append(O._RN[i + lo:i + hi + 1, j] - O.EW_R[i + lo:i + hi + 1])
    m = np.mean(np.array(M), 0)
    return {int(k): round(float(v) * 100, 3) for k, v in zip(range(lo, hi + 1), m)}
out['AR_path_A'] = {c: pathA(g) for c, g in T[T.cls.isin(['STD', 'SC'])].groupby('cls')}

# P2 劑量：pressure（全部 MSCI 中位數切、類內中位數切）、E−10..E 漲幅（類內）
M = T[T.cls.isin(['STD', 'SC'])].copy()
pmed = float(M.pressure.median())
out['pressure_def'] = {'formula': 'pressure = K_class ÷ E−1 的 20 日均成交值', 'K_class_NTD': O.K_ADD, 'median_select_MSCI_add': pmed}
M['p_hi'] = M.pressure >= pmed
M['p_hi_in'] = M.groupby('cls').pressure.transform(lambda s: s >= s.median())
M['run_hi_in'] = M.groupby('cls').car_runup.transform(lambda s: s >= s.median())
M['liq_lo_in'] = M.groupby('cls').v20.transform(lambda s: s < s.median())
M['ae_hi_in'] = M.groupby('cls').car_A_E.transform(lambda s: s >= s.median())
def split(col):
    r = {}
    for k, g in M.groupby(col):
        r[str(bool(k))] = {'n_by_cls': g.groupby('cls').size().to_dict(), 'car_p1_5': O.clus(g.car_p1_5, g.grp), 'car_p1_20': O.clus(g.car_p1_20, g.grp)}
    # 同審核配對差（高 − 低）
    a = M[M[col]].groupby('grp').car_p1_5.mean(); b = M[~M[col]].groupby('grp').car_p1_5.mean()
    d = (a - b).dropna()
    r['paired_hi_minus_lo_p1_5'] = {'mean%': round(float(d.mean()) * 100, 2), 't': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if len(d) > 2 else None, 'n_reviews': int(len(d))}
    return r
out['P2_pressure_overall_median'] = split('p_hi')
out['P2_pressure_within_class'] = split('p_hi_in')
out['P2_runup_within_class'] = split('run_hi_in')
out['P3_low_liquidity_within_class'] = split('liq_lo_in')
out['P3_ann_to_E_runup_within_class'] = split('ae_hi_in')
out['spearman_select'] = {c: {'pressure_vs_p1_5': round(float(g[['pressure', 'car_p1_5']].corr('spearman').iloc[0, 1]), 3),
                              'runup_vs_p1_5': round(float(g[['car_runup', 'car_p1_5']].corr('spearman').iloc[0, 1]), 3),
                              'pressure_vs_vol_ratio_E': round(float(g[['pressure', 'vol_ratio_E']].corr('spearman').iloc[0, 1]), 3)} for c, g in M.groupby('cls')}
out['vol_ratio_E_median'] = T.groupby('cls').vol_ratio_E.median().round(2).to_dict()

# 對照組：同一生效日、前 300 流動性池中非納入股，E−10..E 漲幅在池內前 20%（「近期漲多」的反轉基準）
def winners_control(reviews):
    rows = []
    allinc = set(zip(INC.review, INC.code))
    for rv in reviews:
        E = INC[INC.review == rv].eff.iloc[0]; i = IDX.get_loc(E)
        v60 = O.V60[i - 1]; px = K.Cn[i - 1]
        ok = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10) & K.TRADED[i]
        order = np.argsort(-np.where(ok, v60, -np.inf)); top = order[:300]
        top = [j for j in top if ok[j] and (rv, COLS[j]) not in allinc]
        ru = np.array([O.ar(j, i - 10, i) for j in top]); p1 = np.array([O.ar(j, i + 1, i + 5) for j in top])
        q = np.quantile(ru, 0.8)
        for j, a, b in zip(top, ru, p1):
            if a >= q: rows.append(dict(grp=rv, code=COLS[j], runup=a, car_p1_5=b))
    return pd.DataFrame(rows)
WC = winners_control(sorted(M.grp.unique()))
out['control_recent_winners_top20pct_runup'] = {'car_p1_5': O.clus(WC.car_p1_5, WC.grp), 'runup_mean%': round(float(WC.runup.mean()) * 100, 2),
                                                'note': '同生效日、前 300 流動性池、非 MSCI 納入、E−10..E 異常漲幅前 20%'}
out['MSCI_add_runup_mean%'] = {c: round(float(g.car_runup.mean()) * 100, 2) for c, g in M.groupby('cls')}
T.to_csv(O.HERE / 'explore_events_select.csv', index=False)

# ---------------- (b) 主策略層級 ----------------
RT = O.review_table()
mains = {}
hits_all = []
for n in (10, 30):
    S = O.strategy(True)
    pl, _ = O.plan_mod(S, n)
    ex = [p[0] for p in pl]
    t0, t1, sd = S.bounds()
    for k, (e, pk, pool, d, ranked) in enumerate(pl):
        e1 = ex[k + 1] if k + 1 < len(ex) else t1
        held = [COLS[j] for j in pk]
        h = INC[(INC.iE >= e) & (INC.iE <= e1 - 1) & INC.code.isin(held)]
        rank = {COLS[j]: r + 1 for r, j in enumerate(ranked[:200])}
        for r in h.itertuples():
            j = CI[r.code]; iE = r.iE
            A = K.MYADJ.to_numpy()
            # E 收盤 → 下一成交日開盤（同 o1 P4）
            def r_e_next(q):
                a0 = A[iE, q]; o1 = K.On[e1, q] * (A[e1, q] / K.Cn[e1, q]) if K.TRADED[e1, q] else A[e1 - 1, q]
                return o1 / a0 - 1 if np.isfinite(a0) and np.isfinite(o1) else np.nan
            Ri = r_e_next(j); Rp = np.nanmean([r_e_next(CI[c]) for c in held if c != r.code])
            hits_all.append(dict(n=n, sig=d, exec=IDX[e], E=r.eff, code=r.code, name=r.name, cls=r.cls, review=r.review, rank=rank.get(r.code),
                                 known_at_exec=bool(pd.notna(r.ann_tw) and r.ann_tw <= IDX[e]), src3=bool(r.src3), period=r.period,
                                 R_i=Ri, R_peers=Rp, diff=Ri - Rp, car_p1_5=O.ar(j, iE + 1, iE + 5), car_runup=O.ar(j, iE - 10, iE)))
H = pd.DataFrame(hits_all)
H.to_csv(O.HERE / 'explore_main_hits_all_counts.csv', index=False, columns=['n', 'sig', 'exec', 'E', 'code', 'name', 'cls', 'review', 'rank', 'known_at_exec', 'src3', 'period'])
cnt = {}
for n, g in H.groupby('n'):
    cnt[f'n{n}'] = {'by_cls_period': g.groupby(['cls', 'period']).size().unstack(fill_value=0).to_dict(),
                    'MSCI_STD_SC_by_year': g[g.cls.isin(['STD', 'SC'])].groupby(g.E.dt.year).size().to_dict(),
                    'MSCI_known_at_exec_by_period': {f'{a}|known={b}': int(v) for (a, b), v in g[g.cls.isin(['STD', 'SC'])].groupby(['period', 'known_at_exec']).size().items()},
                    'MSCI_event_months_by_period': g[g.cls.isin(['STD', 'SC'])].groupby('period').review.nunique().to_dict(),
                    'STD_event_months_by_period': g[g.cls == 'STD'].groupby('period').review.nunique().to_dict()}
out['main_hits_counts_all_periods_no_returns'] = cnt
Hs = H[H.period == 'select'].copy(); Hs['grp'] = Hs.sig.astype(str)
out['main_hits_select'] = {}
for n, g in Hs.groupby('n'):
    o = {}
    for c, gg in g.groupby('cls'):
        o[c] = {'n': int(len(gg)), 'rank_median': float(gg['rank'].median()), 'diff_E_to_next_exec': O.clus(gg['diff'], gg.grp),
                'car_runup_E-10_E': O.clus(gg.car_runup, gg.grp), 'car_p1_5': O.clus(gg.car_p1_5, gg.grp),
                'list': gg[['code', 'name', 'review', 'rank', 'known_at_exec']].astype(str).values.tolist()}
    m = g[g.cls.isin(['STD', 'SC'])]
    o['MSCI_STD_SC'] = {'n': int(len(m)), 'diff_E_to_next_exec': O.clus(m['diff'], m.grp)}
    out['main_hits_select'][f'n{n}'] = o
Hs.to_csv(O.HERE / 'explore_main_hits_select.csv', index=False)

# A 規則會跳過的（公告已知且未生效、原本會入選）——全期只數筆數
askip = {}
for n in (10, 30):
    S = O.strategy(True)
    for cls_set, nm in ((('STD',), 'A1'), (('STD', 'SC'), 'A2')):
        _, sk = O.plan_mod(S, n, INC, cls_set)
        askip[f'n{n}_{nm}'] = {'n_skips': sum(len(x[1]) for x in sk), 'n_months': len(sk),
                               'select_months': sum(1 for x in sk if x[0] <= '2018-12-31'), 'select_skips': sum(len(x[1]) for x in sk if x[0] <= '2018-12-31'),
                               'select_list': [x for x in sk if x[0] <= '2018-12-31']}
out['A_rule_skips_counts'] = askip

# P6（挑選期）：主策略 n10 前 20 名單股貢獻中，屬於持有期遇到 MSCI 納入者
S = O.strategy(True); pl10 = S.plan(10, 'official'); t0, t1, sd = S.bounds()
con, per = V.stock_contrib(pl10, t1, a='2010-11-01', b='2018-12-31')
top20 = sorted(con.items(), key=lambda x: -x[1])[:20]
inc_codes_sel = set(Hs[(Hs.n == 10) & Hs.cls.isin(['STD', 'SC'])].code)
out['P6_select'] = {'top20': [(c, V.SL['name'].get(c, ''), round(v * 100, 1), c in inc_codes_sel) for c, v in top20],
                    'n_top20_that_were_MSCI_inclusion_hits': int(sum(c in inc_codes_sel for c, _ in top20))}
json.dump(out, open(O.HERE / 'explore.json', 'w'), ensure_ascii=False, indent=1, default=str)

# 列印摘要
for c, b in out['CAR_by_class'].items():
    print(c, {w: (v.get('n'), v.get('mean_clustered%'), v.get('t_clustered')) for w, v in b.items()})
for k in ['P2_pressure_overall_median', 'P2_pressure_within_class', 'P2_runup_within_class', 'P3_low_liquidity_within_class', 'P3_ann_to_E_runup_within_class']:
    print(k, {kk: (vv['car_p1_5'] if isinstance(vv, dict) and 'car_p1_5' in vv else vv) for kk, vv in out[k].items()})
print('control', out['control_recent_winners_top20pct_runup'], out['MSCI_add_runup_mean%'])
print(json.dumps(out['main_hits_counts_all_periods_no_returns'], default=str, ensure_ascii=False))
for n, o in out['main_hits_select'].items():
    for c, v in o.items(): print(n, c, {k: vv for k, vv in v.items() if k != 'list'})
print(json.dumps(out['A_rule_skips_counts'], ensure_ascii=False, default=str))
print(out['P6_select'])

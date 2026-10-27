#!/usr/bin/env python3
"""樣本外（只能在 PREREG 之後跑）：事件層級 P1～P4（兩期）、主策略層級 B2 與鄰近版本（驗證／保留／合併／跨循環）、P5、P6、穩健性。"""
import sys, os, json, math, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, 'results.json')))
assert 'PREREG' in RES and RES['PREREG']['sha256'] == hashlib.sha256(RES['PREREG']['text'].encode('utf-8')).hexdigest(), '先執行 prereg.py'
import numpy as np, pandas as pd
import oa_lib as O
import vers as W
K, V = O.K, O.V
IDX, COLS, CI = O.IDX, O.COLS, O.CI
out = {}

# ================= 事件層級 =================
INC = W.INC.copy()
T = O.event_metrics(INC)
T = T[T.ok == True].copy()
T['grp'] = T.eff.dt.strftime('%Y-%m-%d')
def blk(g):
    return {w: O.clus(g[w], g.grp) for w in ['car_preA10', 'ar_A', 'car_A_E', 'car_pre10', 'ar_E', 'car_p1_5', 'car_p1_10', 'car_p1_20', 'bhar_p1_5'] if w in g}
main = T[~T.src3]
out['events_by_class_period'] = {f'{c}|{p}': blk(g) for (c, p), g in main.groupby(['cls', 'period'])}
out['events_src3_only'] = {c: blk(g) for c, g in T[T.src3].groupby('cls')}
out['events_by_class_all_excl_src3'] = {c: blk(g) for c, g in main.groupby('cls')}
# P1
p1 = {}
for p in ('select', 'oos'):
    g = main[(main.cls == 'STD') & (main.period == p)]
    s = O.clus(g.car_p1_5, g.grp); p1[p] = s
P1_ok = all(p1[p]['mean_clustered%'] <= -2 and (p1[p]['t_clustered'] or 0) <= -2 for p in p1)
P1_wrong = any(p1[p]['mean_clustered%'] > -1 for p in p1)
# P4
p4 = {}
for c in ('TW50', 'HDETF'):
    g = main[main.cls == c]
    p4[c] = {'all': O.clus(g.car_p1_5, g.grp), **{p: O.clus(gg.car_p1_5, gg.grp) for p, gg in g.groupby('period')},
             'vol_ratio_E_median': round(float(g.vol_ratio_E.median()), 2), 'runup_mean%': round(float(g.car_runup.mean()) * 100, 2)}
P4_ok = all(abs(p4[c]['all']['mean_clustered%']) < 1 for c in p4)
out['P1'] = {'stats': p1, 'pass': bool(P1_ok), 'mechanism_wrong_flag(any >-1%)': bool(P1_wrong)}
out['P4_placebo'] = {'stats': p4, 'pass': bool(P4_ok)}
out['event_level_pass'] = bool(P1_ok and P4_ok)
# P2/P3（樣本外）
M = main[main.cls.isin(['STD', 'SC'])].copy()
def split(df, col):
    r = {}
    for k, g in df.groupby(col):
        r[str(bool(k))] = {'n_by_cls': g.groupby('cls').size().to_dict(), 'car_p1_5': O.clus(g.car_p1_5, g.grp)}
    a = df[df[col]].groupby('grp').car_p1_5.mean(); b = df[~df[col]].groupby('grp').car_p1_5.mean(); d = (a - b).dropna()
    r['paired_hi_minus_lo'] = {'mean%': round(float(d.mean()) * 100, 2), 't': round(float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))), 2) if len(d) > 2 else None, 'n': int(len(d))}
    return r
for p in ('select', 'oos'):
    D = M[M.period == p].copy()
    D['p_hi_in'] = D.groupby('cls').pressure.transform(lambda s: s >= s.median())
    D['run_hi_in'] = D.groupby('cls').car_runup.transform(lambda s: s >= s.median())
    D['liq_lo_in'] = D.groupby('cls').v20.transform(lambda s: s < s.median())
    D['ae_hi_in'] = D.groupby('cls').car_A_E.transform(lambda s: s >= s.median())
    out[f'P2P3_{p}'] = {k: split(D, k) for k in ['p_hi_in', 'run_hi_in', 'liq_lo_in', 'ae_hi_in']}
    out[f'P3_STD_vs_SC_{p}'] = {c: O.clus(g.car_p1_5, g.grp) for c, g in D.groupby('cls')}
# 近期漲多對照組（樣本外）
def winners_control(reviews):
    rows = []; allinc = set(zip(INC.review, INC.code))
    for rv in reviews:
        E = INC[INC.review == rv].eff.iloc[0]; i = IDX.get_loc(E)
        v60 = O.V60[i - 1]; px = K.Cn[i - 1]
        ok = np.isfinite(v60) & (v60 >= 1e7) & np.isfinite(px) & (px >= 10) & K.TRADED[i]
        order = np.argsort(-np.where(ok, v60, -np.inf))[:300]
        top = [j for j in order if ok[j] and (rv, COLS[j]) not in allinc]
        ru = np.array([O.ar(j, i - 10, i) for j in top]); p1_ = np.array([O.ar(j, i + 1, i + 5) for j in top]); q = np.quantile(ru, 0.8)
        rows += [dict(grp=rv, car_p1_5=b) for a, b in zip(ru, p1_) if a >= q]
    return pd.DataFrame(rows)
for p in ('select', 'oos'):
    WC = winners_control(sorted(M[M.period == p].review.unique()))
    out[f'control_recent_winners_{p}'] = O.clus(WC.car_p1_5, WC.grp)
# 補充事件
SUP = pd.read_csv(os.path.join(HERE, 'supp_events.csv'), dtype={'code': str}, parse_dates=['eff', 'ann_tw'])
SUP['iE'] = [IDX.get_loc(e) for e in SUP.eff]; SUP['iA'] = [O.cal_on_or_after(a) for a in SUP.ann_tw]
SUP['src3'] = False; SUP['period'] = 'oos'; SUP['in_panel'] = True
TS = O.event_metrics(SUP)
out['supplement_events'] = TS[['code', 'name', 'eff', 'ok', 'car_pre10', 'ar_E', 'car_p1_5', 'car_A_E']].astype(str).values.tolist()
T.to_csv(os.path.join(HERE, 'oos_events_all.csv'), index=False)

# ================= 主策略層級 =================
ew = W.ew_nav(10, W.T1_FULL)
PER = {'select': ('2010-11-01', '2018-12-31'), 'val': ('2019-01-01', '2023-12-31'), 'hold': ('2024-01-01', '2026-10-12'),
       'oos': ('2019-01-01', '2026-10-12'), 'ALL': ('2006-04-01', '2026-10-12'), 'S3_2021': ('2021-03-01', '2026-10-12')}
strat = {}; navs = {}
INC_nosrc3 = INC[~INC.src3].copy()
SUPI = SUP.assign(event_id=SUP.event_id, review=SUP.review)
INC_sup = pd.concat([INC, SUPI[['event_id', 'review', 'cls', 'code', 'name', 'eff', 'iE', 'ann_tw', 'iA', 'src3', 'period', 'in_panel']]], ignore_index=True)
def evaluate(tag, r, nb, plb):
    aff = W.affected_periods(r, plb)
    o = {}
    for p, (a, b) in PER.items():
        sm = W.seg(r['nav'], ew, a, b); sb = W.seg(nb, ew, a, b)
        pr = W.paired(r['nav'], nb, W.SD, aff, a, b)
        # 對原版主策略的 t（訊號窗配對差，全部月份）
        pm, pb = W.period_returns(r['nav'], W.SD), W.period_returns(nb, W.SD)
        d = (pm - pb).dropna(); d = d[(d.index >= pd.Timestamp(a)) & (d.index <= pd.Timestamp(b))]
        t_main = float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))) if d.std(ddof=1) > 0 else float('nan')
        o[p] = {'CAGR': sm['CAGR'], 'maxDD': sm['maxDD'], 'base_CAGR': sb['CAGR'], 'base_maxDD': sb['maxDD'], 'dCAGR_pp': round(sm['CAGR'] - sb['CAGR'], 3),
                'dMaxDD_pp': round(sm['maxDD'] - sb['maxDD'], 3), 't_0050': sm['t_0050'], 't_EW': sm['t_EW'], 'base_t_0050': sb['t_0050'],
                'E50_CAGR': sm['E50_CAGR'], 'EW_CAGR': sm['EW_CAGR'], 't_vs_main_all_months': round(t_main, 2), 'paired': pr}
    return o, aff
for n in (10, 30):
    for cost in ('normal', 'stress'):
        nb, plb = W.base(n, cost)
        navs[f'BASE|n{n}|{cost}'] = nb
        for v in W.VERSIONS:
            r = W.run(v, n, cost)
            o, aff = evaluate(v, r, nb, plb)
            o['events'] = [x for x in r['log'] if x[0] in ('sell_E', 'postpone', 'buy_0050', 'buy_repl')]
            o['skips'] = r['skips']
            strat[f'{v}|n{n}|{cost}'] = o
            navs[f'{v}|n{n}|{cost}'] = r['nav']
            if v == 'B2':
                print(v, n, cost, {p: (o[p]['CAGR'], o[p]['dCAGR_pp'], o[p]['dMaxDD_pp'], o[p]['paired']['n_affected_months'], o[p]['paired']['mean_affected%'], o[p]['paired']['t_affected']) for p in PER}, flush=True)
        # 敏感度（B2）：src 3 期間不執行；加入補充事件
        if cost == 'normal':
            for tag, inc in (('B2_no_src3', INC_nosrc3), ('B2_plus_supp', INC_sup)):
                r = W.run('B2', n, cost, inc=inc); o, _ = evaluate(tag, r, nb, plb)
                o['events'] = [x for x in r['log'] if x[0] in ('sell_E', 'postpone')]
                strat[f'{tag}|n{n}|{cost}'] = o
out['strategy'] = strat
pd.DataFrame(navs).to_parquet(os.path.join(HERE, 'navs.parquet'))

# 判定 (2)
it = json.load(open(os.path.join(HERE, 'iterations.json')))
sel = it['results']['B2|n10|normal']['paired']
o = strat['B2|n10|normal']
pa = o['oos']['paired']['affected_list']
vals = sorted(pa.values())
drop_best = (sum(vals) - max(vals)) / (len(vals) - 1) if len(vals) > 1 else float('nan')
c2 = {'a_select_diff_gt0': (sel['mean_affected%'] or 0) > 0, 'a_select_months_ge8': sel['n_affected_months'] >= 8,
      'b_oos_diff_gt0': (o['oos']['paired']['mean_affected%'] or 0) > 0,
      'c_ALL_CAGR_ge_base': o['ALL']['dCAGR_pp'] >= 0, 'c_maxDD_not_deeper_1pp': o['ALL']['dMaxDD_pp'] >= -1.0,
      'd_drop_best_month_gt0': drop_best > 0, 'drop_best_mean%': round(drop_best, 3)}
out['strategy_conditions_B2_n10'] = c2
# 檢定力
x = pd.Series(list(pa.values())) / 100
sel_list = pd.Series(list(sel['affected_list'].values())) / 100
allaff = pd.concat([x, sel_list])
sd_ = float(allaff.std(ddof=1)) if len(allaff) > 1 else float('nan')
out['power'] = {'n_affected_months_select': int(len(sel_list)), 'n_affected_months_oos': int(len(x)), 'sd_affected_paired%': round(sd_ * 100, 3),
                'MDE_oos%(2.8*sd/sqrt(n))': round(2.8 * sd_ / math.sqrt(max(len(x), 1)) * 100, 3),
                'MDE_all%': round(2.8 * sd_ / math.sqrt(len(allaff)) * 100, 3), 'n_events_lt_15': bool(len(allaff) < 15)}
EV = out['event_level_pass']
if not EV: verdict = '不可用'
else:
    rest = c2['b_oos_diff_gt0'] and c2['c_ALL_CAGR_ge_base'] and c2['c_maxDD_not_deeper_1pp'] and c2['d_drop_best_month_gt0']
    if rest and c2['a_select_diff_gt0'] and c2['a_select_months_ge8']: verdict = '待觀察'
    elif rest and c2['a_select_diff_gt0']: verdict = '待觀察（檢定力不足，只做前瞻紙上追蹤）'
    else: verdict = '主策略層級不可用（事件層級留紙上追蹤）'
out['verdict_by_prereg_table'] = verdict

# ================= P5 =================
H = []
for n in (10, 30):
    pl, _ = O.plan_mod(W.S, n); ex = [p[0] for p in pl]
    A_ = K.MYADJ.to_numpy()
    for k, (e, pk, pool, d, ranked) in enumerate(pl):
        e1 = ex[k + 1] if k + 1 < len(ex) else W.T1_FULL
        held = [COLS[j] for j in pk]
        h = INC[(INC.iE >= e) & (INC.iE <= e1 - 1) & INC.code.isin(held)]
        def rn(q, iE):
            a0 = A_[iE, q]; o1 = K.On[e1, q] * (A_[e1, q] / K.Cn[e1, q]) if (e1 < len(IDX) and K.TRADED[e1, q]) else A_[e1 - 1, q]
            return o1 / a0 - 1 if np.isfinite(a0) and np.isfinite(o1) else np.nan
        for r in h.itertuples():
            j = CI[r.code]; Ri = rn(j, r.iE); Rp = np.nanmean([rn(CI[c], r.iE) for c in held if c != r.code])
            r50 = O.E50[e1 - 1] / O.E50[r.iE] - 1
            H.append(dict(n=n, sig=str(d.date()), E=str(r.eff.date()), code=r.code, name=r.name, cls=r.cls, src3=bool(r.src3), period=r.period,
                          R_i=Ri, R_peers=Rp, diff=Ri - Rp, r50=r50, known_at_exec=bool(r.ann_tw <= IDX[e]), rank=int(list(ranked).index(j) + 1)))
H = pd.DataFrame(H); H.to_csv(os.path.join(HERE, 'p5_hits.csv'), index=False)
p5 = {}
for n, g in H.groupby('n'):
    for c in (('STD',), ('STD', 'SC'), ('TW50',), ('HDETF',)):
        gg = g[g.cls.isin(c)]
        for p in ('select', 'oos', 'all'):
            g3 = gg if p == 'all' else gg[gg.period == p]
            p5[f'n{n}|{"+".join(c)}|{p}'] = O.clus(g3['diff'], g3.sig)
out['P5'] = p5
# P6：跨循環前 20 名單股貢獻
pl10 = W.S.plan(10, 'official')
con, per = V.stock_contrib(pl10, W.T1_FULL, a='2006-04-01', b='2026-10-12')
top20 = sorted(con.items(), key=lambda x: -x[1])[:20]
std_hit = set(H[(H.n == 10) & (H.cls == 'STD')].code); any_hit = set(H[(H.n == 10) & H.cls.isin(['STD', 'SC'])].code)
out['P6'] = {'top20': [(c, V.SL['name'].get(c, ''), round(v * 100, 1), c in std_hit, c in any_hit) for c, v in top20],
             'n_top20_STD_hits': int(sum(c in std_hit for c, _ in top20)), 'n_top20_MSCI_hits': int(sum(c in any_hit for c, _ in top20))}
# 每筆 B2 賣出：E→下一成交日，賣出股 vs 0050（扣額外成本 0.5%＋0050 0.19%）
g = H[(H.n == 10) & (H.cls == 'STD')].copy()
g['gain_from_rule'] = g.r50 - g.R_i - 0.005 - 0.0019
out['P6_decomposition_n10_STD'] = {'n': int(len(g)), 'savings_sum%(rule helped)': round(float(g.gain_from_rule[g.gain_from_rule > 0].sum()) * 100, 2),
                                    'lost_sum%(rule hurt)': round(float(g.gain_from_rule[g.gain_from_rule <= 0].sum()) * 100, 2),
                                    'per_event': g[['sig', 'E', 'code', 'name', 'period', 'src3', 'R_i', 'r50', 'gain_from_rule']].round(4).astype(str).values.tolist(),
                                    'note': '單位是「該股部位報酬」，乘上約 1/10 權重才是組合影響'}
# 穩健性：分年、分產業、去掉最大貢獻事件
g['year'] = g.E.str[:4]
g['ind'] = [V.SL['industry'].get(c, '') for c in g.code]
out['robust_n10_STD'] = {'by_year': g.groupby('year').gain_from_rule.agg(['count', 'mean']).round(4).to_dict('index'),
                         'by_industry': g.groupby('ind').gain_from_rule.agg(['count', 'mean']).round(4).to_dict('index'),
                         'drop_best_event_mean': round(float(g.gain_from_rule.drop(g.gain_from_rule.idxmax()).mean()), 4) if len(g) > 1 else None,
                         'drop_worst_event_mean': round(float(g.gain_from_rule.drop(g.gain_from_rule.idxmin()).mean()), 4) if len(g) > 1 else None}
RES = json.load(open(os.path.join(HERE, 'results.json')))
RES['OOS'] = out
json.dump(RES, open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: out[k] for k in ['P1', 'P4_placebo', 'event_level_pass', 'strategy_conditions_B2_n10', 'power', 'verdict_by_prereg_table']}, ensure_ascii=False, indent=1, default=str))

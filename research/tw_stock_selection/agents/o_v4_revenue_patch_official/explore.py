#!/usr/bin/env python3
"""探索（不跑任何策略淨值）。只用挑選期（訊號 2006-04～2018-12）做報酬類描述統計；2019 起只報筆數。
(1) 下市股缺營收月數表：修補前（R_DEL）與修補後（R_FIX）並列，依池內缺口股票月加權。
(2) 新來源 N 與面板重疊月的一致率（取自 iterations_data.json）。
(3) 每月合格池增加幾檔、增加的股票落在什麼分數位置（R_FIX、B_Q 季代理、B05、B10 各自放回時的綜合分數名次）。
(4) 機制：池內缺營收下市股的下月報酬（成交日開盤 → 下一成交日開盤）減同池等權，按訊號月分群；依 52 週比百分位、季代理營收百分位、終止型態切分。
(5) 處置日口徑：成交日口徑與訊號日口徑不同的名額（rem=0、公告日＝訊號日），以及其是否會進 n10（不看報酬）。
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V = L.K, L.V

fz = L.check_freeze()
XR = L.setup_px()
YRD = L.load_rdel(); YFX, Y5F = L.load_rfix(); Y5 = L.y5_orig()
GAP = L.gap_dict()
QP = L.qproxy_fn()
dm_t, dm_s = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
E = L._STATE['E']
SELB = pd.Timestamp('2018-12-31')
out = {}

# ---------- (1) 缺營收月數表 ----------
O = pd.read_csv(HERE / 'gap_delisted.csv', dtype={'code': str})
Gc = pd.read_csv(HERE / 'gap_pool_stockmonths_cov.csv', dtype={'code': str})
sel = fz['merge_rule_selected']
O['pool_gap_after'] = O.code.map(lambda c: int(((Gc.code == c) & ~Gc['ok_' + sel]).sum()))
O['pool_gap_filled'] = O.pool_gap_sigs - O.pool_gap_after
cells = pd.read_csv(HERE / 'rfix_cells.csv', dtype={'code': str})
O['rfix_yoy_cells'] = O.code.map(cells.code.value_counts()).fillna(0).astype(int)
O['type'] = O.code.map(lambda c: E.loc[c, 'type'] if c in E.index else '')
O.to_csv(HERE / 'gap_delisted_before_after.csv', index=False)
top5 = ['2325', '2311', '3474', '6286', '2473']
out['1_gap_table'] = {
    'n_delisted_2006plus': int(len(O)), 'n_with_missing_level_months': int((O.miss_level > 0).sum()), 'n_all_missing': int((O.miss_level == O.months).sum()),
    'n_missing_ge3_months': int((O.miss_level >= 3).sum()), 'n_with_pool_gap': int((O.pool_gap_sigs > 0).sum()),
    'pool_gap_stockmonths_before': int(O.pool_gap_sigs.sum()), 'pool_gap_stockmonths_after_RFIX': int(O.pool_gap_after.sum()),
    'priority5': O[O.code.isin(top5)][['code', 'name', 'delist', 'miss_level', 'pool_gap_sigs', 'pool_gap_after', 'rfix_yoy_cells', 'pool_gap_s52pct_mean']].to_dict(orient='records'),
    'top25_by_pool_gap': O.head(25)[['code', 'name', 'delist', 'type', 'miss_level', 'pool_gap_sigs', 'pool_gap_after', 'rfix_yoy_cells', 'pool_gap_s52pct_mean']].to_dict(orient='records')}

# ---------- (2) 一致率 ----------
itd = json.loads((HERE / 'iterations_data.json').read_text(encoding='utf-8'))
out['2_overlap_agreement'] = {v['version']: {**v['gate'], 'passed': v['passed']} for v in itd['versions']}
out['2_quarterly_vs_monthly_sum_agree_0.5pct'] = 0.846
out['2_qproxy_validation'] = fz['qproxy']['validation_survivors_2014_04_2018_12']

# ---------- (3) 合格池增加與分數位置 ----------
def rank_if_added(cs, j, prr_val):
    el = cs['el'].copy(); el[j] = True
    s52 = cs['s52']
    pr52 = pd.Series(np.where(el, s52, np.nan)).rank(pct=True).to_numpy()
    sc_j = 0.5 * pr52[j] + 0.5 * prr_val
    others = cs['sc'][cs['el']]
    return int((others > sc_j).sum()) + 1
S = L.V4(YFX, Y5F, dm_t)            # R_FIX、成交日口徑
S0 = L.V4(YRD, Y5, dm_t)            # R_DEL
rows = []
for dd in S.sig_days:
    cs, c0 = S.cross_section(dd), S0.cross_section(dd)
    add_fix = np.where(cs['el'] & ~c0['el'])[0]
    ok = cs['ok_other']; y3 = cs['y3']
    gmask = np.array([c in GAP and dd <= GAP[c] for c in K.COLS]) & ok & ~np.isfinite(y3)
    yp = QP(dd)
    vals = np.sort(y3[cs['el']])
    for j in np.where(gmask)[0]:
        rq = float(np.searchsorted(vals, yp[j], side='right') / len(vals)) if np.isfinite(yp[j]) else np.nan
        rows.append(dict(sig=str(dd.date()), code=K.COLS[j], kind='gap', s52_pct=float(pd.Series(np.where(ok, cs['s52'], np.nan)).rank(pct=True).iloc[j]),
                         rank_B05=rank_if_added(cs, j, 0.5), rank_B10=rank_if_added(cs, j, 1.0),
                         qproxy_pct=rq, rank_BQ=rank_if_added(cs, j, rq) if np.isfinite(rq) else np.nan, pool=int(cs['el'].sum())))
    for j in add_fix:
        r = int((cs['sc'][cs['el']] > cs['sc'][j]).sum()) + 1
        rows.append(dict(sig=str(dd.date()), code=K.COLS[j], kind='rfix_added', s52_pct=np.nan, rank_B05=np.nan, rank_B10=np.nan, qproxy_pct=np.nan,
                         rank_BQ=np.nan, rank_RFIX=r, pool=int(cs['el'].sum())))
A = pd.DataFrame(rows)
A.to_csv(HERE / 'explore_pool_additions.csv', index=False)
A['y'] = A.sig.str[:4].astype(int)
sig_all = [str(d.date()) for d in S.sig_days]
def per_day(mask, lo, hi):
    ds = [s for s in sig_all if lo <= int(s[:4]) <= hi]
    return float(A[mask & A.sig.isin(ds)].groupby('sig').size().reindex(ds, fill_value=0).mean())
g = A.kind == 'gap'
out['3_pool_additions_per_signal_day'] = {
    'RFIX_added_2006_2012': per_day(A.kind == 'rfix_added', 2006, 2012), 'RFIX_added_2013_2018': per_day(A.kind == 'rfix_added', 2013, 2018),
    'RFIX_added_2019_2026': per_day(A.kind == 'rfix_added', 2019, 2026),
    'remaining_gap_2006_2012': per_day(g, 2006, 2012), 'remaining_gap_2013_2018': per_day(g, 2013, 2018), 'remaining_gap_2019_2026': per_day(g, 2019, 2026),
    'BQ_covered_2013_2018': per_day(g & A.qproxy_pct.notna(), 2013, 2018),
    'RFIX_added_rank_list': A[A.kind == 'rfix_added'][['sig', 'code', 'rank_RFIX', 'pool']].to_dict(orient='records'),
    'gap_rank_le10_B05': int((g & (A.rank_B05 <= 10)).sum()), 'gap_rank_le10_B10': int((g & (A.rank_B10 <= 10)).sum()),
    'gap_rank_le10_BQ': int((g & (A.rank_BQ <= 10)).sum()),
    'gap_rank_le10_BQ_by_code': A[g & (A.rank_BQ <= 10)].code.value_counts().to_dict(),
    'gap_rank_le10_B10_by_year': A[g & (A.rank_B10 <= 10)].groupby('y').size().to_dict(),
    'gap_rank_le10_B05_by_year': A[g & (A.rank_B05 <= 10)].groupby('y').size().to_dict(),
    'gap_rank_le10_BQ_by_year': A[g & (A.rank_BQ <= 10)].groupby('y').size().to_dict(),
    'gap_qproxy_pct_mean_2013_2018': float(A[g & (A.y.between(2013, 2018))].qproxy_pct.mean())}

# ---------- (4) 機制：缺營收下市股的下月超額（挑選期）----------
plan_days = S.sig_days
e_of = {d: K.IDX.get_loc(d) + 1 for d in plan_days}
def hold_ret(j, d, d2):
    e, e2 = e_of[d], e_of[d2]
    r = K.Cn[e, j] / K.On[e, j]
    r *= np.prod(K.GROW[e + 1:e2, j])
    if K.TRADED[e2, j] and np.isfinite(K.On[e2, j]): r *= K.GROW[e2, j] * K.On[e2, j] / K.Cn[e2, j]
    elif j in XR: r *= XR[j][0]
    return r - 1
ev = []
for i, d in enumerate(plan_days[:-1]):
    if d > SELB: break
    d2 = plan_days[i + 1]
    cs = S.cross_section(d); el = cs['el']
    pool = np.where(el)[0]
    if len(pool) < 50: continue
    ew = np.nanmean([hold_ret(j, d, d2) for j in pool])
    sub = A[(A.sig == str(d.date())) & (A.kind == 'gap')]
    for _, r in sub.iterrows():
        j = K.CI[r.code]
        ev.append(dict(sig=r.sig, code=r.code, ex=hold_ret(j, d, d2) - ew, s52_pct=r.s52_pct, qproxy_pct=r.qproxy_pct, rank_B10=r.rank_B10,
                       typ=E.loc[r.code, 'type'] if r.code in E.index else 'none'))
EV = pd.DataFrame(ev)
EV.to_csv(HERE / 'explore_events.csv', index=False)
def clus(x):
    m = x.groupby('sig').ex.mean()
    return dict(n_events=int(len(x)), n_months=int(len(m)), mean_pp=float(m.mean() * 100), t=L.tstat(m.to_numpy()), median_event_pp=float(x.ex.median() * 100))
out['4_gap_stock_next_month_excess_vs_pool_EW_SEL'] = {
    'all': clus(EV), 's52_top_tercile': clus(EV[EV.s52_pct >= 2 / 3]), 's52_bottom_two': clus(EV[EV.s52_pct < 2 / 3]),
    'qproxy_ge_0.7': clus(EV[EV.qproxy_pct >= 0.7]), 'qproxy_lt_0.3': clus(EV[EV.qproxy_pct < 0.3]),
    'B10_rank_le10': clus(EV[EV.rank_B10 <= 10]), 'B10_rank_le30': clus(EV[EV.rank_B10 <= 30]),
    'type_cash': clus(EV[EV.typ == 'cash']), 'type_swap_or_transfer': clus(EV[EV.typ.isin(['swap', 'transfer'])]), 'type_failure': clus(EV[EV.typ == 'failure']),
    '2013_2018': clus(EV[EV.sig >= '2013-01-01']), '2006_2012': clus(EV[EV.sig < '2013-01-01']),
    'top5_codes': {c: clus(EV[EV.code == c]) for c in top5 if (EV.code == c).any()}}
# 併購標的在最後幾個換股日的 52 週比
mm = {}
for c in ('2325', '2311', '3474', '6286', '2473'):
    x = A[(A.code == c) & (A.kind == 'gap')].tail(6)
    mm[c] = x[['sig', 's52_pct', 'qproxy_pct', 'rank_B05', 'rank_B10', 'rank_BQ']].to_dict(orient='records')
out['4_priority5_last_signals'] = mm

# ---------- (5) 處置日口徑差異 ----------
St = L.V4(YFX, Y5F, dm_t); Ss = L.V4(YFX, Y5F, dm_s)
diff = []
for dd in St.sig_days:
    mt, ms = dm_t(dd), dm_s(dd)
    for j in np.where(mt != ms)[0]:
        c1, c2 = St.cross_section(dd), Ss.cross_section(dd)
        el = c1['el'] | c2['el']
        sc = np.where(c1['el'], c1['sc'], c2['sc'])
        r = int((np.nan_to_num(np.where(el, sc, np.nan), nan=-9) > sc[j]).sum()) + 1 if np.isfinite(sc[j]) else None
        diff.append(dict(sig=str(dd.date()), code=K.COLS[j], case='signal_only(rem=0)' if ms[j] else 'trade_only(announce=d,start=e)', score_rank=r))
DF = pd.DataFrame(diff)
DF.to_csv(HERE / 'explore_disposal_rule_diff.csv', index=False)
out['5_disposal_rule_diff'] = {'n_total': int(len(DF)), 'by_case': DF.case.value_counts().to_dict() if len(DF) else {},
                               'n_rank_le10': int((DF.score_rank.fillna(999) <= 10).sum()) if len(DF) else 0,
                               'rank_le10_list': DF[DF.score_rank.fillna(999) <= 10].to_dict(orient='records') if len(DF) else [],
                               'n_SEL_2006_2018': int((DF.sig <= '2018-12-31').sum()) if len(DF) else 0}
(HERE / 'explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
print(json.dumps({k: v for k, v in out.items() if k not in ('1_gap_table',)}, ensure_ascii=False, indent=1, default=float)[:9000])
print(json.dumps(out['1_gap_table']['priority5'], ensure_ascii=False))

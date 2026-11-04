#!/usr/bin/env python3
"""凍結後一次跑完：D1a-2006（留一錨點）、D1c、D2（含主策略金融旗標不一致計數）、D1-retro（路徑 B，o12 原協定）、P2 錯誤分布、P3 覆蓋。
先斷言 data/GATE_SPEC.json 存在且 sha256 與 results.json PREREG 相符。只用 close_raw/open/value/flag 與 raw/；不讀 adj、ret；不算任何報酬。"""
import sys, json, time, gzip
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import plib as P, pbuild as B
t00 = time.time()
res_p = P.HERE / 'results.json'
res = json.loads(res_p.read_text(encoding='utf-8'))
assert (P.DATA / 'GATE_SPEC.json').exists(), 'GATE_SPEC 不存在：不得計算閘門'
assert P.sha(P.DATA / 'GATE_SPEC.json') == res['PREREG']['sha256'], 'GATE_SPEC sha 不符'
SPEC = json.loads((P.DATA / 'GATE_SPEC.json').read_text(encoding='utf-8'))
VER = SPEC['construction']['version']
G = dict(spec_sha256=res['PREREG']['sha256'], version=VER)

A = B.load_anchor_rows(); M = B.merge_anchors(A)
tlo, thi = P.ti('2005-06-01'), P.ti('2021-06-30')
mult = B.event_multipliers(tlo, thi)
ADV = P.adv120()
S_full, SRC = B.build(M, tlo, thi, mult, VER)

# ---------------- D1a-2006 ----------------
cps = [P.ti(d) for d in SPEC['D1a_2006']['checkpoints']]
rows, det = B.loo_eval(M, cps, mult, VER, S_full, ADV)
E = pd.DataFrame(det); ov = np.array([r['overlap'] for r in rows])
th = SPEC['D1a_2006']['thresholds']
d1a = dict(pooled_med_abs_log=float(E.abs_log.median()), pooled_p90=float(E.abs_log.quantile(0.9)), overlap_median=float(np.median(ov)), overlap_min=int(ov.min()),
           n_err=len(E), by_q=rows)
d1a['pass'] = bool(d1a['pooled_med_abs_log'] <= th['pooled_med_abs_log_le'] and d1a['overlap_median'] >= th['overlap_median_ge'] and d1a['overlap_min'] >= th['overlap_min_ge'])
d1a['truthless_in_top100_mean'] = float(np.mean([r['n_truthless_in_top'] for r in rows]))
G['D1a_2006'] = d1a
print('D1a', {k: v for k, v in d1a.items() if k != 'by_q'}, round(time.time() - t00), flush=True)
# 誤差依名次桶
E['bucket'] = pd.cut(E['rank'], [0, 20, 60, 100], labels=['1-20', '21-60', '61-100'])
G['D1a_error_by_rank'] = E.groupby('bucket', observed=True).abs_log.agg(['median', 'count']).round(5).reset_index().astype(str).to_dict('records')
G['D1a_worst'] = E.sort_values('abs_log', ascending=False).head(20).round(4).to_dict('records')

# ---------------- D1c ----------------
crow = []; extrap = []
for d in P.month_last_days('2006-01-01', '2013-12-31'):
    t = P.ti(d); al = P.alive_at(t)
    mc = np.where(al, S_full[t] * P.CFF[t], np.nan); mc, imp = B.impute_mcap(t, mc, al, ADV)
    T = B.top_k(mc, 100)
    # 錨點距離（交易日）
    far = []
    for j in T:
        c = P.COLS[j]
        if imp[j]: continue
        ts = M.t[M.code == c].to_numpy()
        if len(ts) and np.min(np.abs(ts - t)) > 250: far.append(c)
    crow.append(dict(d=str(d.date()), n_imputed=int(imp[T].sum()), imputed=[P.COLS[j] for j in T if imp[j]], n_far_gt_1y=len(far), far=far))
CR = pd.DataFrame(crow); CR['y'] = CR.d.str[:4].astype(int)
d1c = dict(mean_imputed=float(CR.n_imputed.mean()), max_imputed_from_2008=int(CR[CR.y >= 2008].n_imputed.max()),
           by_year=CR.groupby('y').n_imputed.mean().round(3).to_dict(), imputed_codes=pd.Series([c for x in CR.imputed for c in x]).value_counts().to_dict(),
           far_gt_1y_by_year=CR.groupby('y').n_far_gt_1y.mean().round(3).to_dict(), far_codes=pd.Series([c for x in CR.far for c in x]).value_counts().head(20).to_dict())
d1c['pass'] = bool(d1c['mean_imputed'] <= SPEC['D1c']['thresholds']['mean_le'] and d1c['max_imputed_from_2008'] == 0)
G['D1c'] = d1c
print('D1c', {k: v for k, v in d1c.items() if k in ('mean_imputed', 'max_imputed_from_2008', 'pass', 'by_year')}, flush=True)

# ---------------- D2 ----------------
snaps = [s for s in B.load_snapshots() if not s['evidence'].startswith('C')]
SL = pd.read_csv(P.RAW / 'center' / 'stock_list_202610.csv', dtype=str).drop_duplicates('code').set_index('code')
FIN_MAIN = set(SL.index[SL['industry'] == '金融保險業']) | {c for c in P.COLS if c[:2] == '28' or c in ('5876', '5880')}
O12I = pd.read_csv(P.RAW / 'center' / 'industry_pit_o12.csv', dtype=str)
def o12_ind(c, d):
    x = O12I[(O12I.code == c) & (pd.to_datetime(O12I.start) <= d) & (pd.to_datetime(O12I.end.fillna('2099-12-31')) > d)]
    return x.industry.iloc[0] if len(x) else None
V60 = P.v60_main()
drows = []
for d in P.main_rebalance_days('2006-01-01', '2013-12-31'):
    t = P.IDX.get_loc(d); v = V60[t]; px = P.Cn[t]
    base = np.isfinite(v) & (v >= 1e7) & np.isfinite(px) & (px >= 10) & ~P.DRMASK
    o = np.argsort(-np.where(base, v, -np.inf), kind='stable')[:min(300, int(base.sum()))]
    for j in o:
        c = P.COLS[j]
        before = [s for s in snaps if c in s['map'] and s['date'] <= d]
        after = [s for s in snaps if c in s['map'] and s['date'] > d and (s['date'] - d).days <= 731]
        if before and (d - before[-1]['date']).days <= 731: s = before[-1]
        elif after: s = after[0]
        else: s = None
        ind = s['map'][c] if s else None
        drows.append(dict(d=str(d.date()), y=d.year, code=c, ind=ind, snap=str(s['date'].date()) if s else None, dist=abs((s['date'] - d).days) if s else None,
                          fin_pit=(ind == '金融保險業') if ind else None, fin_main=c in FIN_MAIN, o12=o12_ind(c, d)))
DR = pd.DataFrame(drows)
dated = DR.ind.notna()
d2 = dict(n=len(DR), dated_share=float(dated.mean()), undeterminable_share=float((~dated).mean()), by_year=DR.groupby('y').ind.apply(lambda s: round(float(s.notna().mean()), 4)).to_dict())
d2['pass'] = bool(d2['dated_share'] >= SPEC['D2']['thresholds']['dated_share_ge'] and d2['undeterminable_share'] <= SPEC['D2']['thresholds']['undeterminable_le'])
mm = DR[dated & (DR.fin_pit != DR.fin_main)]
d2['fin_flag_mismatch'] = dict(n_stocks=int(mm.code.nunique()), n_stock_months=len(mm),
                               detail=mm.groupby('code').agg(first=('d', 'min'), last=('d', 'max'), n=('d', 'size'), pit=('ind', 'first'), fin_main=('fin_main', 'first')).reset_index().to_dict('records'))
und = DR[~dated]
d2['undeterminable_fin_relevant'] = dict(n_stock_months=len(und), n_stocks=int(und.code.nunique()),
                                         note='無 dated 快照的股月無法判定 PIT 金融旗標；其中主策略旗標＝金融者', n_fin_main_stock_months=int(und.fin_main.sum()))
# o12 回溯表 vs dated 快照 的產業差異（只在 dated 股月）
x = DR[dated & DR.o12.notna()]
d2['o12_vs_dated_mismatch'] = dict(n_stock_months=int((x.o12 != x.ind).sum()), share=float((x.o12 != x.ind).mean()),
                                  top=x[x.o12 != x.ind].groupby(['code', 'o12', 'ind']).size().sort_values(ascending=False).head(25).reset_index().astype(str).to_dict('records'))
# 描述：寬鬆口徑（2011-07 快照的電子 8 類往前適用到 2007-07-02 改制日；不計入閘門）
d2['lenient_reform_desc'] = 'strict 口徑不採計；2006-01～2009-07 沒有任何 dated 快照，寬鬆口徑也只能把 2007-07-02 以後電子股推回，非電子股與 2007-07 以前仍無法判定'
G['D2'] = d2
DR.to_csv(P.SCR / 'd2_rows.csv', index=False)
print('D2', {k: v for k, v in d2.items() if k in ('dated_share', 'pass', 'by_year')}, 'fin mismatch', d2['fin_flag_mismatch']['n_stocks'], flush=True)

# ---------------- D1-retro（路徑 B：o12 原協定，TWT49U 官方事件） ----------------
S8 = pd.read_parquet(P.RAW / 'center' / 'o8_shares_pit.parquet').reindex(index=P.IDX, columns=P.COLS)
FIX4763 = dict(a0='2019-09-04', a1='2023-07-27', div_a=12.65, b1='2025-06-30', div_b=10.0)
ix = S8.index
a = (ix >= pd.Timestamp(FIX4763['a0'])) & (ix < pd.Timestamp(FIX4763['a1'])); b = (ix >= pd.Timestamp(FIX4763['a1'])) & (ix < pd.Timestamp(FIX4763['b1']))
S8.loc[a, '4763'] = S8.loc[a, '4763'] / FIX4763['div_a']; S8.loc[b, '4763'] = S8.loc[b, '4763'] / FIX4763['div_b']
pv = float(S8.loc[:pd.Timestamp('2018-06-29'), '2317'].dropna().iloc[-1]); m2317 = (ix > pd.Timestamp('2018-06-29')) & (ix <= pd.Timestamp('2018-10-25')); S8.loc[m2317, '2317'] = pv
S8 = S8.ffill(limit=10); S8[ix < pd.Timestamp('2013-01-02')] = np.nan
S13 = S8.to_numpy()
tw = []
for f in sorted((P.RAW / 'twt49u').glob('*.csv.gz')):
    tw.append(pd.read_csv(f, dtype={'code': str}))
TW = pd.concat(tw); TW['date'] = pd.to_datetime(TW.date); TW = TW[TW.code.isin(P.CI) & TW.date.isin(P.IDX)]
TW['f'] = TW.prev_close / TW.ref_price; TW['y'] = TW.date.dt.year
cash = TW[TW.type.str.strip() == '息']
def m_official():
    out = {}
    cy = {}
    for r in cash.itertuples():
        if 0.0 < r.f - 1 <= 0.15: cy.setdefault(r.code, []).append((r.y, r.f - 1))
    mk = cash.assign(x=cash.f - 1).query('x > 0 and x <= 0.15').groupby('y').x.median().to_dict()
    for r in TW.itertuples():
        ty = str(r.type).strip()
        if ty == '息' or not np.isfinite(r.f): continue
        if ty == '權': m = r.f
        else:
            own = [x for (y, x) in cy.get(r.code, []) if abs(y - r.y) <= 3]
            yh = float(np.median(own)) if own else mk.get(r.y, 0.03)
            m = r.f / (1 + yh)
        if m <= 1.02: continue
        out.setdefault(P.CI[r.code], []).append((P.IDX.get_loc(r.date), float(min(max(m, 0.2), 3.0))))
    return out
mo = m_official()
# flag 3（參考價重設；TWT49U 沒有的減資等）用開盤估計，且當日沒有 TWT49U 事件
F = P.open_factor(); tw_days = {(P.CI[c], P.IDX.get_loc(d)) for c, d in zip(TW.code, TW.date)}
ta = P.ti('2021-03-31'); tl = P.ti('2014-01-02') - 1
n3 = 0
for t, j in np.argwhere((P.Fn[tl:ta + 1] == 3) & P.TRADED[tl:ta + 1]):
    t = int(t) + tl; j = int(j)
    if (j, t) in tw_days or not np.isfinite(F[t, j]) or abs(F[t, j] - 1) <= 0.02: continue
    mo.setdefault(j, []).append((t, float(min(max(F[t, j], 0.2), 3.0)))); n3 += 1
def backcast(anc_vec, tan, t_lo, mult):
    S = np.full((len(P.IDX), P.N), np.nan)
    for j in range(P.N):
        if not np.isfinite(anc_vec[j]): continue
        evs = sorted([(t, m) for (t, m) in mult.get(j, []) if t_lo < t <= tan], reverse=True)
        cur = anc_vec[j]; hi = tan
        for t, m in evs:
            S[t:hi + 1, j] = cur; cur = cur / m; hi = t - 1
        S[t_lo:hi + 1, j] = cur
    return S
Sh = backcast(S13[ta].copy(), ta, tl, mo)
mult_o12 = B.event_multipliers(tl, ta)
Sh_open = backcast(S13[ta].copy(), ta, tl, mult_o12)
def retro(ShX, oracle_noanchor=False):
    rr = []; ee = []
    for q in pd.date_range('2014-03-31', '2020-12-31', freq='QE'):
        t = P.ti(q); al = P.alive_at(t)
        mt = np.where(al, S13[t] * P.CFF[t], np.nan)
        hx = ShX[t].copy()
        if oracle_noanchor: hx = np.where(np.isfinite(hx), hx, S13[t])
        mh = np.where(al, hx * P.CFF[t], np.nan)
        mh, imp = B.impute_mcap(t, mh, al, ADV)
        T = B.top_k(mt); H = B.top_k(mh)
        e = np.abs(np.log(mh[T] / mt[T])); e = e[np.isfinite(e)]; ee.append(e)
        rr.append(dict(q=str(q.date()), overlap=len(set(T) & set(H)), med_abs_log=float(np.median(e)), n_imputed_true_top=int(imp[T].sum()),
                       missed=[P.COLS[j] for j in set(T) - set(H)], missed_imputed=[P.COLS[j] for j in set(T) - set(H) if imp[j]]))
    ov = np.array([r['overlap'] for r in rr]); Ea = np.concatenate(ee)
    return dict(pooled_med_abs_log=float(np.median(Ea)), overlap_median=float(np.median(ov)), overlap_min=int(ov.min()), by_q=rr)
G['D1_retro'] = dict(n_official_events=int(sum(len(v) for v in mo.values())), n_flag3_added=n3,
                     pathB_official=retro(Sh), open_factor_o12_rule=retro(Sh_open),
                     pathB_official_oracle_noanchor=retro(Sh, True), open_factor_oracle_noanchor=retro(Sh_open, True))
for k in ('pathB_official', 'open_factor_o12_rule', 'pathB_official_oracle_noanchor', 'open_factor_oracle_noanchor'):
    print('retro', k, {x: G['D1_retro'][k][x] for x in ('pooled_med_abs_log', 'overlap_median', 'overlap_min')}, flush=True)
G['D1_retro']['P1_pass'] = bool(G['D1_retro']['pathB_official']['overlap_min'] >= SPEC['D1_retro']['threshold_P1']['overlap_min_ge'])

# ---------------- P2：o12 回推股數（2006～2012）的錯誤分布 ----------------
O12S = pd.read_parquet(P.RAW / 'center' / 'o12_shares_full.parquet').reindex(index=P.IDX, columns=P.COLS).to_numpy().astype(float)
prow = []
for q in pd.date_range('2006-03-31', '2012-12-31', freq='QE'):
    t = P.ti(q); al = P.alive_at(t)
    tru = M[M.t == t].set_index('code')['shares']
    tv = np.full(P.N, np.nan)
    for c, v in tru.items(): tv[P.CI[c]] = v
    mt = np.where(al, np.where(np.isfinite(tv), tv, S_full[t]) * P.CFF[t], np.nan)
    T = B.top_k(mt, 120)
    for r, j in enumerate(T):
        if np.isfinite(tv[j]) and np.isfinite(O12S[t, j]):
            prow.append(dict(q=str(q.date()), y=q.year, code=P.COLS[j], rank=r + 1, ld=float(np.log(O12S[t, j] / tv[j]))))
PR = pd.DataFrame(prow); PR['abs'] = PR.ld.abs()
PR['bucket'] = pd.cut(PR['rank'], [0, 20, 59, 120], labels=['1-20', '21-59', '60-120'])
ELEC = {'半導體業', '電腦及週邊設備業', '光電業', '通信網路業', '電子零組件業', '電子通路業', '資訊服務業', '其他電子業', '電子工業'}
PR['elec'] = [(o12_ind(c, pd.Timestamp(q)) in ELEC) for c, q in zip(PR.code, PR.q)]
p2 = dict(n=len(PR), by_bucket=PR.groupby('bucket', observed=True)['abs'].median().round(5).to_dict(),
          by_year=PR.groupby('y')['abs'].median().round(5).to_dict(), signed_by_year=PR.groupby('y').ld.median().round(5).to_dict(),
          elec_vs_other_2006_2008=PR[PR.y <= 2008].groupby('elec')['abs'].median().round(5).to_dict(),
          worst=PR.sort_values('abs', ascending=False).head(15).round(4).astype(str).to_dict('records'))
b = p2['by_bucket']
p2['P2_top20_le_1pct'] = bool(b.get('1-20', 1) <= 0.01); p2['P2_60_120_ge_2x_top20'] = bool(b.get('60-120', 0) >= 2 * b.get('1-20', 1))
G['P2_o12_backcast_errors'] = p2
print('P2', p2['by_bucket'], p2['by_year'], flush=True)

# ---------------- P3：覆蓋 ----------------
ex = json.loads((P.HERE / 'explore.json').read_text(encoding='utf-8'))
G['P3'] = dict(coverage_le_1q=ex['coverage_adv150']['all']['le_1q'], coverage_le_1y=ex['coverage_adv150']['all']['le_1y'],
               survivors_le_1q=ex['coverage_adv150']['survivors']['le_1q'], delisted_le_1q=ex['coverage_adv150']['delisted_before_2021']['le_1q'])
G['P5'] = '臺灣50 2006～2013 官方納入／剔除名單找不到 → 不做'
G['D1_pass'] = bool(d1a['pass'] and d1c['pass'])
G['overall_pass'] = bool(d1a['pass'] and d1c['pass'] and d2['pass'])
G['label_2006_2013'] = '近似 PIT（D1 通過）' if G['overall_pass'] else '2006～2013 仍非 PIT'
G['elapsed_s'] = round(time.time() - t00)
(P.DATA / 'gate.json').write_text(P.dumps(G), encoding='utf-8')
if G['overall_pass']:
    pd.DataFrame(S_full.astype(np.float32), index=P.IDX, columns=P.COLS).loc['2006-01-01':'2013-12-31'].to_parquet(P.DATA / 'shares_pit_2006_2013.parquet')
    DR.to_csv(P.DATA / 'industry_pit_2006_2013.csv', index=False)
print('overall', G['overall_pass'], G['label_2006_2013'], G['elapsed_s'], 's')

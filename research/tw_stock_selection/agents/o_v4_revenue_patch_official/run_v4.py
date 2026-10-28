#!/usr/bin/env python3
"""樣本外（一次跑完、不改規格）：V4 正式數字與必報項目。PREREG 必須已存在。"""
import sys, json, hashlib, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import v4lib as L
K, V, R = L.K, L.V, L.R

RES_F = HERE / 'results.json'
res = json.loads(RES_F.read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
canon = json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()
assert hashlib.sha256(canon).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
assert L.sha(HERE / 'data_freeze_v4.json') == res['PREREG']['data_freeze_v4_sha256'], 'data_freeze_v4 與 PREREG 不符'
fz = L.check_freeze()
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)

XR = L.setup_px()
YFX, Y5F = L.load_rfix(); YRD = L.load_rdel(); Y5 = L.y5_orig()
GAP = L.gap_dict(); QP = L.qproxy_fn()
DMT, DMS = L.disp_mask_fn('trade'), L.disp_mask_fn('signal')
cells = pd.read_csv(HERE / 'rfix_cells.csv', dtype={'code': str})
YNO13 = YFX.copy()      # IFRS 敏感度：拿掉 2013 年的修補格，以及基期是 2013 年新聞水準的 2014 年修補格
_lv = pd.read_csv(K.LONG, dtype={'code': str, 'market': str}); _lv = _lv[_lv.market == 'sii']
_has = set(zip(_lv.code, _lv.month.str[:7]))
for c, y in zip(cells.code, cells.ym):
    base = str(pd.Period(y, 'M') - 12)
    if y[:4] == '2013' or (y[:4] == '2014' and (c, base) not in _has):
        YNO13.at[pd.Period(y, 'M'), c] = np.nan
NORMAL, STRESS = 'normal', 'stress'
A_, B_ = '2006-04-01', '2026-10-12'

def mk(name, **kw):
    YL = kw.pop('YL', YFX); Y5x = kw.pop('Y5', Y5F); dm = kw.pop('dm', DMT)
    return L.V4(YL, Y5x, dm, **kw)
SPECS = {
    'V4': dict(),
    'V4_B05': dict(gap=GAP, pfix=0.5),
    'V4_B10': dict(gap=GAP, pfix=1.0),
    'V4_BQ': dict(gap=GAP, yproxy=QP),
    'V4_SIG': dict(dm=DMS),
    'V4_RDEL': dict(YL=YRD, Y5=Y5),
    'V4_NO2013': dict(YL=YNO13),
    'V4_SPEC0': dict(fin2=False, exclude_foreign=False),
    'O3REP': dict(YL=YRD, Y5=Y5, dm=DMS, fin2=False, exclude_foreign=False),
}
navs = {}; plans = {}; objs = {}; logs = {}
t0 = t1 = sd0 = None
for k, kw in SPECS.items():
    S = mk(k, **kw)
    ta, tb, sd = S.bounds()
    pl = S.plan(10, 'official')
    if t0 is None: t0, t1, sd0 = ta, tb, sd
    assert (ta, tb) == (t0, t1) and sd == sd0
    for c in (NORMAL, STRESS):
        nav, lg = L.sim_plan(pl, ta, tb, c, XR)
        navs[f'{k}|{c}'] = nav; logs[f'{k}|{c}'] = lg
    ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, ta, tb)
    navs[f'{k}|EW'] = ew; plans[k] = pl; objs[k] = S
    log(k, round(L.seg(navs[f'{k}|normal'], ew, sd, A_, B_)['CAGR'], 3))
sd = sd0
o3 = json.loads((L.AG / 'o_disposal_rejudge' / 'v3_results.json').read_text())['runs']['R_DEL|E1']['stats']['ALL']
st = L.seg(navs['O3REP|normal'], navs['O3REP|EW'], sd, A_, B_)
assert abs(st['CAGR'] - o3['CAGR']) < 1e-6 and abs(st['t_0050'] - o3['vs0050']['t']) < 1e-6, 'o3 重現失敗'

# ---------------- AB（mc 40）----------------
def ab(kw, nseed=40, costs=(NORMAL, STRESS), slips=()):
    rets = {c: [] for c in costs}; rs = {s: [] for s in slips}; ewr = []; nold = []
    for seed in range(nseed):
        S = mk('AB', unavail=V.make_unavail('mc', seed), **kw)
        pl = S.plan(10, 'official')
        for c in costs:
            nav, _ = L.sim_plan(pl, t0, t1, c, XR); rets[c].append(nav.pct_change())
        for s in slips:
            nav, _ = L.sim_plan(pl, t0, t1, ('slip', s), XR); rs[s].append(nav.pct_change())
        ew, _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1); ewr.append(ew.pct_change())
        nold.append(np.mean([S.cross_section(d)['nold'] for d in S.sig_days]))
    mean_nav = lambda L_: (1 + pd.concat(L_, axis=1).mean(axis=1).fillna(0)).cumprod()
    return {c: mean_nav(rets[c]) for c in costs}, {s: mean_nav(rs[s]) for s in slips}, mean_nav(ewr), float(np.mean(nold))
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]        # 0.10%～0.40%
AB = {}
for k in ('V4', 'V4_B05', 'V4_B10', 'V4_BQ'):
    sl = SLIPS if k == 'V4' else ()
    nv, ns, ew, no = ab(SPECS[k], slips=sl)
    for c in nv: navs[f'AB_{k}|{c}'] = nv[c]
    navs[f'AB_{k}|EW'] = ew
    AB[k] = {'mean_unavail_in_top300': no, 'slip_navs': ns}
    log('AB', k, round(L.seg(nv[NORMAL], ew, sd, A_, B_)['CAGR'], 3), round(L.seg(nv[NORMAL], ew, sd, A_, B_)['t_0050'], 3))
for k in ('V4', 'V4_B10'):
    S = mk('W', unavail=V.make_unavail('worst'), **SPECS[k]); pl = S.plan(10, 'official')
    for c in (NORMAL, STRESS):
        navs[f'ABW_{k}|{c}'], _ = L.sim_plan(pl, t0, t1, c, XR)
    navs[f'ABW_{k}|EW'], _ = K.simulate([(p[0], p[2]) for p in pl], 0, 0, t0, t1)

# ---------------- 成本損益兩平（V4 主、V4 AB）----------------
def be(navfn):
    pts = [(s, L.seg(navfn(s), navs['V4|EW'], sd, A_, B_)['t_0050']) for s in SLIPS]
    def cross(th):
        for (s1, a), (s2, b) in zip(pts, pts[1:]):
            if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
        return None if pts[0][1] < th else float('inf')
    return {'grid': pts, 'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0)}
pl = plans['V4']
slip_navs_main = {s: L.sim_plan(pl, t0, t1, ('slip', s), XR)[0] for s in SLIPS}
BE = {'V4': be(lambda s: slip_navs_main[s]), 'AB_V4': be(lambda s: AB['V4']['slip_navs'][s])}
navs['V4|fullfee'], _ = L.sim_plan(pl, t0, t1, ('fullfee', 0.001), XR)
log('breakeven', {k: (v['slip_at_t2.3'], v['slip_at_t2.0']) for k, v in BE.items()})

# ---------------- 統計工具 ----------------
E50 = K.E50
def xs(nav, a=A_, b=B_): return L.excess(nav, E50, sd, a, b)
def nw_t(x, lag):
    x = np.asarray(x, float); n = len(x); mu = x.mean(); e = x - mu
    g0 = (e @ e) / n; s = g0
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1); s += 2 * w * (e[l:] @ e[:-l]) / n
    return float(mu / math.sqrt(s / n))
def block_boot(x, B=10000, blk=12, seed=7):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed); c = x - x.mean(); mu = x.mean()
    nb = int(math.ceil(n / blk)); cnt = 0
    for _ in range(B):
        st_ = rng.integers(0, n, nb); idx = (st_[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n
        if c[idx].mean() >= mu: cnt += 1
    return cnt / B
def cal_t(nav):
    me = nav.groupby(nav.index.to_period('M')).last(); be_ = E50.reindex(nav.index).groupby(nav.index.to_period('M')).last()
    r = (me.pct_change() - be_.pct_change()).dropna()
    r = r[(r.index >= pd.Period('2006-04', 'M')) & (r.index <= pd.Period('2026-10', 'M'))]
    return L.tstat(r.to_numpy())
def endpoint(nav):
    x = xs(nav); o = {'t_all': L.tstat(x), 'n': len(x)}
    o['drop_last1'] = L.tstat(x.iloc[:-1])
    xsrt = x.sort_values(ascending=False)
    for k in (1, 3, 5): o[f'drop_best{k}'] = L.tstat(x.drop(xsrt.index[:k]))
    yr = x.groupby(x.index.year).sum(); by = int(yr.idxmax())
    o['best_year'] = by; o['drop_best_year'] = L.tstat(x[x.index.year != by])
    o['best_month'] = str(xsrt.index[0].date()); o['best_month_pp'] = float(xsrt.iloc[0] * 100)
    o['to_2025_end'] = L.tstat(x[x.index <= pd.Timestamp('2025-12-31')])
    o['cal_month_t'] = cal_t(nav)
    return o
def full(nav, ew, extra=True):
    o = L.allseg(nav, ew, sd)
    if extra:
        x = xs(nav)
        o['ALL'].update(NW_t={l: nw_t(x, l) for l in (3, 6, 12)}, boot12_p_one_sided=block_boot(x),
                        bonferroni_m10_p=min(1.0, 10 * L.p_one_sided(o['ALL']['t_0050'])), p_one_sided=L.p_one_sided(o['ALL']['t_0050']))
        o['endpoint'] = endpoint(nav)
        r = nav.pct_change(); r50 = E50.reindex(nav.index).pct_change()
        msk = r.index >= pd.Timestamp(A_)
        o['yearly'] = {int(y): {'V4': float((1 + g).prod() - 1) * 100, '0050': float((1 + r50[g.index].fillna(0)).prod() - 1) * 100} for y, g in r[msk].groupby(r[msk].index.year)}
        o['yearly_lose_to_0050'] = [y for y, v in o['yearly'].items() if v['V4'] < v['0050']]
    return o
def paired(a, b, x0=A_, x1=B_):
    ra, rb = L.win_ret(a, sd), L.win_ret(b.reindex(a.index), sd)
    d = (ra - rb); d = d[(d.index >= pd.Timestamp(x0)) & (d.index <= pd.Timestamp(x1))].dropna()
    nz = d[d.abs() > 1e-12]
    return dict(mean_pp=float(d.mean() * 100), t=L.tstat(d), n=int(len(d)), n_nonzero=int(len(nz)))

out = {'runtime_note': '一次跑完，不改規格', 'n_patched_cells_P': L._STATE['npatch'], 'n_exit_rules_X': len(XR)}
tab = {}
for k in SPECS:
    for c in (NORMAL, STRESS):
        tab[f'{k}|{c}'] = full(navs[f'{k}|{c}'], navs[f'{k}|EW'], extra=(c == NORMAL and k in ('V4', 'V4_B05', 'V4_B10', 'V4_BQ', 'V4_SIG')))
    tab[f'{k}|normal']['turnover'] = logs[f'{k}|normal']['turnover']
    tab[f'{k}|normal']['n_exits'] = len(logs[f'{k}|normal']['exits']); tab[f'{k}|normal']['n_frozen'] = len(logs[f'{k}|normal']['frozen'])
for k in ('V4', 'V4_B05', 'V4_B10', 'V4_BQ'):
    for c in (NORMAL, STRESS):
        tab[f'AB_{k}|{c}'] = full(navs[f'AB_{k}|{c}'], navs[f'AB_{k}|EW'], extra=(c == NORMAL))
    tab[f'AB_{k}|normal']['mean_unavail_in_top300'] = AB[k]['mean_unavail_in_top300']
for k in ('V4', 'V4_B10'):
    for c in (NORMAL, STRESS): tab[f'ABW_{k}|{c}'] = full(navs[f'ABW_{k}|{c}'], navs[f'ABW_{k}|EW'], extra=False)
tab['V4|fullfee'] = full(navs['V4|fullfee'], navs['V4|EW'], extra=False)
out['table'] = tab
out['breakeven'] = BE
# 成本損益兩平（AB）也報全額手續費
out['breakeven_fullfee'] = {'V4': tab['V4|fullfee']['ALL']['t_0050']}

# ---------------- 配對比較 ----------------
mv2 = pd.read_parquet(L.AG / 'main_v2' / 'navs_gapfix.parquet')
m2 = mv2['EXFIN|n10|normal'].reindex(navs['V4|normal'].index)
m2s = mv2['EXFIN|n10|stress'].reindex(navs['V4|normal'].index)
out['paired'] = {
    'V4_minus_main_v2': {p: paired(navs['V4|normal'], m2, *L.PER[p]) for p in ('ALL', 'SEL', 'VAL', 'HOLD', 'S2')},
    'V4_stress_minus_main_v2_stress': paired(navs['V4|stress'], m2s),
    'V4_minus_O3REP(V3官方名單版)': {p: paired(navs['V4|normal'], navs['O3REP|normal'], *L.PER[p]) for p in ('ALL', 'SEL', 'VAL', 'HOLD')},
    'RFIX_effect(V4_minus_V4_RDEL)': {p: paired(navs['V4|normal'], navs['V4_RDEL|normal'], *L.PER[p]) for p in ('ALL', 'S1', 'S2', 'S3', 'S4')},
    'trade_vs_signal(V4_minus_V4_SIG)': {p: paired(navs['V4|normal'], navs['V4_SIG|normal'], *L.PER[p]) for p in ('ALL', 'SEL', 'VAL', 'HOLD')},
    'spec_explicit(V4_minus_V4_SPEC0)': paired(navs['V4|normal'], navs['V4_SPEC0|normal']),
    'AB_V4_minus_V4': paired(navs['AB_V4|normal'], navs['V4|normal']),
    'B05_minus_V4': paired(navs['V4_B05|normal'], navs['V4|normal']), 'B10_minus_V4': paired(navs['V4_B10|normal'], navs['V4|normal']),
    'BQ_minus_V4': paired(navs['V4_BQ|normal'], navs['V4|normal'])}

# ---------------- 名單差異：修補股、處置口徑、界限股 ----------------
def picks(pl): return {d: set(K.COLS[j] for j in p[1]) for p in pl for d in [p[3]]}
P4, P0 = picks(plans['V4']), picks(plans['V4_RDEL'])
patched = set(cells.code)
sel_patched = [(str(d.date()), c) for d in P4 for c in P4[d] - P0.get(d, set()) if c in patched]
out['rfix_picks'] = {'patched_stock_months_selected': sel_patched, 'n': len(sel_patched),
                     'n_slots_changed_vs_RDEL': int(sum(len(P4[d] ^ P0.get(d, set())) // 2 for d in P4)),
                     'months_changed_vs_RDEL': [str(d.date()) for d in P4 if P4[d] != P0.get(d, set())]}
Ps = picks(plans['V4_SIG'])
out['disposal_rule_picks'] = {'months_changed': [str(d.date()) for d in P4 if P4[d] != Ps.get(d, set())],
                              'diff': {str(d.date()): {'trade_only': sorted(P4[d] - Ps[d]), 'signal_only': sorted(Ps[d] - P4[d])} for d in P4 if P4[d] != Ps.get(d, set())}}
for k in ('V4_B05', 'V4_B10', 'V4_BQ'):
    Pk = picks(plans[k]); gp = [(str(d.date()), c) for d in Pk for c in Pk[d] if c in GAP and c not in P4.get(d, set()) and
                                 not np.isfinite(objs['V4'].cross_section(d)['y3'][K.CI[c]])]
    out[f'{k}_gap_picks'] = {'n': len(gp), 'by_year': pd.Series([s[:4] for s, _ in gp]).value_counts().sort_index().to_dict() if gp else {},
                             'by_code': pd.Series([c for _, c in gp]).value_counts().head(20).to_dict() if gp else {}}

# ---------------- 可證偽：修補股入選股月的下月超額 vs 其他入選股 ----------------
con, per = V.stock_contrib(plans['V4'], t1)
rows = []
for d, pk, r in per:
    codes = [K.COLS[j] for j in pk]
    for c, x in zip(codes, r):
        rows.append((d, c, x, (str(d.date()), c) in set(sel_patched)))
PR = pd.DataFrame(rows, columns=['d', 'code', 'r', 'patched'])
def vs_others(mask):
    g = []
    for d, x in PR.groupby('d'):
        a = x[mask.loc[x.index]]; b = x[~mask.loc[x.index]]
        if len(a) and len(b): g.append(a.r.mean() - b.r.mean())
    return {'n_months': len(g), 'mean_pp': float(np.mean(g) * 100) if g else None, 't': L.tstat(g) if len(g) >= 3 else None}
out['P5_patched_vs_other_picks'] = vs_others(PR.patched)
# 同法用 B10／B_Q 的缺營收入選股（界限情境裡的「修補股」）
for k in ('V4_B10', 'V4_BQ'):
    c2, per2 = V.stock_contrib(plans[k], t1)
    rr = []
    for d, pk, r in per2:
        y3 = objs['V4'].cross_section(d)['y3']
        for j, x in zip(pk, r): rr.append((d, K.COLS[j], x, K.COLS[j] in GAP and not np.isfinite(y3[j])))
    Q = pd.DataFrame(rr, columns=['d', 'code', 'r', 'patched'])
    g = []
    for d, x in Q.groupby('d'):
        a, b = x[x.patched], x[~x.patched]
        if len(a) and len(b): g.append(a.r.mean() - b.r.mean())
    out[f'P5_{k}_gap_picks_vs_other'] = {'n_months': len(g), 'mean_pp': float(np.mean(g) * 100) if g else None, 't': L.tstat(g) if len(g) >= 3 else None,
                                         'n_months_SEL': int(sum(1 for d, x in Q.groupby('d') if x.patched.any() and d <= pd.Timestamp('2018-12-31')))}

# ---------------- 穩健性：分產業、最大貢獻股 ----------------
SL = V.SL
ind = {}
for c, v in con.items():
    i = SL['industry'].get(c) if c in SL.index else None
    if not isinstance(i, str): i = L.DL_NAME.get(c, '（下市）') and '（已下市）'
    ind[i] = ind.get(i, 0) + v
tot = sum(con.values())
out['robust'] = {'industry_share_top10': sorted(((k, round(v / tot, 3)) for k, v in ind.items()), key=lambda x: -x[1])[:10],
                 'top_contrib_ALL': V.top_contrib(plans['V4'], t1, A_, B_),
                 'top_contrib_SEL': V.top_contrib(plans['V4'], t1, '2006-04-01', '2018-12-31'),
                 'top_contrib_HOLD': V.top_contrib(plans['V4'], t1, '2024-01-01', B_)}

# ---------------- 同分序區間（main_v2 的 rand 同分）----------------
tie = []
S = objs['V4']
for s in range(20):
    plr = S.plan(10, f'rand:{s}')
    nav, _ = L.sim_plan(plr, t0, t1, NORMAL, XR)
    st_ = L.seg(nav, navs['V4|EW'], sd, A_, B_); tie.append((st_['CAGR'], st_['t_0050']))
out['tie_interval'] = {'CAGR_min': min(x[0] for x in tie), 'CAGR_max': max(x[0] for x in tie), 't_min': min(x[1] for x in tie), 't_max': max(x[1] for x in tie), 'n': len(tie)}
log('tie', out['tie_interval'])

# ---------------- 判定 ----------------
tAB = tab['AB_V4|normal']['ALL']['t_0050']
tAB_d1 = tab['AB_V4|normal']['endpoint']['drop_best1']
def grade(t): return '第 1 級' if t >= 2.58 else ('第 1 級邊緣' if t >= 2.3 else '第 2 級')
out['decision'] = {'AB_V4_normal_t': tAB, 'grade': grade(tAB), 'AB_V4_drop_best1_t': tAB_d1, 'grade_drop_best1': grade(tAB_d1),
                   'endpoint_sensitive': grade(tAB) != grade(tAB_d1),
                   'conservative_AB_B10': {'CAGR': tab['AB_V4_B10|normal']['ALL']['CAGR'], 't': tab['AB_V4_B10|normal']['ALL']['t_0050']},
                   'bonferroni_m10_p_AB': tab['AB_V4|normal']['ALL']['bonferroni_m10_p']}
res['RESULTS'] = out
RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
NV = pd.DataFrame({k: v for k, v in navs.items()})
NV.to_parquet(HERE / 'navs_v4.parquet')
# 名單（給複製員比對）
pd.DataFrame([(str(p[3].date()), str(K.IDX[p[0]].date()), ','.join(K.COLS[j] for j in p[1])) for p in plans['V4']], columns=['signal', 'exec', 'picks']).to_csv(HERE / 'plan_V4.csv', index=False)
for k in ('V4_B05', 'V4_B10', 'V4_SIG', 'V4_BQ'):
    pd.DataFrame([(str(p[3].date()), str(K.IDX[p[0]].date()), ','.join(K.COLS[j] for j in p[1])) for p in plans[k]], columns=['signal', 'exec', 'picks']).to_csv(HERE / f'plan_{k}.csv', index=False)
log('done', out['decision'])

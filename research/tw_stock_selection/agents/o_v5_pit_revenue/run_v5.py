#!/usr/bin/env python3
"""樣本外（一次跑完、不改規格）：V5 正式數字與必報項目。PREREG 與 data_freeze_v5 必須已存在且未被改過。
輸出 results.json 的 RESULTS、navs_v5.parquet、seed_navs_v5.parquet、plan_V5*.csv。"""
import sys, json, hashlib, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
from multiprocessing import Pool
import v5lib as L
K, V, L4 = L.K, L.V, L.L4

RES_F = HERE / 'results.json'
res = json.loads(RES_F.read_text(encoding='utf-8'))
assert 'PREREG' in res and 'PREREG_sha256' in res, 'PREREG 不存在：先跑 prereg.py'
assert hashlib.sha256(json.dumps(res['PREREG'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == res['PREREG_sha256'], 'PREREG 被改過'
assert L.sha(L.FREEZE5) == res['PREREG']['data_freeze_v5_sha256'], 'data_freeze_v5 與 PREREG 不符'
L.check_freeze5()
T0 = time.time()
def log(*a): print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
g = L.setup()
NORMAL, STRESS = 'normal', 'stress'
SLIPS = [round(0.0010 + 0.0002 * i, 4) for i in range(16)]

def fv_of(name):
    if name == 'none': return None
    if name == 'news': return L.FVAdj()
    if name == 'news_baselatest': return L.FVAdj(base='latest')
    if name.startswith('mc'): return L.FVAdj(use_news=True, mc=L.load_mc(name[2:]))
    raise ValueError(name)

S_ = L.make(fv=None); t0, t1, sd = S_.bounds()

def job(a):
    """a＝(fvname, bound, seed or None or 'worst', costs tuple)。回傳 (key, {cost: 日報酬}, picks, ew 日報酬, nold)。"""
    fvn, bound, seed, costs = a
    U = None if seed is None else (V.make_unavail('worst') if seed == 'worst' else V.make_unavail('mc', seed))
    S = L.make(fv=fv_of(fvn), unavail=U, bound=bound)
    ta, tb, sdx = S.bounds(); assert (ta, tb) == (t0, t1) and sdx == sd
    pl = S.plan(10, 'official')
    out = {}
    for c in costs:
        out[str(c)] = L.sim(pl, c, t0, t1).pct_change()
    ew = L.ew_of(pl, t0, t1).pct_change()
    nold = float(np.mean([S.cross_section(d)['nold'] for d in S.sig_days])) if U is not None else 0.0
    return (fvn, bound, seed), out, [(p[0], p[1].copy(), p[3]) for p in pl], ew, nold

def run_many(args):
    with Pool(4) as P: return P.map(job, args, chunksize=1)

R = {}
def add(results):
    for k, o, pl, ew, no in results: R[k] = (o, pl, ew, no)

SEEDS = list(range(40))
jobs = []
# 非 AB
for fvn, b in (('news', None), ('news', 'B05'), ('news', 'B10'), ('none', None), ('none', 'B10')):
    jobs.append((fvn, b, None, (NORMAL, STRESS)))
# AB 主欄（含成本格點）
for s in SEEDS: jobs.append(('news', None, s, (NORMAL, STRESS) + tuple(('slip', x) for x in SLIPS) + (('fullfee', 0.001),)))
for b in ('B05', 'B10'):
    for s in SEEDS: jobs.append(('news', b, s, (NORMAL, STRESS)))
for s in SEEDS: jobs.append(('none', None, s, (NORMAL,)))                 # AB_V4 重現
for s in SEEDS: jobs.append(('news_baselatest', None, s, (NORMAL,)))
for b in (None, 'B10'): jobs.append(('news', b, 'worst', (NORMAL, STRESS)))
log('main jobs', len(jobs)); add(run_many(jobs)); log('main done')
# 首次值界限：MC 20 個擾動 × AB 40 seed，固定方向 neg／pos
jobs = [(f'mc{t}', None, s, (NORMAL,)) for t in [f's{i}' for i in range(20)] + ['neg', 'pos'] for s in SEEDS]
log('fv bound jobs', len(jobs)); add(run_many(jobs)); log('fv bound done')

navs = {}
def ab_nav(fvn, b, cost):
    return L.mean_nav([R[(fvn, b, s)][0][str(cost)] for s in SEEDS])
def ab_ew(fvn, b):
    return L.mean_nav([R[(fvn, b, s)][2] for s in SEEDS])
def cum(r): return (1 + r.fillna(0)).cumprod()
for fvn, b, nm in (('news', None, 'V5'), ('news', 'B05', 'V5_B05'), ('news', 'B10', 'V5_B10'), ('none', None, 'V4REF'), ('none', 'B10', 'V4REF_B10')):
    for c in (NORMAL, STRESS): navs[f'{nm}|{c}'] = cum(R[(fvn, b, None)][0][c])
    navs[f'{nm}|EW'] = cum(R[(fvn, b, None)][2])
for c in (NORMAL, STRESS): navs[f'AB_V5|{c}'] = ab_nav('news', None, c)
navs['AB_V5|EW'] = ab_ew('news', None)
for x in SLIPS: navs[f'AB_V5|slip{x}'] = ab_nav('news', None, ('slip', x))
navs['AB_V5|fullfee'] = ab_nav('news', None, ('fullfee', 0.001))
for b in ('B05', 'B10'):
    for c in (NORMAL, STRESS): navs[f'AB_V5_{b}|{c}'] = ab_nav('news', b, c)
    navs[f'AB_V5_{b}|EW'] = ab_ew('news', b)
navs['AB_V4REF|normal'] = ab_nav('none', None, NORMAL); navs['AB_V4REF|EW'] = ab_ew('none', None)
navs['AB_V5_baselatest|normal'] = ab_nav('news_baselatest', None, NORMAL)
for b, nm in ((None, 'ABW_V5'), ('B10', 'ABW_V5_B10')):
    for c in (NORMAL, STRESS): navs[f'{nm}|{c}'] = cum(R[('news', b, 'worst')][0][c])
    navs[f'{nm}|EW'] = cum(R[('news', b, 'worst')][2])
for t in [f's{i}' for i in range(20)] + ['neg', 'pos']: navs[f'AB_V5_FVMC_{t}|normal'] = ab_nav(f'mc{t}', None, NORMAL)

# 參考：o_v4 已發布淨值（AB_V4、V4），與本題 fv=None 的重現
NV4 = pd.read_parquet(L.V4DIR / 'navs_v4.parquet')
ref_diff = {'AB_V4': float((navs['AB_V4REF|normal'] / NV4['AB_V4|normal'].reindex(navs['AB_V4REF|normal'].index) - 1).abs().max()),
            'V4': float((navs['V4REF|normal'] / NV4['V4|normal'].reindex(navs['V4REF|normal'].index) - 1).abs().max())}
assert ref_diff['AB_V4'] < 1e-9 and ref_diff['V4'] < 1e-9, ref_diff
ABV4 = NV4['AB_V4|normal'].reindex(navs['AB_V5|normal'].index); ABV4s = NV4['AB_V4|stress'].reindex(navs['AB_V5|normal'].index)
mv2 = pd.read_parquet(L.AG / 'main_v2' / 'navs_gapfix.parquet')['EXFIN|n10|normal'].reindex(navs['AB_V5|normal'].index)
EWm = navs['AB_V5|EW']

out = {'runtime_note': '一次跑完，不改規格', 'ref_reproduction_maxreldiff': ref_diff}
tab = {}
for k in ('V5|normal', 'V5|stress', 'AB_V5|normal', 'AB_V5|stress', 'AB_V5_B10|normal', 'AB_V5_B10|stress', 'AB_V5_B05|normal', 'V5_B10|normal', 'V5_B05|normal',
          'ABW_V5|normal', 'ABW_V5_B10|normal', 'AB_V5_baselatest|normal', 'AB_V5|fullfee', 'AB_V4REF|normal'):
    nm = k.split('|')[0]; ewk = f'{nm}|EW' if f'{nm}|EW' in navs else 'AB_V5|EW'
    tab[k] = L.full(navs[k], navs[ewk], sd, extra=k in ('V5|normal', 'AB_V5|normal', 'AB_V5|stress', 'AB_V5_B10|normal', 'AB_V4REF|normal'))
out['table'] = tab
# ---- 逐 seed 分布（AB_V5 一般成本）----
def seed_stats(fvn, b, cost=NORMAL):
    ts, td1, cg = [], [], []
    for s in SEEDS:
        nv = cum(R[(fvn, b, s)][0][str(cost)]); x = L.xs(nv, sd)
        ts.append(L.tstat(x)); td1.append(L.tstat(x.drop(x.sort_values(ascending=False).index[:1]))); cg.append(R_cagr(nv))
    ts, td1 = np.array(ts), np.array(td1)
    return {'t_median': float(np.median(ts)), 't_min': float(ts.min()), 't_max': float(ts.max()), 't_p10': float(np.percentile(ts, 10)),
            'n_seed_t_below_2.30': int((ts < 2.30).sum()), 'n_seed_drop_best1_below_2.30': int((td1 < 2.30).sum()),
            'CAGR_median': float(np.median(cg)), 'CAGR_min': float(min(cg)), 'CAGR_max': float(max(cg)), 't_by_seed': ts.round(4).tolist()}
def R_cagr(nv): return L.R.cagr_mdd(nv, L.A0, L.B0)[0]
out['seed_dist'] = {'AB_V5|normal': seed_stats('news', None), 'AB_V5|stress': seed_stats('news', None, STRESS),
                    'AB_V5_B10|normal': seed_stats('news', 'B10'), 'AB_V4REF|normal': seed_stats('none', None)}
log('seed', out['seed_dist']['AB_V5|normal']['t_median'])
# ---- 成本損益兩平 ----
pts = [(x, L.full(navs[f'AB_V5|slip{x}'], EWm, sd, extra=False)['ALL']['t_0050']) for x in SLIPS]
def cross(th):
    for (s1, a), (s2, b) in zip(pts, pts[1:]):
        if a >= th > b: return s1 + (s2 - s1) * (a - th) / (a - b)
    return None if pts[0][1] < th else float('inf')
out['breakeven_AB_V5'] = {'grid': pts, 'slip_at_t2.3': cross(2.3), 'slip_at_t2.0': cross(2.0),
                          'fullfee_t': tab['AB_V5|fullfee']['ALL']['t_0050'], 'fullfee_CAGR': tab['AB_V5|fullfee']['ALL']['CAGR']}
# ---- 配對比較 ----
P = L.PER if hasattr(L, 'PER') else L4.PER
out['paired'] = {
    'AB_V5_minus_AB_V4': {p: L.paired(navs['AB_V5|normal'], ABV4, sd, *L4.PER[p]) for p in ('ALL', 'S1', 'S2', 'S3', 'S4', 'SEL', 'VAL', 'HOLD')},
    'AB_V5_stress_minus_AB_V4_stress': L.paired(navs['AB_V5|stress'], ABV4s, sd),
    'V5_minus_V4': {p: L.paired(navs['V5|normal'], NV4['V4|normal'].reindex(navs['V5|normal'].index), sd, *L4.PER[p]) for p in ('ALL', 'S1', 'S2', 'S3', 'S4')},
    'AB_V5_minus_main_v2': L.paired(navs['AB_V5|normal'], mv2, sd),
    'AB_V5_B10_minus_AB_V5': {p: L.paired(navs['AB_V5_B10|normal'], navs['AB_V5|normal'], sd, *L4.PER[p]) for p in ('ALL', 'S1', 'S2', 'S3', 'S4')},
    'AB_V5_baselatest_minus_AB_V5': L.paired(navs['AB_V5_baselatest|normal'], navs['AB_V5|normal'], sd)}
# ---- 首次值界限 ----
fvb = {}
for t in [f's{i}' for i in range(20)] + ['neg', 'pos']:
    st = L.full(navs[f'AB_V5_FVMC_{t}|normal'], EWm, sd, extra=False)['ALL']
    fvb[t] = {'CAGR': st['CAGR'], 't': st['t_0050'], 'paired_vs_AB_V5': L.paired(navs[f'AB_V5_FVMC_{t}|normal'], navs['AB_V5|normal'], sd)['mean_pp']}
mcs = [fvb[f's{i}'] for i in range(20)]
out['fv_bounds'] = {'per': fvb, 'MC_t_median': float(np.median([x['t'] for x in mcs])), 'MC_t_min': float(min(x['t'] for x in mcs)), 'MC_t_max': float(max(x['t'] for x in mcs)),
                    'MC_CAGR_min': float(min(x['CAGR'] for x in mcs)), 'MC_CAGR_max': float(max(x['CAGR'] for x in mcs)),
                    'worst_t': float(min(x['t'] for x in fvb.values())), 'best_t': float(max(x['t'] for x in fvb.values())),
                    'worst_CAGR': float(min(x['CAGR'] for x in fvb.values())), 'best_CAGR': float(max(x['CAGR'] for x in fvb.values()))}
log('fv bounds', out['fv_bounds']['worst_t'], out['fv_bounds']['best_t'])
# ---- 名單差異與機制（邊際名額）----
def plans(fvn, b): return [[(e, pk, None, d) for e, pk, d in R[(fvn, b, s)][1]] for s in SEEDS]
P5, P4 = plans('news', None), plans('none', None)
chg = [(str(a[3].date()), s) for s, (pa, pb) in enumerate(zip(P5, P4)) for a, b in zip(pa, pb) if set(a[1].tolist()) != set(b[1].tolist())]
nz = {}
for d, s in chg: nz.setdefault(d, set()).add(s)
pn, po = R[('news', None, None)][1], R[('none', None, None)][1]
chg_nonab = [str(a[2].date()) for a, b in zip(pn, po) if set(a[1].tolist()) != set(b[1].tolist())]
out['fv_pick_changes'] = {'nonAB_months_changed': chg_nonab, 'AB_months_changed_any_seed': sorted(nz), 'AB_seedmonths_changed': len(chg),
                          'AB_mean_months_changed_per_seed': len(chg) / 40}
out['mechanism'] = {'FV_in_vs_out（AB 逐 seed）': L.marginal_swaps([x for x in P5], [x for x in P4], t1),
                    'B10_gap_in_vs_displaced（AB 逐 seed）': L.marginal_swaps(plans('news', 'B10'), P5, t1),
                    'B10_gap_in_vs_displaced（非 AB）': L.marginal_swaps([[(e, pk, None, d) for e, pk, d in R[('news', 'B10', None)][1]]], [[(e, pk, None, d) for e, pk, d in R[('news', None, None)][1]]], t1)}
for t in ('s0', 'neg', 'pos'):
    out['mechanism'][f'FVMC_{t}_in_vs_out（AB 逐 seed）'] = L.marginal_swaps(plans(f'mc{t}', None), P5, t1)
# B10 缺口股入選股月（非 AB、AB 逐 seed）：數量、年份、2006～2012
GAP = g['GAP']
def gap_picks(pls_b, pls_0):
    rows = []
    for pa, pb in zip(pls_b, pls_0):
        for a, b in zip(pa, pb):
            for j in set(a[1].tolist()) - set(b[1].tolist()):
                if K.COLS[j] in GAP: rows.append((str(a[3].date()), K.COLS[j]))
    return rows
gp = gap_picks([[(e, pk, None, d) for e, pk, d in R[('news', 'B10', None)][1]]], [[(e, pk, None, d) for e, pk, d in R[('news', None, None)][1]]])
gpa = gap_picks(plans('news', 'B10'), P5)
out['B10_gap_picks'] = {'nonAB_n': len(gp), 'nonAB_2006_2012': sum(1 for d, c in gp if d <= '2012-12-31'), 'nonAB_by_year': pd.Series([d[:4] for d, c in gp]).value_counts().sort_index().to_dict(),
                        'AB_seedmean_n': len(gpa) / 40, 'AB_seedmean_2006_2012': sum(1 for d, c in gpa if d <= '2012-12-31') / 40}
# ---- 穩健性：分產業（V5 非 AB 名單貢獻）、最大貢献股 ----
plV5 = [(e, pk, np.array([], int), d) for e, pk, d in R[('news', None, None)][1]]
con, per = V.stock_contrib(plV5, t1)
ind = {}
for c, v in con.items():
    i = V.SL['industry'].get(c) if c in V.SL.index else None
    if not isinstance(i, str): i = '（已下市）'
    ind[i] = ind.get(i, 0) + v
tot = sum(con.values())
out['robust'] = {'industry_share_top10': sorted(((k, round(v / tot, 3)) for k, v in ind.items()), key=lambda x: -x[1])[:10],
                 'top_contrib_ALL': V.top_contrib(plV5, t1, L.A0, L.B0), 'top_contrib_HOLD': V.top_contrib(plV5, t1, '2024-01-01', L.B0)}
EL = {'半導體業', '電子零組件業', '其他電子業', '光電業', '通信網路業', '電腦及週邊設備業', '電子通路業', '資訊服務業'}
def ind_of(j):
    c = K.COLS[j]; i = V.SL['industry'].get(c) if c in V.SL.index else None
    return 'elec' if i in EL else 'non_elec'
sw = {'elec': 0, 'non_elec': 0}
for pa, pb in zip(P5, P4):
    for a, b in zip(pa, pb):
        for j in set(a[1].tolist()) ^ set(b[1].tolist()): sw[ind_of(j)] += 1
out['fv_swaps_by_industry_group（AB 逐 seed 合計名額）'] = sw
res['RESULTS'] = out
RES_F.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o)), encoding='utf-8')
pd.DataFrame(navs).to_parquet(HERE / 'navs_v5.parquet')
pd.DataFrame({f'AB_V5|normal|s{s}': cum(R[('news', None, s)][0][NORMAL]) for s in SEEDS}).to_parquet(HERE / 'seed_navs_v5.parquet')
for (fvn, b), nm in ((('news', None), 'V5'), (('news', 'B10'), 'V5_B10')):
    pd.DataFrame([(str(d.date()), str(K.IDX[e].date()), ','.join(K.COLS[j] for j in pk)) for e, pk, d in R[(fvn, b, None)][1]], columns=['signal', 'exec', 'picks']).to_csv(HERE / f'plan_{nm}.csv', index=False)
pd.DataFrame([(s, str(d.date()), ','.join(K.COLS[j] for j in pk)) for s in SEEDS for e, pk, d in R[('news', None, s)][1]], columns=['seed', 'signal', 'picks']).to_csv(HERE / 'plan_AB_V5_seeds.csv', index=False)
log('done', tab['AB_V5|normal']['ALL']['CAGR'], tab['AB_V5|normal']['ALL']['t_0050'])

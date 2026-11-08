#!/usr/bin/env python3
"""第 4 步：凍結後一次跑完（斷言 SPEC 已凍結、所有輸入 sha 未變、v7surv.json 在凍結後產生）。
R1 營收可得時點、J0、R2 存活者缺口（面板 A；面板 B 讀 v7surv.json）、R3 處置、描述附表。不計算任何策略報酬
（J0 排序位置只讀既有 j-scan 淨值算 13 個 j 的年化超額排序，不重算策略）。輸出 results.json（ERRATUM 區塊）、data/slot_avail.csv。
python3 -I -B s4_count.py"""
import sys, json, time, hashlib, importlib.util
from pathlib import Path
HERE = Path(__file__).resolve().parent
AG = HERE.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
assert 'ERRATUM' not in R, '只跑一次：ERRATUM 已存在'
assert sha(HERE / 'SPEC.json') == R['PREREG']['sha256'], 'SPEC 未凍結或被改'
SPEC = json.loads((HERE / 'SPEC.json').read_text(encoding='utf-8'))
PFX = json.loads((HERE / 'POSTFREEZE_FIXES.json').read_text(encoding='utf-8')) if (HERE / 'POSTFREEZE_FIXES.json').exists() else {'fixes': []}
ALLOWED = {fx['file']: fx['new_sha256'] for fx in PFX['fixes']}
for f, h in SPEC['files_sha256'].items():
    assert sha(HERE / f) == h or (f in ALLOWED and sha(HERE / f) == ALLOWED[f]), f'{f} 在凍結後被改（且未登記在 POSTFREEZE_FIXES.json）'
V7S = json.loads((HERE / 'v7surv.json').read_text(encoding='utf-8'))
assert V7S['utc'] > SPEC['frozen_utc'], 'v7surv.json 必須在凍結後產生'
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import pclib as P
import avail as A
assert P.ENGINE_SHA == SPEC['engine_sha256']
T0 = time.time()
RUN_UTC = pd.Timestamp.now('UTC').isoformat()
DD = pd.read_csv(HERE / 'data/census_days.csv', dtype=str)
SL = pd.read_csv(HERE / 'data/c0_slots.csv', dtype={'code': str, 'sig': str})
PL = pd.read_csv(P.SCR / 'c0_pool.csv.gz', dtype={'code': str, 'sig': str}).merge(DD[['sig', 'case', 'rev_month', 'seg4', 'formal']], on='sig')
SL = SL.merge(DD[['sig', 'rev_month']], on='sig')
E6 = A.Evidence('sig'); E5 = A.Evidence('15')

# ================= R1：營收可得時點 =================
X = SL[SL.case.isin(['D=S', 'D>S'])].copy()
res = [E6.classify(pd.Timestamp(s), c, y, 'V6') for s, c, y in zip(X.sig, X.code, X.rev_month)]
X['state'] = [r[0] for r in res]; X['second'] = [r[2] for r in res]
X['evidence'] = [json.dumps({k: v for k, v in r[1].items()}, ensure_ascii=False, default=str) for r in res]
X['state_V5'] = [E5.classify(pd.Timestamp(s), c, y, 'V5')[0] for s, c, y in zip(X.sig, X.code, X.rev_month)]
X['upper'] = X.state != 'AVAIL'; X['lower'] = X.state == 'UNAVAIL'
X.drop(columns=['evidence']).to_csv(HERE / 'data/slot_avail.csv', index=False)
X[['sig', 'code', 'state', 'evidence']].to_csv(HERE / 'data/slot_avail_evidence.csv', index=False)
F = X[X.formal.astype(str) == 'True']
def ratio(sub, denom, col='upper'):
    n = int(sub[col].sum()); by = sub[sub[col]].code.value_counts()
    top = by.index[0] if len(by) else None
    return {'n': n, 'denom': denom, 'share': n / denom, 'n_days': int(sub[sub[col]].sig.nunique()), 'n_issuers': int(len(by)),
            'top_issuer': top, 'top_n': int(by.iloc[0]) if len(by) else 0, 'share_ex_top': (n - (int(by.iloc[0]) if len(by) else 0)) / denom}
den = SPEC['denominators']
r1 = {}
s_a = F[F.sig <= '2013-12-31']; s_b = F; s_c = F[F.sig >= '2019-01-01']
assert len(s_a) == 330 and (s_a.case == 'D=S').all() and len(s_b) == den['C0_ALL_2006_2026']['DS_slots'] + den['C0_ALL_2006_2026']['DgtS_slots']
r1['r1a_2006_2013'] = {'upper': ratio(s_a, 930), 'lower': ratio(s_a, 930, 'lower')}
r1['r1b_ALL'] = {'upper': ratio(s_b, 2450), 'lower': ratio(s_b, 2450, 'lower')}
r1['r1c_2019_on'] = {'upper': ratio(s_c, 920), 'lower': ratio(s_c, 920, 'lower')}
dist = {}
for k, g in [('2006_2013', s_a), ('2014_2018', F[(F.sig >= '2014-01-01') & (F.sig <= '2018-12-31')]), ('2019_2024-05', F[(F.sig >= '2019-01-01') & (F.sig <= '2024-05-31')]),
             ('2024-06_', F[F.sig >= '2024-06-01']), ('ALL', F), ('latest_2026-10-12', X[X.formal.astype(str) != 'True'])]:
    dist[k] = {'n_slots': int(len(g))} | {s: int((g.state == s).sum()) for s in ('AVAIL', 'UNDET', 'UNAVAIL', 'NONE')} | \
              {'share_of_DS_slots': {s: round(float((g.state == s).mean()), 4) for s in ('AVAIL', 'UNDET', 'UNAVAIL', 'NONE')} if len(g) else {}} | \
              {'V5_AVAIL': int((g.state_V5 == 'AVAIL').sum()), 'second': {k2: int(v) for k2, v in g.second.value_counts().items()}}
th = 0.05
J1 = {'G5_lookahead_not_excluded': r1['r1a_2006_2013']['upper']['share'] > th or r1['r1b_ALL']['upper']['share'] > th,
      'G5_lookahead_proven': (r1['r1a_2006_2013']['upper']['share'] > th and r1['r1a_2006_2013']['lower']['share'] > th) or
                             (r1['r1b_ALL']['upper']['share'] > th and r1['r1b_ALL']['lower']['share'] > th),
      'exec_note': r1['r1c_2019_on']['upper']['share'] > th,
      'exec_proven': r1['r1c_2019_on']['upper']['share'] > th and r1['r1c_2019_on']['lower']['share'] > th}
# 子樣本（描述）：同日股池
Q = PL[PL.case.isin(['D=S', 'D>S']) & (PL.formal.astype(str) == 'True')].copy()
Q['state'] = [E6.classify(pd.Timestamp(s), c, y, 'V6')[0] for s, c, y in zip(Q.sig, Q.code, Q.rev_month)]
def rate(g, s='AVAIL'): return round(float((g.state == s).mean()), 4) if len(g) else None
sub = {}
for k, (a, b) in {'2013_2018': ('2013-01-01', '2018-12-31'), '2019_2024-05': ('2019-01-01', '2024-05-31'), '2024-06_': ('2024-06-01', '2026-12-31'), '2006_2013': ('2006-01-01', '2013-12-31')}.items():
    f = F[(F.sig >= a) & (F.sig <= b)]; q = Q[(Q.sig >= a) & (Q.sig <= b)]
    pm = q.groupby('sig').apply(lambda g: (g.state == 'AVAIL').mean(), include_groups=False)
    sub[k] = {'picks_AVAIL': rate(f), 'pool_AVAIL': rate(q), 'pool_AVAIL_dayavg': round(float(pm.mean()), 4) if len(pm) else None,
              'picks_UNAVAIL': rate(f, 'UNAVAIL'), 'pool_UNAVAIL': rate(q, 'UNAVAIL'),
              'picks_rev_ge_0.9_AVAIL': rate(f[f.pr_rev >= 0.9]), 'n_rev_ge_0.9': int((f.pr_rev >= 0.9).sum()),
              'picks_rev_lt_0.5_AVAIL': rate(f[f.pr_rev < 0.5]), 'n_rev_lt_0.5': int((f.pr_rev < 0.5).sum()),
              'pool_rev_ge_0.9_AVAIL': rate(q[q.pr_rev >= 0.9]), 'pool_rev_lt_0.5_AVAIL': rate(q[q.pr_rev < 0.5]),
              'DgtS_picks_UNAVAIL': rate(f[f.case == 'D>S'], 'UNAVAIL'), 'DS_picks_UNAVAIL': rate(f[f.case == 'D=S'], 'UNAVAIL'),
              'DgtS_picks_AVAIL': rate(f[f.case == 'D>S']), 'DS_picks_AVAIL': rate(f[f.case == 'D=S'])}
falsify = {k: (v['picks_AVAIL'] is not None and v['pool_AVAIL'] is not None and v['picks_AVAIL'] < v['pool_AVAIL'] - 0.10) for k, v in sub.items()}

# ================= J0 =================
PJ = pd.read_csv(AG / 'o_v7_listing_exec_engine/data/plans_j7.csv', dtype=str)
PJ = PJ[(PJ.strat == 'AB_V7')]
PJ['rev_month'] = [str(pd.Period(s, 'M') - 1) for s in PJ.signal]
def j0_class(df):
    rows = []
    for r in df.itertuples():
        for c in (r.picks or '').split(','):
            if c: rows.append((r.seed, r.signal, c, E6.classify(pd.Timestamp(r.signal), c, r.rev_month, 'V6')[0]))
    return pd.DataFrame(rows, columns=['seed', 'signal', 'code', 'state'])
j0s = PJ[(PJ.j == '0') & (PJ.seed == '0') & (PJ.signal >= '2019-01-01')]
JC = j0_class(j0s)
allj = den['jscan_AB_V7_seed0']['2019_']['all_j_slots']
J0 = {'share_J0_of_all_j_2019_on': den['jscan_AB_V7_seed0']['2019_']['share'], 'share_J0_of_all_j_ALL': den['jscan_AB_V7_seed0']['ALL']['share'],
      'J0_slots_2019_on_seed0': int(len(JC)), 'states': {s: int((JC.state == s).sum()) for s in ('AVAIL', 'UNDET', 'UNAVAIL', 'NONE')},
      'upper_share_of_all_j': float((JC.state != 'AVAIL').sum() / allj), 'lower_share_of_all_j': float((JC.state == 'UNAVAIL').sum() / allj),
      'upper_share_of_J0': float((JC.state != 'AVAIL').mean()), 'lower_share_of_J0': float((JC.state == 'UNAVAIL').mean()),
      'by_period': {k: {s: int((g.state == s).sum()) for s in ('AVAIL', 'UNDET', 'UNAVAIL', 'NONE')} for k, g in
                    (('2019_2024-05', JC[JC.signal <= '2024-05-31']), ('2024-06_', JC[JC.signal >= '2024-06-01']))}}
JA = j0_class(PJ[(PJ.j == '0') & (PJ.signal >= '2019-01-01')])
J0['all_40_seeds'] = {'n': int(len(JA)), 'states': {s: int((JA.state == s).sum()) for s in ('AVAIL', 'UNDET', 'UNAVAIL', 'NONE')}}
# J0 在既有 j-scan（可執行 V7X，100 萬一般成本 GROW）的排序位置（只讀）
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
FC = _load('fam_common_ro', AG / 'o_main_spa_breaks/fam_common.py')
EX = pd.read_parquet(AG / 'o_exec_engine_rewrite_v7x/out/navs_oos_R4.parquet')
r50 = FC.win_ret(FC.e50_nav()).to_numpy(); WIN = FC.WIN
def ann(r): return float(np.prod(1 + r) ** (12 / len(r)) - 1) * 100
rank = {}
for pn, (a, b) in {'ALL': ('2006-04-01', '2026-10-12'), 'SEG3_2019_on': ('2019-01-01', '2026-10-12'), 'SEG1': ('2006-04-01', '2012-12-31')}.items():
    mk = (WIN >= pd.Timestamp(a)) & (WIN <= pd.Timestamp(b))
    gx = {}
    for j in range(13):
        r = FC.win_ret(EX[f'B10|J{j}|K5T|normal|100萬|GROW']).to_numpy(); r = np.where(np.isfinite(r), r, 0.0)
        gx[j] = ann(r[mk]) - ann(r50[mk])
    v = np.array([gx[j] for j in range(13)]); order = list(np.argsort(v))
    med = float(np.median(v)); p10 = float(np.percentile(v, 10))
    pos = order.index(0) + 1
    rank[pn] = {'gex_by_j': {f'J{j}': round(gx[j], 3) for j in range(13)}, 'median': med, 'p10': p10, 'J0_gex': gx[0], 'J0_rank_from_low': pos,
                'J0_is_median_j': pos == 7, 'J0_is_p10_bracket': pos in (2, 3), 'p10_interp_between': [f'J{order[1]}', f'J{order[2]}'], 'median_j': f'J{order[6]}'}
J0['jscan_rank_exec_V7X_100w_normal_GROW'] = rank
J0['check_vs_integrate_o11'] = {'ALL_median_gex_mine': rank['ALL']['median'], 'integrate_o11': 0.7665795735346084, 'p10_mine': rank['ALL']['p10'], 'p10_integrate_o11': -3.128070402690514}

# ================= R2：存活者缺口（面板 A） =================
def u_run(mode, which='GOV'):
    S = P.VU(mode, lo='2013-01-01', hi='2026-12-31', which=which)
    UL = P.u_lists(S)
    return UL
UA = u_run('opt'); BASE = {str(r['sig'].date()): r['picks'] for r in P.c0_cached(False)}
assert {k: v['picks'] for k, v in u_run('base').items()} == BASE
def useg(UL, a, b, denom):
    x = {k: v for k, v in UL.items() if a <= k <= b}
    ins = [(k, c, v['causes'].get(c)) for k, v in x.items() for c in v['picks'] if c in set(v['aff'])]
    by = pd.Series([c for _, c, _ in ins]).value_counts() if ins else pd.Series(dtype=int)
    top = by.index[0] if len(by) else None
    n = len(ins)
    return {'U_opt': n, 'denom': denom, 'share': n / denom, 'n_days': len({k for k, _, _ in ins}), 'n_issuers': int(len(by)), 'top_issuer': top,
            'top_n': int(by.iloc[0]) if len(by) else 0, 'share_ex_top': (n - (int(by.iloc[0]) if len(by) else 0)) / denom,
            'changed': sum(len(set(BASE[k]) - set(v['picks'])) for k, v in x.items()),
            'by_cause': pd.Series([z for _, _, z in ins]).value_counts().to_dict() if ins else {},
            'by_issuer': {c: int(n_) for c, n_ in by.items()}, 'by_year': pd.Series([k[:4] for k, _, _ in ins]).value_counts().sort_index().to_dict() if ins else {},
            'n_aff_stockmonths': sum(len(v['aff']) for v in x.values()),
            'aff_by_cause': pd.Series([z for v in x.values() for c, z in v['causes'].items() if c in set(v['aff'])]).value_counts().to_dict()}
R2A = {'2013_2026': useg(UA, '2013-01-01', '2026-12-31', 1640), '2013_2018': useg(UA, '2013-01-01', '2018-12-31', 720),
       '2019_2026': useg(UA, '2019-01-01', '2026-12-31', 920), '2014_2018': useg(UA, '2014-01-01', '2018-12-31', 600)}
desc = {}
for md, wh in (('p90', 'GOV'), ('carry', 'GOV'), ('opt', 'ALL')):
    UL = u_run(md, wh)
    desc[f'U_{md}_{wh}'] = {k: {kk: vv for kk, vv in useg(UL, a, b, dn).items() if kk in ('U_opt', 'share', 'n_issuers', 'by_cause')}
                            for k, (a, b, dn) in {'2013_2026': ('2013-01-01', '2026-12-31', 1640), '2013_2018': ('2013-01-01', '2018-12-31', 720), '2019_2026': ('2019-01-01', '2026-12-31', 920)}.items()}
R2B = V7S['seg']
gov = {k: max(R2A[k]['share'], R2B[k]['share']) for k in ('2013_2026', '2013_2018', '2019_2026')}
J2 = {'G4_seg3_survivorship_not_excluded': gov['2013_2026'] > th, 'G4_seg2_survivorship_not_excluded': gov['2013_2018'] > th,
      'governing_share_2013_2026': gov['2013_2026'], 'governing_share_2013_2018': gov['2013_2018'], 'share_2019_2026_desc': gov['2019_2026'],
      'panelA_vs_B_same_side': {k: (R2A[k]['share'] > th) == (R2B[k]['share'] > th) for k in gov}}

# ================= R3：處置 =================
DU = pd.read_csv(HERE / 'raw/disposal_official_universe.csv', dtype={'code': str}, parse_dates=['start', 'end'])
DH = pd.read_csv(HERE / 'raw/disposal_hist_rejudge.csv', dtype={'code': str}, parse_dates=['announce', 'start', 'end'])
byc = {c: g for c, g in DU.groupby('code')}; byh = {c: g for c, g in DH.groupby('code')}
def indisp(g, d):
    if g is None: return False
    return bool(((g.start <= d) & (g.end >= d)).any())
SLf = SL[SL.formal.astype(str) == 'True'].copy()
SLf['d_sig'] = [indisp(byc.get(c), pd.Timestamp(s)) for c, s in zip(SLf.code, SLf.sig)]
SLf['d_trd'] = [indisp(byc.get(c), pd.Timestamp(e)) for c, e in zip(SLf.code, SLf.exec)]
SLf['h_sig'] = [indisp(byh.get(c), pd.Timestamp(s)) for c, s in zip(SLf.code, SLf.sig)]
SLf['h_trd'] = [indisp(byh.get(c), pd.Timestamp(e)) for c, e in zip(SLf.code, SLf.exec)]
def dseg(a, b):
    x = SLf[(SLf.sig >= a) & (SLf.sig <= b)]; n = len(x)
    o = {'slots': n}
    for col in ('d_sig', 'd_trd', 'h_sig', 'h_trd'):
        y = x[x[col]]; by = y.code.value_counts()
        o[col] = {'n': int(len(y)), 'share': len(y) / n if n else None, 'n_days': int(y.sig.nunique()), 'n_issuers': int(len(by)),
                  'share_ex_top': (len(y) - (int(by.iloc[0]) if len(by) else 0)) / n if n else None,
                  'slots': [f'{s} {c}' for s, c in zip(y.sig, y.code)]}
    return o
R3 = {'2006_2013': dseg('2006-01-01', '2013-12-31'), '2014_2021-09-23': dseg('2014-01-01', '2021-09-23'), '2021-09-24_': dseg('2021-09-24', '2026-12-31'),
      '2006_2021-09-23': dseg('2006-01-01', '2021-09-23')}
gov3 = max(R3['2006_2013']['d_sig']['share'], R3['2006_2013']['d_trd']['share'], R3['2014_2021-09-23']['d_sig']['share'], R3['2014_2021-09-23']['d_trd']['share'])
J3 = {'governing_share': gov3, 'G5_disposal_not_PIT': gov3 > th, 'action': ('G5 加「處置排除未 PIT」' if gov3 > th else '只更新 o16 第 5 節註記的數字'),
      'known_4_reproduced': sorted(R3['2006_2013']['d_sig']['slots']) == sorted(['2009-05-11 2383', '2009-07-13 2545', '2009-11-11 1702', '2010-12-13 3406'])}

# ================= 描述：main_v2 已發布的最大貢獻股 =================
MV = json.loads((AG / 'main_v2/results.json').read_text(encoding='utf-8'))['RUN']['gapfix']['official']['EXFIN']['10']['contrib']
top = {}
for sg, v in MV.items():
    for c, pp in v.get('top5', []): top.setdefault(c, []).append((sg, pp))
upd = F[F.upper]
topx = {c: {'segments_top5': v, 'C0_slots_total': int((SL[(SL.formal.astype(str) == 'True')].code == c).sum()),
            'upper_unavail_DS_slots': [f'{s}:{st}' for s, st in zip(upd[upd.code == c].sig, upd[upd.code == c].state)],
            'lower_unavail_DS_slots': list(F[(F.code == c) & F.lower].sig)} for c, v in top.items()}

ER = {'run_utc': RUN_UTC, 'spec_sha256': R['PREREG']['sha256'], 'no_returns': '本程式未計算任何策略報酬、淨值或 t 值（J0 排序只讀既有 j-scan 淨值）',
      'R1': {'ratios': r1, 'distribution': dist, 'judgement': J1, 'subsamples_desc': sub, 'falsifier_picks_10pp_below_pool': falsify},
      'J0': J0,
      'R2': {'panelA_C0': R2A, 'panelB_V7': {k: {kk: vv for kk, vv in v.items()} for k, v in R2B.items()}, 'descriptive_panelA': desc, 'judgement': J2},
      'R3': {'counts': R3, 'judgement': J3},
      'desc_top_contributors': topx,
      'runtime_s': round(time.time() - T0, 1)}
R['ERRATUM'] = ER
(HERE / 'results.json').write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
print(json.dumps({'R1': {k: {kk: round(vv['share'], 4) for kk, vv in v.items()} for k, v in r1.items()}, 'J1': J1, 'dist': dist, 'J0': {k: J0[k] for k in J0 if k != 'jscan_rank_exec_V7X_100w_normal_GROW'},
                  'J0rank': {k: {kk: v[kk] for kk in ('J0_rank_from_low', 'J0_is_median_j', 'J0_is_p10_bracket', 'median', 'p10', 'J0_gex', 'median_j', 'p10_interp_between')} for k, v in rank.items()},
                  'R2A': {k: {kk: v[kk] for kk in ('U_opt', 'share', 'n_days', 'n_issuers', 'top_issuer', 'top_n', 'share_ex_top', 'by_cause')} for k, v in R2A.items()},
                  'R2B': {k: {kk: v[kk] for kk in ('U_opt', 'share', 'n_days', 'n_issuers', 'top_issuer', 'top_n', 'share_ex_top', 'by_cause')} for k, v in R2B.items()},
                  'J2': J2, 'R3': {k: {c: (v[c]['n'], round(v[c]['share'], 4)) for c in ('d_sig', 'd_trd', 'h_sig', 'h_trd')} for k, v in R3.items()}, 'J3': J3,
                  'sub': sub}, ensure_ascii=False, indent=1, default=str))

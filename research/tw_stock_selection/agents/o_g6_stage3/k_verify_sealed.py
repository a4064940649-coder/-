#!/usr/bin/env python3
"""K 保管員（接手，2026-11-14）：逐項核對 sealed/ 內的封存目標，全部通過才寫 sealed/SHA256.txt。只讀；不 import 任何研究員程式、不跑引擎、不算新績效數字。
核對項：
 A 紙上名單 paper_lists_M1／M2：逐列等於紙上正式名單 o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz 的 M1／M2 列；由封存檔重算的 lists_sha 等於
   integrate_o20/results.json RESULTS_PAPER.lists_sha 與 k_paper_stats.json；M1→M2 換股等於 RESULTS_PAPER.list_diffs；exec＝紙上面板的下一列；inserted ⊆ picks。
 B V7X 名單 v7x_plans_M1／M2：解壓後 sha256 等於 integrate_o20/v7x_plans_check.json 記錄的 plans_base.csv／plans_M2.csv；每變體列數、放回槽平均數、ends 等於
   該檔與 o_delist_rev_backfill_allpit/results.json RESULTS_V7X_PLANS；由封存檔重算 M1↔M2 差異（列數、組數、訊號日數、換股組數）並逐列等於
   integrate_o20/data/v7x_plan_diff_M2.csv.gz；exec／exec_idx 與 V7 面板索引一致；k_v7x_plans_check.json 的檢查等於 integrate_o20/gen_v7x_plans.log 的檢查。
 C 淨值／逐窗檔：三個檔與 integrate_o20/data/ 同名檔位元組相同。
 D targets.json：每個帶 src 的數字回到來源檔逐一比對（完全相等）；涵蓋格數；列出「未保存」項。
python3 -I -B k_verify_sealed.py"""
import gzip, hashlib, io, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; AG = HERE.parent; SEAL = HERE / 'sealed'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
R20 = json.loads((AG / 'integrate_o20/results.json').read_text(encoding='utf-8'))
res = {}; ok = {}
VP = ['ALLPIT_REAL_opt', 'ALLPIT_REAL_p90', 'ALLPIT_REAL_carry']; VX = ['REAL_opt', 'REAL_p90', 'REAL_carry']

# ---------------- A 紙上名單
OFF = pd.read_csv(AG / 'o_g6_allpit_list_trace_stage2/out/k_paper_lists.csv.gz', dtype=str)
KS = json.loads((AG / 'o_g6_allpit_list_trace_stage2/out/k_paper_stats.json').read_text(encoding='utf-8'))
PIDX = pd.read_parquet('/tmp/claude-0/-home-user--/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/panel/flag.parquet', columns=['2330']).index
A = {}; PL = {}
for tag in ('M1', 'M2'):
    D = pd.read_csv(SEAL / f'paper_lists_{tag}.csv.gz', dtype=str, keep_default_na=False); PL[tag] = D
    assert list(D.columns) == ['mask', 'variant', 'signal', 'exec', 'picks', 'inserted'] and set(D['mask']) == {tag} and list(dict.fromkeys(D.variant)) == VP
    for v in VP:
        d = D[D.variant == v].reset_index(drop=True); o = OFF[(OFF['mask'] == tag) & (OFF.variant == v)].reset_index(drop=True)
        ls = hashlib.sha256(json.dumps({s: p.split(',') for s, p in zip(d.signal, d.picks)}, sort_keys=True).encode()).hexdigest()
        loc = PIDX.get_indexer(pd.to_datetime(d.signal))
        A[f'{tag}|{v}'] = dict(
            n_rows=len(d), n_rows_official=len(o),
            rows_eq_official=bool(len(d) == len(o) and (d.signal.values == o.signal.values).all() and (d.picks.values == o.picks.values).all()),
            lists_sha=ls, lists_sha_eq_integrate_o20=ls == R20['RESULTS_PAPER']['lists_sha'][f'{tag}|{v}'], lists_sha_eq_k_paper_stats=ls == KS[f'{tag}|{v}|lists_sha'],
            exec_is_next_panel_row=bool((loc >= 0).all() and (PIDX[loc + 1] == pd.to_datetime(d['exec'])).all()),
            ten_picks_every_row=bool(all(len(p.split(',')) == 10 and len(set(p.split(','))) == 10 for p in d.picks)),
            inserted_subset_of_picks=bool(all(set(q for q in i.split(',') if q) <= set(p.split(',')) for p, i in zip(d.picks, d.inserted))),
            signals_sorted_unique=bool(d.signal.is_monotonic_increasing and d.signal.is_unique))
ld = {}
for v in VP:
    a = PL['M1'][PL['M1'].variant == v].set_index('signal').picks; b = PL['M2'][PL['M2'].variant == v].set_index('signal').picks
    assert list(a.index) == list(b.index)
    mine = [dict(signal=s, out=sorted(set(x.split(',')) - set(y.split(','))), into=sorted(set(y.split(',')) - set(x.split(',')))) for s, x, y in zip(a.index, a, b) if set(x.split(',')) != set(y.split(','))]
    ref = R20['RESULTS_PAPER']['list_diffs'][f'{v}|M1->M2']
    ld[v] = dict(n_diff_days=len(mine), n_ref=R20['RESULTS_PAPER']['n_list_diffs'][f'{v}|M1->M2'],
                 eq_integrate_o20_list_diffs=[dict(signal=r['signal'], out=sorted(r['out']), into=sorted(r['into'])) for r in ref] == mine,
                 n_rows_order_only_diff=int(sum((x != y) and set(x.split(',')) == set(y.split(',')) for x, y in zip(a, b))))
res['A_paper_lists'] = dict(per_list=A, M1_to_M2=ld)
ok['A_paper_lists'] = all(all(v[k] for k in ('rows_eq_official', 'lists_sha_eq_integrate_o20', 'lists_sha_eq_k_paper_stats', 'exec_is_next_panel_row', 'ten_picks_every_row',
                                             'inserted_subset_of_picks', 'signals_sorted_unique')) for v in A.values()) \
    and all(v['eq_integrate_o20_list_diffs'] and v['n_diff_days'] == v['n_ref'] for v in ld.values())

# ---------------- B V7X 名單
REF = json.loads((AG / 'integrate_o20/v7x_plans_check.json').read_text(encoding='utf-8'))
T1 = json.loads((AG / 'o_delist_rev_backfill_allpit/results.json').read_text(encoding='utf-8'))['RESULTS_V7X_PLANS']
KC = json.loads((HERE / 'k_v7x_plans_check.json').read_text(encoding='utf-8'))
LOGCHK = json.loads([l for l in (AG / 'integrate_o20/gen_v7x_plans.log').read_text().splitlines() if l.startswith('{')][-1])
VIDX = pd.read_parquet(AG / 'o_v7_listing_exec_engine/data/flag.parquet', columns=['2330']).index
B = {}; XP = {}
for tag, dv in (('M1', 'base'), ('M2', 'M2')):
    raw = gzip.decompress((SEAL / f'v7x_plans_{tag}.csv.gz').read_bytes()); h = hashlib.sha256(raw).hexdigest()
    D = pd.read_csv(io.BytesIO(raw), dtype={'variant': str, 'signal': str, 'exec': str, 'picks': str, 'inserted': str}, keep_default_na=False); XP[tag] = D
    assert list(D.columns) == ['variant', 'j', 'seed', 'signal', 'exec', 'exec_idx', 'picks', 'inserted']
    cnt = {v: dict(n_rows=int((D.variant == v).sum()),
                   inserted_slots_mean_per_plan=float(D[D.variant == v].groupby(['j', 'seed']).inserted.apply(lambda s: s.str.split(',').apply(lambda z: len([q for q in z if q])).sum()).mean())) for v in VX}
    loc = VIDX.get_indexer(pd.to_datetime(D.signal))
    g = D.groupby(['variant', 'j', 'seed'])
    last_sig = D.groupby(['variant', 'j']).signal.max()
    B[tag] = dict(csv_sha256=h, csv_sha256_eq_integrate_o20=h == REF['out'][dv]['sha256'], csv_bytes=len(raw), n_rows=len(D), counts=cnt,
                  counts_eq_integrate_o20=cnt == REF['out'][dv]['counts'],
                  n_plans=int(g.ngroups), variants=sorted(D.variant.unique()), j_set=sorted(int(x) for x in D.j.unique()), seed_set=sorted(int(x) for x in D.seed.unique()),
                  key_unique=bool(not D.duplicated(['variant', 'j', 'seed', 'signal']).any()),
                  exec_idx_is_signal_row_plus_1=bool((loc >= 0).all() and (loc + 1 == D.exec_idx.values).all()),
                  exec_eq_panel_date_at_exec_idx=bool((VIDX[D.exec_idx.values] == pd.to_datetime(D['exec'])).all()),
                  ten_picks_every_row=bool(D.picks.str.count(',').eq(9).all()),
                  inserted_subset_of_picks=bool(all(set(q for q in i.split(',') if q) <= set(p.split(',')) for p, i in zip(D.picks, D.inserted))),
                  signal_days_same_across_seeds_and_variants=bool(D.groupby(['j', 'signal']).size().eq(60).all()),
                  all_signals_before_end_day=bool(all(last_sig[(v, j)] < REF['out'][dv]['ends'][f'{v}|J{j}'] for v in VX for j in range(13))),
                  ends_k_eq_integrate_o20=KC['out'][dv]['ends'] == REF['out'][dv]['ends'],
                  k_check_sha_eq=KC['out'][dv]['sha256'] == h, k_check_counts_eq=KC['out'][dv]['counts'] == cnt)
B['M1']['counts_eq_T1_RESULTS_V7X_PLANS'] = all(B['M1']['counts'][v] == T1['counts'][v] for v in VX)
B['M1']['ends_eq_T1_RESULTS_V7X_PLANS'] = all(T1['ends'].get(k) == e for k, e in KC['out']['base']['ends'].items()) and len(KC['out']['base']['ends']) == 39
a, b = XP['M1'], XP['M2']
m = a.merge(b, on=['variant', 'j', 'seed', 'signal'], suffixes=('_b', '_m'), how='outer', indicator=True)
dd = m[(m._merge != 'both') | (m.picks_b != m.picks_m)].copy()
sd = lambda x, y: ','.join(sorted(set(str(x).split(',')) - set(str(y).split(','))))
dd['out'] = [sd(x, y) for x, y in zip(dd.picks_b, dd.picks_m)]; dd['into'] = [sd(y, x) for x, y in zip(dd.picks_b, dd.picks_m)]
ev = dd.groupby(['signal', 'out', 'into']).size().reset_index(name='n_plans')
mine = dict(rows_changed=int(len(dd)), plans_changed=int(dd.groupby(['variant', 'j', 'seed']).ngroups),
            plans_changed_by_variant={v: int(dd[dd.variant == v].groupby(['j', 'seed']).ngroups) for v in VX},
            n_signal_days=int(dd.signal.nunique()), n_distinct_swaps=int(len(ev)), ends_eq_base=KC['out']['M2']['ends'] == KC['out']['base']['ends'],
            swaps_pre2021_09_24=int((ev.signal < '2021-09-24').sum()))
RD = pd.read_csv(AG / 'integrate_o20/data/v7x_plan_diff_M2.csv.gz', dtype=str, keep_default_na=False)
key = ['variant', 'j', 'seed', 'signal']
x = dd.assign(j=dd.j.astype(str), seed=dd.seed.astype(str))[key + ['exec_b', 'exec_m', 'picks_b', 'picks_m', 'out', 'into']].sort_values(key).reset_index(drop=True)
y = RD[key + ['exec_b', 'exec_m', 'picks_b', 'picks_m', 'out', 'into']].sort_values(key).reset_index(drop=True)
B['M1_vs_M2'] = dict(recomputed_from_sealed=mine, eq_integrate_o20_log=mine == LOGCHK['M2'], eq_k_gen_plans_check=mine == KC['checks']['M2'],
                     n_rows_only_in_one=int((m._merge != 'both').sum()), exec_same_on_common_rows=bool((m.exec_b == m.exec_m).all()),
                     n_rows_order_only_diff=int(((dd.out == '') & (dd.into == '')).sum()), n_rows_set_diff=int(((dd.out != '') | (dd.into != '')).sum()),
                     diff_rows_eq_integrate_o20_diff_file=bool(x.shape == y.shape and (x.values == y.values).all()), integrate_o20_diff_rows=len(RD))
B['k_gen_plans_checks'] = dict(base_counts_eq_T1=KC['checks']['base_counts_eq_T1'], base_ends_eq_T1=KC['checks']['base_ends_eq_T1'],
                               eq_integrate_o20_log_line=KC['checks'] == LOGCHK, n_proc=KC.get('n_proc'), resumed=KC.get('resumed'), runtime_s=KC['runtime_s'])
res['B_v7x_plans'] = B
ok['B_v7x_plans'] = all(B[t][k] for t in ('M1', 'M2') for k in ('csv_sha256_eq_integrate_o20', 'counts_eq_integrate_o20', 'key_unique', 'exec_idx_is_signal_row_plus_1', 'exec_eq_panel_date_at_exec_idx',
                                                              'ten_picks_every_row', 'inserted_subset_of_picks', 'signal_days_same_across_seeds_and_variants', 'all_signals_before_end_day',
                                                              'ends_k_eq_integrate_o20', 'k_check_sha_eq', 'k_check_counts_eq')) \
    and all(B[t]['n_plans'] == 780 and B[t]['variants'] == sorted(VX) and B[t]['j_set'] == list(range(13)) and B[t]['seed_set'] == list(range(20)) and
            all(B[t]['counts'][v]['n_rows'] == 63820 for v in VX) for t in ('M1', 'M2')) \
    and B['M1']['counts_eq_T1_RESULTS_V7X_PLANS'] and B['M1']['ends_eq_T1_RESULTS_V7X_PLANS'] \
    and B['M1_vs_M2']['eq_integrate_o20_log'] and B['M1_vs_M2']['eq_k_gen_plans_check'] and B['M1_vs_M2']['diff_rows_eq_integrate_o20_diff_file'] \
    and B['M1_vs_M2']['n_rows_only_in_one'] == 0 and B['M1_vs_M2']['exec_same_on_common_rows'] \
    and mine == dict(rows_changed=2816, plans_changed=780, plans_changed_by_variant={v: 260 for v in VX}, n_signal_days=58, n_distinct_swaps=75, ends_eq_base=True, swaps_pre2021_09_24=75) \
    and B['k_gen_plans_checks']['eq_integrate_o20_log_line'] and all(KC['checks']['base_counts_eq_T1'].values()) and KC['checks']['base_ends_eq_T1']

# ---------------- C 淨值／逐窗檔
C = {f: dict(sealed=sha(SEAL / f), integrate_o20_data=sha(AG / 'integrate_o20/data' / f)) for f in ('v7x_navs_jmean_100w.csv.gz', 'paper_daily_navs.csv.gz', 'paper_windows.npz')}
for v in C.values(): v['byte_identical'] = v['sealed'] == v['integrate_o20_data']
N = pd.read_csv(SEAL / 'v7x_navs_jmean_100w.csv.gz', index_col=0, nrows=2); PN = pd.read_csv(SEAL / 'paper_daily_navs.csv.gz', index_col=0, nrows=2); Z = np.load(SEAL / 'paper_windows.npz')
want_x = {f'{dv}|{v}|{j}|{c}|100萬|std' for dv in ('base', 'M2') for v in VX for j in range(13) for c in ('normal', 'stress')}
want_p = {f'{mk}|{v}|{c}' for mk in ('M1', 'M2') for v in VP for c in ('normal', 'stress')}
C['columns'] = dict(v7x_navs_cols=len(N.columns), v7x_navs_cols_eq_expected=set(N.columns) == want_x, paper_navs_cols=len(PN.columns), paper_navs_has_M1_M2_12=want_p <= set(PN.columns),
                    paper_windows_has_M1_M2_r_r50=all(f'{k}|{q}' in Z.files for k in want_p for q in ('r', 'r50')), paper_windows_n=sorted({int(len(Z[f'{k}|r'])) for k in want_p}))
res['C_nav_files'] = C
ok['C_nav_files'] = all(C[f]['byte_identical'] for f in ('v7x_navs_jmean_100w.csv.gz', 'paper_daily_navs.csv.gz', 'paper_windows.npz')) and C['columns']['v7x_navs_cols_eq_expected'] \
    and C['columns']['paper_navs_has_M1_M2_12'] and C['columns']['paper_windows_has_M1_M2_r_r50']

# ---------------- D targets.json
T = json.loads((SEAL / 'targets.json').read_text(encoding='utf-8'))
PRE = 'integrate_o20/results.json :: '
def resolve(path):
    cur = R20
    for tok in re.findall(r"\['([^']*)'\]|([A-Za-z_][A-Za-z0-9_]*)", path):
        cur = cur[tok[0] if tok[0] != '' else tok[1]]
    return cur
st = dict(n_src=0, n_numbers=0, n_mismatch=0, n_value_without_src=0, bad=[])
def walk(x, p):
    if isinstance(x, dict):
        if 'src' in x:
            assert x['src'].startswith(PRE), (p, x['src']); r = resolve(x['src'][len(PRE):]); st['n_src'] += 1
            if 'value' in x: pairs = [(x['value'], r)]
            elif 'values' in x: assert isinstance(r, list) and len(r) == len(x['values']) == 13, p; pairs = list(zip(x['values'], r))
            else: pairs = [(x['med'], r['med']), (x['p10'], r['p10'])]
            for a_, b_ in pairs:
                st['n_numbers'] += 1
                if not (a_ == b_ and type(a_) == type(b_)):
                    st['n_mismatch'] += 1; st['bad'].append(p)
        elif any(k in x for k in ('value', 'values', 'med')): st['n_value_without_src'] += 1; st['bad'].append(p)
        else:
            for k, v in x.items(): walk(v, f'{p}/{k}')
for sec in ('v7x', 'paper'): walk(T[sec], sec)
CELLS = ['100萬一般', '100萬壓力', '300萬一般', '1000萬一般', '30萬標準', '30萬整股', '30萬壓力']
num = lambda z: isinstance(z, float) and np.isfinite(z)
cov = dict(v7x_cells=0, v7x_missing=[], per_j_saved=set(), per_j_not_saved=set(), paper_cells=0, paper_missing=[])
for mk in ('M1', 'M2'):
    for v in VX:
        for cn in CELLS:
            o = T['v7x'][mk][v][cn]; cov['v7x_cells'] += 1
            for f in ('exc', 't', 'CAGR_Y19on', 'exc_Y19on'):
                if not (num(o['j_median'][f]['value']) and num(o['j_p10'][f]['value']) and isinstance(o['per_j'][f], dict) and len(o['per_j'][f]['values']) == 13 and all(num(q) for q in o['per_j'][f]['values'])):
                    cov['v7x_missing'].append(f'{mk}|{v}|{cn}|{f}')
            for f, pj in o['per_j'].items(): (cov['per_j_saved'] if isinstance(pj, dict) else cov['per_j_not_saved']).add(f)
    for v in VP:
        for c in ('normal', 'stress'):
            o = T['paper'][mk][v][c]['ALL']; cov['paper_cells'] += 1
            for f in ('ann_cal', 'ann', 't_vs_0050'):
                if not num(o[f]['value']): cov['paper_missing'].append(f'{mk}|{v}|{c}|{f}')
cov['per_j_saved'] = sorted(cov['per_j_saved']); cov['per_j_not_saved'] = sorted(cov['per_j_not_saved'])
res['D_targets'] = dict(src_check=st, coverage=cov, sources_sha_ok={k: sha(AG / k) == v['sha256'] for k, v in T['sources'].items()},
                        builder_checks=T['checks'], not_saved_items=[q['item'] for q in T['not_saved']], sealed_files_described=sorted(T['files']))
ok['D_targets'] = st['n_mismatch'] == 0 and st['n_value_without_src'] == 0 and cov['v7x_cells'] == 42 and cov['paper_cells'] == 12 and not cov['v7x_missing'] and not cov['paper_missing'] \
    and all(res['D_targets']['sources_sha_ok'].values()) and T['checks']['med_eq_median_of_per_j_maxabs'] == 0.0 and T['checks']['p10_eq_percentile_of_per_j_maxabs'] == 0.0 \
    and T['checks']['o20_base_vs_o19_strict_maxabs'] == 0.0 and T['checks']['o19_keys_compared'] == 21 and T['checks']['decision_D1_all_within_rounding'] \
    and set(T['files']) == {p.name for p in SEAL.iterdir() if p.name not in ('targets.json', 'SHA256.txt')}

# ---------------- 結果與 SHA256.txt
res['ok'] = ok; res['all_ok'] = all(ok.values()); res['utc'] = pd.Timestamp.now('UTC').isoformat()
files = sorted(p for p in SEAL.iterdir() if p.name != 'SHA256.txt')
res['sealed'] = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in files}
res['sealed_total_bytes'] = sum(v['bytes'] for v in res['sealed'].values()); res['sealed_max_bytes'] = max(v['bytes'] for v in res['sealed'].values())
ok['size_limits'] = res['sealed_max_bytes'] < 45 * 2**20 and res['sealed_total_bytes'] < 100 * 2**20; res['all_ok'] = all(ok.values())
(HERE / 'k_verify_sealed.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(dict(ok=ok, all_ok=res['all_ok'], targets_src_check={k: v for k, v in st.items() if k != 'bad'}, bad=st['bad'][:5], coverage=cov, sealed=res['sealed'],
                      total_bytes=res['sealed_total_bytes']), ensure_ascii=False, indent=1))
if not res['all_ok']: sys.exit('核對未全部通過：不寫 SHA256.txt')
L = [f"{v['sha256']}  {n}" for n, v in res['sealed'].items()]
L += [f"{sha(HERE / 'SPEC_v2_blind.md')}  ../SPEC_v2_blind.md", f"{sha(HERE / 'k_blind_data_sha256.json')}  ../k_blind_data_sha256.json"]
L += [f"# 產生時間（UTC）：{pd.Timestamp.now('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')}；產生程式 k_verify_sealed.py（逐項核對全部通過後才寫本檔；核對明細 ../k_verify_sealed.json）。",
      '# 前 %d 列＝sealed/ 內每個檔（本檔除外）；後 2 列＝盲版規格與其附檔（交給複製員 A 的兩個檔）。路徑相對於 sealed/；可用 sha256sum -c 檢查。' % len(res['sealed']),
      '# v7x_plans_M1／M2 解壓後的 CSV sha256：%s（M1）、%s（M2），等於 integrate_o20/v7x_plans_check.json 記錄的 plans_base.csv／plans_M2.csv。' % (B['M1']['csv_sha256'], B['M2']['csv_sha256']),
      '# 未遮蔽版規格 o_g6_spec_v2/SPEC_v2.md 的 sha256：%s。' % sha(AG / 'o_g6_spec_v2/SPEC_v2.md')]
(SEAL / 'SHA256.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
print('SHA256.txt written')

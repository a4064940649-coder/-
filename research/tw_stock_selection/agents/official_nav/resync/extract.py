#!/usr/bin/env python3
"""從 harness.py 兩種模式的輸出（scratchpad/resync/run/<題>_<模式>/）取出網站風險表要的數字，
驗收：exright 模式要和舊數字（原資料夾／S26 重跑的結果檔）逐位相同；再寫 resync_results.json（舊＝只修除權息、新＝正式）。
計算方式逐字照 s26_exright_robustness/compare.py（S7、S9）、s27 analyze.py（現值）、s42 results.json 的 grid。
python3 -I -B extract.py"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parents[1]
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/resync')
RUN = SCR / 'run'
J = lambda p: json.load(open(p))
pp = lambda x: round(float(x) * 100, 2)
LAG = {'A10': 0.0, 'A11': 0.6201550387596899, 'A12': 1.2945736434108528, 'A13': 1.9767441860465116, 'A15': 3.356589147286822}   # 同 S26 compare.py


def deep_diff(a, b, p='', out=None, skip=()):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f'{p}/{k}: 缺'); continue
            deep_diff(a[k], b[k], f'{p}/{k}', out, skip)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f'{p}: 長度 {len(a)}≠{len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                deep_diff(x, y, f'{p}[{i}]', out, skip)
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
        out.append(f'{p}: {a!r}≠{b!r}')
    return out


def s7(mode):
    r = J(RUN / f's7_{mode}' / 'results.json'); q = J(RUN / f's7_{mode}' / 'results_posthoc.json')['Q7_delay_capitals']['W10_70L']
    c = r['cells']['W10_70L']
    return dict(late1_cost_5cap_median_pp=round(-q['1']['median_cost_pp'], 2), late2_cost_5cap_median_pp=round(-q['2']['median_cost_pp'], 2),
                late1_cost_100w_pp=round(-c['delay_cost_pp']['1']['since2019'], 2), late2_cost_100w_pp=round(-c['delay_cost_pp']['2']['since2019'], 2),
                late2_maxdd=pp(c['delay']['2']['since2019']['maxdd']))


def s9(mode):
    r = J(RUN / f's9_{mode}' / 'results.json')
    s3 = r['family']['W10_70|open']['S3']
    xs, ys = [], []
    for d, lag in LAG.items():
        for k, v in r['combos'][f'{d}|open|W10_70']['paths'].items():
            xs.append(lag + int(k.split('+')[1])); ys.append(v * 100)
    bb = np.polyfit(xs, ys, 1)
    return dict(late_median=pp(s3['fam2_r19_median']), late_p10=pp(s3['fam2_r19_p10']), late_dd_median=pp(s3['fam2_maxdd_median']),
                late_dd_p10=pp(s3['fam2_maxdd_p10']), ontime_intercept=round(float(bb[1]), 2), ontime_slope=round(float(bb[0]), 2),
                A10_paths20_median=pp(s3['A10_paths_r19_median']), A10_paths20_dd_median=pp(s3['A10_paths_maxdd_median']))


def s27(mode):
    r = J(RUN / f's27_{mode}' / 'results.json'); p3 = r['part3_alerts']; std = p3['history_check_std']
    d = pd.read_pickle(SCR / f'tmp_{mode}' / 's27' / 'daily_paths.pkl')['100萬|+0']
    main = r['part2_simulation']['main_2016_b6to12']; S = main['scenarios']; AL = main['alerts']['fa10']
    return dict(A_now=pp(std['A']['current_2026_10']), E_now=pp(std['E']['current_2026_10']), B_now=pp(std['B']['current_2026_10']),
                C_now=pp(std['C']['current_2026_10']), C_now_daily=pp(d.iloc[-1] / d.cummax().iloc[-1] - 1), D_now=std['D']['current_2026_10'],
                thresholds=p3['thresholds_fa10'], range1y_median=pp(S['EFF']['12m']['strat_ann']['p50']),
                range1y_p10_p90=[pp(S['EFF']['12m']['strat_ann']['p10']), pp(S['EFF']['12m']['strat_ann']['p90'])],
                p_lose_1y=pp(S['EFF']['12m']['p_lose_to_0050']),
                false_alarm_5y={k: pp(AL[k]['prob']['EFF']['5y']) for k in 'ABCE'},
                red_false_alarm_5y=pp(AL['TWO_OR_MORE_posthoc']['prob']['EFF']['5y']), any_12m_behind_5y=pp(S['EFF']['events_5y']['any_12m_behind']),
                hist=p3['history_check_summary'])


def s42(mode):
    r = J(RUN / f's42_{mode}' / 'results.json'); g = r['grid']
    m = lambda k, f, q='median': pp(g[k][f][q])
    return dict(neutral=m('COMB|0.004|LATE', 'since2019'), neutral_maxdd=m('COMB|0.004|LATE', 'maxdd_full'), neutral_maxdd_p10=m('COMB|0.004|LATE', 'maxdd_full', 'p10'),
                neutral_full=m('COMB|0.004|LATE', 'full'),
                half=m('COMB|0.004|LATE', 'HALF_since2019'), half_maxdd=m('COMB|0.004|LATE', 'HALF_maxdd_full'),
                conservative=m('COMB|0.005|LATE', 'HALF_since2019', 'p10'), conservative_maxdd=m('COMB|0.005|LATE', 'HALF_maxdd_full', 'p10'),
                optimistic=m('COMB|0.003|ON', 'since2019'), optimistic_maxdd=m('COMB|0.003|ON', 'maxdd_full'),
                cur_std_since2019=pp(r['acceptance']['A0']['since2019']), bench_since2019=pp(r['bench_0050']['since2019']), bench_maxdd=pp(r['bench_0050']['maxdd_full']),
                stack=[dict(step=x['step'], since2019_median=pp(x['since2019_median']), d_vs_CUR_pp=round(x['d_vs_CUR_since2019_median_pp'], 2)) for x in r['stack']],
                paired_effects={k: round(v['median'], 2) if isinstance(v, dict) else round(v, 2) for k, v in r['paired_effects'].items()},
                acceptance=r['acceptance'])


def xas(mode):
    r = J(RUN / f'xas_{mode}' / 'out' / 'res_fix1_XAS.json')['strats']['XAS']
    return dict(since2019=pp(r['std']['cagr_since2019']), full=pp(r['std']['cagr_full']), maxdd=pp(r['std']['maxdd']),
                paths_median=pp(r['paths_summary']['cagr19_median']), paths_min=pp(r['paths_summary']['cagr19_min']), paths_max=pp(r['paths_summary']['cagr19_max']),
                paths_dd_median=pp(r['paths_summary']['maxdd_median']))


# ---------------------------------------------------------------- 驗收 1：exright 模式＝舊結果檔（全檔深度比對，跳過執行時間類欄位）
SK = ('runtime_s', 'maxrss_mb', 'run_runtime_s', 'run_maxrss_mb', 'peak_mb', 'secs')
acc = {}
acc['S9_vs_s26_rerun'] = deep_diff(J(RUN / 's9_exright' / 'results.json'), J(AG / 's26_exright_robustness' / 'rerun' / 's9_run_fix1' / 'results.json'), skip=SK)
acc['S7_posthoc_vs_s26_rerun'] = deep_diff(J(RUN / 's7_exright' / 'results_posthoc.json'), J(AG / 's26_exright_robustness' / 'rerun' / 's7_run_fix1' / 'results_posthoc.json'), skip=SK)
a7 = J(RUN / 's7_exright' / 'results.json'); b7 = J(AG / 's26_exright_robustness' / 'rerun' / 's7_run_fix1' / 'results.json')
acc['S7_vs_s26_rerun_except_S6_tradeoff'] = deep_diff(a7, b7, skip=SK + ('rows_b', 'frontier_b_since2019', 'frontier_b_full'))
acc['S27_vs_original'] = deep_diff(J(RUN / 's27_exright' / 'results.json'), J(AG / 's27_expectation_ranges' / 'results.json'), skip=SK + ('build_meta',))
acc['S27_site_table_vs_original'] = deep_diff(J(RUN / 's27_exright' / 'site_table.json'), J(AG / 's27_expectation_ranges' / 'site_table.json'))
acc['S42_vs_original'] = deep_diff(J(RUN / 's42_exright' / 'results.json'), J(AG / 's42_realistic_expectation' / 'results.json'), skip=SK)
acc['XAS_vs_original'] = deep_diff(J(RUN / 'xas_exright' / 'out' / 'res_fix1_XAS.json')['strats']['XAS'],
                                   J(AG / 'exright_engine_fix' / 'out' / 'res_fix1_XTS_XAS.json')['strats']['XAS'])
OLD = dict(S7=s7('exright'), S9=s9('exright'), S27=s27('exright'), S42=s42('exright'), XAS=xas('exright'))
NEW = dict(S7=s7('official'), S9=s9('official'), S27=s27('official'), S42=s42('official'), XAS=xas('official'))
# 網站目前寫的舊數字（四捨五入後）必須由 exright 模式重現
SITE = [('回撤_壞的時候', -36.1, OLD['S9']['late_dd_p10'], 1), ('晚1天少', 3.5, OLD['S7']['late1_cost_5cap_median_pp'], 1),
        ('晚2天少', 5.5, OLD['S7']['late2_cost_5cap_median_pp'], 1), ('晚2天回撤', -34.1, OLD['S7']['late2_maxdd'], 1),
        ('準時可預期', 46.7, OLD['S9']['ontime_intercept'], 1), ('有時晚可預期', 43, OLD['S9']['late_median'], 0),
        ('警戒A_3年超額', 15.3, OLD['S27']['A_now'], 1), ('警戒E_相對前高', -6.6, OLD['S27']['E_now'], 1), ('警戒B_1年超額', 34.5, OLD['S27']['B_now'], 1),
        ('警戒C_月底回撤', -5.3, OLD['S27']['C_now'], 1), ('警戒C_日內', -10.6, OLD['S27']['C_now_daily'], 1), ('正常範圍_1年', 30, OLD['S27']['range1y_median'], 0),
        ('實盤可預期_中性', 41.8, OLD['S42']['neutral'], 1), ('實盤可預期_保守', 36.1, OLD['S42']['half'], 1),
        ('XQ純衛星_2019起', 43.08, OLD['XAS']['since2019'], 2), ('XQ純衛星_中位數', 38.07, OLD['XAS']['paths_median'], 2)]
acc['site_numbers_reproduced'] = [dict(項目=a, 網站=b, 重算=c, ok=bool(round(c, n) == b)) for a, b, c, n in SITE]
# 驗收 2：正式模式的基準＝official_nav（S27 的 20 條路徑、S42 的 CUR 標準路徑、XAS 不在 official_nav 裡）
ro = J(AG / 'official_nav' / 'out' / 'res_official.json')['strats']['E0|W10_70']
d = pd.read_pickle(SCR / 'tmp_official' / 's27' / 'daily_paths.pkl')
from_off = {k: v for k, v in ro['paths'].items()}
D19 = pd.Timestamp('2018-12-28'); DE = d.index[-1]
cg = lambda s: float((s.loc[DE] / s.loc[D19]) ** (365.25 / (DE - D19).days) - 1)
nv = pd.read_csv(Path('/home/user/-/xq_scripts/site/navs_official.csv.gz'), index_col=0, parse_dates=True)['E0|W10_70']
acc['official_S27_std_vs_navs_official_max_abs'] = float((d['100萬|+0'].round(2) - nv.reindex(d.index)).abs().max())
acc['official_S42_CUR_std_since2019'] = NEW['S42']['cur_std_since2019']
acc['official_S42_CUR_vs_4876_ok'] = bool(abs(NEW['S42']['acceptance']['A0']['since2019'] - ro['std']['cagr_since2019']) < 1e-12)
acc['official_S9_A10_paths20_median_vs_official_nav'] = [NEW['S9']['A10_paths20_median'], pp(ro['paths_summary']['cagr19_median'])]
ok1 = all(not acc[k] for k in ('S9_vs_s26_rerun', 'S7_posthoc_vs_s26_rerun', 'S7_vs_s26_rerun_except_S6_tradeoff', 'S27_vs_original',
                                'S27_site_table_vs_original', 'S42_vs_original', 'XAS_vs_original'))
ok2 = all(x['ok'] for x in acc['site_numbers_reproduced'])
acc['pass_exright_reproduces_old'] = bool(ok1 and ok2)
acc['shim_check'] = {m: J(HERE / 'out' / f'check_shim_{m}.json') for m in ('exright', 'official')}
acc['pass_official_matches_official_nav'] = bool(acc['shim_check']['official']['pass'] and acc['shim_check']['exright']['pass'] and acc['official_S27_std_vs_navs_official_max_abs'] < 0.006 and acc['official_S42_CUR_vs_4876_ok']
                                                 and abs(NEW['S9']['A10_paths20_median'] - pp(ro['paths_summary']['cagr19_median'])) < 0.005)
soft = {p.stem.replace('__soft', ''): dict(shim=J(p)['shim'], n_soft=len(J(p)['soft_asserts']), soft_lines=[x['line'] for x in J(p)['soft_asserts']], soft_msgs=[x['msg'][:400] for x in J(p)['soft_asserts']], peak_mb=J(p)['peak_mb'])
        for p in sorted((HERE / 'out').glob('*__soft.json'))}
res = dict(meta=dict(說明='網站風險表「只修除權息時算的」各列，用正式引擎（cslib.EXRIGHT_FIX、DEADLINE_FIX、INS15_MASK 全開）重算；2026-11-17。'
                         '舊＝exright 模式（只開 EXRIGHT_FIX，逐位重現舊結果檔）；新＝official 模式。'
                         'S7／S9 用 s4lib.make_signal、S9／S42 自算 A_D 訊號日，共用引擎的期限開關不會作用到，harness.py 以墊片補上（開關關閉時與原函式逐位相同）；'
                         'XQ 純衛星（XAS）同 official_nav 的 XQ 版：期限例外表有套用、15 日期限遮蔽不適用。',
                     程式='agents/official_nav/resync/harness.py、batch.sh、extract.py'),
           acceptance=acc, old=OLD, new=NEW, soft=soft)
json.dump(res, open(HERE / 'resync_results.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print(json.dumps(dict(pass1=acc['pass_exright_reproduces_old'], pass2=acc['pass_official_matches_official_nav'],
                      diffs={k: v[:5] for k, v in acc.items() if isinstance(v, list) and v and isinstance(v[0], str)},
                      site=[(x['項目'], x['網站'], x['重算'], x['ok']) for x in acc['site_numbers_reproduced']]), ensure_ascii=False, indent=0))
for k in OLD:
    print(k, {a: (OLD[k][a], NEW[k][a]) for a in OLD[k] if a not in ('stack', 'paired_effects', 'acceptance', 'hist', 'thresholds')})
print('thresholds', OLD['S27']['thresholds'], NEW['S27']['thresholds'])

#!/usr/bin/env python3
"""數據稽核部 2026-11-21：從 harness.py 兩種模式的輸出（scratchpad/v2recalc/run/<題>_<v1|v2>/）取出網站風險表各列的數字。
v1＝改版前（三個開關、舊 0050 序列）：必須重現 official_nav/resync/resync_results.json 的 new（S7／S9／S27／S42）與各題原 results.json（S46／S62／S64／S39）；
v2＝2026-11-21 新口徑（五個開關全開）。S7／S9／S27／S42 的取數函式逐字照 official_nav/resync/extract.py。
python3 -I -B extract.py → recalc_results.json（本資料夾）"""
import json, math
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
AG = HERE.parent
SCR = Path('/tmp/claude-0/-home-user/4a4e211a-f98b-5aa5-8802-26f871eea688/scratchpad/v2recalc')
RUN = SCR / 'run'
J = lambda p: json.load(open(p))
ND = 2                                                       # 取數小數位；先用 2 位（和 resync_results.json 比對），再用 4 位存一份（寫網站文字時判斷四捨五入用）
pp = lambda x: round(float(x) * 100, ND)
LAG = {'A10': 0.0, 'A11': 0.6201550387596899, 'A12': 1.2945736434108528, 'A13': 1.9767441860465116, 'A15': 3.356589147286822}


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
    return dict(late1_cost_5cap_median_pp=round(-q['1']['median_cost_pp'], ND), late2_cost_5cap_median_pp=round(-q['2']['median_cost_pp'], ND),
                late1_cost_100w_pp=round(-c['delay_cost_pp']['1']['since2019'], ND), late2_cost_100w_pp=round(-c['delay_cost_pp']['2']['since2019'], ND),
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
                late_dd_p10=pp(s3['fam2_maxdd_p10']), ontime_intercept=round(float(bb[1]), ND), ontime_slope=round(float(bb[0]), ND),
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
                stack=[dict(step=x['step'], since2019_median=pp(x['since2019_median']), d_vs_CUR_pp=round(x['d_vs_CUR_since2019_median_pp'], ND)) for x in r['stack']],
                paired_effects={k: round(v['median'], ND) if isinstance(v, dict) else round(v, ND) for k, v in r['paired_effects'].items()},
                acceptance=r['acceptance'])


def s46(mode):
    f = J(RUN / f's46_{mode}' / 'results.json'); m = f['families']['main45']; c = f['cells']['g10_L20_D10']
    return dict(all_paths_median=pp(m['all_paths_median']), all_paths_p10=pp(m['all_paths_p10']), n_all_paths=m['n_all_paths'], current_rank=m['current_rank'],
                current_minus_median_pp=round(m['current_minus_median_pp'], ND), family_median=pp(m['family_median']), best=m['best'],
                current_main_since2019=pp(c['main']['since2019']), current_paths_median=pp(c['path']['median']), current_n_closed=c['main']['n_closed'])


def s62(mode):
    r = J(RUN / f's62_{mode}' / 'results.json'); b = r['bench_0050']['since2019']; o = dict(bench_since2019=pp(b))
    for k in ('B_基準', 'D_只買整張', 'R0_現行嚴成本'):
        cid = r['named'][k] if isinstance(r['named'][k], str) else r['named'][k].get('id')
        for cap in ('10萬', '100萬'):
            if cap in r['cells'][cid]:
                v = r['cells'][cid][cap]['summary']['since2019_median']
                o[f'{k}|{cap}'] = pp(v); o[f'{k}|{cap}|excess_pp'] = round((v - b) * 100, ND)
    return o


def s64(mode):
    r = J(RUN / f's64_{mode}' / 'results.json'); sc = r['scenarios']; o = dict(bench_since2019=pp(r['bench_0050']['since2019']), baseline_median=pp(r['baseline']['since2019']),
                                                                           baseline_maxdd=pp(r['baseline']['maxdd_full']), baseline_main=pp(r['baseline']['main']['since2019']))
    for k in ('PICK_RAND', 'STOP_20_6M', 'SKIP_25', 'GAP_3', 'SKIP_10', 'SKIP_50', 'GAP_5'):
        o[k] = dict(since2019=pp(sc[k]['since2019']), d19_pp=round(sc[k]['d19_pp'], ND), ddd_full_pp=round(sc[k]['ddd_full_pp'], ND))
    return o


def s39(mode):
    r = J(RUN / f's39_{mode}' / 'results.json'); e = r['engine']
    return dict(paths_pairs_FPM=e['paths_pairs']['FPM']['cagr'], window=e['window'], CUR_since2019=pp(e['standard']['CUR']['since2019']), CUR_window=pp(e['standard']['CUR']['cagr_window']), FPM_window=pp(e['standard']['FPM']['cagr_window']),
                d_window_pp=round(e['standard']['FPM']['d_cagr_window_pp'], ND), d_maxdd_window_pp=round(e['standard']['FPM']['d_maxdd_window_pp'], ND),
                d_since2019_pp=round(e['standard']['FPM']['d_since2019_pp'], ND), paths_median_cagr=e['paths_median_cagr'])


def s42fpm(mode):
    """首次公布營收（S39 的做法）在 S42 鏈上的效果：標準路徑視窗差、20 條視窗差中位數與最差。"""
    r = J(RUN / f's42_{mode}' / 'results.json'); a = r['acceptance']['A1_FPM']; ad = r['additivity_paths20']['since2024_07']
    return dict(d_window_pp=round(a['d_window_pp'], ND), paths20_window_diff_median_pp=round(a['paths20_window_diff_median_pp'], ND),
                paths20_window_diff_min_pp=round(ad['COMB_min_pp'], ND), paths20_window_diff_max_pp=round(ad['COMB_max_pp'], ND), n_lower=ad['COMB_n_lower'],
                FPM_window=pp(a['window_2024_07']))


FUN = dict(S7=s7, S9=s9, S27=s27, S42=s42, S42FPM=s42fpm, S46=s46, S62=s62, S64=s64, S39=s39)
res = dict(v1={}, v2={}, errors={})
for mode in ('v1', 'v2'):
    for k, f in FUN.items():
        try:
            res[mode][k] = f(mode)
        except Exception as e:                                # noqa: BLE001
            res['errors'][f'{k}_{mode}'] = repr(e)[:300]
ND = 4
for mode in ('v1', 'v2'):
    res[mode + '_p4'] = {}
    for k, f in FUN.items():
        try:
            res[mode + '_p4'][k] = f(mode)
        except Exception as e:                                # noqa: BLE001
            pass
ND = 2
# ---- 驗收：v1 重現舊值
rs = J(AG / 'official_nav' / 'resync' / 'resync_results.json')['new']
acc = {}
for k in ('S7', 'S9', 'S27', 'S42'):
    if k in res['v1']:
        acc[f'{k}_v1_vs_resync_new'] = deep_diff(json.loads(json.dumps(res['v1'][k])), rs[k])
SK = ('runtime_s', 'maxrss_mb', 'run_runtime_s', 'run_maxrss_mb', 'peak_mb', 'secs', 'run_sha256', 'lib_sha256', 'describe_1_runtime_s',
      'describe_1', 'paths_trade_diff_100萬')   # 後兩個鍵是 S39 的 describe.py／diag_paths.py 另外寫進 results.json 的，run.py 不產生
for k, d in (('S46', 's46_momentum_gate_family'), ('S62', 's62_small_capital_retail'), ('S64', 's64_retail_behavior_tax'), ('S39', 's39_first_print_revenue')):
    p = RUN / f'{k.lower()}_v1' / 'results.json'
    if p.exists():
        dd = deep_diff(J(p), J(AG / d / 'results.json'), skip=SK)
        acc[f'{k}_v1_results_vs_original'] = dict(n_diff=len(dd), first=dd[:8])
res['acceptance'] = acc
res['soft'] = {p.stem.replace('__soft', ''): dict(flags=J(p)['flags'], self_flags=J(p).get('self_flags'), shim=J(p)['shim'], n_soft=len(J(p)['soft_asserts']),
                                                  soft=[(x['line'], x['msg'][:200]) for x in J(p)['soft_asserts']], peak_mb=J(p)['peak_mb'], secs=J(p)['secs'],
                                                  panel_reduce_fix=J(p).get('panel_reduce_fix'), panel_e50_source=J(p).get('panel_e50_source'))
               for p in sorted((HERE / 'out').glob('*__soft.json'))}
json.dump(res, open(HERE / 'recalc_results.json', 'w'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('errors', res['errors'])
print('acceptance', json.dumps(acc, ensure_ascii=False)[:3000])
for k in res['v1']:
    if k in res['v2'] and k != 'S39':
        a, b = res['v1'][k], res['v2'][k]
        print(k, {x: (a[x], b[x]) for x in a if x not in ('stack', 'paired_effects', 'acceptance', 'hist')})

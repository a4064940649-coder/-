#!/usr/bin/env python3
"""S27 獨立複製（第 3 步）：results_rep.json 對 s27 results.json（只讀數字）逐項比對，寫進 results_rep.json['comparison']。
容許：確定性數字（歷史、門檻、現值）|差| ≤ 0.001（0.1 個百分點；月數／段數須相等）；模擬機率與報酬分位 ≤ 0.02（2 個百分點）。"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.load(open(HERE.parent / 'results.json')); M = json.load(open(HERE / 'results_rep.json'))
rows = []


def get(d, path):
    for p in path.split('/'):
        d = d[p]
    return d


def cmp(group, path, tol, label=None, mpath=None):
    a = get(R, path); b = get(M, mpath or path)
    if isinstance(a, (list, str)) or isinstance(b, (list, str)):
        ok = a == b; diff = None
    else:
        diff = None if a is None or b is None else round(b - a, 6); ok = (a == b) if diff is None else abs(diff) <= tol + 1e-12
    rows.append(dict(group=group, item=label or path, orig=a, rep=b, diff=diff, tol=tol, ok=bool(ok)))


DET = 0.001; MC = 0.02
# 1. 歷史描述
for per in ('since2016', 'since2019'):
    base = f'part1_history/{per}'
    for k in get(R, f'{base}/summary_20paths'):
        for s in ('median', 'worst', 'best'):
            cmp(f'歷史{per}', f'{base}/summary_20paths/{k}/{s}', 0 if ('n_' in k or 'lag_' in k or 'run' in k) else DET)
    for s in ('median', 'worst'):
        cmp(f'歷史{per}', f'{base}/daily_minus_monthly_maxdd/{s}', DET)
    sp = get(R, f'{base}/std_path')
    for k, v in sp.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                    cmp(f'歷史{per}標準路徑', f'{base}/std_path/{k}/{kk}', 0 if kk == 'n_windows' else DET)
                elif isinstance(vv, str):
                    cmp(f'歷史{per}標準路徑', f'{base}/std_path/{k}/{kk}', 0)
        elif isinstance(v, (int, float)):
            cmp(f'歷史{per}標準路徑', f'{base}/std_path/{k}', DET)
        elif isinstance(v, str):
            cmp(f'歷史{per}標準路徑', f'{base}/std_path/{k}', 0)
    for k in ('maxdd_daily', 'maxdd_monthly', 'n_dd_ge10', 'n_dd_ge20', 'longest_underwater_m', 'recover_ge20_max_m', 'underwater_ge20_max_m'):
        cmp(f'歷史{per}0050', f'{base}/bench_0050/{k}', DET)
    for p in get(R, f'{base}/per_path'):
        for k in ('maxdd_daily', 'maxdd_monthly', 'rel_maxdd', 'lag_longest_m', 'lag_current_m', 'R12_longest_neg_run_m'):
            cmp(f'歷史{per}逐路徑', f'{base}/per_path/{p}/{k}', DET)
        for k in ('min', 'p10', 'p50', 'frac_neg'):
            cmp(f'歷史{per}逐路徑', f'{base}/per_path/{p}/R12/{k}', DET); cmp(f'歷史{per}逐路徑', f'{base}/per_path/{p}/R36/{k}', DET)
# 2. 模擬
for sim in ('main_2016_b6to12', 'sens_2016_b6', 'sens_2016_b12', 'sens_2019_b6to12'):
    base = f'part2_simulation/{sim}'
    for k in ('median', 'min', 'max'):
        cmp(f'模擬{sim}', f'{base}/dbar_monthly_by_path/{k}', 0.00005)
    for sc in ('EFF', 'HALF', 'ZERO'):
        for h in ('12m', '36m', '60m'):
            b = f'{base}/scenarios/{sc}/{h}'
            for q in ('p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95'):
                cmp(f'模擬{sim}', f'{b}/strat_ann/{q}', MC); cmp(f'模擬{sim}', f'{b}/b0050_ann/{q}', MC)
                if sim == 'main_2016_b6to12':
                    cmp(f'模擬{sim}', f'{b}/excess_ann/{q}', MC)
            for k in ('p_lose_to_0050', 'p_strat_loss', 'p_0050_loss', 'maxdd_p50', 'maxdd_p90', 'maxdd_p95', 'p_dd_worse_than_hist', 'p_lag_ge12', 'p_lag_ge24'):
                cmp(f'模擬{sim}', f'{b}/{k}', MC)
            for k in ('lag_p50', 'lag_p90'):
                cmp(f'模擬{sim}', f'{b}/{k}', 2)
        for k in get(R, f'{base}/scenarios/{sc}/events_5y'):
            cmp(f'模擬{sim}事件', f'{base}/scenarios/{sc}/events_5y/{k}', MC)
    # 3. 警戒線
    for fa in ('fa10', 'fa5'):
        A = get(R, f'{base}/alerts/{fa}')
        for k, v in A.items():
            if 'threshold' in v:
                cmp(f'警戒線{sim}{fa}門檻', f'{base}/alerts/{fa}/{k}/threshold', 0 if k == 'D' else 0.0)
            for sc in ('EFF', 'HALF', 'ZERO'):
                for h in ('1y', '3y', '5y'):
                    cmp(f'警戒線{sim}{fa}', f'{base}/alerts/{fa}/{k}/prob/{sc}/{h}', MC)
# 4. 歷史警戒線檢查與 2026-10 現值
cmp('現值', 'part3_alerts/hist_worst_mdd_monthly', DET); cmp('現值', 'part3_alerts/hist_worst_mdd_daily', DET)
for k in 'ABCDE':
    for kk in ('n_months', 'worst', 'current_2026_10'):
        cmp('現值標準路徑', f'part3_alerts/history_check_std/{k}/{kk}', DET)
    cmp('現值標準路徑', f'part3_alerts/history_check_std/{k}/months', 0)
    for kk in ('paths_triggered', 'current_median', 'current_worst', 'worst_value_median', 'worst_value_worst'):
        cmp('現值20條', f'part3_alerts/history_check_summary/{k}/{kk}', DET)
for p in get(R, 'part3_alerts/history_check_per_path'):
    for k in 'ABCDE':
        cmp('現值逐路徑', f'part3_alerts/history_check_per_path/{p}/{k}/months', 0)
        cmp('現值逐路徑', f'part3_alerts/history_check_per_path/{p}/{k}/current_2026_10', DET)

summ = {}
for r in rows:
    g = summ.setdefault(r['group'], dict(n=0, ok=0, max_abs_diff=0.0))
    g['n'] += 1; g['ok'] += r['ok']
    if r['diff'] is not None:
        g['max_abs_diff'] = round(max(g['max_abs_diff'], abs(r['diff'])), 6)
M['comparison'] = dict(tolerance=dict(deterministic=DET, monte_carlo=MC), n=len(rows), n_ok=sum(r['ok'] for r in rows),
                       by_group=summ, failures=[r for r in rows if not r['ok']])
json.dump(M, open(HERE / 'results_rep.json', 'w'), ensure_ascii=False, indent=1)
print(len(rows), sum(r['ok'] for r in rows))
for g, v in summ.items():
    print(g, v)
for r in rows:
    if not r['ok']:
        print('FAIL', r['item'], r['orig'], r['rep'], r['diff'])

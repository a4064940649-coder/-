#!/usr/bin/env python3
"""把 results.json 的正式結果整理成 summary.txt（只讀）。用法：python3 -I -B summarize.py [--tag V4]"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent

def f(x, d=3):
    return 'NA' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='V4'); a = ap.parse_args()
    sfx = '' if a.tag == 'V4' else '_' + a.tag
    R = json.loads((HERE / 'results.json').read_text(encoding='utf-8'))
    P, A = R['RESULTS_POSTERIOR' + sfx], R['RESULTS_ALLOC' + sfx]
    L = []
    L.append('== 事後分布（pp／窗）==')
    for fam in ('MAIN', 'ALL', 'LOCO', 'THEORY'):
        fo = P['families'][fam]
        L.append(f"[{fam}] n {fo['n']} 共同誤差 sd {f(fo['common_sd_pp'])} K_eff {f(fo['Keff_equiv'], 1)}")
        for m in ('G1', 'G2', 'G3'):
            pr = fo['priors'][m]
            L.append(f"  {m} 先驗 mean {f(pr['mean'])} sd {f(pr['sd'])} P(>0) {f(pr['P_pos'])} q95 {f(pr.get('q95'))}")
            for c in ('AB', 'B10', 'EXEC', 'EXEC_J', 'AB_stress', 'B10_stress', 'EXEC_stress'):
                p = fo['post'][m][c]
                L.append(f"    {c:11s} raw {f(p['alpha_hat_pp'])} post {f(p['mean'])} sd {f(p['sd'])} [{f(p['q10'])},{f(p['q90'])}] P(>0) {f(p['p_pos'])} post/raw {f(p['post_over_raw'])}")
    L.append('外部先驗：' + json.dumps({c: {k: (round(v['mean'], 3), round(v['p_pos'], 3)) for k, v in P['external'][c].items()} for c in ('AB', 'B10', 'EXEC', 'EXEC_J')}, ensure_ascii=False))
    L.append('迭代版本敏感度：' + json.dumps({k: {c: (round(v['mean'], 3), round(v['p_pos'], 3)) for c, v in d.items()} for k, d in P['sens_versions'].items()}, ensure_ascii=False))
    L.append('可證偽（post ≥ 80% raw）：' + json.dumps(P['falsify_post_ge_80pct_raw']))
    keys = ['P_win', 'ratio_p10', 'ratio_p50', 'ratio_p90', 'P_ratio_lt_0.75', 'P_ratio_lt_0.5', 'mdd_p50', 'mdd_p90_severe', 'rel_mdd_p50', 'rel_mdd_p90_severe', 'P_roll5y_lose', 'P_under_ge60w', 'Elog_ratio_per_year', 'cagr_p50_pp']
    for blk in ('main', 'noshrink', 'annual'):
        for c in ('AB', 'B10', 'EXEC', 'EXEC_J'):
            for hy in ('10y', '20y'):
                L.append(f'== {blk} {c} {hy} ==  x | ' + ' | '.join(keys))
                for x, r in A[blk][c][hy].items():
                    L.append(f'  {x:>3} | ' + ' | '.join(f(r[k]) for k in keys))
    L.append('== x*（10pp 格點）==')
    for c in ('AB', 'B10', 'EXEC', 'EXEC_J'):
        L.append(f'  {c}: ' + ', '.join(f"{k} {v['x_star']} (Pwin100 {v['P_win_x100']:.3f}, Pwin50 {v['P_win_x50']:.3f})" for k, v in A['kelly_grid'][c].items()))
        L.append(f"     fine: " + ', '.join(f"{k} {v['x_star']}" for k, v in A['kelly_fine'][c].items()))
        L.append('     sens: ' + ', '.join(f"{k} x*{v['x_star_10y']} Pwin100 {v['P_win_x100_10y']:.3f}" for k, v in A['sens'][c].items()))
    L.append('== 判定 ==\n' + json.dumps(A['decision'], ensure_ascii=False, indent=1))
    L.append('MDD 校準：' + json.dumps(A['mdd_calibration']))
    if 'RESULTS_ROBUST' + sfx in R:
        L.append('穩健性：' + json.dumps(R['RESULTS_ROBUST' + sfx]['drop_year_range'], ensure_ascii=False))
    (HERE / ('summary' + sfx + '.txt')).write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))

if __name__ == '__main__':
    main()

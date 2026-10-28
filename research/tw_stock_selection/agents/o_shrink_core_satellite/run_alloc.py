#!/usr/bin/env python3
"""正式計算 2：核心衛星配置模擬（PREREG 之後）。
用法：python3 -I -B run_alloc.py [--focus data/focus_windows.parquet] [--tag V4] [--paths 20000]
需要先跑 run_posterior.py（同 tag）產生 data/post_dens_<tag>.npz。輸出 RESULTS_ALLOC[_tag]。"""
import sys, json, argparse, math, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

XS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.6, 0.7, 0.8, 0.9, 1.0]
KOU = ['AB', 'B10', 'EXEC', 'EXEC_J']
SEEDS = {6: 20261029, 12: 20261030}
ASEED = 7

def alpha_draws(D, key, n):
    rng = np.random.default_rng(ASEED)
    return S.sample_post(D[key.replace('|', '__')], n, rng) / 100.0

def table(r50, resid, alpha, idx, H, xs=XS, rebal='M'):
    I = idx[:, :H]
    r50p = r50[I]; ep = resid[I]
    rows = {}
    for x in xs:
        o = S.simulate(r50p, ep, alpha, x, rebal)
        rows[f'{int(round(x * 100))}'] = S.summarize_sim(o, H)
    return rows

def kelly(r50, resid, alpha, idx, H, step=0.01, rebal='M'):
    """E[log 財富比]／年 在細格點上的最大者（只算期末，不算回撤）。"""
    I = idx[:, :H]
    r50p = r50[I]; ep = resid[I]
    xs = np.round(np.arange(0, 1 + 1e-9, step), 4)
    el = []
    for x in xs:
        o = S.simulate(r50p, ep, alpha, float(x), rebal)
        el.append(float(o['logR'].mean() / (H / 12)))
    el = np.array(el)
    return {'x_star': float(xs[int(np.argmax(el))]), 'Elog_at_star': float(el.max()), 'curve': {f'{int(round(x * 100))}': float(v) for x, v in zip(xs, el)}}

def grid_star(rows):
    k = max(rows, key=lambda z: rows[z]['Elog_ratio_per_year'] if int(z) % 10 == 0 else -1e9)
    return int(k)

def run(focus_path, tag, B, check=True):
    if check: S.check_prereg()
    F = S.load_focus(focus_path)
    D = np.load(S.DATA / f'post_dens_{tag}.npz')
    r50 = F['r0050'].values
    n = len(r50)
    idx = {L: S.sb_index(n, 240, L, B, SEEDS[L]) for L in (6, 12)}
    out = {'tag': tag, 'paths': B, 'x_grid': XS, 'main': {}, 'noshrink': {}, 'flat': {}, 'annual': {}, 'kelly_grid': {}, 'kelly_fine': {}, 'sens': {}}
    t0 = time.time()
    for c in KOU:
        e = F[c].values - r50; ahat = e.mean(); resid = e - ahat
        a_main = alpha_draws(D, f'MAIN|G2|{c}', B)
        a_raw = np.full(B, ahat)
        a_flat = alpha_draws(D, f'FLAT|N|{c}', B)
        for H in (120, 240):
            hy = f'{H // 12}y'
            out['main'].setdefault(c, {})[hy] = table(r50, resid, a_main, idx[6], H)
            out['noshrink'].setdefault(c, {})[hy] = table(r50, resid, a_raw, idx[6], H)
            out['flat'].setdefault(c, {})[hy] = table(r50, resid, a_flat, idx[6], H, xs=[0.0, 0.3, 0.56, 0.8, 1.0])
            out['annual'].setdefault(c, {})[hy] = table(r50, resid, a_main, idx[6], H, rebal='A')
        # 先驗 × 區塊長度的 x*（10pp 格點，10 年與 20 年）
        kg = {}
        for model in ('G1', 'G2', 'G3'):
            a = alpha_draws(D, f'MAIN|{model}|{c}', B)
            for L in (6, 12):
                for H in (120, 240):
                    rows = table(r50, resid, a, idx[L], H, xs=[i / 10 for i in range(11)])
                    kg[f'{model}|L{L}|{H // 12}y'] = {'x_star': grid_star(rows), 'P_win_x100': rows['100']['P_win'],
                                                     'P_win_x50': rows['50']['P_win'], 'Elog': {k: v['Elog_ratio_per_year'] for k, v in rows.items()}}
        out['kelly_grid'][c] = kg
        out['kelly_fine'][c] = {'G2|L6|10y': kelly(r50, resid, a_main, idx[6], 120), 'NOSHRINK|L6|10y': kelly(r50, resid, a_raw, idx[6], 120)}
        # 敏感度族群與外部先驗：x* 與 x＝100% 勝率
        sens = {}
        for key in ['ALL|G2', 'LOCO|G2', 'THEORY|G2', 'ALL|G1', 'LOCO|G1', 'THEORY|G1', 'ALL|G3', 'LOCO|G3', 'THEORY|G3', 'EXT_MP|N', 'EXT_FUND|N', 'FLAT|N']:
            a = alpha_draws(D, f'{key}|{c}', B)
            rows = table(r50, resid, a, idx[6], 120, xs=[i / 10 for i in range(11)])
            sens[key] = {'x_star_10y': grid_star(rows), 'P_win_x100_10y': rows['100']['P_win'], 'P_win_x50_10y': rows['50']['P_win'],
                         'ratio_p10_x100_10y': rows['100']['ratio_p10']}
        out['sens'][c] = sens
        print(c, 'done', round(time.time() - t0), flush=True)
    # 歷史校準：窗頻 vs 日頻最大回撤
    V4 = pd.read_parquet(S.AG / 'o_v4_revenue_patch_official' / 'navs_v4.parquet')
    e50 = S.FC.e50_nav(); e50 = e50[e50.index >= '2006-04-11']
    def mdd(s): s = s.dropna(); return float((s / s.cummax() - 1).min())
    wn = (1 + F[['AB', 'r0050']]).cumprod()
    out['mdd_calibration'] = {'AB_daily': mdd(V4['AB_V4|normal']), 'AB_window': mdd(wn['AB']), '0050_daily': mdd(e50), '0050_window': mdd(wn['r0050']),
                              'B10_daily': mdd(V4['AB_V4_B10|normal'])}
    return out

def decide(out, post):
    """事前登記的判定（只讀結果）。"""
    m = out['main']['EXEC_J']
    kg = out['kelly_grid']['EXEC_J']
    xstar = kg['G2|L6|10y']['x_star']
    cand = [kg[f'{g}|L6|10y']['x_star'] for g in ('G1', 'G2', 'G3')]
    candL = [kg[f'G2|L{L}|10y']['x_star'] for L in (6, 12)]
    cond = (max(cand) - min(cand) > 40) or (max(candL) - min(candL) > 40)
    cmp = {}
    for ref in ('80', '100', '56'):
        cmp[ref] = {}
        for hy in ('10y', '20y'):
            a = m[hy][str(xstar) if str(xstar) in m[hy] else str(xstar)]; b = m[hy][ref]
            imp_p10 = a['ratio_p10'] - b['ratio_p10']
            imp_mdd = (a['mdd_p50'] - b['mdd_p50']) * 100
            dwin = (a['P_win'] - b['P_win']) * 100
            cmp[ref][hy] = {'x_star': xstar, 'd_ratio_p10': imp_p10, 'd_mdd_p50_pp': imp_mdd, 'd_P_win_pp': dwin,
                            'rule_met': bool(imp_p10 >= 0.05 or (imp_mdd >= 5 and dwin >= -2))}
    now = cmp['80']
    both = now['10y']['rule_met'] and now['20y']['rule_met']; one = now['10y']['rule_met'] or now['20y']['rule_met']
    return {'x_star_EXEC_J_G2_L6_10y': xstar, 'x_star_by_prior': dict(zip(('G1', 'G2', 'G3'), cand)), 'x_star_by_L': dict(zip(('L6', 'L12'), candL)),
            'conditional_range_rule': bool(cond), 'x_star_AB_upper_ref': out['kelly_grid']['AB']['G2|L6|10y']['x_star'],
            'compare_vs_current': cmp, 'rewrite_allocation': 'yes' if both else ('conditional' if one else 'no'),
            'P_pos_EXEC_J': post['families']['MAIN']['post']['G2']['EXEC_J']['p_pos'],
            'trigger_0050_primary': bool(post['families']['MAIN']['post']['G2']['EXEC_J']['p_pos'] < 0.5)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus', default=str(S.DATA / 'focus_windows.parquet'))
    ap.add_argument('--tag', default='V4')
    ap.add_argument('--paths', type=int, default=20000)
    a = ap.parse_args()
    out = run(a.focus, a.tag, a.paths)
    res = S.load_res()
    post = res['RESULTS_POSTERIOR' + ('' if a.tag == 'V4' else '_' + a.tag)]
    out['decision'] = decide(out, post)
    S.save_key('RESULTS_ALLOC' + ('' if a.tag == 'V4' else '_' + a.tag), out)
    print(json.dumps(out['decision'], ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()

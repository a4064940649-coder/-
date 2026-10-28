#!/usr/bin/env python3
"""正式計算 1：主策略 α 的事後分布（PREREG 之後）。
用法：python3 -I -B run_posterior.py [--focus data/focus_windows.parquet] [--meta data/focus_meta.json] [--tag V4]
輸出：results.json 的 RESULTS_POSTERIOR[_tag]、data/post_dens_<tag>.npz（供配置模擬抽樣）。"""
import sys, json, argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sc_lib as S

FOCI = ['AB', 'B10', 'EXEC', 'EXEC_J', 'AB_stress', 'B10_stress', 'EXEC_stress', 'B10_V5', 'EXEC_30F', 'EXEC_300F', 'EXEC_100G', 'EW']
COST_FOCI = {'EXEC', 'EXEC_J', 'EXEC_stress', 'EXEC_30F', 'EXEC_300F', 'EXEC_100G'}
FAMS = ['MAIN', 'ALL', 'LOCO', 'THEORY']
MODELS = ['G1', 'G2', 'G3']

def prior_summary(pr, model):
    th = S.theta_grid()
    if model == 'G1':
        m, sd = pr['mu'], math.sqrt(pr['tau2'])
        return {'mean': m, 'sd': sd, 'P_pos': float(1 - stats.norm.cdf(0, m, max(sd, 1e-9))), 'tau': sd, 'var_mu': pr['var_mu']}
    g = pr['g']; m = float((th * g).sum()); sd = float(math.sqrt(((th - m) ** 2 * g).sum()))
    c = np.cumsum(g)
    return {'mean': m, 'sd': sd, 'P_pos': float(g[th > 0].sum()),
            'q05': float(np.interp(0.05, c, th)), 'q50': float(np.interp(0.5, c, th)), 'q95': float(np.interp(0.95, c, th)),
            'q99': float(np.interp(0.99, c, th)), 'loglik': pr['loglik']}

def run(focus_path, meta_path, tag, check=True):
    if check: S.check_prereg()
    X, M = S.load_family()
    F = S.load_focus(focus_path)
    meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    sd_cost = float(meta['cost_uncertainty']['sd_cost_pp'])
    fam = json.loads((S.DATA / 'families.json').read_text(encoding='utf-8'))
    assert (F.index == X.index).all()
    foci = [c for c in FOCI if c in F.columns]
    fx = {c: ((F[c] - F['r0050']).values * 100) for c in foci}
    th = S.theta_grid()
    out = {'tag': tag, 'focus_file': str(focus_path), 'sd_cost_pp': sd_cost, 'raw': {}, 'families': {}, 'external': {}}
    for c in foci:
        out['raw'][c] = {'alpha_hat_pp': float(fx[c].mean()), 'se_hac_pp': float(math.sqrt(S.hac_var_mean(fx[c], 6))), 't_iid': S.tstat(fx[c])}
    dens = {}
    for fname in FAMS:
        cols = fam[fname]
        dec = S.decompose(X[cols], fx, mode='decomp')
        xs = dec['xhat']; s2 = dec['s2']
        assert xs.min() > th[0] + 1 and xs.max() < th[-1] - 1, '格點範圍不足'
        fo = {'n': len(cols), 'vP': dec['vP'], 'common_sd_pp': math.sqrt(dec['vP']), 'Keff_equiv': dec['Keff'],
              'xhat_mean': float(xs.mean()), 'xhat_sd': float(xs.std(ddof=1)), 'idio_se_median': float(np.median(np.sqrt(s2))),
              'priors': {}, 'post': {}}
        for model in MODELS:
            pr = S.fit_prior(model, xs, s2)
            fo['priors'][model] = prior_summary(pr, model)
            if model == 'G2': fo['priors'][model]['converged'] = pr['converged']
            if model == 'G3': fo['priors'][model]['iters'] = pr['iters']
            fo['post'][model] = {}
            for c in foci:
                s2f = dec['focus_s2'][c] + (sd_cost ** 2 if c in COST_FOCI else 0.0)
                p = S.posterior(pr, model, dec['focus_xhat'][c], s2f, dec['vP'])
                raw = dec['focus_xhat'][c]
                d = p.pop('dens')
                p.update({'alpha_hat_pp': raw, 'se_idio_pp': math.sqrt(s2f), 'shrink_frac': float(1 - p['mean'] / raw) if raw != 0 else None,
                          'post_over_raw': float(p['mean'] / raw) if raw != 0 else None})
                fo['post'][model][c] = p
                dens[f'{fname}|{model}|{c}'] = d
        out['families'][fname] = fo
    # 迭代 v1（誤差獨立、c0＝1）與 v2（拆解、c0＝1）的 MAIN×G2 事後（敏感度，揭露選版影響）
    out['sens_versions'] = {}
    for vname, mode, c0 in (('v1_naive_c0_1', 'naive', 1.0), ('v2_decomp_c0_1', 'decomp', 1.0)):
        dv = S.decompose(X[fam['MAIN']], fx, mode=mode)
        pr = S.fit_g2(dv['xhat'], dv['s2'], c0=c0)
        out['sens_versions'][vname] = {}
        for c in ['AB', 'B10', 'EXEC', 'EXEC_J']:
            s2f = dv['focus_s2'][c] + (sd_cost ** 2 if c in COST_FOCI else 0.0)
            p = S.posterior(pr, 'G2', dv['focus_xhat'][c], s2f, dv['vP']); p.pop('dens')
            out['sens_versions'][vname][c] = {'mean': p['mean'], 'p_pos': p['p_pos'], 'q10': p['q10'], 'q90': p['q90']}
    # 外部先驗（B 級，敏感度）
    dm = S.decompose(X[fam['MAIN']], fx, mode='decomp')
    for c in foci:
        a = float(fx[c].mean()); s2f = dm['focus_s2'][c] + dm['vP'] + (sd_cost ** 2 if c in COST_FOCI else 0.0)
        s = math.sqrt(s2f)
        mp_m, mp_s = 0.42 * a, s
        pm, ps = 0.375, 0.125
        fm = (pm / ps ** 2 + a / s2f) / (1 / ps ** 2 + 1 / s2f); fs = math.sqrt(1 / (1 / ps ** 2 + 1 / s2f))
        out['external'][c] = {'EXT_MP': {'mean': mp_m, 'sd': mp_s, 'p_pos': float(1 - stats.norm.cdf(0, mp_m, mp_s))},
                              'EXT_FUND': {'mean': fm, 'sd': fs, 'p_pos': float(1 - stats.norm.cdf(0, fm, fs))}}
        for k, (m_, s_) in (('EXT_MP', (mp_m, mp_s)), ('EXT_FUND', (fm, fs))):
            d = stats.norm.pdf(th, m_, s_); dens[f'{k}|N|{c}'] = d / d.sum()
        # 不收縮對照：FLAT＝N(α̂, s²)（含共同誤差與成本不確定）
        d = stats.norm.pdf(th, a, s); dens[f'FLAT|N|{c}'] = d / d.sum()
        out['external'][c]['FLAT'] = {'mean': a, 'sd': s, 'p_pos': float(1 - stats.norm.cdf(0, a, s))}
    # 預測對照與可證偽
    mp = out['families']['MAIN']['post']['G2']
    out['falsify_post_ge_80pct_raw'] = {c: bool(mp[c]['mean'] >= 0.8 * mp[c]['alpha_hat_pp']) for c in ['AB', 'B10', 'EXEC', 'EXEC_J'] if c in mp}
    out['decision_inputs'] = {'EXEC_J_P_pos_G2_MAIN': mp.get('EXEC_J', {}).get('p_pos'), 'EXEC_P_pos_G2_MAIN': mp.get('EXEC', {}).get('p_pos'),
                              'trigger_0050_primary': bool(mp['EXEC_J']['p_pos'] < 0.5) if 'EXEC_J' in mp else None}
    np.savez_compressed(S.DATA / f'post_dens_{tag}.npz', theta=th, **{k.replace('|', '__'): v for k, v in dens.items()})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--focus', default=str(S.DATA / 'focus_windows.parquet'))
    ap.add_argument('--meta', default=str(S.DATA / 'focus_meta.json'))
    ap.add_argument('--tag', default='V4')
    a = ap.parse_args()
    out = run(a.focus, a.meta, a.tag)
    S.save_key('RESULTS_POSTERIOR' + ('' if a.tag == 'V4' else '_' + a.tag), out)
    for fname in FAMS:
        fo = out['families'][fname]
        print(fname, 'n', fo['n'], 'common_sd %.3f' % fo['common_sd_pp'], {m: {k: round(v, 3) for k, v in fo['priors'][m].items() if isinstance(v, float)} for m in MODELS})
        for m in MODELS:
            print('  ', m, {c: (round(fo['post'][m][c]['mean'], 3), round(fo['post'][m][c]['p_pos'], 3)) for c in ['AB', 'B10', 'EXEC', 'EXEC_J']})
    print('external', {c: {k: (round(v['mean'], 3), round(v['p_pos'], 3)) for k, v in out['external'][c].items()} for c in ['AB', 'B10', 'EXEC', 'EXEC_J']})
    print(out['falsify_post_ge_80pct_raw'], out['decision_inputs'])

if __name__ == '__main__':
    main()

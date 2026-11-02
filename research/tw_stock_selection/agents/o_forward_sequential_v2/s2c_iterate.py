#!/usr/bin/env python3
"""步驟 2c：迭代（只用挑選期）。挑選期先驗（CV_SEL 選定版本、s、k，族群 SEL 窗）、σ_SEL、SEL 殘差；
在三種 DGP（D2＝SEL 殘差）下對各迭代版本跑 OC（每 DGP×α 20,000 條路徑、60 個月）。寫 iterations.json 與 results.json 的 ITER。
版本：
  I1  o7 原規則 E7（V5 先驗、e-process 門檻、σ 5.264；o_tier PREREG_FORWARD L1 100 萬 FIX）——基準
  I2  R1～R4，設計目標＝限制本身（誤升級 5%、檢定力 50%），σ_SEL，j 中位數線
  I3  R1～R4，緩衝目標（4%／55%），σ_SEL，j 中位數線
  I4  同 I3，σ_design＝1.15×σ_SEL
  I5  同 I3，決策線改官方日代理（J1 可執行 K5T 100 萬 FIX）
用法：python3 -I -B s2c_iterate.py"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q, prior as PR

P, HSIM, SEED = 20_000, 60, 20261103
DGPS = ['D1', 'D2', 'D3']


def sel_line(kind):
    M, _, _ = F.cell_jmatrix(*F.MAIN_CELL)
    s = F.jmed(M) if kind == 'jmed' else M[1].dropna()
    s.index = pd.PeriodIndex(s.index, freq='M')
    return s[F.per_mask(s.index, None, F.SEL_END)]


def oc_for(cfg, line, prior, th, sigma_rule):
    sig_dgp = F.hac_sd(line)
    resid = (line - line.mean()).to_numpy()
    par = dict(alpha_min=0.25, up_P=0.975, down_P=0.80)
    if cfg['rules'] == ['E7']:
        pf = json.loads((F.VEN / 'o_tier_results.json').read_text())['PREREG_FORWARD']
        L = pf['lines_L1_variants']['L1_MAIN_B10_EXEC@100萬|FIX']
        par.update(thr=pf['thresholds'], alpha_min_e=L['alpha_min'])
    else:
        par.update(Q.calibrate(sigma_rule, 36, cfg['target_up'], cfg['target_pow'], HSIM))
    # DGP 的 σ：D1／D3 用該線 SEL 的 HAC sd（真實雜訊），規則用 sigma_rule（可含安全係數）
    res = {}
    for dgp in DGPS:
        for a in Q.ALPHAS:
            rng = np.random.default_rng([SEED, int(round((a + 2) * 100)), ord(dgp[-1])])
            X = Q.gen(dgp, a, sig_dgp, P, HSIM, rng, resid=resid)
            for r in cfg['rules']:
                up, dn = Q.raw_triggers(r, X, sigma_rule, par, prior, th)
                u, d = Q.first_events(up, dn)
                res[(r, dgp, a)] = Q.oc_curve(u, d, list(range(12, HSIM + 1, 6)))
    summ = {}
    for r in cfg['rules']:
        ok, det = Q.check_constraints(res, r, DGPS)
        summ[r] = dict(pass_all=ok, worst_false_down_36=max(v['false_down_a0'] for v in det.values()),
                       worst_power_36=min(v['power_down_a_m1'] for v in det.values()), worst_false_up_36=max(v['false_up_a0'] for v in det.values()),
                       min_power_up_p1_36=min(v['power_up_a_p1'] for v in det.values()), by_dgp=det,
                       down_a_m1_24=min(res[(r, g, -1.0)][24]['down'] for g in DGPS), down_a_m1_60=min(res[(r, g, -1.0)][60]['down'] for g in DGPS),
                       false_up_a0_60=max(res[(r, g, 0.0)][60]['up'] for g in DGPS))
    return summ, {k: v for k, v in par.items() if k != 'thr'}, sig_dgp


def main():
    F.check_prereg('PREREG_DESIGN', files=False)
    res = F.load_res(); cv = res['CV_SEL']
    X, lab, med, anc = F.family()
    sel = F.per_mask(anc, None, F.SEL_END)
    lines = {'jmed': sel_line('jmed'), 'J1': sel_line('J1')}
    priors = {}
    for kind, ln in lines.items():
        b = PR.build(sel, cv['selected'], cv['s'], cv['k'], ln)
        priors[kind] = b
    zf = np.load(F.VEN / 'forward_priors.npz')
    o7th, o7pr = zf['theta'], zf['L1_MAIN_B10_EXEC_AT_100萬__FIX']
    sig = {k: F.hac_sd(v) for k, v in lines.items()}
    cfgs = {
        'I1_o7_E7': dict(rules=['E7'], line='J1', sigma_rule=5.2641713361224, prior='o7', note='o7 原規則；決策線以 J1 代理官方日'),
        'I2_exact': dict(rules=['R1', 'R2', 'R3', 'R4'], line='jmed', target_up=0.05, target_pow=0.50, sig_mult=1.0),
        'I3_buffer': dict(rules=['R1', 'R2', 'R3', 'R4'], line='jmed', target_up=0.04, target_pow=0.55, sig_mult=1.0),
        'I4_buffer_sig115': dict(rules=['R1', 'R2', 'R3', 'R4'], line='jmed', target_up=0.04, target_pow=0.55, sig_mult=1.15),
        'I5_buffer_official': dict(rules=['R1', 'R2', 'R3', 'R4'], line='J1', target_up=0.04, target_pow=0.55, sig_mult=1.0),
    }
    out = {'n_versions': len(cfgs), 'scope': '挑選期（SEL 錨定月 ≤ 2019-04）', 'cv_sel': {k: cv[k] for k in ('selected', 's', 'k', 'cov80_at_s_k')},
           'sigma_sel_hac': sig, 'priors_sel': {k: dict(raw=v['raw'], cal=v['cal'], ahat=v['ahat'], n_missing=v['n_missing_windows']) for k, v in priors.items()},
           'o7_prior_100FIX': F.T.dens_summary(o7pr, o7th), 'versions': {}}
    for nm, cfg in cfgs.items():
        t0 = time.time()
        line = lines[cfg['line']]
        if cfg.get('prior') == 'o7':
            pr, th, sr = o7pr, o7th, cfg['sigma_rule']
        else:
            pr, th, sr = priors[cfg['line']]['prior'], priors[cfg['line']]['th'], sig[cfg['line']] * cfg['sig_mult']
        summ, par, sdg = oc_for(cfg, line, pr, th, sr)
        out['versions'][nm] = dict(config={k: v for k, v in cfg.items()}, sigma_rule=sr, sigma_dgp=sdg, params=par,
                                   prior_P_pos=Q.prior_probs(pr, th, 0.25)[2], results=summ, runtime_s=time.time() - t0)
        print(nm, {r: (v['pass_all'], round(v['worst_power_36'], 3), round(v['worst_false_up_36'], 3), round(v['worst_false_down_36'], 3), round(v['min_power_up_p1_36'], 3)) for r, v in summ.items()}, flush=True)
    # 選版（挑選期）：有規則三 DGP 皆過的版本中，取最低誤下修；同分取較簡單（I3 優先於 I4）
    cand = []
    for nm, v in out['versions'].items():
        for r, s in v['results'].items():
            if s['pass_all']: cand.append((s['worst_false_down_36'], nm, r))
    cand.sort()
    out['candidates_passing'] = cand
    out['chosen'] = dict(version=cand[0][1], rule=cand[0][2], false_down=cand[0][0]) if cand else None
    out['utc'] = F.now_utc()
    (F.HERE / 'iterations.json').write_text(json.dumps(out, ensure_ascii=False, indent=1, default=F._js), encoding='utf-8')
    F.save_key('ITER', {'chosen': out['chosen'], 'candidates_passing': cand[:12], 'n_versions': len(cfgs)}, need=['CV_SEL'])
    print('chosen', out['chosen'])


if __name__ == '__main__':
    main()

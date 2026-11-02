#!/usr/bin/env python3
"""步驟 5（樣本外）：最終先驗、σ_design、門檻校準、三 DGP OC（每 DGP×α 40,000 條、60 個月）、選規則；不通過時找最短可行期限。
輸出：results.json 的 OC_FINAL、data/oc_final.csv、data/forward_prior_main.npz（theta、prior）。
用法：python3 -I -B s5_oc_final.py"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q, prior as PR

P, HSIM, SEED = 40_000, 60, 20261105
DGPS = ['D1', 'D2', 'D3']
RULES = ['R1', 'R2', 'R3', 'R4']
FALLBACK_H = [42, 48, 54, 60, 72, 84, 96, 120]


def decision_line(cost='normal', tier=None, mode=None):
    tier, mode = (tier, mode) if tier else F.MAIN_CELL
    M, _, _ = F.cell_jmatrix(tier, mode, cost=cost)
    s = F.jmed(M); s.index = pd.PeriodIndex(s.index, freq='M')
    return s


def sim_rules(par, sigma, prior, th, resid, rules, alphas, Hsim, P_, seed):
    res = {}
    for dgp in DGPS:
        for a in alphas:
            rng = np.random.default_rng([seed, int(round((a + 2) * 100)), ord(dgp[-1])])
            X = Q.gen(dgp, a, sigma, P_, Hsim, rng, resid=resid)
            for r in rules:
                up, dn = Q.raw_triggers(r, X, sigma, par, prior, th)
                u, d = Q.first_events(up, dn)
                res[(r, dgp, a)] = Q.oc_curve(u, d, list(range(12, Hsim + 1, 6)))
    return res


def main():
    PRE = F.check_prereg('PREREG')
    cfg = PRE['config']
    res0 = F.load_res(); cv = res0['CV_FINAL']
    X, lab, med, anc = F.family()
    line = decision_line()
    post = line[line.index >= F.VAL[0]]
    sig_full, sig_post = F.hac_sd(line), F.hac_sd(post)
    sigma = max(sig_full, sig_post) * cfg['sig_mult']
    b = PR.build(np.ones(len(anc), bool), cv['selected'], cv['s'], cv['k'], line)
    th, prior = b['th'], b['prior']
    np.savez(F.DATA / 'forward_prior_main.npz', theta=th, prior=prior, raw=b['raw_dens'])
    resid = (post - post.mean()).to_numpy()
    par = dict(alpha_min=cfg['alpha_min'], up_P=cfg['up_P'], down_P=cfg['down_P'])
    par.update(Q.calibrate(sigma, 36, cfg['target_up'], cfg['target_pow'], 120))
    t0 = time.time()
    res = sim_rules(par, sigma, prior, th, resid, RULES, Q.ALPHAS, HSIM, P, SEED)
    rows = Q.oc_table(res); pd.DataFrame(rows).to_csv(F.DATA / 'oc_final.csv', index=False)
    summ = {}
    for r in RULES:
        ok, det = Q.check_constraints(res, r, DGPS)
        summ[r] = dict(pass_all=ok, by_dgp=det, worst_false_down_36=max(v['false_down_a0'] for v in det.values()),
                       worst_power_36=min(v['power_down_a_m1'] for v in det.values()), worst_false_up_36=max(v['false_up_a0'] for v in det.values()),
                       power_up_p1_36=[v['power_up_a_p1'] for v in det.values()],
                       n_looks_60=len(Q.looks12(60)) if r == 'R4' else (49 if r == 'R3' else len(Q.looks6(60))))
    passing = sorted([(summ[r]['worst_false_down_36'], summ[r]['n_looks_60'], r) for r in RULES if summ[r]['pass_all']])
    chosen = passing[0][2] if passing else None
    out = dict(sigma_full=sig_full, sigma_post2019=sig_post, sigma_design=sigma, prior_raw=b['raw'], prior_cal=b['cal'], ahat_line=b['ahat'],
               prior_probs=dict(zip(['P_gt_alpha_min', 'P_lt0', 'P_gt0'], Q.prior_probs(prior, th, cfg['alpha_min']))),
               cv_used={k: cv[k] for k in ('selected', 's', 'k')}, params=par, P=P, Hsim=HSIM, summary=summ, passing=passing, chosen=chosen,
               resid_post_sd=float(np.std(resid, ddof=1)), resid_post_n=len(resid), runtime_s=time.time() - t0)
    # 誤下修 > 30% 或沒有規則通過 → 最短可行期限
    need_fb = (chosen is None) or (summ[chosen]['worst_false_down_36'] > 0.30)
    fb = {}
    if need_fb or True:            # 一律報告（PREREG：「在現有雜訊下要多少個月才能判定」）
        for Hc in FALLBACK_H:
            pr_ = dict(par); pr_.update(Q.calibrate(sigma, Hc, cfg['target_up'], cfg['target_pow'], 120))
            rr = sim_rules(pr_, sigma, prior, th, resid, ['R2', 'R4'], [-1.0, 0.0], 120, 20_000, SEED + Hc)
            fb[Hc] = {}
            for r in ('R2', 'R4'):
                pw = min(rr[(r, g, -1.0)][Hc]['down'] for g in DGPS)
                fu = max(rr[(r, g, 0.0)][Hc]['up'] for g in DGPS)
                fd = max(rr[(r, g, 0.0)][Hc]['down'] for g in DGPS)
                fb[Hc][r] = dict(power=pw, false_up=fu, false_down=fd, feasible=bool(pw >= 0.5 and fu <= 0.05 and fd <= 0.30),
                                 c=dict(c_u=pr_['c_u'], c_d=pr_['c_d'], c_u4=pr_['c_u4'], c_d4=pr_['c_d4']))
        feas = [Hc for Hc in FALLBACK_H if any(v['feasible'] for v in fb[Hc].values())]
        out['fallback'] = dict(needed=need_fb, by_H=fb, shortest_feasible_H=(36 if not need_fb else (feas[0] if feas else None)))
    # 低誤下修的參考：α＝0 時「誤下修 ≤ 10%」所需期限（同法，下修門檻改成 α＝0 時 10%，看 α＝−1 檢定力 ≥ 50% 的最短期限）
    out['utc'] = F.now_utc()
    F.save_key('OC_FINAL', out, need=['CV_FINAL'])
    for r in RULES:
        s = summ[r]; print(r, s['pass_all'], round(s['worst_power_36'], 3), round(s['worst_false_up_36'], 3), round(s['worst_false_down_36'], 3), [round(x, 3) for x in s['power_up_p1_36']])
    print('sigma', sig_full, sig_post, sigma, 'chosen', chosen, 'prior', b['cal'], out['prior_probs'])
    print('fallback', {h: {r: (round(v['power'], 3), round(v['false_up'], 3), round(v['false_down'], 3)) for r, v in d.items()} for h, d in fb.items()})


if __name__ == '__main__':
    main()

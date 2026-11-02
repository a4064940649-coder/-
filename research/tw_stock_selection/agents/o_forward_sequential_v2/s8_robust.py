#!/usr/bin/env python3
"""步驟 8（樣本外）：穩健性。
(a) OC 敏感度（選定規則；每格 20,000 條）：真實 sd＝0.8／1.2／1.5 × σ_design（規則仍用 σ_design）；D2 改用全期殘差；D2 區塊 3／12 個月。
(b) CV 選版穩健：CV_FINAL 只用 half、只用 roll、只用 medoid 目標時的最佳版本；各版本下主決策線的校準先驗 P(α>0)。
(c) 偽前瞻：去掉 2021 年各月後的 z_end；最大貢獻月（已在 s6）。
(d) 分產業：決策線是組合層級的月超額，不適用（照實寫）。
用法：python3 -I -B s8_robust.py"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q, prior as PR, cvlib as CV
from s5_oc_final import decision_line

DGPS = ['D1', 'D2', 'D3']


def oc_one(rule, par, sig_rule, sig_true, prior, th, dgp, resid, P=20_000, H=60, seed=7, block=6):
    out = {}
    for a in (-1.0, 0.0, 1.0):
        rng = np.random.default_rng([seed, int((a + 2) * 100)])
        if dgp == 'D2':
            X = Q.gen('D2', a, sig_true, P, H, rng, resid=resid, block=block)
        else:
            X = Q.gen(dgp, a, sig_true, P, H, rng)
        up, dn = Q.raw_triggers(rule, X, sig_rule, par, prior, th)
        u, d = Q.first_events(up, dn)
        out[a] = dict(up36=float(np.mean((u > 0) & (u <= 36))), down36=float(np.mean((d > 0) & (d <= 36))))
    return dict(power36=out[-1.0]['down36'], false_up36=out[0.0]['up36'], false_down36=out[0.0]['down36'], up_power36=out[1.0]['up36'])


def main():
    PRE = F.check_prereg('PREREG'); cfg = PRE['config']
    res = F.load_res(); oc = res['OC_FINAL']; rule = oc['chosen'] or 'R2'; par = dict(oc['params'])
    z = np.load(F.DATA / 'forward_prior_main.npz'); th, prior = z['theta'], z['prior']
    sig = oc['sigma_design']
    line = decision_line()
    post = line[line.index >= F.VAL[0]]
    out = {'rule': rule}
    sens = {}
    for m in (0.8, 1.2, 1.5):
        sens[f'D1_sd_x{m}'] = oc_one(rule, par, sig, sig * m, prior, th, 'D1', None)
        sens[f'D3_sd_x{m}'] = oc_one(rule, par, sig, sig * m, prior, th, 'D3', None)
    sens['D2_full_resid'] = oc_one(rule, par, sig, sig, prior, th, 'D2', (line - line.mean()).to_numpy())
    for bl in (3, 12):
        sens[f'D2_post_block{bl}'] = oc_one(rule, par, sig, sig, prior, th, 'D2', (post - post.mean()).to_numpy(), block=bl)
    # 其他規則在 sd×1.2 下（看選定規則是否特別脆弱）
    for r in ('R1', 'R2', 'R3', 'R4'):
        sens[f'{r}_D1_sd_x1.2'] = oc_one(r, par, sig, sig * 1.2, prior, th, 'D1', None)
    out['oc_sensitivity'] = sens
    # (b) CV 選版穩健
    D = pd.read_parquet(F.DATA / 'cv_final.parquet'); dens = np.load(F.DATA / 'cv_final_dens.npy')
    X, lab, med, anc = F.family()
    th0 = F.T.G_DEFAULT.th
    cvr = {}
    for nm, msk in (('half_only', D.split.str.startswith('half')), ('roll_only', D.split.str.startswith('roll'))):
        S, _, _, _ = CV.summarize(D[msk.to_numpy()].reset_index(drop=True), dens[msk.to_numpy()], th0, 'sel')
        cvr[nm] = dict(best=S.sum_logpd.idxmax(), sum_logpd=S.sum_logpd.round(2).to_dict(), slope=S.slope.round(3).to_dict())
    S, _, _, _ = CV.summarize(D, dens, th0, 'medoid')
    cvr['medoid_target'] = dict(best=S.sum_logpd.idxmax(), sum_logpd=S.sum_logpd.round(2).to_dict())
    cvf = res['CV_FINAL']
    pv = {}
    for v in F.H.VERSIONS:
        b = PR.build(np.ones(len(anc), bool), v, cvf['s'], cvf['k'], line)
        b0 = PR.build(np.ones(len(anc), bool), v, 1.0, 1.0, line)
        pv[v] = dict(cal_mean=b['cal']['mean'], cal_P=b['cal']['p_pos'], raw_mean=b0['raw']['mean'], raw_P=b0['raw']['p_pos'])
    cvr['prior_by_version_main_line'] = pv
    out['cv_robust'] = cvr
    # (c) 偽前瞻去掉 2021
    pf = res['PSEUDO_FORWARD']['VAL_2019-05']
    fwd = line[line.index >= F.VAL[0]]
    keep = np.array([p.year != 2021 for p in fwd.index])
    x = fwd.to_numpy()[keep]
    out['pseudo_drop2021'] = dict(n=int(keep.sum()), mean=float(x.mean()), z_end=float(x.sum() / (pf['sigma'] * math.sqrt(len(x)))))
    out['industry'] = '不適用：決策線是組合層級月超額；產業拆解屬選股題，本題不改選股'
    out['utc'] = F.now_utc()
    F.save_key('ROBUST', out, need=['PSEUDO_FORWARD'])
    for k, v in sens.items(): print(k, {a: round(b_, 3) for a, b_ in v.items()})
    print({k: v['best'] for k, v in cvr.items() if isinstance(v, dict) and 'best' in v})
    print({k: {a: round(b_, 3) for a, b_ in v.items()} for k, v in pv.items()})
    print(out['pseudo_drop2021'])


if __name__ == '__main__':
    main()

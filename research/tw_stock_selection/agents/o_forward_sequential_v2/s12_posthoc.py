#!/usr/bin/env python3
"""步驟 12（事後，看過 OC_FINAL 失敗之後才設計；不改判定，只給下一輪登記用）：
D2 失敗的來源：2019-05 起殘差 iid sd 大於 HAC 長期 sd（σ_design 低估短期雜訊），加上右尾。
試：(a) σ_design' ＝ max(HAC 全期, HAC 2019+, iid sd 2019+)；(b) target_up 3%、2.5%；組合後 R2、R4 在三 DGP 的 OC（20,000 條）。
用法：python3 -I -B s12_posthoc.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import fslib as F, seq2 as Q
from s5_oc_final import decision_line, sim_rules, DGPS


def main():
    res = F.load_res(); oc = res['OC_FINAL']; cfg = res['PREREG']['config']
    z = np.load(F.DATA / 'forward_prior_main.npz'); th, prior = z['theta'], z['prior']
    line = decision_line(); post = line[line.index >= F.VAL[0]]
    resid = (post - post.mean()).to_numpy()
    s_iid = float(post.std(ddof=1))
    sig_opts = {'hac': oc['sigma_design'], 'max_iid': max(oc['sigma_design'], s_iid)}
    out = {'iid_sd_post': s_iid, 'skew_post': float(pd.Series(resid).skew()), 'ar1_post': float(pd.Series(resid).autocorr(1)), 'grid': {}}
    for sn, sig in sig_opts.items():
        for tu in (0.04, 0.03, 0.025):
            par = dict(alpha_min=0.25, up_P=0.975, down_P=0.80); par.update(Q.calibrate(sig, 36, tu, 0.55, 60))
            rr = sim_rules(par, sig, prior, th, resid, ['R2', 'R4'], [-1.0, 0.0, 1.0], 60, 20_000, 777)
            for r in ('R2', 'R4'):
                ok, det = Q.check_constraints(rr, r, DGPS)
                key = f'{sn}|up{tu}|{r}'
                out['grid'][key] = dict(sigma=sig, pass_all=ok, worst_power=min(v['power_down_a_m1'] for v in det.values()),
                                        worst_false_up=max(v['false_up_a0'] for v in det.values()), worst_false_down=max(v['false_down_a0'] for v in det.values()),
                                        up_power_p1=min(v['power_up_a_p1'] for v in det.values()), by_dgp=det)
                g = out['grid'][key]
                print(key, ok, round(g['worst_power'], 3), round(g['worst_false_up'], 3), round(g['worst_false_down'], 3), round(g['up_power_p1'], 3), flush=True)
    out['note'] = '事後診斷：不改本題判定；可作為下一輪 PREREG 的候選設計（需新登記）'
    out['utc'] = F.now_utc()
    F.save_key('POSTHOC', out, need=['OC_FINAL'])


if __name__ == '__main__':
    main()
